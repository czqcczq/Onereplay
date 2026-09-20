"""OSFT baseline, executed out of the authors' own repository.

OSFT (Orthogonal Subspace Fine-Tuning) is the continual-learning baseline from
"Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual
Learning" (arXiv:2504.07097). It replaces every targeted 2D weight W with its
SVD U S V^T, freezes all but the smallest ``unfreeze_rank_ratio`` fraction of the
singular directions as old knowledge, trains only that remainder, and projects
both gradients and parameters back into the orthogonal complement of the frozen
subspace at every optimizer step. DeltaW is therefore confined by construction
rather than by a penalty -- which is what makes it the right sibling to compare
OneReplay and EWC against, and also why it is a full fine-tuning method that
cannot wear a LoRA adapter.

Every line that implements the method runs from the authors' repository, kept as
an unmodified git clone:

    upstream  https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer.git
    commit    fd5b552177bde9a551d30e73660d6a96205c97c0  (2026-09-12)
    license   Apache-2.0 (baseline/mini_trainer/LICENSE.md)

This module is glue. It does not reimplement, port or paraphrase any part of the
algorithm, so an OSFT number cannot be wrong because of a transcription mistake
on our side. What it actually does:

  * import their ``osft_utils`` without running their package ``__init__``,
  * hand our model path to their loader, in their documented call order, so the
    decomposition is theirs,
  * wrap our optimizer with their ``optim_wrapper``,
  * call their ``prepare_state_dict_for_save`` before writing a checkpoint.

The one upstream step we deliberately skip is ``align_model_and_tokenizer``;
see build_osft_model for why, and for the assertion that proves the skip is a
no-op on our checkpoints.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

UPSTREAM_URL = "https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer.git"
UPSTREAM_COMMIT = "fd5b552177bde9a551d30e73660d6a96205c97c0"

# Populated by load_upstream on first use. Holding the modules rather than the
# individual functions keeps every call site an explicit `upstream.osft_utils.x`,
# so a reader can tell at a glance which side of the boundary a symbol is on.
_UPSTREAM: dict[str, Any] | None = None


def upstream_source_dir() -> Path:
    """Directory to put on sys.path so that `import mini_trainer` resolves.

    Defaults to the clone inside this repository. MINI_TRAINER_SRC overrides it
    for the case where the package is installed or lives elsewhere on a cluster.
    """

    override = os.environ.get("MINI_TRAINER_SRC", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "baseline" / "mini_trainer" / "src"


def _register_namespace_only_package(package_dir: Path) -> bool:
    """Make `mini_trainer` importable without executing its __init__.py.

    Their __init__ eagerly imports the whole training stack -- typer, numba,
    aiofiles and `datasets>=5.0.0` -- none of which OSFT itself touches. Letting
    pip resolve those would upgrade the datasets and transformers already pinned
    into this environment, and every earlier arm's numbers were produced against
    those versions. So we register a namespace-only stand-in for the package and
    let the normal import machinery find the submodules underneath it from disk.

    Their source files are untouched by this: `mini_trainer.osft_utils` and the
    four siblings it imports are loaded from their own .py files, by Python, in
    the ordinary way. The only thing that does not run is the convenience
    re-export block in __init__.py, which OSFT never reads.

    Returns True when the stand-in was installed, False when something had
    already imported the real package and we left it alone.
    """

    if "mini_trainer" in sys.modules:
        return False

    spec = importlib.machinery.ModuleSpec("mini_trainer", loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(package_dir)]
    sys.modules["mini_trainer"] = package
    return True


def _install_gpt_oss_stub() -> None:
    """Stand in for transformers' gpt_oss module when this version lacks it.

    osft_utils imports GptOssForCausalLM at module scope, so on a transformers
    older than 4.55 the import fails before any OSFT code can run. The symbol is
    read in exactly one place -- the `is_gpt_oss` branch of their from_pretrained
    -- which a Qwen or Llama checkpoint never enters, so a placeholder that
    refuses to be constructed is safe and cannot silently change behavior.

    Upgrading transformers instead would move every arm's tokenizer and modeling
    code underneath the comparison, which is a far larger risk than this.
    """

    for name in ("transformers.models.gpt_oss", "transformers.models.gpt_oss.modeling_gpt_oss"):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module

    placeholder = sys.modules["transformers.models.gpt_oss.modeling_gpt_oss"]
    if not hasattr(placeholder, "GptOssForCausalLM"):

        class GptOssForCausalLM:  # noqa: D401 - placeholder, never instantiated
            """Absent in this transformers version; OSFT only names it for gpt-oss."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "GptOssForCausalLM is not available in this transformers version. "
                    "onereplay.core.osft installed a placeholder so that OSFT could be "
                    "imported for a non-gpt-oss model; loading an actual gpt-oss "
                    "checkpoint requires transformers>=4.55."
                )

        placeholder.GptOssForCausalLM = GptOssForCausalLM
    print(
        "OSFT: transformers has no models.gpt_oss; installed a placeholder for the "
        "one symbol osft_utils imports. Not reachable for Qwen/Llama checkpoints.",
        flush=True,
    )


def load_upstream() -> dict[str, Any]:
    """Import the authors' osft_utils and return it alongside their model helper.

    Cached, because the decomposition classes are created per call to
    create_osft_model_class and we want one module instance behind all of them.
    """

    global _UPSTREAM
    if _UPSTREAM is not None:
        return _UPSTREAM

    source_dir = upstream_source_dir()
    package_dir = source_dir / "mini_trainer"
    if not (package_dir / "osft_utils.py").is_file():
        raise FileNotFoundError(
            f"mini_trainer source not found at {source_dir}. Clone it to "
            "baseline/mini_trainer or point MINI_TRAINER_SRC at an existing copy:\n"
            f"  git clone {UPSTREAM_URL} baseline/mini_trainer\n"
            f"  git -C baseline/mini_trainer checkout {UPSTREAM_COMMIT}"
        )
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    _register_namespace_only_package(package_dir)

    try:
        osft_utils = importlib.import_module("mini_trainer.osft_utils")
    except ModuleNotFoundError as error:
        if "gpt_oss" not in str(error):
            raise
        _install_gpt_oss_stub()
        osft_utils = importlib.import_module("mini_trainer.osft_utils")

    mini_trainer_utils = importlib.import_module("mini_trainer.utils")
    _UPSTREAM = {
        "osft_utils": osft_utils,
        # Resolves e.g. Qwen3ForCausalLM from a checkpoint's config. Theirs
        # rather than ours so the class OSFT subclasses is the one they tested.
        "get_model_class_from_config": mini_trainer_utils.get_model_class_from_config,
        "source_dir": source_dir,
    }
    return _UPSTREAM


def resolved_target_patterns(model_path: str, target_patterns: list[str] | None) -> list[str]:
    """The layer-name patterns OSFT will decompose, resolved through their table.

    Worth resolving up front rather than letting from_pretrained do it silently,
    because their lookup is a substring scan over the *path* against an ordered
    table in which "opt" precedes "qwen". A checkpoint living under any directory
    containing "opt" -- /scratch/opt/models/Qwen3-8B, say -- would be decomposed
    with OPT's attention names, match nothing, and train as plain full
    fine-tuning while still being logged as OSFT. Returning the list lets the
    caller assert on it.
    """

    upstream = load_upstream()
    return list(
        upstream["osft_utils"].get_model_config(
            model_name_or_class=model_path,
            target_patterns=target_patterns,
        )
    )


def parse_target_patterns(raw: str) -> list[str] | None:
    """Split a comma-separated pattern list the way upstream's CLI does.

    Their train.py strips quotes and spaces before splitting, so a value copied
    out of one of their command lines lands on the same list here. Empty means
    "let their per-architecture table decide".
    """

    cleaned = raw.replace("'", "").replace('"', "").replace(" ", "")
    if not cleaned:
        return None
    return [item for item in cleaned.split(",") if item]


def frozen_rank_ratio(unfreeze_rank_ratio: float) -> float:
    """Convert the user-facing unfreeze ratio into the ratio their code wants.

    Their two names run in opposite directions and only one line connects them
    (mini_trainer/train.py: ``osft_rank_ratio = 1.0 - osft_unfreeze_rank_ratio``).
    ``unfreeze_rank_ratio`` is the fraction that *trains* -- the documented knob,
    0.25 in their README -- while ``rank_ratio`` reaching their
    auto_generate_target_osft_config is the fraction that stays *frozen*, since
    it becomes ``top_k = floor(min(shape) * rank_ratio)`` and the top-k singular
    directions are the frozen ones. Exposing the internal name would silently
    invert every sweep, so the flip lives here and nowhere else.
    """

    if not 0.0 <= unfreeze_rank_ratio <= 1.0:
        raise ValueError(
            f"--osft_unfreeze_rank_ratio must be in [0, 1]; got {unfreeze_rank_ratio}"
        )
    return 1.0 - unfreeze_rank_ratio


def build_osft_model(
    model_path: str,
    *,
    unfreeze_rank_ratio: float,
    target_patterns: list[str] | None = None,
    torch_dtype: torch.dtype | None = None,
    config: Any | None = None,
    upcast_dtype: torch.dtype = torch.float32,
    tokenizer: Any | None = None,
) -> nn.Module:
    """Load ``model_path`` as an OSFT model, following upstream's own call order.

    The four statements that matter are copied in sequence from their
    non-distributed branch in setup_model_for_training.load_osft_model:
    resolve the concrete class, subclass it, load with ``initialize_osft=False``,
    then decompose explicitly. Their comment explains the split -- "initialize
    outside from_pretrained for consistency" -- and the dtypes are set afterwards
    exactly as they set them.

    ``unfreeze_rank_ratio`` is their user-facing knob: the fraction of each
    matrix's singular directions that train. See frozen_rank_ratio for the
    inversion their code applies underneath.

    We do not call their ``align_model_and_tokenizer``. It resizes embeddings
    only when the tokenizer is larger than the checkpoint's vocab, then syncs
    pad/bos/eos onto the config. Our loader already sets pad_token_id, every
    other arm goes through that loader, and adding a second token-syncing step
    on this arm alone would be a second variable in the comparison. The
    assertions below prove the skip changes nothing for the checkpoint at hand
    rather than assuming it.
    """

    upstream = load_upstream()
    osft_utils = upstream["osft_utils"]
    rank_ratio = frozen_rank_ratio(unfreeze_rank_ratio)

    patterns = resolved_target_patterns(model_path, target_patterns)
    print(
        f"OSFT: unfreeze_rank_ratio={unfreeze_rank_ratio} -> frozen rank_ratio={rank_ratio}; "
        f"decomposing layers matching {patterns}",
        flush=True,
    )

    base_cls = upstream["get_model_class_from_config"](model_path)
    osft_cls = osft_utils.create_osft_model_class(base_cls)

    # rank_ratio and target_patterns go straight to their from_pretrained rather
    # than through their _build_osft_kwargs helper, which drops the value when it
    # is falsy:
    #
    #     if osft_rank_ratio:
    #         osft_kwargs["rank_ratio"] = osft_rank_ratio
    #
    # rank_ratio 0.0 is exactly the "freeze nothing" setting -- our
    # --osft_unfreeze_rank_ratio 1.0 control, and their own
    # --osft-unfreeze-rank-ratio 1.0 -- so it would silently fall back to
    # from_pretrained's default of 0.5 and half the model would stay frozen in
    # the run that is supposed to reproduce vanilla. Passing the keyword
    # directly uses the same public entry point with the value intact.
    osft_kwargs = {"rank_ratio": rank_ratio, "target_patterns": patterns}

    load_kwargs: dict[str, Any] = {}
    if config is not None:
        load_kwargs["config"] = config
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    # Their non-distributed loader builds the base model with torch_dtype, then
    # constructs the OSFT subclass with a plain `actual_osft_cls(config=config,
    # ...)` and copies the weights in with load_state_dict. That constructor is
    # ordinary nn.Module construction, so it honors torch.get_default_dtype()
    # rather than torch_dtype -- and load_state_dict casts *down* to the
    # destination, so a bf16 checkpoint would silently become an fp32 model.
    # Their own runs never hit this because torchrun takes the distributed branch,
    # which threads train_dtype through. Setting the default dtype for the
    # duration makes the non-distributed branch land on the same state: bf16
    # parameters, bf16 SVD factors, and the SVD itself still computed in fp32
    # because create_svd_dict upcasts internally.
    with _default_dtype(torch_dtype):
        model = osft_cls.from_pretrained(
            model_path,
            fsdp2_lazy_init=False,
            initialize_osft=False,
            **osft_kwargs,
            **load_kwargs,
        )
    _assert_parameter_dtype(model, torch_dtype)
    # Order follows upstream: decompose first, set the dtype attributes after.
    # One consequence worth knowing is that --osft_upcast_dtype only reaches the
    # reconstruction on save, not this SVD, which always runs at the __init__
    # default of fp32. Reordering would make our arm diverge from theirs.
    model.reinitialize_osft(decompose_existing_weights=True)
    osft_utils._set_osft_dtypes(model, upcast_dtype, torch_dtype or model.dtype)

    if not model.osft_config:
        raise RuntimeError(
            f"OSFT matched no weight matrices in {model_path} with patterns {patterns}. "
            "The run would be indistinguishable from vanilla full fine-tuning. Pass "
            "--osft_target_patterns explicitly."
        )
    _assert_no_tied_targets(model)
    _assert_token_ids_already_aligned(model, tokenizer)
    _report_decomposition(model, unfreeze_rank_ratio)
    # Recorded on the model so the base class name survives into the saved
    # config.json; save_pretrained would otherwise stamp "...WithOSFT" there.
    model._onereplay_base_class_name = base_cls.__name__
    _install_save_pretrained(model)
    return model


def _install_save_pretrained(model: nn.Module) -> None:
    """Point save_pretrained at the reconstruction path.

    The shared trainer ends a run with ``self.model.save_pretrained(save_path)``,
    which on a decomposed model would write SVD factors under a "...WithOSFT"
    architecture -- unreadable by our eval path. Rebinding the method here means
    the trainer, the eval loader and every PBS script stay as they are, and an
    OSFT model honors the same contract as every other arm's.
    """

    def save_pretrained(save_directory: str, **_: Any) -> None:
        save_osft_checkpoint(model, str(save_directory))

    model.save_pretrained = save_pretrained


def _assert_no_tied_targets(model: nn.Module) -> None:
    """Refuse to decompose a weight that is tied to another parameter.

    Live for the small Qwen2.5 and Qwen3 checkpoints, which set
    tie_word_embeddings=true so that lm_head.weight *is* embed_tokens.weight.
    Adding lm_head to --osft_target_patterns there would fail in two silent
    stages: the decomposition pops lm_head.weight and replaces the module's
    forward, so the output head and the input embedding stop being the same
    tensor and drift apart during training; then save writes a reconstructed
    lm_head.weight into a config that still says the weights are tied, and
    from_pretrained re-ties on load and throws the trained head away. The
    evaluated model would not be the trained one, with nothing in any log to say
    so.

    Upstream's own pattern tables never name lm_head, so this only triggers when
    --osft_target_patterns asks for it. Matching the penalty arms' coverage is
    not worth this failure mode; leave the head out and record the difference.
    """

    seen: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        seen.setdefault(parameter.data_ptr(), []).append(name)

    named = dict(model.named_parameters(remove_duplicate=False))
    for name, top_k in model.osft_config.items():
        if top_k <= 0 or name not in named:
            continue
        aliases = [other for other in seen.get(named[name].data_ptr(), []) if other != name]
        if aliases:
            raise RuntimeError(
                f"OSFT would decompose {name}, which shares storage with {aliases}. "
                "Decomposing one side of a tied weight breaks the tie during training "
                "and the saved checkpoint would be re-tied on load, discarding what was "
                f"trained. Drop it from --osft_target_patterns (tie_word_embeddings="
                f"{getattr(model.config, 'tie_word_embeddings', None)})."
            )


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype | None):
    """Run a block with torch's default dtype set, restoring it afterwards."""

    if dtype is None:
        yield
        return
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _assert_parameter_dtype(model: nn.Module, dtype: torch.dtype | None) -> None:
    """Fail if the loaded model is not in the dtype the run asked for.

    Cheap insurance against the load path quietly widening the model: an fp32
    Qwen3-8B plus fp32 SVD factors is roughly four times the resident size of the
    intended bf16 pair, which would show up as an out-of-memory error far from
    its cause, or worse, as a run that fits and is simply not the same arm.
    """

    if dtype is None:
        return
    wrong = {
        name: parameter.dtype
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    }
    if wrong:
        sample = list(wrong.items())[:3]
        raise RuntimeError(
            f"OSFT loaded {len(wrong)} parameters in a dtype other than {dtype}, e.g. {sample}. "
            "The run would not be comparable with the other arms."
        )


def _assert_token_ids_already_aligned(model: nn.Module, tokenizer: Any | None) -> None:
    """Fail if skipping upstream's tokenizer alignment would have changed anything.

    Their step would resize embeddings for an enlarged tokenizer and overwrite
    pad/bos/eos on the config. If none of that applies, skipping it is provably
    a no-op and this arm's model is byte-identical to what the other arms load.
    """

    if tokenizer is None:
        return
    config = model.config
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is not None and len(tokenizer) > vocab_size:
        raise RuntimeError(
            f"tokenizer has {len(tokenizer)} tokens but the checkpoint's vocab_size is "
            f"{vocab_size}. Upstream OSFT resizes embeddings here; our loader does not, "
            "so the OSFT arm would differ from the others by more than the decomposition."
        )
    # pad_token_id is left out on purpose: the shared loader assigns it from the
    # tokenizer immediately after this function returns, which is the same thing
    # upstream's alignment step would have done. bos/eos are the two the shared
    # loader never touches, so a disagreement there would be a real difference
    # between this arm and upstream's.
    for attribute in ("bos_token_id", "eos_token_id"):
        wanted = getattr(tokenizer, attribute, None)
        if wanted is None:
            continue
        current = getattr(config, attribute, None)
        # eos_token_id is a list on some chat checkpoints; a membership test is
        # the honest comparison there.
        if isinstance(current, (list, tuple)):
            if wanted in current:
                continue
        elif current == wanted:
            continue
        raise RuntimeError(
            f"config.{attribute}={current!r} disagrees with tokenizer.{attribute}={wanted!r}. "
            "Upstream OSFT syncs these onto the config and we skip that step, so the "
            "disagreement has to be resolved in the shared loader instead of here."
        )


def _report_decomposition(model: nn.Module, unfreeze_rank_ratio: float) -> None:
    """Print what the decomposition actually did, and what it costs.

    The frozen/trainable split is the method's only hyperparameter, so a run log
    that does not contain the realized ranks cannot be checked afterwards
    against the ratio it claims to have used. They also diverge by construction:
    top_k is floored per matrix and clamped to full_rank - 1, so the realized
    frozen share never exactly equals 1 - unfreeze_rank_ratio.
    """

    summary = describe_osft(model, unfreeze_rank_ratio)
    if summary["osft_matrices"] == 0:
        # top_k = floor(min(shape) * 0) = 0 makes every matrix fail their
        # is_osft_param test, so nothing is decomposed and nothing is frozen.
        # That is the correct degenerate behavior and it is what makes a 1.0 run
        # the control whose loss curve must match the vanilla arm's -- but it
        # would read as a silent failure in a log without saying so.
        print(
            "OSFT: no matrix was decomposed, so this run is plain full fine-tuning. "
            "Expected at --osft_unfreeze_rank_ratio 1.0, where it is the arm's own "
            "vanilla control; at any other ratio it means the patterns matched nothing.",
            flush=True,
        )
    print(
        f"OSFT: {summary['osft_matrices']} matrices decomposed, "
        f"frozen rank {summary['osft_rank_high_total']} / "
        f"{summary['osft_rank_total']} singular directions "
        f"({summary['osft_frozen_rank_share']:.3f} frozen), "
        f"factors hold {summary['osft_factor_memory_gb']:.2f} GiB vs "
        f"{summary['osft_dense_memory_gb']:.2f} GiB dense",
        flush=True,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"OSFT: trainable params: {trainable:,} || all params: {total:,} "
        f"|| trainable%: {100.0 * trainable / max(total, 1):.4f}",
        flush=True,
    )


def describe_osft(model: nn.Module, unfreeze_rank_ratio: float) -> dict[str, Any]:
    """Facts about the decomposition, for the run's metrics record."""

    matrices = 0
    rank_high = 0
    rank_total = 0
    factor_elements = 0
    dense_elements = 0
    for module in _osft_modules(model):
        matrices += 1
        high = int(module.osft_params.rank_high)
        low = int(module.osft_params.S_low.numel())
        rank_high += high
        rank_total += high + low
        for tensor in (
            module.osft_U_high,
            module.osft_S_high,
            module.osft_V_high,
            module.osft_params.U_low,
            module.osft_params.S_low,
            module.osft_params.V_low,
        ):
            factor_elements += tensor.numel()
        dense_elements += module.osft_U_high.shape[0] * module.osft_V_high.shape[1]

    element_size = 2 if next(model.parameters()).dtype == torch.bfloat16 else 4
    return {
        # Both directions, because the two names are one minus the other and a
        # table that records only one of them is ambiguous to a later reader.
        "osft_unfreeze_rank_ratio": unfreeze_rank_ratio,
        "osft_frozen_rank_ratio": frozen_rank_ratio(unfreeze_rank_ratio),
        "osft_matrices": matrices,
        "osft_rank_high_total": rank_high,
        "osft_rank_total": rank_total,
        "osft_frozen_rank_share": rank_high / max(rank_total, 1),
        "osft_factor_memory_gb": factor_elements * element_size / 1024**3,
        "osft_dense_memory_gb": dense_elements * element_size / 1024**3,
        "osft_upstream_commit": UPSTREAM_COMMIT,
    }


def _osft_modules(model: nn.Module):
    """Modules carrying a decomposition, found the way upstream finds them."""

    for module in model.modules():
        if (
            hasattr(module, "osft_params")
            and hasattr(module, "osft_U_high")
            and hasattr(module, "osft_S_high")
            and hasattr(module, "osft_V_high")
        ):
            yield module


def is_osft_model(model: nn.Module) -> bool:
    """True when the model went through the decomposition. Upstream's own test."""

    return bool(load_upstream()["osft_utils"].is_osft_model(model))


def wrap_optimizer(optimizer: torch.optim.Optimizer, model: nn.Module) -> torch.optim.Optimizer:
    """Install the two projections around optimizer.step, using their wrapper.

    Their optim_wrapper mutates the optimizer in place: project_gradients before
    the step so the moment estimates stay in the allowed subspace, and
    project_parameters after it because Adam's element-wise rescaling can rotate
    the update back out. Call this before any LR scheduler is constructed, so the
    scheduler's own step counter ends up outermost where torch expects it.
    """

    wrapped = load_upstream()["osft_utils"].optim_wrapper(optimizer, model)
    if not hasattr(model, "project_gradients"):
        raise RuntimeError(
            "optim_wrapper found no project_gradients on the model, so it returned the "
            "optimizer untouched and the subspace constraint would never be applied. "
            "The model was not built by build_osft_model."
        )
    return wrapped


def max_frozen_subspace_leak(model: nn.Module) -> float:
    """Largest component of a trainable factor inside the frozen subspace.

    The method's whole claim is that U_low stays orthogonal to U_high and V_low
    to V_high. This measures that directly, and is deliberately our own
    arithmetic rather than a call into theirs: a check that reuses the code it is
    checking cannot fail. Read it after a few optimizer steps; it should sit at
    the storage dtype's noise floor.

    The Gram products are formed one matrix at a time in at least float32, and in
    float64 when the factors themselves are float64. Measuring in a narrower type
    than the factors are stored in would report that type's epsilon and hide a
    real leak underneath it.
    """

    worst = 0.0
    with torch.no_grad():
        for module in _osft_modules(model):
            stored = module.osft_U_high.dtype
            measure_in = torch.float64 if stored == torch.float64 else torch.float32
            u_high = module.osft_U_high.detach().to(measure_in)
            v_high = module.osft_V_high.detach().to(measure_in)
            u_low = module.osft_params.U_low.detach().to(measure_in)
            v_low = module.osft_params.V_low.detach().to(measure_in)
            if u_high.numel() and u_low.numel():
                worst = max(worst, float((u_high.transpose(0, 1) @ u_low).abs().max()))
            if v_high.numel() and v_low.numel():
                worst = max(worst, float((v_low @ v_high.transpose(0, 1)).abs().max()))
    return worst


def _model_float_dtype(model: nn.Module) -> torch.dtype:
    """The dtype the model's floating-point parameters are actually in."""

    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.bfloat16


def _config_dtype_key(config_dict: dict[str, Any]) -> str:
    """Whichever key this transformers version stores the checkpoint dtype under.

    4.x writes ``torch_dtype``; 5.x renamed it to ``dtype``. Guessing wrong
    leaves a key that from_pretrained ignores, and the checkpoint would load at
    the default dtype instead of the one it was trained in. Prefer whatever
    ``config.to_dict()`` already emitted, and only fall back to a version test
    when it emitted neither.
    """

    if "dtype" in config_dict:
        return "dtype"
    if "torch_dtype" in config_dict:
        return "torch_dtype"
    import transformers

    major = int(str(transformers.__version__).split(".", 1)[0])
    return "dtype" if major >= 5 else "torch_dtype"


def save_osft_checkpoint(
    model: nn.Module,
    save_path: str,
    save_dtype: torch.dtype | None = None,
) -> None:
    """Write a plain HuggingFace checkpoint from the decomposed model.

    An OSFT model's state_dict holds SVD factors, not weights, and its class is
    named "...WithOSFT", so save_pretrained would produce something our eval
    path cannot read. Upstream's save_model has the same problem and solves it
    the same way: call prepare_state_dict_for_save to fold U S V back into a
    dense W under its original parameter name, cast to the save dtype, then shard
    with huggingface_hub and write config.json from config.to_dict(). We follow
    that, minus their distributed gather, because the reconstruction -- the only
    part where fidelity matters -- is their function.
    """

    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    directory = Path(save_path)
    directory.mkdir(parents=True, exist_ok=True)

    state_dict = model.prepare_state_dict_for_save(model.state_dict())

    # Read the dtype off the parameters rather than off the config. transformers
    # renamed the config field from torch_dtype to dtype at 5.0, so either name
    # may be absent, and what we want to write is what was actually trained.
    target_dtype = save_dtype or _model_float_dtype(model)
    cpu = torch.device("cpu")
    state_dict = {
        key: (value.to(dtype=target_dtype, device=cpu) if value.is_floating_point() else value.to(cpu))
        for key, value in state_dict.items()
    }

    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size="5GB",
    )
    for filename, tensor_names in split.filename_to_tensors.items():
        save_file(
            {name: state_dict[name] for name in tensor_names},
            str(directory / filename),
            metadata={"format": "pt"},
        )
    if split.is_sharded:
        with open(directory / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
            json.dump(
                {"metadata": split.metadata, "weight_map": split.tensor_to_filename},
                handle,
                indent=2,
                sort_keys=True,
            )

    config_dict = model.config.to_dict()
    # from_pretrained recorded the checkpoint's own architecture; the OSFT
    # subclass name must not leak into a checkpoint meant to load as a plain
    # causal LM.
    base_name = getattr(model, "_onereplay_base_class_name", None)
    if base_name:
        config_dict["architectures"] = [base_name]
    # Write back under whichever key this transformers version already used, so
    # the checkpoint is read the same way it would be for any other arm.
    config_dict[_config_dtype_key(config_dict)] = str(target_dtype).replace("torch.", "")
    with open(directory / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2, sort_keys=True)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(str(directory))

    print(
        f"OSFT: wrote a reconstructed full checkpoint to {save_path} "
        f"({len(state_dict)} tensors, {'sharded' if split.is_sharded else 'single file'}, "
        f"{target_dtype})",
        flush=True,
    )
