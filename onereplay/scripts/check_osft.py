"""Preflight and correctness self-check for the OSFT baseline.

Run this once per environment before spending cluster time on an OSFT arm. It
answers the only question that matters about a vendored baseline: is the arm
labelled OSFT actually running OSFT, and is it running it on the same model the
other arms run on?

  1. env        python / torch / transformers, and whether osft_utils imports.
  2. commit     the clone is at the commit onereplay/core/osft.py was written for.
  3. patterns   upstream's path-substring lookup resolved the architecture we meant.
  4. rebuild    U S V^T reproduces each original weight. This is the decisive
                correctness test, and it is separate from the logit test below
                on purpose: it isolates the decomposition from the arithmetic of
                running it, so a real bug cannot hide behind dtype noise.
  5. parity     the decomposed model and the plain model agree at step 0, scored
                by next-token agreement and by loss rather than by raw logit
                distance -- see compare_models for why.
  6. frozen     after real optimizer steps the high-rank factors have not moved,
                and the trainable ones have.
  7. leak       U_low stays orthogonal to U_high and V_low to V_high, computed
                here rather than by calling the projection being tested.
  8. save       a reconstructed checkpoint reloads as a plain causal LM and
                behaves like the in-memory decomposed model, which is what
                onereplay.scripts.evaluate will read.
  9. degenerate --osft_unfreeze_rank_ratio 1.0 freezes nothing, so the arm
                collapses onto vanilla full fine-tuning.

Usage:
    python -m onereplay.scripts.check_osft \
        --model_dir /home/weiliu1/huggingface/models/ --model_name Qwen3-1.7B \
        --osft_unfreeze_rank_ratio 0.25

If check 4 or 5 fails in bf16, re-run with --use_bf16 0 before concluding
anything. In float32 the factored and dense paths agree to roughly 1e-5, so a
float32 failure is a real bug while a bf16-only failure is the storage precision
the method itself runs at.

Prefer the smallest checkpoint of the family you plan to train, not the 8B: the
load path holds the base model and the decomposed model at once, so peak host
memory is roughly twice the checkpoint before the SVD factors are even built.

For the glue-only checks that need neither GPU nor checkpoint, run
test_code/check_osft_integration.py instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from onereplay.core.modeling import join_model_path, set_seed  # noqa: E402

_results: list[tuple[str, bool, str]] = []

# Text rather than random token ids. On random ids the model is maximally
# uncertain, so next-token agreement is dominated by near-ties and reports noise
# instead of behavior. These are ordinary sentences the model has real opinions
# about, which is the regime the experiment runs in.
PROBE_TEXTS = [
    "The capital of France is Paris, and the capital of Japan is",
    "def add(a, b):\n    return a +",
    "Question: If a train travels 60 km in 2 hours, what is its average speed?\nAnswer:",
    "Summarize the following in one sentence: the quarterly report showed revenue growth",
]


def record(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSFT integration self-check")
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--osft_unfreeze_rank_ratio", type=float, default=0.25)
    parser.add_argument("--osft_target_patterns", type=str, default="")
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Optimizer steps to take before the frozen/leak checks.",
    )
    parser.add_argument(
        "--check_save",
        type=int,
        default=1,
        help=(
            "1 also writes a reconstructed checkpoint to a temporary directory and "
            "reloads it. Costs one checkpoint's worth of scratch space and a second "
            "model load."
        ),
    )
    parser.add_argument(
        "--min_argmax_agreement",
        type=float,
        default=0.0,
        help=(
            "Required next-token agreement for checks 5 and 8. 0 picks a default "
            "from the dtype: 0.99 in bf16, 0.999 otherwise."
        ),
    )
    parser.add_argument(
        "--max_loss_gap",
        type=float,
        default=0.0,
        help=(
            "Required agreement in cross-entropy, in nats, for checks 5 and 8. 0 "
            "picks a default from the dtype."
        ),
    )
    parser.add_argument(
        "--max_rebuild_error",
        type=float,
        default=0.0,
        help=(
            "Required relative accuracy of U S V^T against the original weight. 0 "
            "picks a default from the dtype: the factors are stored in the training "
            "dtype, so bf16 caps this at a few times 2^-8."
        ),
    )
    return parser.parse_args()


def thresholds(dtype: torch.dtype, args: argparse.Namespace) -> dict[str, float]:
    bf16 = dtype == torch.bfloat16
    return {
        "argmax": args.min_argmax_agreement or (0.99 if bf16 else 0.999),
        "loss_gap": args.max_loss_gap or (0.02 if bf16 else 1e-3),
        "rebuild": args.max_rebuild_error or (3e-2 if bf16 else 1e-5),
        "leak": 5e-2 if bf16 else 1e-4,
    }


def check_environment() -> bool:
    import transformers

    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}")
    print(f"transformers {transformers.__version__}")
    print(f"cuda         {torch.cuda.is_available()}")

    ok = True
    try:
        import rich  # noqa: F401
    except ImportError:
        ok = record(
            "env: rich installed",
            False,
            "mini_trainer.utils imports rich.logging; pip install rich",
        )
    try:
        import huggingface_hub  # noqa: F401
        from safetensors.torch import save_file  # noqa: F401
    except ImportError as error:
        ok = record("env: safetensors + huggingface_hub", False, str(error))
    if not torch.cuda.is_available():
        print(
            "\nnote: no GPU visible. Every check still runs, but the SVD takes far "
            "longer on CPU and nothing is learned about memory. Prefer a compute node.\n"
        )
    return ok


def check_upstream_import() -> bool:
    from onereplay.core import osft

    try:
        upstream = osft.load_upstream()
    except Exception as error:  # noqa: BLE001 - the message is the whole point
        return record("upstream: import mini_trainer.osft_utils", False, str(error))

    module = upstream["osft_utils"]
    required = [
        "create_osft_model_class",
        "optim_wrapper",
        "reconstruct_weight_matrix",
        "get_model_config",
        "_set_osft_dtypes",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        return record("upstream: expected symbols present", False, f"missing {missing}")
    return record(
        "upstream: import mini_trainer.osft_utils",
        True,
        f"from {upstream['source_dir']}",
    )


def check_commit() -> bool:
    """The glue was written against one commit; say so if the clone moved."""

    import subprocess

    from onereplay.core import osft

    repo = osft.upstream_source_dir().parent
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return record("commit: clone is a git checkout", False, str(error))

    if head != osft.UPSTREAM_COMMIT:
        return record(
            "commit: matches the one core/osft.py was written for",
            False,
            f"clone is at {head[:10]}, expected {osft.UPSTREAM_COMMIT[:10]}. Re-read "
            "core/osft.py's call order against their setup_model_for_training before "
            "trusting results, then update UPSTREAM_COMMIT.",
        )
    return record("commit: matches the one core/osft.py was written for", True, head[:10])


def check_patterns(model_path: str, raw_patterns: str) -> bool:
    from onereplay.core.osft import load_upstream, parse_target_patterns, resolved_target_patterns

    osft_utils = load_upstream()["osft_utils"]
    resolved = resolved_target_patterns(model_path, parse_target_patterns(raw_patterns))

    from transformers import AutoConfig

    model_type = str(getattr(AutoConfig.from_pretrained(model_path), "model_type", "")).lower()
    table_key = None
    for identifier, key in osft_utils.MODEL_NAME_MAPPINGS.items():
        if identifier in model_type:
            table_key = key
            break

    if raw_patterns.strip():
        return record("patterns: explicit list used", True, f"{resolved}")
    if table_key is None:
        return record(
            "patterns: architecture recognised",
            False,
            f"model_type {model_type!r} is not in upstream's table; pass "
            "--osft_target_patterns explicitly",
        )
    expected = osft_utils.MODEL_CONFIGS[table_key]["patterns"]
    if list(resolved) != list(expected):
        return record(
            "patterns: path lookup agrees with the architecture",
            False,
            f"model_type {model_type!r} implies {table_key!r} patterns {expected}, but the "
            f"path resolved to {resolved}. Their lookup scans the *path* and tests 'opt' "
            "before 'qwen', so a directory name collided. Pass --osft_target_patterns.",
        )
    return record("patterns: path lookup agrees with the architecture", True, f"{table_key}")


def load_plain(model_path: str, dtype, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    return model.to(device).eval()


def load_osft(model_path: str, args, dtype, device, unfreeze_rank_ratio: float):
    from onereplay.core.osft import build_osft_model, parse_target_patterns

    model = build_osft_model(
        model_path,
        unfreeze_rank_ratio=unfreeze_rank_ratio,
        target_patterns=parse_target_patterns(args.osft_target_patterns),
        torch_dtype=dtype,
    )
    return model.to(device).eval()


def compare_models(first, second, input_ids, attention_mask) -> dict[str, float]:
    """Score two models against each other on the same batch.

    Raw logit distance is the wrong yardstick here. The decomposed model stores
    U, S and V in the training dtype, so in bf16 the rebuilt weight differs from
    the original by a fraction of a percent per element, and the factored
    forward does four matmuls where the dense one does a single one. Both effects
    are inherent to the method as the authors implement it, and over a couple of
    dozen layers they move individual logits by whole units without changing what
    the model does.

    Next-token agreement and cross-entropy are what the experiment actually
    reads, so those are what is compared. The relative logit gap is reported
    alongside for context rather than gated on.
    """

    with torch.no_grad():
        logits_a = first(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        logits_b = second(input_ids=input_ids, attention_mask=attention_mask).logits.float()

    difference = (logits_a - logits_b).abs()
    scale = logits_a.abs().max().clamp_min(1e-6)
    supervised = attention_mask[:, 1:].bool()
    agreement = (
        (logits_a.argmax(-1) == logits_b.argmax(-1))[:, :-1][supervised].float().mean()
    )

    def cross_entropy(logits):
        flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
        targets = input_ids[:, 1:].reshape(-1).clone()
        targets[~supervised.reshape(-1)] = -100
        return float(torch.nn.functional.cross_entropy(flat_logits, targets, ignore_index=-100))

    loss_a = cross_entropy(logits_a)
    loss_b = cross_entropy(logits_b)
    return {
        "max_abs": float(difference.max()),
        "rel": float(difference.max() / scale),
        "argmax": float(agreement),
        "loss_a": loss_a,
        "loss_b": loss_b,
        "loss_gap": abs(loss_a - loss_b),
    }


def format_comparison(metrics: dict[str, float]) -> str:
    return (
        f"next-token agreement {metrics['argmax']:.4f}, "
        f"loss {metrics['loss_a']:.5f} vs {metrics['loss_b']:.5f} "
        f"(gap {metrics['loss_gap']:.2e}), "
        f"relative logit gap {metrics['rel']:.3e} "
        f"(max abs {metrics['max_abs']:.3g})"
    )


def gate_comparison(name: str, metrics: dict[str, float], limits: dict[str, float]) -> bool:
    ok = metrics["argmax"] >= limits["argmax"] and metrics["loss_gap"] <= limits["loss_gap"]
    detail = format_comparison(metrics)
    if not ok:
        detail += (
            f" | needed agreement >= {limits['argmax']} and loss gap <= {limits['loss_gap']}. "
            "Re-run with --use_bf16 0: passing there means this is the storage precision "
            "the method runs at, not an integration bug."
        )
    return record(name, ok, detail)


def check_weight_rebuild(osft_model, original: dict[str, torch.Tensor], limit: float) -> bool:
    """Does U S V^T reproduce the weight it was built from?

    The decisive correctness test, and the cheapest. It compares one matrix at a
    time with no forward pass, so nothing compounds across layers and nothing
    depends on the arithmetic of the factored linear. A real mistake in the
    decomposition -- a transpose, a wrong rank split, a lost singular value --
    shows up here at order 1, far above any dtype noise.
    """

    worst = 0.0
    worst_layer = ""
    checked = 0
    with torch.no_grad():
        for name, top_k in osft_model.osft_config.items():
            if top_k <= 0 or name not in original:
                continue
            rebuilt = osft_model._reconstruct_weight(
                name, upcast_dtype=torch.float32, output_dtype=torch.float32
            )
            reference = original[name].to(device=rebuilt.device, dtype=torch.float32)
            denominator = reference.abs().max().clamp_min(1e-12)
            relative = float((rebuilt - reference).abs().max() / denominator)
            checked += 1
            if relative > worst:
                worst, worst_layer = relative, name
    if checked == 0:
        return record("rebuild: U S V^T reproduces the original weights", False, "no layer checked")
    return record(
        "rebuild: U S V^T reproduces the original weights",
        worst <= limit,
        f"worst relative error {worst:.3e} at {worst_layer} over {checked} matrices "
        f"(limit {limit:g})",
    )


def check_buffers_match(plain, osft_model) -> bool:
    """Every buffer must survive the decomposition unchanged.

    Worth its own check because core/osft.py sets torch's default dtype while
    the OSFT wrapper is constructed, to stop their non-distributed loader from
    quietly widening a bf16 checkpoint to fp32. Non-persistent buffers -- rotary
    inv_freq above all -- are not in the state dict, so they are *created* under
    that default rather than copied. Qwen builds inv_freq with an explicit
    .float(), which makes it immune, but that is an implementation detail of one
    transformers version and a bf16 inv_freq would rotate every position
    slightly differently. That reads as a plausible-looking logit gap and would
    be blamed on dtype noise, which is exactly the trap this check exists for.
    """

    plain_buffers = dict(plain.named_buffers())
    mismatched: list[str] = []
    for name, tensor in osft_model.named_buffers():
        reference = plain_buffers.get(name)
        if reference is None:
            continue
        if tensor.dtype != reference.dtype:
            mismatched.append(f"{name}: {tensor.dtype} vs {reference.dtype}")
            continue
        if tensor.shape != reference.shape:
            mismatched.append(f"{name}: shape {tuple(tensor.shape)} vs {tuple(reference.shape)}")
            continue
        if tensor.is_floating_point():
            gap = float(
                (tensor.float() - reference.to(tensor.device).float()).abs().max()
            )
            if gap > 0.0:
                mismatched.append(f"{name}: max |d| {gap:.3g}")

    return record(
        "buffers: unchanged by the decomposition",
        not mismatched,
        f"{len(plain_buffers)} buffers compared"
        if not mismatched
        else f"{len(mismatched)} differ, e.g. {mismatched[:3]}",
    )


def take_steps(model, input_ids, attention_mask, steps: int, lr: float = 1e-5) -> None:
    from onereplay.core.osft import wrap_optimizer

    model.train()
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=lr
    )
    optimizer = wrap_optimizer(optimizer, model)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for _ in range(steps):
        optimizer.zero_grad()
        model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.backward()
        optimizer.step()
    model.eval()


def snapshot_factors(model) -> dict[str, torch.Tensor]:
    from onereplay.core.osft import _osft_modules

    snapshot: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for index, module in enumerate(_osft_modules(model)):
            snapshot[f"{index}.U_high"] = module.osft_U_high.detach().clone()
            snapshot[f"{index}.V_high"] = module.osft_V_high.detach().clone()
            snapshot[f"{index}.U_low"] = module.osft_params.U_low.detach().clone()
            snapshot[f"{index}.V_low"] = module.osft_params.V_low.detach().clone()
    return snapshot


def check_frozen_and_moved(model, before: dict[str, torch.Tensor]) -> bool:
    from onereplay.core.osft import _osft_modules

    frozen_drift = 0.0
    trainable_drift = 0.0
    with torch.no_grad():
        for index, module in enumerate(_osft_modules(model)):
            for field, tensor in (
                ("U_high", module.osft_U_high),
                ("V_high", module.osft_V_high),
                ("U_low", module.osft_params.U_low),
                ("V_low", module.osft_params.V_low),
            ):
                key = f"{index}.{field}"
                if key not in before:
                    continue
                drift = float((tensor.float() - before[key].float()).abs().max())
                if "_high" in field:
                    frozen_drift = max(frozen_drift, drift)
                else:
                    trainable_drift = max(trainable_drift, drift)

    ok = record(
        "frozen: high-rank factors unchanged after optimizer steps",
        frozen_drift == 0.0,
        f"max drift {frozen_drift:.4g}",
    )
    return (
        record(
            "trainable: low-rank factors moved",
            trainable_drift > 0.0,
            f"max drift {trainable_drift:.4g}",
        )
        and ok
    )


def check_leak(model, limit: float) -> bool:
    from onereplay.core.osft import max_frozen_subspace_leak

    leak = max_frozen_subspace_leak(model)
    return record(
        "leak: trainable factors stay orthogonal to the frozen subspace",
        leak <= limit,
        f"max overlap {leak:.4g} (budget {limit:g})",
    )


def check_save_roundtrip(model, input_ids, attention_mask, dtype, device, limits) -> bool:
    """The reconstructed checkpoint must behave like the model we just trained."""

    from onereplay.core.osft import save_osft_checkpoint

    directory = Path(tempfile.mkdtemp(prefix="osft_roundtrip_"))
    try:
        save_osft_checkpoint(model, str(directory))
        if not (directory / "config.json").is_file():
            return record("save: wrote config.json", False, str(directory))
        with open(directory / "config.json", encoding="utf-8") as handle:
            written = json.load(handle)
        architectures = written.get("architectures")
        if architectures and any("WithOSFT" in name for name in architectures):
            return record(
                "save: config.json names the base architecture",
                False,
                f"architectures={architectures}",
            )
        reloaded = load_plain(str(directory), dtype, device)
        metrics = compare_models(model, reloaded, input_ids, attention_mask)
        del reloaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return gate_comparison(
            "save: reconstructed checkpoint behaves like the trained model", metrics, limits
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def check_degenerate_full_ft(model_path: str, args, dtype, device) -> bool:
    """unfreeze_rank_ratio 1.0 has to leave nothing frozen.

    That is the setting under which OSFT is vanilla full fine-tuning, so it is
    the arm's own control: a run at 1.0 should reproduce the vanilla loss curve
    already in results_log. Here we only confirm the rank bookkeeping.

    This check is also what catches upstream's _build_osft_kwargs dropping a
    rank_ratio of 0.0 as falsy; core/osft.py bypasses that helper for exactly
    this reason, and if the bypass is ever removed this goes red again.
    """

    from onereplay.core.osft import describe_osft

    model = load_osft(model_path, args, dtype, device, unfreeze_rank_ratio=1.0)
    summary = describe_osft(model, 1.0)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record(
        "degenerate: unfreeze_rank_ratio 1.0 freezes nothing",
        summary["osft_rank_high_total"] == 0,
        f"frozen rank total {summary['osft_rank_high_total']} over "
        f"{summary['osft_matrices']} matrices",
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    model_path = join_model_path(args.model_dir, args.model_name)
    dtype = torch.bfloat16 if args.use_bf16 == 1 else torch.float32
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    limits = thresholds(dtype, args)
    print(f"checking OSFT against {model_path} on {device} in {dtype}\n", flush=True)

    if not check_environment():
        print("\nenvironment is not ready; fix the above before continuing")
        return 1
    if not check_upstream_import():
        return 1
    check_commit()
    if not Path(model_path).exists():
        record("model: checkpoint exists", False, f"{model_path} not found")
        return 1
    check_patterns(model_path, args.osft_target_patterns)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(PROBE_TEXTS, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    print("\nloading the plain model for the step-0 comparison", flush=True)
    plain = load_plain(model_path, dtype, device)

    print("\nloading the OSFT model", flush=True)
    osft_model = load_osft(model_path, args, dtype, device, args.osft_unfreeze_rank_ratio)

    # Kept on CPU so the two models and the reference weights do not have to be
    # resident on the GPU together.
    original = {
        name: tensor.detach().to("cpu")
        for name, tensor in plain.state_dict().items()
        if name in osft_model.osft_config
    }
    check_weight_rebuild(osft_model, original, limits["rebuild"])
    del original
    check_buffers_match(plain, osft_model)

    gate_comparison(
        "parity: decomposed model behaves like the plain model at step 0",
        compare_models(plain, osft_model, input_ids, attention_mask),
        limits,
    )
    del plain
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU memory with the OSFT model alone: "
              f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB", flush=True)

    print(f"\ntaking {args.steps} optimizer steps", flush=True)
    before = snapshot_factors(osft_model)
    take_steps(osft_model, input_ids, attention_mask, args.steps)
    check_frozen_and_moved(osft_model, before)
    del before
    check_leak(osft_model, limits["leak"])
    if torch.cuda.is_available():
        print(
            f"peak GPU memory through training steps: "
            f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB",
            flush=True,
        )

    if args.check_save == 1:
        print("\nround-tripping a reconstructed checkpoint", flush=True)
        check_save_roundtrip(osft_model, input_ids, attention_mask, dtype, device, limits)
    del osft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nchecking the degenerate setting", flush=True)
    check_degenerate_full_ft(model_path, args, dtype, device)

    failed = [name for name, ok, _ in _results if not ok]
    print("\n" + "=" * 72)
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
        print(
            "\nIf only the parity or save checks failed in bf16, re-run with "
            "--use_bf16 0 before concluding anything: a float32 pass means the gap is "
            "the factored arithmetic the method runs at, and the number to report is "
            "how far the step-0 loss moved. A float32 failure is an integration bug."
        )
        return 1
    print(
        "OSFT is wired correctly. The remaining end-to-end check is a training run at "
        "--osft_unfreeze_rank_ratio 1.0, whose loss curve must match the vanilla arm's."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
