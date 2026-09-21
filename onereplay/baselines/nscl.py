"""Adam-NSCL baseline, executed out of the authors' own optimizer.

Adam-NSCL is the continual-learning baseline from "Training Networks in Null
Space of Feature Covariance for Continual Learning" (CVPR 2021 oral). After a
task is learned it estimates the uncentered feature covariance C = E[x x^T] of
every layer's *input*, takes the eigenvectors whose eigenvalues sit near the
bottom of that spectrum as an approximate null space, and multiplies every later
Adam update by the projector onto that subspace. DeltaW is therefore confined to
directions the old data barely occupies, so DeltaW x stays near zero for old x.

That makes it the hard-constraint sibling of our own penalty: OneReplay adds
lambda * tr(DeltaW C DeltaW^T) to the loss and lets the optimizer trade the two
terms off, while Adam-NSCL forces the same quadratic form toward zero by
construction. Same C, same coverage, no lambda.

The code that implements the method runs from the authors' repository, kept as
an unmodified git clone:

    upstream  https://github.com/ShipengWang/Adam-NSCL.git
    commit    a2f39b4273aa300739c460b33a5e8d2c674632b2  (2021-07-24)

This module is glue. It does not reimplement their optimizer: the eigen-
decomposition (Adam.get_eigens), the null-space selection and projector
construction (Adam.get_transforms) and the projected step (Adam.step) are all
called on their class, so an Adam-NSCL number cannot be wrong because of a
transcription mistake on our side.

Three things this module has to do that their code does not, each with its own
reason and its own assertion:

  * Feed C from our estimator instead of their forward hook. Their
    ``SVDAgentAvg.compute_cov`` averages over the batch dimension before the
    outer product and then calls ``torch.mm`` on the result, which cannot run at
    all on a transformer's (batch, seq, hidden) activations -- mm rejects the 3D
    tensor. Our scripts/collect_cov.py already accumulates x^T x per target
    Linear over every supervised token, which is the estimator the paper's
    equations describe, and reusing it means the OneReplay arm and this arm are
    weighted by the same matrix. Their sum is unnormalized and ours is divided
    by the token count, which changes nothing here: the selection rule compares
    eigenvalues against the smallest eigenvalue, and the projector is divided by
    its own norm, so both are invariant to a global rescaling of C.
  * Establish the precondition their ``p.grad is None`` guards assume. Both
    get_eigens and get_transforms skip parameters without a gradient, because
    upstream calls them *after* a task has been trained. We call them before the
    first step, where .grad is still None and every layer would be skipped
    silently -- producing a run that is plain full fine-tuning while the log says
    Adam-NSCL. See install_null_space_transforms.
  * Cast their projectors to the training dtype. The SVD needs fp32, the model
    is bf16, and ``torch.mm(update, transform)`` rejects the mismatch. The cast
    happens to their output after get_transforms returns, not inside it.

One upstream behavior deliberately kept even though it looks like a bug:
get_transforms stores ``basis @ basis.T / ||basis @ basis.T||_F`` rather than the
plain projector. For an orthogonal projector of rank k that Frobenius norm is
sqrt(k), so every update is additionally shrunk by 1/sqrt(k) and the realized
step size varies per layer with k. That is why upstream carries a separate
``svd_lr`` (5e-5 against a model_lr of 1e-4), why --nscl_svd_lr exists here, and
why describe_nscl puts the realized k in the run record: without k, the
effective learning rate of a finished run cannot be reconstructed.

Not applicable to LoRA. The guarantee is that the *accumulated* DeltaW lies in
the null space, which holds because every update from W0 onward is projected.
Under LoRA, DeltaW = B A with A randomly initialized and B zero; A's initial
draw is never projected and B is unconstrained, so B A x != 0 for old x however
the updates are projected. Making it hold would mean projecting A itself at
init, which is a different idea and could not be labelled Adam-NSCL. train.py
refuses the combination.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

UPSTREAM_URL = "https://github.com/ShipengWang/Adam-NSCL.git"
UPSTREAM_COMMIT = "a2f39b4273aa300739c460b33a5e8d2c674632b2"

# Populated by load_upstream on first use.
_UPSTREAM: dict[str, Any] | None = None

# The name their module is registered under in sys.modules. Deliberately not
# "optim": their package directory is called that, and putting their parent
# directory on sys.path would register a top-level `optim` for the whole
# process. Loading the one file by path avoids the question, which their layout
# makes cheap -- adam_svd.py imports only math, collections and torch.
_UPSTREAM_MODULE_NAME = "onereplay_upstream_adam_nscl_adam_svd"


def upstream_source_dir() -> Path:
    """Directory holding the authors' clone.

    Defaults to the clone inside this repository. ADAM_NSCL_SRC overrides it for
    a cluster that keeps the checkout elsewhere.
    """

    override = os.environ.get("ADAM_NSCL_SRC", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "baseline" / "Adam-NSCL"


def load_upstream() -> dict[str, Any]:
    """Import the authors' adam_svd module and return it.

    Cached, because re-executing the file would hand out a second Adam class
    that isinstance checks against the first would not recognize.
    """

    global _UPSTREAM
    if _UPSTREAM is not None:
        return _UPSTREAM

    source_dir = upstream_source_dir()
    module_path = source_dir / "optim" / "adam_svd.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Adam-NSCL source not found at {module_path}. Clone it to "
            "baseline/Adam-NSCL or point ADAM_NSCL_SRC at an existing copy:\n"
            f"  git clone {UPSTREAM_URL} baseline/Adam-NSCL\n"
            f"  git -C baseline/Adam-NSCL checkout {UPSTREAM_COMMIT}"
        )

    module = sys.modules.get(_UPSTREAM_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(_UPSTREAM_MODULE_NAME, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not build an import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        # Registered before exec so a failure part-way through does not leave a
        # half-initialized module behind for the next call to find.
        sys.modules[_UPSTREAM_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[_UPSTREAM_MODULE_NAME]
            raise

    _UPSTREAM = {"adam_svd": module, "source_dir": source_dir, "module_path": module_path}
    return _UPSTREAM


class _CovarianceView:
    """Hand upstream one covariance at a time, on the device the SVD runs on.

    ``Adam.get_eigens(fea_in)`` walks its own parameter groups and reads
    ``fea_in[p]`` for each. A plain dict would therefore need every C resident at
    once on that device, on top of the model, the Adam moments and the
    eigenvectors their loop is busy accumulating. Serving them through
    __getitem__ instead keeps the peak at one matrix, because the previous one is
    released before the next is materialized.

    Their loop is untouched: it still does exactly ``fea_in[p]``, which is the
    only operation this class has to support.
    """

    def __init__(
        self,
        matrices: dict[int, torch.Tensor],
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._matrices = matrices
        self._device = device
        self._dtype = dtype
        self._held: torch.Tensor | None = None

    def __getitem__(self, parameter: torch.nn.Parameter) -> torch.Tensor:
        matrix = self._matrices[id(parameter)]
        self._held = None
        self._held = matrix.to(device=self._device, dtype=self._dtype)
        return self._held

    def release(self) -> None:
        self._held = None


def covered_linear_modules(
    model: nn.Module,
    covariances: dict[str, torch.Tensor],
) -> list[tuple[str, nn.Linear, str]]:
    """Linear modules that have a covariance, as (module_name, module, cov_key).

    The covariance file decides this arm's coverage, exactly as it decides the
    penalty arms' coverage in snapshot_reference_weights, so the two are
    comparable by construction and --target_modules means the same thing on both.
    Name matching goes through the shared lookup_covariance, so a key written by
    collect_cov resolves here the way it resolves in the regularizer.

    Weights sharing storage are yielded once. Qwen3's small checkpoints tie
    lm_head to embed_tokens; neither is an nn.Linear that collect_cov targets
    today, but a parameter reaching the optimizer twice is an error torch raises
    far from its cause, and one reaching the same group twice would have its
    update projected twice per step.
    """

    from onereplay.core.regularizer import lookup_covariance

    covered: list[tuple[str, nn.Linear, str]] = []
    seen_storage: set[int] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        key, covariance = lookup_covariance(covariances, module_name)
        if covariance is None:
            continue
        pointer = module.weight.data_ptr()
        if pointer in seen_storage:
            continue
        seen_storage.add(pointer)
        covered.append((module_name, module, str(key)))
    return covered


def _assert_shapes_match(
    covered: list[tuple[str, nn.Linear, str]],
    covariances: dict[str, torch.Tensor],
) -> None:
    """Fail if any C is not d_in x d_in for the weight it will project.

    Their step computes ``update @ transform`` with update of shape
    (d_out, d_in), so the transform -- and therefore C -- has to be square in
    d_in. A mismatch is what a covariance collected for a different model or a
    different target set looks like, and torch would otherwise report it as a
    bare matmul shape error from inside their optimizer.
    """

    for module_name, module, key in covered:
        covariance = covariances[key]
        expected = module.weight.shape[1]
        if tuple(covariance.shape) != (expected, expected):
            raise ValueError(
                f"covariance '{key}' has shape {tuple(covariance.shape)} but "
                f"{module_name}.weight is {tuple(module.weight.shape)}, so the projector "
                f"would have to be {expected} x {expected}. The covariance file was "
                "collected for a different model or a different target set."
            )


def build_nscl_optimizer(
    model: nn.Module,
    covariances: dict[str, torch.Tensor],
    *,
    lr: float,
    svd_lr: float,
    thres: float,
) -> tuple[torch.optim.Optimizer, list[tuple[str, nn.Linear, str]]]:
    """Build the authors' Adam with the projected and unprojected groups split.

    Mirrors the split in their SVDAgent.init_model_optimizer: the feature
    extractor's weights get ``svd=True`` and their own learning rate, everything
    else -- their classifier head and batch-norm groups, our embeddings, norms
    and lm_head -- trains as ordinary Adam.

    Membership is decided by the covariance file rather than by a name regex,
    because a projected parameter without a projector is the one failure mode
    their step cannot survive: it reads ``self.transforms[p]`` for every member
    of an svd group as soon as *any* transform exists, so a single uncovered
    member is a KeyError mid-run. install_null_space_transforms asserts that the
    two sets coincide.

    Betas, eps and weight decay stay at their defaults, which are torch's
    defaults, which is what the other arms' ``torch.optim.Adam(params, lr=...)``
    uses. Weight decay in particular has to stay 0: their get_update applies it
    with an in-place ``grad.add_`` on .grad itself, so a non-zero value would
    also perturb the gradient every other diagnostic reads.
    """

    covered = covered_linear_modules(model, covariances)
    if not covered:
        raise ValueError(
            "Adam-NSCL matched no Linear module against the covariance file, so every "
            "update would be unprojected and the run would be plain full fine-tuning. "
            "Check that --nscl_cov_path was collected from this model with the same "
            "--target_modules."
        )
    _assert_shapes_match(covered, covariances)

    frozen = [name for name, module, _ in covered if not module.weight.requires_grad]
    if frozen:
        raise ValueError(
            f"{len(frozen)} covered weights do not require grad, e.g. {frozen[:3]}. "
            "Adam-NSCL projects the update of every covered weight, so a frozen one "
            "would be counted as protected while never moving. This arm requires "
            "--full_finetune 1."
        )

    projected = [module.weight for _, module, _ in covered]
    projected_ids = {id(parameter) for parameter in projected}
    ordinary = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in projected_ids
    ]

    upstream = load_upstream()
    optimizer = upstream["adam_svd"].Adam(
        [
            {"params": projected, "svd": True, "lr": svd_lr, "thres": thres},
            {"params": ordinary, "lr": lr},
        ],
        lr=lr,
    )
    print(
        f"Adam-NSCL: {len(projected)} weights projected at lr={svd_lr:g} "
        f"(thres={thres:g}), {len(ordinary)} tensors trained as ordinary Adam at "
        f"lr={lr:g}",
        flush=True,
    )
    return optimizer, covered


def install_null_space_transforms(
    optimizer: torch.optim.Optimizer,
    covered: list[tuple[str, nn.Linear, str]],
    covariances: dict[str, torch.Tensor],
    *,
    thres: float,
    svd_device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run their eigendecomposition and projector construction, then check it.

    Order follows their SVDAgent.update_optim_transforms: get_eigens over the
    covariances, then get_transforms. What surrounds those two calls is ours.

    Zero gradients go on first. Their two methods both ``continue`` past a
    parameter whose .grad is None, which is never the case at their call site --
    they run after a task has been trained -- and always the case at ours, which
    is before the first step. Without this, every layer is skipped, transforms
    stays empty, their step takes the ``len(self.transforms) > 0`` false branch,
    and the run is plain full fine-tuning under an Adam-NSCL label. The buffers
    are released again below, so the first real backward sees the state it would
    have seen anyway.

    The eigenvectors come off afterwards. get_eigens keeps a d_in x d_in fp32
    eigenvector matrix per covered layer alive in optimizer.eigens, which
    get_transforms consumes and nothing reads again in a single-stage run; they
    would otherwise outlive the projectors they produced, at the same size, for
    the whole length of training.
    """

    svd_params = [
        parameter
        for group in optimizer.param_groups
        if group.get("svd", False)
        for parameter in group["params"]
    ]
    if not svd_params:
        raise RuntimeError("the optimizer has no svd group; it was not built by this module")

    matrices = {id(module.weight): covariances[key] for _, module, key in covered}
    uncovered = [index for index, p in enumerate(svd_params) if id(p) not in matrices]
    if uncovered:
        raise RuntimeError(
            f"{len(uncovered)} parameters are in the projected group without a "
            "covariance. Their step reads transforms[p] for every member of an svd "
            "group, so this would be a KeyError on the first optimizer step."
        )

    device = svd_device if svd_device is not None else svd_params[0].device
    fea_in = _CovarianceView(matrices, device=device, dtype=torch.float32)

    touched = [parameter for parameter in svd_params if parameter.grad is None]
    for parameter in touched:
        parameter.grad = torch.zeros_like(parameter)
    try:
        optimizer.get_eigens(fea_in)
        optimizer.get_transforms()
    finally:
        fea_in.release()
        for parameter in touched:
            parameter.grad = None

    stats = _null_space_stats(optimizer, covered, thres)
    optimizer.eigens.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _assert_projectors_usable(optimizer, covered)
    _cast_transforms_to_parameter_dtype(optimizer, covered)
    return stats


def _transform_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    """Identities of the parameters that have a projector.

    By id rather than by ``p in optimizer.transforms``: the keys are tensors,
    and a dict lookup that misses the identity fast path falls back to ``==``,
    which on tensors returns an elementwise mask that bool() then refuses.
    """

    return {id(key) for key in optimizer.transforms}


def _assert_projectors_usable(
    optimizer: torch.optim.Optimizer,
    covered: list[tuple[str, nn.Linear, str]],
) -> None:
    """Fail unless every covered weight got a projector and every projector is finite.

    Two silent degradations land here. A parameter skipped by get_transforms
    keeps training unprojected while being counted as protected, and if all of
    them were skipped the run is vanilla full fine-tuning with an Adam-NSCL label
    on it. Neither is visible in a loss curve.

    The finiteness check covers a genuine edge of their selection rule. When no
    eigenvalue satisfies the threshold, the basis is d x 0, ``basis @ basis.T``
    is all zeros, and ``transform / torch.norm(transform)`` is 0/0 -- a NaN
    projector that turns the weights to NaN on the first step rather than
    raising. thres >= 1 always keeps the smallest eigenvalue and so cannot reach
    this, which is why train.py refuses a smaller one, but the guard costs
    nothing and the failure it prevents is expensive to diagnose.
    """

    present = _transform_ids(optimizer)
    absent = [name for name, module, _ in covered if id(module.weight) not in present]
    if absent:
        raise RuntimeError(
            f"{len(absent)} of {len(covered)} covered weights have no projector after "
            f"get_transforms, e.g. {absent[:3]}. They would train unprojected while "
            "being counted as protected."
        )

    degenerate = [
        name
        for name, module, _ in covered
        if not bool(torch.isfinite(optimizer.transforms[module.weight]).all())
    ]
    if degenerate:
        raise RuntimeError(
            f"{len(degenerate)} projectors are not finite, e.g. {degenerate[:3]}. This is "
            "get_transforms dividing an all-zero projector by its own zero norm, which "
            "happens when the threshold selects no eigenvector. Raise --nscl_thres."
        )


def _cast_transforms_to_parameter_dtype(
    optimizer: torch.optim.Optimizer,
    covered: list[tuple[str, nn.Linear, str]],
) -> None:
    """Put each projector in the dtype of the update it will multiply.

    get_transforms builds the projector out of the eigenvectors, which are fp32
    because the SVD is, while full fine-tuning keeps parameters -- and therefore
    updates -- in bf16. Their step does ``torch.mm(update, transforms[p])``,
    which rejects mixed dtypes rather than promoting.

    Casting down costs mantissa bits, so a bf16 run's DeltaW only lands in the
    null space to bf16 precision. That is the precision the weights themselves
    carry, and check_nscl measures the residual leak rather than assuming it is
    negligible.
    """

    for _, module, _ in covered:
        parameter = module.weight
        transform = optimizer.transforms[parameter]
        if transform.dtype != parameter.dtype:
            optimizer.transforms[parameter] = transform.to(dtype=parameter.dtype)


def _null_space_stats(
    optimizer: torch.optim.Optimizer,
    covered: list[tuple[str, nn.Linear, str]],
    thres: float,
) -> dict[str, Any]:
    """Per-layer facts about the kept subspace, read off their eigenvalues.

    Recomputing their selection rule here rather than scraping their print is
    deliberate: the run record has to carry the realized k for every layer,
    because the Frobenius normalization in get_transforms makes the effective
    step size a function of k and a finished run is otherwise uninterpretable.

    This is a *report*, not a check. It applies the same rule get_transforms
    applies, so it cannot catch an error in that rule; the independent test is
    check_nscl's leak measurement, which asks whether DeltaW actually ended up
    orthogonal to the dropped directions.
    """

    layers: list[dict[str, Any]] = []
    for name, module, _ in covered:
        eigen = optimizer.eigens.get(module.weight)
        if not eigen:
            continue
        # float64 because the ratio below spans the condition number, which runs
        # to 1e9 on the spectra their own logs report.
        values = eigen["eigen_value"].double()
        # torch.svd returns singular values in descending order, so [-1] is the
        # smallest and the rule keeps everything within thres of that floor.
        keep = values <= values[-1] * thres
        total = int(values.numel())
        kept = int(keep.sum())
        smallest = float(values[-1])
        layers.append(
            {
                "layer": name,
                "kept": kept,
                "total": total,
                "kept_share": kept / max(total, 1),
                # How much of the old data's energy the kept subspace still
                # carries. The method's premise is that this is ~0; a large value
                # means the "null space" is not one, and DeltaW is free to move
                # directions the old task actually uses.
                "energy_share": float(values[keep].sum() / values.sum()) if kept else 0.0,
                "condition": float(values[0] / smallest) if smallest > 0 else float("inf"),
            }
        )
    return {"layers": layers}


def describe_nscl(
    optimizer: torch.optim.Optimizer,
    covered: list[tuple[str, nn.Linear, str]],
    stats: dict[str, Any],
    *,
    thres: float,
    svd_lr: float,
) -> dict[str, Any]:
    """Facts about the projection, for the run's metrics record.

    nscl_kept_share is the one to read first. Near 0 the projected layers can
    barely move and the arm is close to a frozen backbone; near 1 almost nothing
    was dropped and it is close to vanilla. Neither endpoint shows up in a loss
    curve, and thres reaches both for spectra that differ only in conditioning,
    which is why this is recorded per run rather than inferred from the flag.
    """

    layers = stats["layers"]
    kept = sum(layer["kept"] for layer in layers)
    total = sum(layer["total"] for layer in layers)
    shares = [layer["kept_share"] for layer in layers] or [0.0]
    transform_bytes = sum(
        optimizer.transforms[module.weight].numel()
        * optimizer.transforms[module.weight].element_size()
        for _, module, _ in covered
    )
    return {
        "nscl_thres": thres,
        "nscl_svd_lr": svd_lr,
        "nscl_layers": len(covered),
        "nscl_kept_directions": kept,
        "nscl_total_directions": total,
        "nscl_kept_share": kept / max(total, 1),
        "nscl_kept_share_min": min(shares),
        "nscl_kept_share_max": max(shares),
        # The worst layer's retained energy. The paper's claim is that this is
        # negligible; if it is not, the constraint is not protecting what it says
        # it protects, and that has to be visible in the table rather than in a
        # log nobody kept.
        "nscl_energy_share_max": max((layer["energy_share"] for layer in layers), default=0.0),
        "nscl_transform_memory_gb": transform_bytes / 1024**3,
        "nscl_upstream_commit": UPSTREAM_COMMIT,
        # The kept fraction per module type, which is where the aggregate above
        # comes from and the only form of it that explains a result. On a
        # transformer the four distinct inputs inside a block have wildly
        # different spectra -- the two LayerNorm outputs are dominated by a few
        # outlier-feature directions while the attention output and the MLP
        # intermediate are far flatter -- so one global threshold routinely
        # freezes half the module types and leaves the other half unconstrained.
        # A single mean hides exactly that, and it is the thing a reader of the
        # table needs.
        "nscl_kept_share_by_module": _kept_share_by_module(layers),
        # Worst retained energy per module type, same reason. A layer whose kept
        # subspace still carries most of the old second moment is not protected,
        # however many directions it dropped.
        "nscl_energy_share_by_module": _energy_share_by_module(layers),
    }


def _by_module(layers: list[dict[str, Any]], field: str, reduce) -> dict[str, float]:
    """Group per-layer values by the last component of the layer name."""

    groups: dict[str, list[float]] = {}
    for layer in layers:
        groups.setdefault(str(layer["layer"]).rsplit(".", 1)[-1], []).append(float(layer[field]))
    return {module: reduce(values) for module, values in sorted(groups.items())}


def _kept_share_by_module(layers: list[dict[str, Any]]) -> dict[str, float]:
    """Median kept fraction per module type; the median, because the spread
    within one module type is small next to the spread between them."""

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    return _by_module(layers, "kept_share", median)


def _energy_share_by_module(layers: list[dict[str, Any]]) -> dict[str, float]:
    """Worst retained energy per module type."""

    return _by_module(layers, "energy_share", max)


def report_null_space(stats: dict[str, Any], record: dict[str, Any]) -> None:
    """Print the realized subspace, per layer and in aggregate."""

    for layer in stats["layers"]:
        print(
            f"  {layer['layer']}: kept {layer['kept']}/{layer['total']} "
            f"({layer['kept_share']:.4f}), energy {layer['energy_share']:.3e}, "
            f"cond {layer['condition']:.3e}",
            flush=True,
        )
    print(
        f"Adam-NSCL: kept {record['nscl_kept_directions']}/"
        f"{record['nscl_total_directions']} directions "
        f"({record['nscl_kept_share']:.4f}; per-layer "
        f"{record['nscl_kept_share_min']:.4f}..{record['nscl_kept_share_max']:.4f}), "
        f"worst-layer retained energy {record['nscl_energy_share_max']:.3e}, "
        f"projectors hold {record['nscl_transform_memory_gb']:.2f} GiB",
        flush=True,
    )
    print("  by module type (median kept / worst retained energy):", flush=True)
    for module, share in record["nscl_kept_share_by_module"].items():
        energy = record["nscl_energy_share_by_module"][module]
        note = ""
        if share < 0.01:
            note = "  <- frozen"
        elif energy > 0.1:
            # Dropping directions is not protection if the ones kept are where
            # the old data lives. This is the reading that a high kept_share
            # alone does not give.
            note = "  <- unconstrained: the kept subspace holds the old energy"
        print(f"    {module:<12} {share:.4f}  {energy:.3e}{note}", flush=True)

    if record["nscl_kept_share_max"] < 0.01:
        print(
            "Adam-NSCL: every layer kept under 1% of its directions, so the projected "
            "weights have almost no room to move and this run will look like a frozen "
            "backbone. Raise --nscl_thres.",
            flush=True,
        )
    elif record["nscl_kept_share_min"] > 0.99:
        print(
            "Adam-NSCL: almost nothing was dropped anywhere, so this run is close to "
            "plain full fine-tuning. Lower --nscl_thres.",
            flush=True,
        )


def step_resolution(
    covered: list[tuple[str, nn.Linear, str]],
    stats: dict[str, Any],
    *,
    svd_lr: float,
) -> dict[str, Any]:
    """Can the weight dtype represent the step this arm is about to take?

    This is the one failure mode that is specific to running the method in bf16,
    and it is not the projector's precision. Their step ends in
    ``p.data.add_(update)``, so the rounding error of the accumulation is
    relative to |W| rather than to |update|. Adam's per-element step is about
    svd_lr, and the Frobenius normalization divides it by a further sqrt(k), so
    once ``svd_lr / sqrt(k)`` falls below the spacing of the weight dtype at the
    weight's own magnitude, most steps round away entirely and the ones that
    survive round to a whole unit in the last place -- in an arbitrary direction,
    which is exactly the isotropic noise the null space is supposed to exclude.

    Measured on the synthetic check: in float32 the realized DeltaW leaks 1e-6
    of its energy outside the kept subspace, and a projector rounded to bfloat16
    only takes that to 1e-3, but bfloat16 *weights* take it to 7e-2 at a step
    ratio near 1 and to 7e-1 once the ratio falls to 1e-3. The guarantee is lost
    to the accumulation, not to the projection.

    Returns the ratio and the numbers behind it; the caller decides how loud to
    be. A ratio comfortably above 1 means the step is resolvable.
    """

    worst: dict[str, Any] | None = None
    by_layer = {layer["layer"]: layer for layer in stats["layers"]}
    for name, module, _ in covered:
        layer = by_layer.get(name)
        if layer is None or layer["kept"] == 0:
            continue
        weight = module.weight
        spacing = float(weight.detach().abs().double().median()) * float(
            torch.finfo(weight.dtype).eps
        )
        step = svd_lr / layer["kept"] ** 0.5
        ratio = step / spacing if spacing > 0 else float("inf")
        if worst is None or ratio < worst["ratio"]:
            worst = {
                "layer": name,
                "ratio": ratio,
                "step": step,
                "spacing": spacing,
                "kept": layer["kept"],
                "dtype": str(weight.dtype),
            }
    return worst or {"ratio": float("inf")}


def report_step_resolution(resolution: dict[str, Any]) -> None:
    """Print the step-resolution finding, loudly when the step is unresolvable."""

    if resolution["ratio"] == float("inf"):
        return
    message = (
        f"Adam-NSCL: tightest layer {resolution['layer']} takes steps of "
        f"{resolution['step']:.2e} against a {resolution['dtype']} spacing of "
        f"{resolution['spacing']:.2e} (ratio {resolution['ratio']:.1f}, k="
        f"{resolution['kept']})"
    )
    print(message, flush=True)
    if resolution["ratio"] >= 10.0:
        return
    print(
        "Adam-NSCL: that ratio is too low for the null-space guarantee to survive. "
        "The update is added to the weight in its storage dtype, so a step near or "
        "below the spacing is mostly rounding, and rounding is isotropic -- DeltaW "
        "picks up exactly the directions the projection removed. Either raise "
        "--nscl_svd_lr or run this arm with --use_bf16 0; a leak measured on the "
        "finished checkpoint is the way to tell which worked.",
        flush=True,
    )


def idempotent_projector(stored: torch.Tensor) -> torch.Tensor:
    """Recover the true projector P from the normalized one their code stores.

    get_transforms stores ``Q = P / ||P||_F`` for an orthogonal projector P of
    rank k. Then ``||P||_F = sqrt(trace(P^T P)) = sqrt(trace(P)) = sqrt(k)``, so
    ``trace(Q) = k / sqrt(k) = sqrt(k)`` and ``P = Q * trace(Q)``.

    Reading the scale off the trace rather than recomputing a spectral norm
    keeps this O(d) instead of O(d^3), and it does not need k to be passed in
    from wherever the selection happened.
    """

    return stored * float(stored.diagonal().sum())


def max_null_space_leak(
    model: nn.Module,
    reference_weights: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    covered: list[tuple[str, nn.Linear, str]],
) -> float:
    """Largest share of DeltaW's energy lying outside the kept subspace.

    The method's whole claim is that the accumulated DeltaW = W - W0 lives in the
    span of the kept eigenvectors. This measures that directly, from the weights
    themselves, and is deliberately our own arithmetic rather than a call into
    theirs: a check that reuses the code it is checking cannot fail.

    Returns ``max_l ||DeltaW_l - DeltaW_l P_l||_F / ||DeltaW_l||_F``, which is 0
    for a perfectly constrained update and 1 for an unconstrained one. Computed
    in at least float32 -- float64 when the weights are -- because measuring in
    bf16 would report bf16's epsilon and hide a real leak beneath it.
    """

    worst = 0.0
    present = _transform_ids(optimizer)
    with torch.no_grad():
        for name, module, _ in covered:
            parameter = module.weight
            if name not in reference_weights or id(parameter) not in present:
                continue
            measure_in = torch.float64 if parameter.dtype == torch.float64 else torch.float32
            delta = (
                parameter.detach() - reference_weights[name].to(parameter.device)
            ).to(measure_in)
            norm = float(delta.norm())
            if norm == 0.0:
                continue
            projector = idempotent_projector(optimizer.transforms[parameter].to(measure_in))
            residual = float((delta - delta @ projector).norm())
            worst = max(worst, residual / norm)
    return worst
