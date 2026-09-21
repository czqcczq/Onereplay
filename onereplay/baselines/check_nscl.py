"""Preflight and correctness self-check for the Adam-NSCL baseline.

Run this once per environment before spending cluster time on an NSCL arm. It
answers the only question that matters about a vendored baseline: is the arm
labelled Adam-NSCL actually running Adam-NSCL?

Everything here is synthetic and runs on CPU in a few seconds. That is a
deliberate choice rather than a shortcut: the property being tested is a
statement about linear algebra -- DeltaW stays in the span of the kept
eigenvectors -- and on a covariance built with a *known* exact null space it can
be checked against ground truth instead of against a tolerance pulled out of the
air. A real checkpoint would add nothing except the parts already covered by
train.py's own assertions.

  1. upstream    the clone is present, at the pinned commit, and adam_svd imports.
  2. adam        with svd off, their optimizer is Adam up to where eps is added,
                 shown by shrinking eps and watching the gap shrink with it.
                 Establishes that any later difference comes from the projection
                 and not from a different optimizer.
  3. selection   on a covariance with an exact null space, the rule recovers
                 exactly that subspace once thres clears the numerical floor --
                 and does not, at upstream's own default of 1.001.
  4. projector   the stored transform is a true projector divided by its
                 Frobenius norm, and trace recovery inverts that exactly.
  5. constraint  after real optimizer steps, DeltaW's energy outside the kept
                 subspace is at the dtype's noise floor, and tr(DeltaW C DeltaW^T)
                 is driven to zero while DeltaW itself is not -- the model moved,
                 just not where the old data lives.
  6. teeth       the same run without the projection leaks badly, so check 5 is
                 not passing for want of anything to detect.
  7. gradients   the .grad precondition is real: without it their own methods
                 silently select nothing, and our path establishes it.
  8. degenerate  a threshold that selects no eigenvector produces upstream's
                 0/0 NaN projector, and the guard catches it instead of letting
                 NaN weights reach the checkpoint.
  9. coverage    a covariance that misses a projected layer raises at setup
                 rather than as a KeyError on the first optimizer step.
 10. precision   which dtype the guarantee survives. float32 is exact; a
                 bf16-rounded projector costs little; bf16 *weights* cost orders
                 of magnitude, because the update is accumulated into the weight
                 in its storage dtype and the rounding that follows is
                 isotropic. Read this one before configuring a real run.

Usage:
    python -m onereplay.baselines.check_nscl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from onereplay.baselines import nscl  # noqa: E402

_results: list[tuple[str, bool, str]] = []

D_IN = 32
D_OUT = 16
# Rank of the synthetic covariance; D_IN - OCCUPIED_RANK directions are exactly
# unoccupied and are the ground truth check 3 compares against.
OCCUPIED_RANK = 20
NULL_RANK = D_IN - OCCUPIED_RANK


def record(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


class TinyNet(nn.Module):
    """One covered projection plus one uncovered head.

    The head is there so the unprojected parameter group is non-empty, which is
    the shape every real run has: embeddings, norms and lm_head train as
    ordinary Adam alongside the projected weights.
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.proj = nn.Linear(D_IN, D_OUT, bias=False)
        self.head = nn.Linear(D_OUT, 4, bias=False)
        self.to(dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.proj(inputs))


def build_covariance(seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A PSD covariance of exactly OCCUPIED_RANK, so its null space is known."""

    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(D_IN, D_IN, generator=generator))[0]
    occupied = basis[:, :OCCUPIED_RANK]
    scale = torch.diag(torch.rand(OCCUPIED_RANK, generator=generator) * 9 + 1)
    covariance = occupied @ scale @ occupied.T
    return ((covariance + covariance.T) / 2).to(dtype)


def train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: int,
    seed: int = 7,
) -> None:
    """A few real optimizer steps on an arbitrary objective.

    What the loss is does not matter -- the claim under test is about where the
    update is allowed to point, not about what it is trying to fit -- but it has
    to be a real backward so the gradients are not the zeros the setup used.
    """

    generator = torch.Generator().manual_seed(seed)
    dtype = next(model.parameters()).dtype
    for _ in range(steps):
        inputs = torch.randn(8, D_IN, generator=generator).to(dtype)
        loss = model(inputs).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_upstream(expect_commit: bool) -> None:
    source_dir = nscl.upstream_source_dir()
    module_path = source_dir / "optim" / "adam_svd.py"
    if not module_path.is_file():
        record("upstream: clone present", False, f"missing {module_path}")
        return
    record("upstream: clone present", True, str(source_dir))

    if expect_commit:
        try:
            head = subprocess.run(
                ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            record("upstream: commit matches the pin", False, f"could not read HEAD: {error}")
        else:
            ok = head == nscl.UPSTREAM_COMMIT
            record(
                "upstream: commit matches the pin",
                ok,
                head[:10] if ok else f"clone at {head[:10]}, nscl.py written for "
                f"{nscl.UPSTREAM_COMMIT[:10]}",
            )

    try:
        upstream = nscl.load_upstream()
    except Exception as error:  # noqa: BLE001 - the point is to report any import failure
        record("upstream: adam_svd imports", False, str(error))
        return
    record("upstream: adam_svd imports", True, type(upstream["adam_svd"].Adam).__name__)


def check_adam_without_projection() -> None:
    """With svd off, their optimizer has to be Adam.

    If it were not, every difference measured later would be confounded: a gap
    between the NSCL arm and the vanilla arm could be the projection or could be
    a different optimizer, and nothing downstream distinguishes those.

    They are not bit-identical, and the reason is worth pinning down rather than
    absorbing into a loose tolerance. Their eps sits outside the second-moment
    bias correction (``sqrt(v) + eps``) while torch's sits inside it
    (``sqrt(v)/sqrt(bc2) + eps``) -- the eps-placement difference that separates
    several Adam implementations. At step 1, bias_correction2 is 1e-3, so the two
    denominators differ by about eps, which is a large *relative* perturbation
    while sqrt(v) is still small.

    So the check is an attribution, not a bound: shrink eps by eight orders of
    magnitude and the disagreement has to shrink with it. If it does, the only
    difference between the two optimizers is where eps is added, and the
    projection is the only thing that separates this arm from the vanilla one.
    """

    upstream = nscl.load_upstream()

    def disagreement(eps: float) -> float:
        torch.manual_seed(3)
        reference = TinyNet()
        theirs = TinyNet()
        theirs.load_state_dict(reference.state_dict())
        train_steps(
            reference, torch.optim.Adam(reference.parameters(), lr=1e-2, eps=eps), steps=5, seed=11
        )
        train_steps(
            theirs,
            upstream["adam_svd"].Adam(theirs.parameters(), lr=1e-2, eps=eps),
            steps=5,
            seed=11,
        )
        worst = 0.0
        for left, right in zip(reference.parameters(), theirs.parameters()):
            scale = float(left.detach().norm()) or 1.0
            worst = max(worst, float((left.detach() - right.detach()).norm()) / scale)
        return worst

    default_eps = disagreement(1e-8)
    tiny_eps = disagreement(1e-16)
    record(
        "adam: svd off differs from torch only in where eps is added",
        tiny_eps < 1e-7 and tiny_eps < default_eps / 100,
        f"relative weight difference {default_eps:.2e} at eps=1e-8, "
        f"{tiny_eps:.2e} at eps=1e-16",
    )


def check_selection() -> None:
    """The rule recovers the exact null space -- but only once thres clears the floor.

    The second half is the one that matters for a run. Upstream's own signature
    defaults thres to 1.001, and on a covariance whose null directions are
    numerically spread over a couple of orders of magnitude that keeps a single
    direction out of NULL_RANK. This is not a bug in their code or ours; it is
    what anchoring the rule to the smallest eigenvalue does once the smallest
    eigenvalue is roundoff, and it is the reason inspect_nscl_nullspace.py
    exists.
    """

    covariance = build_covariance()
    covariances = {"proj": covariance}

    torch.manual_seed(5)
    model = TinyNet()
    optimizer, covered = nscl.build_nscl_optimizer(
        model, covariances, lr=1e-2, svd_lr=1e-2, thres=1e4
    )
    stats = nscl.install_null_space_transforms(
        optimizer, covered, covariances, thres=1e4, svd_device="cpu"
    )
    kept = stats["layers"][0]["kept"]
    record(
        "selection: a generous thres recovers the exact null space",
        kept == NULL_RANK,
        f"kept {kept}, exact null space is {NULL_RANK}",
    )
    record(
        "selection: the kept subspace carries no old-data energy",
        stats["layers"][0]["energy_share"] < 1e-6,
        f"energy share {stats['layers'][0]['energy_share']:.3e}",
    )

    model = TinyNet()
    optimizer, covered = nscl.build_nscl_optimizer(
        model, covariances, lr=1e-2, svd_lr=1e-2, thres=1.001
    )
    tight = nscl.install_null_space_transforms(
        optimizer, covered, covariances, thres=1.001, svd_device="cpu"
    )
    tight_kept = tight["layers"][0]["kept"]
    record(
        "selection: upstream's default 1.001 under-selects on a smooth floor",
        tight_kept < NULL_RANK,
        f"kept {tight_kept} of {NULL_RANK} truly-null directions -- "
        "thres must be swept, not inherited",
    )


def check_projector() -> None:
    """The stored transform is P/||P||_F, and trace recovery inverts that.

    max_null_space_leak and every downstream reading of the projector depend on
    this identity, so it is tested on its own rather than only through its
    consequences.
    """

    covariances = {"proj": build_covariance()}
    torch.manual_seed(5)
    model = TinyNet()
    optimizer, covered = nscl.build_nscl_optimizer(
        model, covariances, lr=1e-2, svd_lr=1e-2, thres=1e4
    )
    nscl.install_null_space_transforms(
        optimizer, covered, covariances, thres=1e4, svd_device="cpu"
    )

    stored = optimizer.transforms[model.proj.weight]
    projector = nscl.idempotent_projector(stored)
    idempotency = float((projector @ projector - projector).norm())
    record(
        "projector: trace recovery yields an idempotent P",
        idempotency < 1e-4,
        f"||P P - P||_F = {idempotency:.2e}",
    )
    rank = int(torch.linalg.matrix_rank(projector, tol=1e-4))
    record(
        "projector: its rank is the number of kept directions",
        rank == NULL_RANK,
        f"rank {rank}, kept {NULL_RANK}",
    )
    symmetry = float((projector - projector.T).norm())
    record(
        "projector: P is symmetric, so it is an orthogonal projection",
        symmetry < 1e-5,
        f"||P - P^T||_F = {symmetry:.2e}",
    )


def check_constraint_and_teeth(steps: int) -> None:
    """DeltaW lands in the kept subspace, and the control shows it need not have.

    Two numbers together make the claim. The leak says DeltaW is inside the
    subspace; ||DeltaW|| says there was something to constrain, because a
    perfectly constrained update of size zero would also pass the first test.
    tr(DeltaW C DeltaW^T) is the quantity our own penalty minimizes, reported
    here so the hard and soft constraints can be read on the same scale.
    """

    covariance = build_covariance()
    covariances = {"proj": covariance}

    torch.manual_seed(5)
    model = TinyNet()
    reference = {"proj": model.proj.weight.detach().clone()}
    optimizer, covered = nscl.build_nscl_optimizer(
        model, covariances, lr=1e-2, svd_lr=1e-2, thres=1e4
    )
    nscl.install_null_space_transforms(
        optimizer, covered, covariances, thres=1e4, svd_device="cpu"
    )
    train_steps(model, optimizer, steps=steps)

    leak = nscl.max_null_space_leak(model, reference, optimizer, covered)
    delta = model.proj.weight.detach() - reference["proj"]
    moved = float(delta.norm())
    quadratic = float((delta @ covariance @ delta.T).diagonal().sum())

    record(
        "constraint: DeltaW stays inside the kept subspace",
        leak < 1e-4,
        f"outside-subspace energy share {leak:.2e}",
    )
    record(
        "constraint: DeltaW is not merely zero",
        moved > 1e-3,
        f"||DeltaW||_F = {moved:.4f}",
    )
    record(
        "constraint: tr(DeltaW C DeltaW^T) is driven to zero",
        quadratic < 1e-6,
        f"{quadratic:.3e} against ||DeltaW||_F = {moved:.4f}",
    )

    # Control: identical setup, plain Adam, measured against the same projector.
    torch.manual_seed(5)
    control = TinyNet()
    control_reference = {"proj": control.proj.weight.detach().clone()}
    train_steps(control, torch.optim.Adam(control.parameters(), lr=1e-2), steps=steps)
    control_delta = control.proj.weight.detach() - control_reference["proj"]
    projector = nscl.idempotent_projector(optimizer.transforms[model.proj.weight])
    control_leak = float((control_delta - control_delta @ projector).norm()) / float(
        control_delta.norm()
    )
    control_quadratic = float((control_delta @ covariance @ control_delta.T).diagonal().sum())
    record(
        "teeth: the unprojected control leaks, so the test can fail",
        control_leak > 0.1,
        f"outside-subspace energy share {control_leak:.3f} vs {leak:.2e} projected",
    )
    record(
        "teeth: the unprojected control has a real penalty value",
        control_quadratic > 1e-3,
        f"tr = {control_quadratic:.3e} vs {quadratic:.3e} projected",
    )


def check_gradient_precondition() -> None:
    """Their .grad guard is real, and our setup is what defeats it.

    Calling their two methods with .grad still None -- which is the state before
    the first backward, and therefore the state at our call site -- selects
    nothing at all. The run that follows is plain full fine-tuning with an
    Adam-NSCL label on it and no error anywhere. This check is what stops that
    from being rediscovered later.
    """

    covariances = {"proj": build_covariance()}
    torch.manual_seed(5)
    model = TinyNet()
    optimizer, covered = nscl.build_nscl_optimizer(
        model, covariances, lr=1e-2, svd_lr=1e-2, thres=1e4
    )

    # Upstream's own call sequence, without the precondition our glue adds.
    matrices = {id(module.weight): covariances[key] for _, module, key in covered}
    view = nscl._CovarianceView(matrices, device="cpu")
    optimizer.get_eigens(view)
    optimizer.get_transforms()
    record(
        "gradients: without .grad their methods select nothing",
        len(optimizer.transforms) == 0,
        f"{len(optimizer.transforms)} transforms built from a model that has not "
        "yet had a backward",
    )

    stats = nscl.install_null_space_transforms(
        optimizer, covered, covariances, thres=1e4, svd_device="cpu"
    )
    record(
        "gradients: install_null_space_transforms establishes it",
        len(optimizer.transforms) == len(covered) and bool(stats["layers"]),
        f"{len(optimizer.transforms)} transforms for {len(covered)} covered layers",
    )
    leftover = [
        name for name, module, _ in covered if module.weight.grad is not None
    ]
    record(
        "gradients: the temporary buffers are released again",
        not leftover,
        "none left" if not leftover else f"still set on {leftover}",
    )


def check_degenerate_threshold() -> None:
    """An empty selection is upstream's 0/0, and the guard has to catch it.

    With no eigenvector selected, ``basis @ basis.T`` is all zeros and
    ``transform / torch.norm(transform)`` is 0/0. NaN then propagates into the
    weights on the first step and out into the checkpoint, with nothing in the
    log to say where it started. thres >= 1 cannot reach this and train.py
    refuses anything smaller, so this check drives it deliberately.
    """

    covariances = {"proj": build_covariance()}
    torch.manual_seed(5)
    model = TinyNet()
    optimizer, covered = nscl.build_nscl_optimizer(
        model, covariances, lr=1e-2, svd_lr=1e-2, thres=0.5
    )
    try:
        nscl.install_null_space_transforms(
            optimizer, covered, covariances, thres=0.5, svd_device="cpu"
        )
    except RuntimeError as error:
        record(
            "degenerate: an empty selection is refused, not returned as NaN",
            "not finite" in str(error),
            str(error).split(".")[0],
        )
        return
    finite = bool(torch.isfinite(optimizer.transforms[model.proj.weight]).all())
    record(
        "degenerate: an empty selection is refused, not returned as NaN",
        False,
        "setup returned without raising; projector finite" if finite else "NaN projector slipped through",
    )


def check_coverage_mismatch() -> None:
    """A projected layer without a covariance has to fail at setup.

    Their step reads ``self.transforms[p]`` for every member of an svd group as
    soon as any transform exists, so the alternative is a KeyError thousands of
    steps into a cluster job.
    """

    covariances = {"proj": build_covariance()}
    torch.manual_seed(5)
    model = TinyNet()
    optimizer, covered = nscl.build_nscl_optimizer(
        model, covariances, lr=1e-2, svd_lr=1e-2, thres=1e4
    )
    # Smuggle an uncovered parameter into the projected group, which is what a
    # future change to how membership is decided would look like.
    optimizer.param_groups[0]["params"].append(model.head.weight)
    try:
        nscl.install_null_space_transforms(
            optimizer, covered, covariances, thres=1e4, svd_device="cpu"
        )
    except RuntimeError as error:
        record(
            "coverage: a projected layer without a covariance fails at setup",
            "without a covariance" in str(error),
            str(error).split(".")[0],
        )
        return
    record(
        "coverage: a projected layer without a covariance fails at setup",
        False,
        "setup accepted a projected parameter that has no covariance",
    )


def _leak_after_training(
    dtype: torch.dtype,
    lr: float,
    steps: int,
    round_projector_to_bf16: bool = False,
) -> tuple[float, float, dict]:
    """Train the synthetic model under one precision setting; return the leak."""

    covariances = {"proj": build_covariance()}
    torch.manual_seed(5)
    model = TinyNet(dtype=dtype)
    reference = {"proj": model.proj.weight.detach().clone()}
    optimizer, covered = nscl.build_nscl_optimizer(
        model, covariances, lr=lr, svd_lr=lr, thres=1e4
    )
    stats = nscl.install_null_space_transforms(
        optimizer, covered, covariances, thres=1e4, svd_device="cpu"
    )
    if round_projector_to_bf16:
        weight = model.proj.weight
        optimizer.transforms[weight] = (
            optimizer.transforms[weight].to(torch.bfloat16).to(dtype)
        )
    train_steps(model, optimizer, steps=steps)
    leak = nscl.max_null_space_leak(model, reference, optimizer, covered)
    moved = float((model.proj.weight.detach() - reference["proj"]).float().norm())
    return leak, moved, nscl.step_resolution(covered, stats, svd_lr=lr)


def check_precision() -> None:
    """Which precision the null-space guarantee actually survives, and why.

    Three measurements that together attribute the loss rather than asserting a
    tolerance nobody can justify:

      float32                       the method is exact, to machine noise.
      float32 + bf16-rounded P      the projector's own precision costs little.
      bfloat16 weights              the guarantee degrades by orders of magnitude.

    The third is not the projector's fault and the second is what proves it.
    Their step ends in ``p.data.add_(update)``, so the accumulation rounds
    relative to |W|, not to |update|; once the step approaches the dtype's
    spacing at the weight's magnitude, DeltaW is mostly rounding, and rounding
    is isotropic. That is the direction set the projection exists to exclude.

    This is a property of the arm as configured, not a bug to fix here, so the
    checks assert the *ordering* and that step_resolution sees it coming. What a
    real run should do about it is in report_step_resolution.
    """

    torch.manual_seed(5)
    model = TinyNet(dtype=torch.bfloat16)
    optimizer, covered = nscl.build_nscl_optimizer(
        model, {"proj": build_covariance()}, lr=1e-2, svd_lr=1e-2, thres=1e4
    )
    nscl.install_null_space_transforms(
        optimizer, covered, {"proj": build_covariance()}, thres=1e4, svd_device="cpu"
    )
    record(
        "precision: the projector is cast to the parameter dtype",
        optimizer.transforms[model.proj.weight].dtype == torch.bfloat16,
        str(optimizer.transforms[model.proj.weight].dtype),
    )

    try:
        exact, exact_moved, _ = _leak_after_training(torch.float32, 1e-2, 3)
    except RuntimeError as error:
        record("precision: float32 is exact", False, str(error))
        return
    record(
        "precision: in float32 the constraint holds to machine noise",
        exact < 1e-4 and exact_moved > 1e-3,
        f"leak {exact:.2e} at ||DeltaW||_F = {exact_moved:.4f}",
    )

    rounded, _, _ = _leak_after_training(torch.float32, 1e-2, 3, round_projector_to_bf16=True)
    record(
        "precision: a bf16-rounded projector costs little on its own",
        rounded < 1e-2,
        f"leak {rounded:.2e}, {rounded / max(exact, 1e-30):.0f}x the float32 floor",
    )

    try:
        bf16_leak, bf16_moved, resolution = _leak_after_training(torch.bfloat16, 1e-2, 3)
    except RuntimeError as error:
        record("precision: a bf16 step runs at all", False, str(error))
        return
    record("precision: a bf16 step runs at all", bf16_moved > 1e-3, f"||DeltaW||_F = {bf16_moved:.4f}")
    record(
        "precision: bf16 weight accumulation, not the projector, is what leaks",
        bf16_leak > 10 * rounded,
        f"leak {bf16_leak:.2e} with bf16 weights vs {rounded:.2e} with a bf16 "
        f"projector alone -- {bf16_leak / max(rounded, 1e-30):.0f}x",
    )

    starved_leak, _, starved = _leak_after_training(torch.bfloat16, 1e-4, 25)
    record(
        "precision: the leak grows as the step falls below the dtype's spacing",
        starved_leak > bf16_leak,
        f"leak {starved_leak:.2e} at step/spacing {starved['ratio']:.2e}, "
        f"vs {bf16_leak:.2e} at {resolution['ratio']:.1f}",
    )
    record(
        "precision: step_resolution flags the starved setting in advance",
        starved["ratio"] < 1.0 <= resolution["ratio"],
        f"ratio {starved['ratio']:.2e} (starved) vs {resolution['ratio']:.1f} (usable)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adam-NSCL integration self-check")
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Optimizer steps before the constraint is measured.",
    )
    parser.add_argument(
        "--check_commit",
        type=int,
        default=1,
        help="0 skips the pinned-commit check, for a copy that is not a git clone.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("checking Adam-NSCL against baseline/Adam-NSCL/optim/adam_svd.py\n")

    check_upstream(bool(args.check_commit))
    if not all(ok for _, ok, _ in _results):
        print("\nupstream is not usable; the remaining checks would be meaningless")
        return 1

    check_adam_without_projection()
    check_selection()
    check_projector()
    check_constraint_and_teeth(args.steps)
    check_gradient_precondition()
    check_degenerate_threshold()
    check_coverage_mismatch()
    check_precision()

    failed = [name for name, ok, _ in _results if not ok]
    print("\n" + "=" * 68)
    if failed:
        print(f"{len(failed)} of {len(_results)} checks FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"all {len(_results)} Adam-NSCL checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
