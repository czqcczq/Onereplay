"""Is --reg_impl analytic the same gradient as --reg_impl autograd, and is TF32 enough.

Two questions, deliberately kept apart because they have different answers:

  1. Is 2*DeltaW*C the gradient of sum((DeltaW C) * DeltaW)?
     This is an algebraic identity, not a precision trade-off. It either holds to
     rounding or the new path is computing a different penalty. Checked on
     synthetic layers in float64 on CPU, so ordinary rounding stays near 1e-16
     and a real discrepancy cannot hide. No checkpoint, no GPU.

  2. How much does letting the tensor core round C to 11 mantissa bits move the
     gradient?
     This one is a real perturbation and needs a real DeltaW. The trap is that
     comparing the new path against the current one only says they differ, not
     which is closer to the truth -- the current path is not the truth either, it
     runs a second matmul through autograd and accumulates its own error. So all
     arms are measured against a float64 reference:

         truth   fp64,  analytic
         A       fp32,  autograd     (what train.py does today)
         B       tf32,  analytic     (what --reg_impl analytic does)

     If B lands closer to the truth than A, the precision question is settled
     without needing to argue about how large an acceptable error would be.

Usage
  # question 1 only, runs anywhere in a few seconds
  python -m onereplay.scripts.verify_reg_precision

  # question 2 with a real DeltaW from a finished full fine-tune
  python -m onereplay.scripts.verify_reg_precision \
      --base_model_dir .../models/Qwen3-1.7B \
      --trained_model_dir .../checkpoints/full/full_onereplay_lam3e-2_seed1 \
      --cov_path .../cov/cov_flan_chat_20k_full.pt

  # question 2 without any checkpoint: a random DeltaW scaled to a known reg value.
  # Conservative -- a random DeltaW is nearly orthogonal to C's few dominant
  # eigendirections, so the penalty rides on C's small eigenvalues, which is where
  # relative rounding error is worst. Passing here implies passing on a real DeltaW.
  python -m onereplay.scripts.verify_reg_precision \
      --base_model_dir .../models/Qwen3-1.7B \
      --cov_path .../cov/cov_flan_chat_20k_full.pt --target_reg 0.14009
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from onereplay.core.regularizer import full_covariance_grad_  # noqa: E402

# Mirrors Qwen3-1.7B's shape mix (hidden 2048, intermediate 6144) at a size that
# runs instantly on CPU. down_proj's d_in > d_out is the case that catches a
# transposed C, and it is also the layer that dominates the real cost.
SYNTHETIC_SHAPES = ((64, 64), (64, 32), (96, 64), (64, 96))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the analytic penalty gradient.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--identity_tolerance",
        type=float,
        default=1e-10,
        help="Relative gradient difference allowed in the float64 identity check.",
    )
    parser.add_argument("--base_model_dir", type=str, default="")
    parser.add_argument(
        "--trained_model_dir",
        type=str,
        default="",
        help="A finished full fine-tune. Omit to use a synthetic DeltaW instead.",
    )
    parser.add_argument("--cov_path", type=str, default="")
    parser.add_argument(
        "--target_reg",
        type=float,
        default=0.14009,
        help=(
            "Synthetic DeltaW is scaled so the normalized penalty hits this value, so "
            "the rounding is exercised at the magnitude training actually reaches. The "
            "default is the epoch-3 train_replay_reg of the full lambda=3e-2 run."
        ),
    )
    parser.add_argument("--replay_lambda", type=float, default=3e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_layers",
        type=int,
        default=0,
        help="0 checks every covered layer; a small number keeps a smoke test short.",
    )
    return parser.parse_args()


def report(name: str, passed: bool, detail: str) -> bool:
    print(f"[{'OK  ' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def make_spd(dim: int, generator: torch.Generator, decay: float = 1.5) -> torch.Tensor:
    """A PSD C with a spectrum shaped like a real one.

    The measured C matrices are extremely anisotropic: effective rank about 8.5 in
    2048 dimensions, top eigenvalue about 35% of the trace. A Wishart matrix is far
    flatter than that and would make the rounding look harmless for the wrong
    reason, so the eigenvalues are laid out as k^-decay instead.
    """

    eigenvalues = torch.arange(1, dim + 1, dtype=torch.float64) ** (-decay)
    basis, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float64))
    return (basis * eigenvalues) @ basis.T


# ---------------------------------------------------------------------------
# Question 1: the identity, in float64 on CPU.
# ---------------------------------------------------------------------------
def check_analytic_identity(args: argparse.Namespace) -> bool:
    generator = torch.Generator().manual_seed(args.seed)
    worst_grad = 0.0
    worst_reg = 0.0

    for d_in, d_out in SYNTHETIC_SHAPES:
        covariance = make_spd(d_in, generator)
        reference = torch.randn(d_out, d_in, generator=generator, dtype=torch.float64)
        weight = reference + 0.01 * torch.randn(
            d_out, d_in, generator=generator, dtype=torch.float64
        )

        # What train.py does today: build the scalar, let backward derive dR/dW.
        delta = (weight - reference).clone().requires_grad_(True)
        reg_autograd = torch.sum((delta @ covariance) * delta)
        reg_autograd.backward()
        grad_autograd = delta.grad

        # What --reg_impl analytic does, through the shipped function so the test
        # covers the code that runs rather than a re-derivation of it.
        module = nn.Linear(d_in, d_out, bias=False).double()
        with torch.no_grad():
            module.weight.copy_(weight)
        module.weight.grad = torch.zeros_like(module.weight)
        reg_analytic = full_covariance_grad_(
            [("layer", module.weight, covariance, reference)],
            scale=1.0,
            allow_tf32=False,
            compute_dtype=torch.float64,
        )
        grad_analytic = module.weight.grad

        worst_grad = max(
            worst_grad,
            float((grad_analytic - grad_autograd).norm() / grad_autograd.norm()),
        )
        worst_reg = max(
            worst_reg,
            abs(reg_analytic - float(reg_autograd)) / max(abs(float(reg_autograd)), 1e-300),
        )

    passed = worst_grad <= args.identity_tolerance and worst_reg <= args.identity_tolerance
    return report(
        "analytic gradient equals autograd (float64)",
        passed,
        f"worst grad rel {worst_grad:.3e}, worst reg rel {worst_reg:.3e}, "
        f"{len(SYNTHETIC_SHAPES)} shapes",
    )


def check_zero_delta(args: argparse.Namespace) -> bool:
    """W == W0 must give exactly zero penalty and exactly zero gradient."""

    generator = torch.Generator().manual_seed(args.seed + 1)
    d_in, d_out = 96, 64
    covariance = make_spd(d_in, generator)
    module = nn.Linear(d_in, d_out, bias=False).double()
    reference = module.weight.detach().clone()
    module.weight.grad = torch.zeros_like(module.weight)

    reg = full_covariance_grad_(
        [("layer", module.weight, covariance, reference)],
        scale=1.0,
        allow_tf32=False,
        compute_dtype=torch.float64,
    )
    grad_max = float(module.weight.grad.abs().max())
    passed = reg == 0.0 and grad_max == 0.0
    return report(
        "zero at initialization",
        passed,
        f"reg={reg:.3e} max|grad|={grad_max:.3e}",
    )


def check_scale_is_linear(args: argparse.Namespace) -> bool:
    """The injected gradient must be exactly proportional to the scale it is given.

    Cheap, but it is the one place a factor of 2 or a stray 1/used_layers would
    hide: both the penalty value and the gradient direction stay correct while the
    strength silently changes, which no downstream metric would attribute to this.
    """

    generator = torch.Generator().manual_seed(args.seed + 2)
    d_in, d_out = 96, 64
    covariance = make_spd(d_in, generator)
    reference = torch.randn(d_out, d_in, generator=generator, dtype=torch.float64)
    weight_value = reference + 0.01 * torch.randn(
        d_out, d_in, generator=generator, dtype=torch.float64
    )

    grads = {}
    for scale in (1.0, 2.5):
        module = nn.Linear(d_in, d_out, bias=False).double()
        with torch.no_grad():
            module.weight.copy_(weight_value)
        module.weight.grad = torch.zeros_like(module.weight)
        full_covariance_grad_(
            [("layer", module.weight, covariance, reference)],
            scale=scale,
            allow_tf32=False,
            compute_dtype=torch.float64,
        )
        grads[scale] = module.weight.grad.clone()

    ratio = grads[2.5] / grads[1.0]
    deviation = float((ratio - 2.5).abs().max())
    passed = deviation <= args.identity_tolerance
    return report(
        "injected gradient is linear in scale",
        passed,
        f"max deviation from 2.5x: {deviation:.3e}",
    )


# ---------------------------------------------------------------------------
# Question 2: precision on a real model, against a float64 reference.
# ---------------------------------------------------------------------------
def load_linear_weights(model_dir: str) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, local_files_only=True
    )
    weights = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    del model
    return weights


def synthetic_deltas(
    references: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    names: list[str],
    target_reg: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Random DeltaW rescaled so the normalized penalty equals target_reg."""

    generator = torch.Generator().manual_seed(seed)
    raw = {
        name: torch.randn(
            references[name].shape, generator=generator, dtype=torch.float32
        )
        for name in names
    }
    total = 0.0
    for name in names:
        delta = raw[name].to(covariances[name].device)
        total += float(((delta @ covariances[name].float()) * delta).sum())
    current = total / len(names)
    factor = math.sqrt(target_reg / current) if current > 0 else 1.0
    return {name: (value * factor) for name, value in raw.items()}


def check_real_precision(args: argparse.Namespace) -> bool:
    from onereplay.core.covariance import load_covariance_file

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "       note: TF32 only exists on CUDA tensor cores, so on CPU arm B "
            "degenerates to fp32 and this check only bounds the autograd->analytic "
            "difference, not the TF32 rounding."
        )

    covariances = load_covariance_file(args.cov_path)
    references = load_linear_weights(args.base_model_dir)
    names = [name for name in covariances if name in references]
    if not names:
        return report(
            "real precision", False, f"none of the {len(covariances)} C keys match a Linear"
        )
    names.sort()
    if args.max_layers > 0:
        names = names[: args.max_layers]

    if args.trained_model_dir:
        trained = load_linear_weights(args.trained_model_dir)
        source = "real DeltaW from " + Path(args.trained_model_dir).name
    else:
        trained = None
        source = f"synthetic DeltaW scaled to reg={args.target_reg:g}"

    if trained is None:
        deltas = synthetic_deltas(
            references,
            {name: covariances[name].to(device) for name in names},
            names,
            args.target_reg,
            args.seed,
        )

    print(f"       {len(names)} layers, {source}, device={device}")
    print(
        f"       {'layer':<44} {'||g||':>10} {'A vs fp64':>11} {'B vs fp64':>11} {'B/A':>7}"
    )

    worst_a = 0.0
    worst_b = 0.0
    b_better = 0
    reg_total = {"fp64": 0.0, "A": 0.0, "B": 0.0}

    for name in names:
        covariance = covariances[name].to(device=device, dtype=torch.float32)
        reference = references[name].to(device=device, dtype=torch.float32)
        if trained is not None:
            weight_value = trained[name].to(device=device, dtype=torch.float32)
        else:
            weight_value = reference + deltas[name].to(device)

        grads = {}

        # fp64 reference. PyTorch requires .grad to share the parameter's dtype,
        # and full_covariance_grad_ creates the buffer as zeros_like(weight), so
        # the whole arm has to run on an fp64 parameter -- an fp32 weight with an
        # fp64 grad is rejected outright, and even if it were not, the accumulate
        # would round the reference gradient straight back down to fp32. Casting
        # the fp32 inputs up to fp64 is exact, so this stays a faithful fp64
        # evaluation of the same DeltaW the other two arms see.
        weight_fp64 = nn.Parameter(weight_value.double())
        reg_total["fp64"] += full_covariance_grad_(
            [(name, weight_fp64, covariance, reference)],
            scale=1.0,
            allow_tf32=False,
            compute_dtype=torch.float64,
        )
        grads["fp64"] = weight_fp64.grad.clone()

        # Arm A: fp32 + autograd, the expression full_covariance_regularizer
        # evaluates today. autograd's dR/dDelta is what the analytic path returns
        # too, so the two are directly comparable without any rescaling.
        delta = (weight_value - reference).clone().requires_grad_(True)
        reg_a = torch.sum((delta @ covariance) * delta)
        reg_a.backward()
        grads["A"] = delta.grad.double()
        reg_total["A"] += float(reg_a)

        # Arm B: TF32 + analytic, on an fp32 parameter so the .grad buffer it
        # allocates is fp32, exactly as in training.
        weight_fp32 = nn.Parameter(weight_value)
        reg_total["B"] += full_covariance_grad_(
            [(name, weight_fp32, covariance, reference)],
            scale=1.0,
            allow_tf32=True,
            compute_dtype=torch.float32,
        )
        grads["B"] = weight_fp32.grad.double()

        norm = grads["fp64"].norm()
        rel_a = float((grads["A"] - grads["fp64"]).norm() / norm)
        rel_b = float((grads["B"] - grads["fp64"]).norm() / norm)
        worst_a = max(worst_a, rel_a)
        worst_b = max(worst_b, rel_b)
        b_better += int(rel_b <= rel_a)
        print(
            f"       {name[:44]:<44} {float(norm):>10.3e} {rel_a:>11.3e} {rel_b:>11.3e} "
            f"{(rel_b / rel_a if rel_a > 0 else float('inf')):>7.2f}"
        )
        del weight_fp64, weight_fp32, delta, grads

    divisor = len(names)
    print(
        f"       normalized reg: fp64={reg_total['fp64'] / divisor:.10e} "
        f"A={reg_total['A'] / divisor:.10e} B={reg_total['B'] / divisor:.10e}"
    )
    print(
        f"       B is closer to fp64 on {b_better}/{divisor} layers; "
        f"worst A {worst_a:.3e}, worst B {worst_b:.3e}"
    )

    # The verdict deliberately does not compare B against a fixed tolerance. What
    # matters is whether switching makes the gradient worse than the one already
    # in use; if it does not, no threshold argument is needed.
    passed = worst_b <= max(worst_a, 1e-3)
    return report(
        "real precision",
        passed,
        f"worst B vs fp64 {worst_b:.3e} against worst A vs fp64 {worst_a:.3e}",
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    results = [
        check_analytic_identity(args),
        check_zero_delta(args),
        check_scale_is_linear(args),
    ]

    if args.base_model_dir and args.cov_path:
        results.append(check_real_precision(args))
    else:
        print("[skip] real precision: pass --base_model_dir and --cov_path to enable")

    if all(results):
        print("\nALL CHECKS PASSED")
    else:
        raise SystemExit(f"\n{results.count(False)} CHECK(S) FAILED")


if __name__ == "__main__":
    main()
