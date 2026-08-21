"""Check that the full fine-tuning penalty is the same quantity as the LoRA one.

OneReplay is defined as one formula:

    R = tr(DeltaW C DeltaW^T)

LoRA reaches it through scale^2 * tr((B^T B)(A C A^T)), exploiting
DeltaW = scale * B A. Full fine-tuning reaches it through
sum((DeltaW C) * DeltaW) with DeltaW = W - W0. If the two ever disagree, the
full path is not implementing the published method.

The synthetic checks run on CPU in a second and need no checkpoint:

    python -m onereplay.scripts.verify_full_regularizer

Pass --model_dir/--model_name/--cov_path to additionally inspect coverage on
the real model, which is where tied weights and missing C show up:

    python -m onereplay.scripts.verify_full_regularizer \
      --model_dir ... --model_name Qwen3-1.7B --cov_path .../cov_..._full.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from onereplay.core.modeling import (  # noqa: E402
    load_causal_lm_and_tokenizer,
    set_seed,
    snapshot_reference_weights,
)
from onereplay.core.regularizer import (  # noqa: E402
    full_covariance_regularizer,
    lora_covariance_regularizer,
)

# One row per layer: (d_in, d_out). Mirrors Qwen3-1.7B's shape mix, including
# the non-square projections, at a size that runs instantly on CPU.
LAYER_SHAPES = ((64, 64), (64, 32), (96, 64))
RANK = 8
ALPHA = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the full fine-tuning OneReplay path.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Relative difference allowed between the two evaluation paths.",
    )
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--cov_path", type=str, default="")
    parser.add_argument("--use_bf16", type=int, default=1)
    return parser.parse_args()


class FakeLoraLinear(nn.Module):
    """Minimal stand-in exposing the attributes get_lora_weight_matrices reads.

    Using this instead of a real PEFT layer keeps the check dependency-free and
    lets us set B to something non-zero; PEFT initializes B to zero, which would
    make the penalty trivially zero and prove nothing.
    """

    def __init__(self, d_in: int, d_out: int, rank: int, alpha: int) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, d_out, bias=False)})
        self.scaling = {"default": float(alpha) / float(rank)}

    def delta_weight(self) -> torch.Tensor:
        a = self.lora_A["default"].weight
        b = self.lora_B["default"].weight
        return self.scaling["default"] * (b @ a)


class LoraStack(nn.Module):
    def __init__(self, shapes) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeLoraLinear(d_in, d_out, RANK, ALPHA) for d_in, d_out in shapes]
        )


class PlainStack(nn.Module):
    def __init__(self, shapes) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out, bias=False) for d_in, d_out in shapes])


def make_spd(dim: int, generator: torch.Generator) -> torch.Tensor:
    """A symmetric positive semi-definite C, the same shape a real one has."""

    samples = torch.randn(dim * 2, dim, generator=generator)
    return samples.T @ samples / (dim * 2)


def build_covariances(generator: torch.Generator, identity: bool = False):
    covariances = {}
    for index, (d_in, _) in enumerate(LAYER_SHAPES):
        covariances[f"layers.{index}"] = (
            torch.eye(d_in) if identity else make_spd(d_in, generator)
        )
    return covariances


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def report(name: str, passed: bool, detail: str) -> bool:
    print(f"[{'OK  ' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def check_lora_matches_full(args: argparse.Namespace) -> bool:
    """The two evaluation paths must agree on the same DeltaW."""

    generator = torch.Generator().manual_seed(args.seed)
    covariances = build_covariances(generator)

    lora_stack = LoraStack(LAYER_SHAPES)
    plain_stack = PlainStack(LAYER_SHAPES)
    references = {}
    with torch.no_grad():
        for index, layer in enumerate(lora_stack.layers):
            # Non-zero B, otherwise DeltaW is zero and the check is vacuous.
            layer.lora_B["default"].weight.normal_(generator=generator)
            layer.lora_A["default"].weight.normal_(generator=generator)
            name = f"layers.{index}"
            base = plain_stack.layers[index].weight
            references[name] = base.detach().clone()
            base.add_(layer.delta_weight())

    lora_reg, lora_stats = lora_covariance_regularizer(lora_stack, covariances)
    full_reg, full_stats = full_covariance_regularizer(plain_stack, covariances, references)

    difference = relative_difference(float(lora_reg), float(full_reg))
    layers_match = lora_stats["used_layers"] == full_stats["used_layers"] == len(LAYER_SHAPES)
    passed = difference <= args.tolerance and layers_match
    return report(
        "LoRA and full paths agree",
        passed,
        f"lora={float(lora_reg):.10e} full={float(full_reg):.10e} rel={difference:.2e} "
        f"layers={int(full_stats['used_layers'])}",
    )


def check_matches_explicit_trace(args: argparse.Namespace) -> bool:
    """The full path must equal a literal tr(DeltaW C DeltaW^T) per layer."""

    generator = torch.Generator().manual_seed(args.seed + 1)
    covariances = build_covariances(generator)

    plain_stack = PlainStack(LAYER_SHAPES)
    references = {}
    expected = 0.0
    with torch.no_grad():
        for index, layer in enumerate(plain_stack.layers):
            name = f"layers.{index}"
            references[name] = layer.weight.detach().clone()
            layer.weight.normal_(generator=generator)
            delta = layer.weight - references[name]
            expected += float(torch.trace(delta @ covariances[name] @ delta.T))
    expected /= len(LAYER_SHAPES)

    full_reg, _ = full_covariance_regularizer(plain_stack, covariances, references)
    difference = relative_difference(expected, float(full_reg))
    return report(
        "full path equals explicit trace",
        difference <= args.tolerance,
        f"explicit={expected:.10e} full={float(full_reg):.10e} rel={difference:.2e}",
    )


def check_identity_gives_frobenius(args: argparse.Namespace) -> bool:
    """identity_cov must degenerate to plain L2 on DeltaW, as it does for LoRA."""

    generator = torch.Generator().manual_seed(args.seed + 2)
    covariances = build_covariances(generator, identity=True)

    plain_stack = PlainStack(LAYER_SHAPES)
    references = {}
    expected = 0.0
    with torch.no_grad():
        for index, layer in enumerate(plain_stack.layers):
            name = f"layers.{index}"
            references[name] = layer.weight.detach().clone()
            layer.weight.normal_(generator=generator)
            expected += float((layer.weight - references[name]).pow(2).sum())
    expected /= len(LAYER_SHAPES)

    full_reg, _ = full_covariance_regularizer(plain_stack, covariances, references)
    difference = relative_difference(expected, float(full_reg))
    return report(
        "identity C degenerates to Frobenius",
        difference <= args.tolerance,
        f"||DeltaW||_F^2={expected:.10e} full={float(full_reg):.10e} rel={difference:.2e}",
    )


def check_zero_at_initialization(args: argparse.Namespace) -> bool:
    """Before the first step W equals W0, so the penalty must be exactly zero."""

    generator = torch.Generator().manual_seed(args.seed + 3)
    covariances = build_covariances(generator)
    plain_stack = PlainStack(LAYER_SHAPES)
    references = snapshot_reference_weights(plain_stack, covariances)

    full_reg, stats = full_covariance_regularizer(plain_stack, covariances, references)
    passed = float(full_reg) == 0.0 and stats["used_layers"] == len(LAYER_SHAPES)
    return report(
        "penalty is zero at initialization",
        passed,
        f"reg={float(full_reg):.3e} layers={int(stats['used_layers'])}",
    )


def check_tied_weights_counted_once(args: argparse.Namespace) -> bool:
    """A shared weight tensor must contribute one term, not one per alias."""

    generator = torch.Generator().manual_seed(args.seed + 4)
    dim = 64
    covariances = {"first": make_spd(dim, generator), "second": make_spd(dim, generator)}

    class TiedPair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(dim, dim, bias=False)
            self.second = nn.Linear(dim, dim, bias=False)
            self.second.weight = self.first.weight

    model = TiedPair()
    references = snapshot_reference_weights(model, covariances)
    passed = len(references) == 1
    return report(
        "tied weights snapshotted once",
        passed,
        f"snapshot holds {len(references)} entries for 2 aliased modules",
    )


def check_real_model(args: argparse.Namespace) -> bool:
    """Report coverage on the actual checkpoint: how many layers, which are missing."""

    from onereplay.core.covariance import load_covariance_file

    set_seed(args.seed)
    model, _ = load_causal_lm_and_tokenizer(args.model_dir, args.model_name, args.use_bf16)
    covariances = load_covariance_file(args.cov_path)
    references = snapshot_reference_weights(model, covariances)

    linear_names = {
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    uncovered = sorted(linear_names - set(references))
    tied = getattr(model.config, "tie_word_embeddings", None)

    print(
        f"       real model: {len(covariances)} C matrices, "
        f"{len(references)} layers snapshotted, {len(linear_names)} Linear modules total, "
        f"tie_word_embeddings={tied}"
    )
    if uncovered:
        preview = ", ".join(uncovered[:5])
        suffix = " ..." if len(uncovered) > 5 else ""
        print(f"       Linear modules without C ({len(uncovered)}): {preview}{suffix}")

    covariance_bytes = sum(value.numel() * value.element_size() for value in covariances.values())
    reference_bytes = sum(value.numel() * value.element_size() for value in references.values())
    print(
        f"       resident cost: C {covariance_bytes / 1024**3:.3f} GiB, "
        f"W0 {reference_bytes / 1024**3:.3f} GiB"
    )

    reg, stats = full_covariance_regularizer(model, covariances, references)
    passed = float(reg) == 0.0 and stats["missing_layers"] == 0.0
    return report(
        "real model coverage",
        passed,
        f"untrained reg={float(reg):.3e} used={int(stats['used_layers'])} "
        f"missing_C={int(stats['missing_layers'])}",
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    results = [
        check_lora_matches_full(args),
        check_matches_explicit_trace(args),
        check_identity_gives_frobenius(args),
        check_zero_at_initialization(args),
        check_tied_weights_counted_once(args),
    ]

    if args.model_dir and args.cov_path:
        results.append(check_real_model(args))
    else:
        print("[skip] real model coverage: pass --model_dir and --cov_path to enable")

    if all(results):
        print("\nALL CHECKS PASSED: the full path computes the same penalty as LoRA")
    else:
        raise SystemExit(f"\n{results.count(False)} CHECK(S) FAILED")


if __name__ == "__main__":
    main()
