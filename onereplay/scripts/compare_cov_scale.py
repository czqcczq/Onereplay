"""Compare two covariance files on the quantity that sets the OneReplay penalty.

The penalty is (1 / L) * sum_l tr(DeltaW_l C_l DeltaW_l^T). Swapping C for a
matrix collected on a different corpus rescales that number even when lambda is
untouched, so two runs at the same lambda are not two runs at the same
regularization strength. Any retention difference between them then mixes two
causes: the directions inside C moved, and the strength moved. Only the first
one is a property of the sampling strategy.

This script separates them:

  scale      per-layer trace ratio, and -- when --adapter_path is given -- the
             exact penalty ratio under a real DeltaW. Both feed the
             equivalent-lambda conversion printed at the end.
  structure  effective rank and top-eigenvalue share, which say whether the two
             matrices spread their mass over a comparable number of directions.
             With --subspace_topk they also report how much of one matrix's
             energy lives in the other's leading subspace.

The trace ratio is the isotropic estimate: if DeltaW had i.i.d. entries then
E[tr(DeltaW C DeltaW^T)] = ||DeltaW||_F^2 / d_in * tr(C), so the penalty scales
with tr(C) alone. A trained LoRA update is not isotropic, which is exactly why
--adapter_path is worth passing: it contracts both matrices against the DeltaW
the method actually produces, and the gap between the two ratios measures how
much the direction change matters on top of the scale change.

Usage:
    python -m onereplay.scripts.compare_cov_scale \
        --ref_cov results/cov/cov_flan_chat_20k_qv.pt \
        --new_cov results/cov/cov_flan_chat_20k_balanced_qv.pt \
        --ref_lambda 3e-2 \
        --adapter_path results/adapters/cs_onereplay_lam3e-2_seed1_regonce
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

LORA_PREFIX = "base_model.model."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref_cov", type=str, required=True, help="baseline C, e.g. the uniform one")
    parser.add_argument("--new_cov", type=str, required=True, help="C under test, e.g. the balanced one")
    parser.add_argument(
        "--ref_lambda",
        type=float,
        default=3e-2,
        help="lambda already tuned on --ref_cov; the equivalent lambda for --new_cov is derived from it",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="optional PEFT LoRA adapter; enables the exact penalty ratio under a real DeltaW",
    )
    parser.add_argument(
        "--subspace_topk",
        type=int,
        default=0,
        help="if > 0, run eigh per layer and report leading-subspace overlap (minutes, not seconds)",
    )
    parser.add_argument(
        "--subspace_layer_stride",
        type=int,
        default=8,
        help="only every k-th layer joins the subspace analysis, since eigh dominates the runtime",
    )
    parser.add_argument("--json_out", type=str, default="", help="optional path for the machine-readable summary")
    return parser.parse_args()


def load_covariances(path: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Return the C dictionary plus whatever provenance metadata was stored."""

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "covariances" in payload:
        return payload["covariances"], payload.get("metadata", {}) or {}
    if isinstance(payload, dict):
        return payload, {}
    raise ValueError(f"Unsupported covariance file format: {path}")


def load_lora_deltas(adapter_path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """Read A, B and the LoRA scale for every adapted layer, keyed by module name.

    The keys are rewritten into the plain module names the covariance files use
    ("model.layers.0.self_attn.q_proj"), so the two sides can be joined without
    the caller knowing anything about PEFT's naming.
    """

    directory = Path(adapter_path)
    weight_file = directory / "adapter_model.safetensors"
    if weight_file.exists():
        from safetensors.torch import load_file

        state = load_file(str(weight_file))
    else:
        weight_file = directory / "adapter_model.bin"
        if not weight_file.exists():
            raise FileNotFoundError(f"No adapter_model.safetensors or .bin under {directory}")
        state = torch.load(str(weight_file), map_location="cpu")

    config_file = directory / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No adapter_config.json under {directory}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    rank = int(config["r"])
    scale = float(config["lora_alpha"]) / rank
    if config.get("use_rslora"):
        scale = float(config["lora_alpha"]) / (rank**0.5)

    factors: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        for tag in ("lora_A", "lora_B"):
            marker = f".{tag}."
            if marker not in key:
                continue
            module = key.split(marker)[0]
            if module.startswith(LORA_PREFIX):
                module = module[len(LORA_PREFIX) :]
            factors.setdefault(module, {})[tag] = tensor.float()

    deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for module, pair in factors.items():
        if "lora_A" in pair and "lora_B" in pair:
            deltas[module] = (pair["lora_A"], pair["lora_B"], scale)
    if not deltas:
        raise ValueError(f"Found no lora_A/lora_B pairs in {weight_file}")
    return deltas


def layer_penalty(
    covariance: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> float:
    """tr(DeltaW C DeltaW^T) for one layer, via the rank x rank shortcut.

    Mirrors lora_covariance_regularizer so the ratio reported here is the ratio
    the trainer would actually see, not an approximation of it.
    """

    aca = lora_a @ covariance @ lora_a.T
    btb = lora_b.T @ lora_b
    return float(scale**2 * torch.sum(btb * aca.T))


def ratio_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])
    return {
        "mean": sum(values) / len(values),
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
    }


def describe_structure(covariance: torch.Tensor) -> dict[str, float]:
    """Cheap shape descriptors that need no eigendecomposition.

    effective_rank is the participation ratio tr(C)^2 / ||C||_F^2: the number of
    equally sized eigenvalues that would produce the same spread. top_share uses
    power iteration for the largest eigenvalue, which converges fast on the very
    anisotropic matrices activation covariances tend to be.
    """

    matrix = covariance.float()
    trace = float(torch.diagonal(matrix).sum())
    frobenius_sq = float(torch.sum(matrix * matrix))
    vector = torch.randn(matrix.shape[-1], generator=torch.Generator().manual_seed(0))
    vector /= vector.norm()
    for _ in range(64):
        vector = matrix @ vector
        norm = vector.norm()
        if norm == 0:
            break
        vector /= norm
    top_eigenvalue = float(vector @ (matrix @ vector))
    return {
        "trace": trace,
        "effective_rank": trace**2 / max(frobenius_sq, 1e-30),
        "top_share": top_eigenvalue / max(trace, 1e-30),
    }


def subspace_overlap(
    ref: torch.Tensor,
    new: torch.Tensor,
    topk: int,
) -> dict[str, float]:
    """How well each matrix's leading subspace covers the other's mass.

    The raw quantity tr(P_ref C_new) / tr(C_new) is not interpretable on its
    own, because a flat spectrum keeps it low no matter how well the two agree:
    with k far below the effective rank even C_new's own top-k captures little.
    Dividing by that self-capture removes the ceiling, so alignment_new_to_ref
    is 1.0 when C_ref's leading directions are as good as C_new's own, and near
    0 when the penalty has moved to a different part of the input space.

    subspace_overlap is the complementary view: the mean squared cosine between
    the two bases, which ignores eigenvalues entirely and asks only whether the
    directions coincide.
    """

    ref_matrix = ref.float()
    new_matrix = new.float()
    ref_values, ref_vectors = torch.linalg.eigh(ref_matrix)
    new_values, new_vectors = torch.linalg.eigh(new_matrix)
    ref_top = ref_vectors[:, -topk:]
    new_top = new_vectors[:, -topk:]

    ref_self = float(ref_values[-topk:].clamp_min(0).sum())
    new_self = float(new_values[-topk:].clamp_min(0).sum())
    new_in_ref = float(torch.sum((ref_top.T @ new_matrix) * ref_top.T))
    ref_in_new = float(torch.sum((new_top.T @ ref_matrix) * new_top.T))
    return {
        "alignment_new_to_ref": new_in_ref / max(new_self, 1e-30),
        "alignment_ref_to_new": ref_in_new / max(ref_self, 1e-30),
        "topk_energy_share_ref": ref_self / max(float(ref_values.clamp_min(0).sum()), 1e-30),
        "topk_energy_share_new": new_self / max(float(new_values.clamp_min(0).sum()), 1e-30),
        "subspace_overlap": float(torch.sum((ref_top.T @ new_top) ** 2)) / topk,
    }


def main() -> None:
    args = parse_args()
    ref_cov, ref_meta = load_covariances(args.ref_cov)
    new_cov, new_meta = load_covariances(args.new_cov)

    shared = sorted(set(ref_cov) & set(new_cov))
    if not shared:
        raise SystemExit(
            f"The two files share no layer names (ref={len(ref_cov)} new={len(new_cov)}); "
            "they were probably collected with different --target_modules"
        )
    print(f"ref = {args.ref_cov}")
    print(f"new = {args.new_cov}")
    for key in ("sample_strategy", "pool_rows", "pool_fingerprint", "max_samples", "sample_seed"):
        if key in ref_meta or key in new_meta:
            print(f"  {key:<18} ref={ref_meta.get(key)}  new={new_meta.get(key)}")
    print(f"shared layers: {len(shared)} (ref={len(ref_cov)} new={len(new_cov)})")

    # ---- scale: trace ----
    trace_ratios = []
    ref_trace_sum = 0.0
    new_trace_sum = 0.0
    for name in shared:
        ref_trace = float(torch.diagonal(ref_cov[name].float()).sum())
        new_trace = float(torch.diagonal(new_cov[name].float()).sum())
        ref_trace_sum += ref_trace
        new_trace_sum += new_trace
        trace_ratios.append(new_trace / max(ref_trace, 1e-30))
    trace_stats = ratio_stats(trace_ratios)
    print()
    print("==== scale: trace(C_new) / trace(C_ref) per layer ====")
    print(
        f"  mean={trace_stats['mean']:.4f}  median={trace_stats['median']:.4f}  "
        f"min={trace_stats['min']:.4f}  max={trace_stats['max']:.4f}"
    )
    print(f"  summed trace: ref={ref_trace_sum:.4e}  new={new_trace_sum:.4e}  ratio={new_trace_sum / ref_trace_sum:.4f}")

    # ---- scale: real DeltaW ----
    penalty_ratio = None
    if args.adapter_path:
        deltas = load_lora_deltas(args.adapter_path)
        joined = [name for name in shared if name in deltas]
        if not joined:
            raise SystemExit(
                f"The adapter's {len(deltas)} layers do not match the covariance layer names; "
                f"adapter example={next(iter(deltas))}  cov example={shared[0]}"
            )
        ref_penalty = sum(layer_penalty(ref_cov[name], *deltas[name]) for name in joined) / len(joined)
        new_penalty = sum(layer_penalty(new_cov[name], *deltas[name]) for name in joined) / len(joined)
        penalty_ratio = new_penalty / max(ref_penalty, 1e-30)
        print()
        print(f"==== scale: penalty under the real DeltaW of {Path(args.adapter_path).name} ====")
        print(f"  layers matched: {len(joined)} / {len(shared)}")
        print(f"  reg(C_ref) = {ref_penalty:.6f}")
        print(f"  reg(C_new) = {new_penalty:.6f}")
        print(f"  ratio      = {penalty_ratio:.4f}")
        drift = abs(penalty_ratio - trace_stats["mean"]) / max(trace_stats["mean"], 1e-30)
        print(
            f"  vs the isotropic trace estimate {trace_stats['mean']:.4f}: {drift:.1%} apart. "
            "A large gap means the two matrices differ in direction, not only in size, "
            "so no single lambda rescaling can make them equivalent."
        )

    # ---- structure ----
    print()
    print("==== structure: how concentrated is each C ====")
    ref_structure = [describe_structure(ref_cov[name]) for name in shared]
    new_structure = [describe_structure(new_cov[name]) for name in shared]
    for label, records in (("ref", ref_structure), ("new", new_structure)):
        erank = sum(record["effective_rank"] for record in records) / len(records)
        share = sum(record["top_share"] for record in records) / len(records)
        print(f"  {label}: effective rank {erank:8.2f}   top eigenvalue holds {share:.1%} of the trace")
    print("  effective rank = tr(C)^2 / ||C||_F^2, the number of equal directions with the same spread")

    overlap_summary = {}
    if args.subspace_topk > 0:
        picked = shared[:: max(args.subspace_layer_stride, 1)]
        print()
        print(f"==== structure: leading-{args.subspace_topk} subspace, {len(picked)} sampled layers ====")
        rows = [subspace_overlap(ref_cov[name], new_cov[name], args.subspace_topk) for name in picked]
        for key in (
            "alignment_new_to_ref",
            "alignment_ref_to_new",
            "subspace_overlap",
            "topk_energy_share_ref",
            "topk_energy_share_new",
        ):
            value = sum(row[key] for row in rows) / len(rows)
            overlap_summary[key] = value
            print(f"  {key:<22} {value:.4f}")
        print(
            "  alignment near 1 = the new corpus excites the same directions and only reweights "
            "them, so a lambda rescaling can undo the difference; well below 1 = the penalty now "
            "guards a different subspace and no lambda makes the two equivalent."
        )
        print(
            f"  topk_energy_share says how much of each trace the leading {args.subspace_topk} "
            "directions hold, i.e. whether k was large enough to be worth reading."
        )

    # ---- equivalent lambda ----
    # The penalty enters the loss as lambda * reg, so holding lambda * reg fixed
    # across the two matrices means dividing lambda by however much reg grew.
    print()
    print("==== equivalent lambda ====")
    basis = penalty_ratio if penalty_ratio is not None else trace_stats["mean"]
    basis_name = "real-DeltaW penalty ratio" if penalty_ratio is not None else "mean trace ratio"
    equivalent = args.ref_lambda / max(basis, 1e-30)
    print(f"  lambda={args.ref_lambda:g} on C_ref  ==  lambda={equivalent:.4g} on C_new   (via the {basis_name} {basis:.4f})")
    if abs(basis - 1.0) < 0.05:
        print("  Within 5% of 1.0, so the two matrices are on the same scale and the same lambda")
        print("  really was the same strength. A retention difference at fixed lambda is then")
        print("  attributable to the directions in C, or to noise -- not to mistuned lambda.")
    else:
        print("  Far enough from 1.0 that comparing the two at one shared lambda compared two")
        print("  different strengths. Put the equivalent lambda above into the sweep grid.")

    if args.json_out:
        summary = {
            "ref_cov": args.ref_cov,
            "new_cov": args.new_cov,
            "shared_layers": len(shared),
            "trace_ratio": trace_stats,
            "summed_trace_ratio": new_trace_sum / ref_trace_sum,
            "penalty_ratio": penalty_ratio,
            "ref_lambda": args.ref_lambda,
            "equivalent_lambda": equivalent,
            "subspace": overlap_summary,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
