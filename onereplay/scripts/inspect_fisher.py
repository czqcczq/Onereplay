"""Inspect and validate a Fisher (or covariance) payload written by Stage 1/1b.

Two modes:

  --print_meta KEY   print one metadata value and exit. Works on cov files too,
                     which is how a job reads C's pool fingerprint before
                     spending GPU hours estimating F on what it hopes are the
                     same rows.

  default            print the estimator diagnostics, check every --expect_*
                     flag, and exit 1 if any of them is wrong.

The checks exist because the ways a Fisher file can be wrong are all silent. A
right-truncated collection still produces a full-looking F, just one whose
long-answer rows contributed nothing; a bf16 run produces an F with a 1% noise
floor baked into every entry; a mismatched --target_modules produces an F the
training run will happily load and then apply to a different set of layers. None
of these raise, and none are visible in the tensors. They are visible in the
metadata, so this script reads the metadata and says so.

With --reference it also reports mean(F)/mean(F_ref). That ratio is not a
diagnostic but a design input: F is built from a sequence-summed NLL, so a domain
whose answers are k times longer lands roughly k^a times larger with a in [1, 2],
and mixing two domains' Fisher matrices with equal coefficients would then weight
them unequally by that factor. The covariance path does not have this problem
because C is a token mean, which is why the C_mix weights cannot be copied over.

The global mean ratio decides equal-contribution weights only if that ratio holds
layer by layer. F is spiky (max/mean around 1e6), so a single scalar weight that
balances the two domains on average can still leave individual layers dominated by
one side. --reference therefore also runs a per-layer diagnostic: it loads both
files' matrices, computes each layer's mass (the sum of Fisher entries, i.e. the
layer's weight in sum_ij F_ij dW_ij^2 under a uniform dW), and reports how the
per-layer ratio spreads around the global one. A tight spread means one scalar
weight is enough; a wide one means the balance a scalar buys on average is not the
balance any given layer gets.

Usage:
    python -m onereplay.scripts.inspect_fisher --path F.pt --print_meta pool_fingerprint
    python -m onereplay.scripts.inspect_fisher --path F.pt --expect_max_len 2048 \
        --expect_truncation_side left --reference F_if.pt
    # per-layer only, skip the metadata checks:
    python -m onereplay.scripts.inspect_fisher --path F_math.pt --reference F_if.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or validate a Fisher payload.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--print_meta",
        type=str,
        default="",
        help="Print this one metadata key and exit; nothing else goes to stdout.",
    )

    # Expected collection settings. Empty or negative means "do not check".
    parser.add_argument("--expect_max_len", type=int, default=0)
    parser.add_argument("--expect_truncation_side", type=str, default="")
    parser.add_argument("--expect_target_modules", type=str, default="")
    parser.add_argument("--expect_require_target", type=int, default=-1)
    parser.add_argument("--expect_sample_shuffle", type=int, default=-1)
    parser.add_argument("--expect_use_bf16", type=int, default=-1)
    parser.add_argument("--expect_pool_fingerprint", type=str, default="")
    parser.add_argument(
        "--expect_zero_supervision_rows",
        type=int,
        default=-1,
        help=(
            "Rows that ended up with no supervised token. Left truncation keeps the "
            "answer's tail, so a left-truncated collection must report 0; a non-zero "
            "count is the signature of --truncation_side never reaching the collector. "
            "-1 skips the check (the right-truncated F_if legitimately has 1408)."
        ),
    )
    parser.add_argument(
        "--expect_mean",
        type=float,
        default=0.0,
        help=(
            "Expected fisher_scale.mean, checked within --mean_rtol. This is the most "
            "direct test that a re-collection reproduced an earlier F: the fingerprint "
            "proves the same rows went in, the mean proves the same numbers came out."
        ),
    )
    parser.add_argument("--mean_rtol", type=float, default=0.02)
    parser.add_argument(
        "--reference",
        type=str,
        default="",
        help="Another Fisher file to compare scales against, e.g. F_if.",
    )
    parser.add_argument(
        "--layer_diagnostic",
        type=int,
        default=1,
        help=(
            "With --reference, also load both files' matrices and report the per-layer "
            "mass ratio. 0 skips it (only reads metadata, so it stays fast on the big "
            "full-scope files)."
        ),
    )
    parser.add_argument(
        "--strict",
        type=int,
        default=1,
        help="1 exits non-zero when a check fails; 0 only reports.",
    )
    return parser.parse_args()


def load_payload(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not a Stage 1/1b payload dict")
    return payload


def report_fisher(path: str, label: str) -> dict:
    """Print the estimator diagnostics for one Fisher file, return its metadata."""

    payload = load_payload(path)
    meta = payload.get("metadata", {})
    report = meta.get("length_weighting", {})
    scale = meta.get("fisher_scale", {})
    if not report or not scale:
        raise SystemExit(
            f"{path} has no length_weighting/fisher_scale metadata; it was written by a "
            "collect_fisher predating those diagnostics and cannot be validated"
        )

    print(f"---- {label}: {path} ----")
    print(
        f"  collection : max_len={meta.get('max_len')} "
        f"truncation_side={meta.get('truncation_side')!r} "
        f"use_bf16={meta.get('use_bf16')} require_target={meta.get('require_target')} "
        f"sample_shuffle={meta.get('sample_shuffle')}"
    )
    print(f"  modules    : {meta.get('target_modules')} ({scale.get('layers', 0):.0f} layers)")
    print(f"  pool       : rows={meta.get('pool_rows')} fingerprint={meta.get('pool_fingerprint')}")
    print(f"  estimator  : {meta.get('estimator')} (reduction={meta.get('loss_reduction')})")
    print(f"  scale      : mean={scale.get('mean', 0):.6e} max={scale.get('max', 0):.6e}")
    print(
        f"  N          : {report.get('examples', 0):.0f} rows, "
        f"{report.get('zero_supervision_rows', 0):.0f} with zero supervised tokens"
    )
    print(
        f"  tokens     : mean={report.get('mean_supervised_tokens', 0):.1f} "
        f"median={report.get('median_supervised_tokens', 0):.0f} "
        f"max={report.get('max_supervised_tokens', 0):.0f}"
    )
    print(
        f"  ESS        : {report.get('effective_sample_size', 0):.0f} / "
        f"{report.get('examples', 0):.0f} (ratio {report.get('ess_ratio', 0):.3f}) "
        f"-- how many rows F actually rests on"
    )
    print(
        f"  length_exp : a={report.get('length_exponent', float('nan')):.3f} "
        f"in ||g||^2 ~ T^a (1 = uncorrelated per-token grads, 2 = perfectly aligned)"
    )
    print(f"  prompt mask: {meta.get('prompt_prefix_mismatches')} prefix mismatches (want ~0)")
    return meta


def check_expectations(meta: dict, args: argparse.Namespace) -> list[str]:
    report = meta.get("length_weighting", {})
    failures: list[str] = []

    if args.expect_max_len > 0 and meta.get("max_len") != args.expect_max_len:
        failures.append(
            f"max_len={meta.get('max_len')}, want {args.expect_max_len}. F and C must see the "
            "same token budget or they protect different amounts of the same answers."
        )
    if args.expect_truncation_side and meta.get("truncation_side") != args.expect_truncation_side:
        failures.append(
            f"truncation_side={meta.get('truncation_side')!r}, want "
            f"{args.expect_truncation_side!r}. Right truncation cuts the answer first, so "
            "overflowing rows contribute nothing; training truncates left."
        )
    if args.expect_require_target >= 0 and meta.get("require_target") != args.expect_require_target:
        failures.append(
            f"require_target={meta.get('require_target')}, want {args.expect_require_target}. "
            "Empty-target rows still count towards N and dilute F."
        )
    if args.expect_sample_shuffle >= 0 and meta.get("sample_shuffle") != args.expect_sample_shuffle:
        failures.append(
            f"sample_shuffle={meta.get('sample_shuffle')}, want {args.expect_sample_shuffle}. "
            "This changes which rows are selected, so the pool fingerprint moves with it."
        )
    if args.expect_use_bf16 >= 0 and meta.get("use_bf16") != args.expect_use_bf16:
        failures.append(
            f"use_bf16={meta.get('use_bf16')}, want {args.expect_use_bf16}. Gradients are "
            "squared here, and bf16's 8-bit mantissa leaves a ~1% noise floor that widening "
            "afterwards cannot remove."
        )
    if args.expect_target_modules:
        want = [item.strip() for item in args.expect_target_modules.split(",") if item.strip()]
        if meta.get("target_modules") != want:
            failures.append(
                f"target_modules={meta.get('target_modules')}, want {want}. The penalty is "
                "normalized by layer count, so this also moves the lambda scale."
            )
    if args.expect_pool_fingerprint:
        got = meta.get("pool_fingerprint", "")
        if got != args.expect_pool_fingerprint:
            failures.append(
                f"pool_fingerprint={got}, want {args.expect_pool_fingerprint}. F and C consumed "
                "different old knowledge, so an EWC-versus-OneReplay comparison built on them "
                "would be confounded by the data rather than isolating the weighting."
            )
    if args.expect_zero_supervision_rows >= 0:
        got = int(report.get("zero_supervision_rows", 0))
        if got != args.expect_zero_supervision_rows:
            failures.append(
                f"zero_supervision_rows={got}, want {args.expect_zero_supervision_rows}. "
                "This count is deterministic once the rows, max_len and truncation side "
                "are fixed, so a mismatch means one of those three moved."
            )
    if args.expect_mean > 0:
        got_mean = float(meta["fisher_scale"]["mean"])
        deviation = abs(got_mean - args.expect_mean) / args.expect_mean
        if deviation > args.mean_rtol:
            failures.append(
                f"fisher_scale.mean={got_mean:.6e}, want {args.expect_mean:.6e} "
                f"(off by {deviation:.1%}, tolerance {args.mean_rtol:.1%}). The lambda that "
                "was calibrated against the old F does not transfer to this one."
            )
    return failures


def compare_scales(meta: dict, reference_path: str) -> None:
    """Report mean(F)/mean(F_ref), which is what the mixing weights have to undo."""

    reference_meta = load_payload(reference_path).get("metadata", {})
    if "fisher_scale" not in reference_meta or "length_weighting" not in reference_meta:
        print(
            "---- scale comparison skipped: reference has no Fisher scale metadata "
            "(a covariance file, or an F predating the diagnostics) ----"
        )
        return
    reference = report_fisher(reference_path, "reference")
    scale = meta["fisher_scale"]["mean"]
    reference_scale = reference["fisher_scale"]["mean"]
    tokens = meta["length_weighting"]["mean_supervised_tokens"]
    reference_tokens = reference["length_weighting"]["mean_supervised_tokens"]
    if not reference_scale or not reference_tokens:
        print("---- scale comparison skipped: reference has a zero mean ----")
        return

    ratio = scale / reference_scale
    token_ratio = tokens / reference_tokens
    print("---- scale comparison ----")
    print(f"  mean(F) / mean(F_ref)       = {ratio:.2f}x")
    print(f"  mean supervised token ratio = {token_ratio:.2f}x")
    print(
        f"  T^a prediction for a in [1,2]: {token_ratio:.1f}x to {token_ratio**2:.0f}x "
        f"-- measured {ratio:.1f}x"
    )
    if ratio > 3:
        weight = ratio / (ratio + 1)
        print(
            f"  equal coefficients would let this file dominate the reference {ratio:.0f}x. "
            "C is a token mean so its 0.5/0.5 mix is genuinely equal-weight; F is a sequence "
            "sum so it is not, and the mix has to be normalized by scale first."
        )
        print(
            f"  inverse-scale weights: ref:this = {ratio:.2f} : 1 "
            f"(normalized {weight:.4f} / {1 - weight:.4f})"
        )
    else:
        print(
            "  the two files are within 3x, so equal coefficients are defensible. That "
            "contradicts the length prediction, so check the collection flags before "
            "relying on it."
        )


def _matrices_and_kind(payload: dict, path: str) -> tuple[dict, str]:
    """Return the per-layer matrices and how a layer's penalty mass is measured.

    An elementwise Fisher weights sum_ij F_ij dW_ij^2, so a layer's weight in the
    penalty is the sum of all its entries. A covariance weights tr(dW C dW^T), so
    the layer's weight is the trace. Reading the mass the way the matrix is
    actually applied keeps the ratio faithful to what training will do.
    """

    if "fishers" in payload:
        return payload["fishers"], "sum"
    if "covariances" in payload:
        return payload["covariances"], "trace"
    raise SystemExit(f"{path} has neither 'fishers' nor 'covariances'")


def _layer_mass(tensor: torch.Tensor, kind: str) -> float:
    matrix = tensor.double()
    if kind == "trace" and matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        return float(matrix.diagonal().sum())
    return float(matrix.sum())


def _percentiles(values: list[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    p0, p10, p50, p90, p100 = np.percentile(array, [0, 10, 50, 90, 100])
    return f"min={p0:.3f} P10={p10:.3f} median={p50:.3f} P90={p90:.3f} max={p100:.3f}"


def compare_layers(target_path: str, reference_path: str) -> None:
    """Report the per-layer mass ratio so a scalar mix weight can be sanity-checked.

    The global mean ratio says what one scalar weight would have to be to balance
    the two domains on average. Whether that scalar actually balances each layer
    is a separate question, and the answer lives in how the per-layer ratios
    spread around the global one.
    """

    target_matrices, target_kind = _matrices_and_kind(load_payload(target_path), target_path)
    reference_matrices, reference_kind = _matrices_and_kind(
        load_payload(reference_path), reference_path
    )
    shared = sorted(set(target_matrices) & set(reference_matrices))
    if not shared:
        print("---- per-layer diagnostic skipped: no shared layers ----")
        return

    ratios: list[tuple[str, float]] = []
    total_target = 0.0
    total_reference = 0.0
    for key in shared:
        mass_target = _layer_mass(target_matrices[key], target_kind)
        mass_reference = _layer_mass(reference_matrices[key], reference_kind)
        total_target += mass_target
        total_reference += mass_reference
        if mass_reference > 0:
            ratios.append((key, mass_target / mass_reference))

    if not ratios or total_reference <= 0:
        print("---- per-layer diagnostic skipped: reference mass is zero ----")
        return

    global_ratio = total_target / total_reference
    ratio_values = [value for _, value in ratios]
    dominated = sum(1 for value in ratio_values if value > 1.0)
    # r_l normalized by the global ratio: 1.0 means the single global weight already
    # balances that layer, so the spread of this quantity is the whole question.
    normalized = [value / global_ratio for value in ratio_values]

    target_weight = 1.0 / (1.0 + global_ratio)

    print(f"---- per-layer mass ({target_kind} target / {reference_kind} reference) ----")
    print(
        "  layer mass = the layer's weight in the penalty under a uniform dW; "
        "r_l = mass(target,l) / mass(reference,l)"
    )
    print(f"  shared layers            : {len(shared)} ({len(ratio_values)} with nonzero reference)")
    print(f"  global mass ratio        : {global_ratio:.3f}x  (target / reference)")
    print(f"  r_l across layers        : {_percentiles(ratio_values)}")
    print(f"  target dominates (r_l>1) : {dominated} / {len(ratio_values)} layers")

    order = sorted(ratios, key=lambda item: item[1])
    lows = ", ".join(f"{name.split('.')[-1] if '.' not in name else name}={value:.2f}"
                     for name, value in order[:3])
    highs = ", ".join(f"{name}={value:.2f}" for name, value in order[-3:])
    print(f"  most reference-heavy     : {lows}")
    print(f"  most target-heavy        : {highs}")

    print(
        f"  equal-contribution mix   : target={target_weight:.4f} "
        f"reference={1 - target_weight:.4f}  (bigger matrix gets the smaller coefficient)"
    )
    print(
        "  scalar-weight stability  : n_l = r_l / global_ratio, where 1.0 means the "
        "global weight already balances that layer"
    )
    print(f"    n_l across layers      : {_percentiles(normalized)}")
    spread = max(normalized) / min(normalized) if min(normalized) > 0 else float("inf")
    print(f"    spread (max/min)       : {spread:.1f}x")
    if spread <= 4:
        print(
            "    -> tight: one scalar weight balances every layer within ~2x, so the "
            "global equal-contribution weight is safe to use as-is."
        )
    else:
        print(
            "    -> wide: the average balance is not what each layer gets. A single "
            "scalar weight will over-protect some layers and under-protect others; "
            "consider whether the mix should be judged on retention rather than mass."
        )


def main() -> None:
    args = parse_args()
    if not Path(args.path).is_file():
        raise SystemExit(f"file not found: {args.path}")

    if args.print_meta:
        payload = load_payload(args.path)
        value = payload.get("metadata", {}).get(args.print_meta)
        if value is None:
            raise SystemExit(f"{args.path} has no metadata key {args.print_meta!r}")
        print(value if not isinstance(value, (dict, list)) else json.dumps(value))
        return

    meta = report_fisher(args.path, "target")
    failures = check_expectations(meta, args)

    if args.reference:
        if Path(args.reference).is_file():
            compare_scales(meta, args.reference)
            if args.layer_diagnostic == 1:
                compare_layers(args.path, args.reference)
        else:
            print(f"---- scale comparison skipped: no reference at {args.reference} ----")

    if failures:
        print("\nFAILED:")
        for item in failures:
            print(f"  - {item}")
        if args.strict == 1:
            raise SystemExit(1)
        return
    print("\nall checks passed")


if __name__ == "__main__":
    main()
