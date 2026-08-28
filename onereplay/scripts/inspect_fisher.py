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

Usage:
    python -m onereplay.scripts.inspect_fisher --path F.pt --print_meta pool_fingerprint
    python -m onereplay.scripts.inspect_fisher --path F.pt --expect_max_len 2048 \
        --expect_truncation_side left --reference F_if.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
        "--reference",
        type=str,
        default="",
        help="Another Fisher file to compare scales against, e.g. F_if.",
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
                "Under left truncation the answer's tail always survives, so any such row "
                "means the truncation side did not take effect."
            )
    return failures


def compare_scales(meta: dict, reference_path: str) -> None:
    """Report mean(F)/mean(F_ref), which is what the mixing weights have to undo."""

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
