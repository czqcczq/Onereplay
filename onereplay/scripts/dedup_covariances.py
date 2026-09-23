"""Make the duplicate entries in a covariance file share one tensor.

C_l = E_x[x x^T] describes the *input* to a layer, and a transformer block feeds
several projections from the same place: q_proj, k_proj and v_proj all read the
output of input_layernorm, gate_proj and up_proj both read the output of
post_attention_layernorm. collect_cov.py registers one hook per module, so each
of those inputs is turned into an identical matrix once per module and then
stored under a separate key. On Qwen3-8B with all seven projections that is
three redundant copies per layer: 6.75 GiB of the 33.75 GiB file, and the same
6.75 GiB resident on the GPU for every step of the run.

This script finds the entries that are equal and points them at one tensor.
torch.save writes a storage once however many tensors reference it, so the
rewritten file is smaller, and move_covariances_to_device carries the sharing
through to the device instead of undoing it with per-key .to() calls.

Nothing downstream changes. Every key survives, so the layer-count assertion in
the pbs preflight still counts 252 matrices, lookup_covariance still resolves
every module, and the penalty is bit-identical. This removes copies, not
coverage.

Equality is measured, never inferred from the module name. Under
--cov_normalization base_output_norm the collected matrix is
E[(x/||W x||)(x/||W x||)^T], whose denominator is the module's own output, so
the siblings are genuinely different matrices; the script then finds no groups
and rewrites nothing, rather than corrupting the file on an assumption that
only holds for the default normalization.

Usage
  # report only, no file written -- also a check on the collection run
  python -m onereplay.scripts.dedup_covariances --cov_path .../cov_..._full.pt

  # rewrite
  python -m onereplay.scripts.dedup_covariances \
      --cov_path .../cov/cov_flan_chat_20k_full.pt \
      --out_path .../cov/cov_flan_chat_20k_full_dedup.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

BYTES_PER_GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Share one tensor between covariance entries that are equal."
    )
    parser.add_argument("--cov_path", type=str, required=True)
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Omit to only report. The input file is never written in place.",
    )
    parser.add_argument("--report_json", type=str, default="")
    parser.add_argument(
        "--expect_shared",
        type=int,
        default=1,
        help=(
            "Exit non-zero when nothing can be shared. A cov file collected with the "
            "default normalization always has q/k/v and gate/up duplicates, so an empty "
            "result means the file was collected differently than assumed. Set 0 when "
            "deduplicating a base_output_norm file, where no sharing is expected."
        ),
    )
    return parser.parse_args()


def class_of(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def load_payload(path: str) -> tuple[dict, dict[str, torch.Tensor]]:
    """Return the whole payload and its covariance dict.

    load_covariance_file drops counts and metadata, which have to survive a
    rewrite: counts is what a later re-normalization would need, and metadata
    carries the pool fingerprint that ties this file to the run that produced it.
    """

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"Unsupported covariance file format: {path}")
    if "covariances" in payload:
        return payload, payload["covariances"]
    return payload, payload


def group_identical(covariances: dict[str, torch.Tensor]) -> list[list[str]]:
    """Group keys whose matrices are equal, preserving first-seen order.

    Bucketed by shape, dtype and a float64 sum before any pairwise comparison: a
    full torch.equal sweep over the 216 matrices of one shape would read more
    than a terabyte. Two equal tensors always produce the same sum, so the
    prefilter has no false negatives; torch.equal inside a bucket removes the
    false positives, and the buckets are small enough that it is free.
    """

    buckets: dict[tuple, list[str]] = {}
    for name, matrix in covariances.items():
        fingerprint = (
            tuple(matrix.shape),
            str(matrix.dtype),
            float(matrix.sum(dtype=torch.float64)),
        )
        buckets.setdefault(fingerprint, []).append(name)

    groups: list[list[str]] = []
    for names in buckets.values():
        pending = list(names)
        while pending:
            head = pending.pop(0)
            same = [head]
            rest = []
            for other in pending:
                if torch.equal(covariances[head], covariances[other]):
                    same.append(other)
                else:
                    rest.append(other)
            pending = rest
            groups.append(same)
    return groups


def share_within_groups(
    covariances: dict[str, torch.Tensor], groups: list[list[str]]
) -> dict[str, torch.Tensor]:
    """Rebuild the dict so every group's keys reference the group's first tensor.

    Key order is taken from the input rather than from the groups, so a file that
    is deduplicated twice is byte-identical to one deduplicated once.
    """

    representative = {name: group[0] for group in groups for name in group}
    return {name: covariances[representative[name]] for name in covariances}


def report_groups(
    covariances: dict[str, torch.Tensor],
    groups: list[list[str]],
    counts: dict[str, int] | None,
) -> dict:
    """Print what was found and return it as a record."""

    shared = [group for group in groups if len(group) > 1]
    total_bytes = sum(matrix.numel() * matrix.element_size() for matrix in covariances.values())
    unique_bytes = sum(
        covariances[group[0]].numel() * covariances[group[0]].element_size() for group in groups
    )

    print(f"entries              : {len(covariances)}")
    print(f"distinct matrices    : {len(groups)}")
    print(f"groups with a copy   : {len(shared)}")
    print(f"size now             : {total_bytes / BYTES_PER_GIB:7.2f} GiB")
    print(f"size after sharing   : {unique_bytes / BYTES_PER_GIB:7.2f} GiB")
    print(f"saved                : {(total_bytes - unique_bytes) / BYTES_PER_GIB:7.2f} GiB")

    # Grouped by which projection types ended up together rather than listed per
    # layer: the answer that matters is "q_proj+k_proj+v_proj, 36 times", and 144
    # lines of module names would bury it.
    shapes: dict[tuple[str, ...], int] = {}
    for group in shared:
        signature = tuple(sorted(class_of(name) for name in group))
        shapes[signature] = shapes.get(signature, 0) + 1
    if shapes:
        print("\nwhat was merged:")
        for signature, occurrences in sorted(shapes.items(), key=lambda item: -item[1]):
            print(f"  {' + '.join(signature):<40} x{occurrences}")

    # A duplicate matrix that rests on a different token count would mean the two
    # hooks did not see the same batches, which contradicts the equality above.
    # Cheap to check here and there is no other place that would notice.
    if counts:
        inconsistent = [
            group for group in shared if len({counts.get(name) for name in group}) > 1
        ]
        if inconsistent:
            print(
                f"\nwarning: {len(inconsistent)} groups have equal matrices but different "
                "token counts; the collection run is not self-consistent"
            )

    return {
        "entries": len(covariances),
        "distinct_matrices": len(groups),
        "shared_groups": len(shared),
        "bytes_before": total_bytes,
        "bytes_after": unique_bytes,
        "merged": {" + ".join(key): value for key, value in shapes.items()},
    }


def main() -> None:
    args = parse_args()

    print(f"==== {args.cov_path} ====")
    payload, covariances = load_payload(args.cov_path)
    counts = payload.get("counts") if isinstance(payload, dict) else None

    groups = group_identical(covariances)
    record = report_groups(covariances, groups, counts)

    if record["shared_groups"] == 0:
        print(
            "\nnothing to share. Under --cov_normalization base_output_norm this is "
            "expected: x is divided by ||W x||, so sibling projections no longer see "
            "the same vectors. Under the default normalization it is not, and means "
            "the hooks did not read what this script assumes."
        )
        if args.expect_shared:
            raise SystemExit(1)
        return

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.report_json}")

    if not args.out_path:
        print("\nno --out_path, nothing written")
        return

    shared = share_within_groups(covariances, groups)
    if "covariances" in payload:
        payload["covariances"] = shared
        metadata = payload.setdefault("metadata", {})
        metadata["dedup_source"] = args.cov_path
        metadata["dedup_distinct_matrices"] = record["distinct_matrices"]
        metadata["dedup_bytes_saved"] = record["bytes_before"] - record["bytes_after"]
    else:
        payload = shared

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_path)
    written = Path(args.out_path).stat().st_size
    print(f"\nwrote {args.out_path} ({written / BYTES_PER_GIB:.2f} GiB on disk)")
    print(
        "every key survives, so the preflight's layer count and the penalty itself "
        "are unchanged; only the number of stored copies moved"
    )


if __name__ == "__main__":
    main()
