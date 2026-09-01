"""Print the fingerprint of the old-knowledge rows a set of flags selects.

OneReplay estimates C and EWC estimates F; the comparison between them only
isolates the weighting if both saw the same rows. On the local-files path that
selection is deterministic (load_dataset gives a map-style Dataset, shuffle is a
full permutation at a fixed seed, select takes a prefix), but a datasets upgrade
that changed the permutation, or a flag that drifted between two slurm scripts,
would move the sample without raising anything.

This script loads nothing but the corpus, so it costs seconds and no GPU. Run it
once with collect_cov's flags and once with collect_fisher's and compare the two
lines it prints.

Usage: python -m onereplay.scripts.check_old_knowledge_pool [same dataset args]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.data.old_knowledge import (  # noqa: E402
    filter_incomplete_rows,
    fingerprint_pool,
    limit_dataset,
    load_old_knowledge_dataset,
)


def parse_args() -> argparse.Namespace:
    """Mirror the dataset-selection flags of collect_cov and collect_fisher."""

    parser = argparse.ArgumentParser(description="Fingerprint an old-knowledge sample.")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="Muennighoff/flan")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--data_files", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--streaming", type=int, default=1)
    parser.add_argument("--text_column", type=str, default="")
    parser.add_argument("--input_column", type=str, default="inputs")
    parser.add_argument("--target_column", type=str, default="targets")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--sample_shuffle", type=int, default=1)
    parser.add_argument("--sample_seed", type=int, default=1)
    parser.add_argument(
        "--sample_strategy",
        type=str,
        choices=["uniform", "balanced"],
        default="uniform",
        help="Mirror collect_cov: uniform or FLAN-style task-balanced sampling.",
    )
    parser.add_argument("--task_column", type=str, default="task")
    parser.add_argument("--mixing_rate_max", type=int, default=3000)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--require_target", type=int, default=0)
    parser.add_argument(
        "--require_target_column",
        type=str,
        default="",
        help="Mirror collect_cov: column --require_target checks, empty means --target_column.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="pool",
        help="Tag printed alongside the hash so two invocations are easy to tell apart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_old_knowledge_dataset(args)
    if args.require_target == 1:
        dataset = filter_incomplete_rows(dataset, args)
    dataset = limit_dataset(dataset, args)
    rows, digest = fingerprint_pool(dataset, args)
    if digest == "streaming-not-fingerprinted":
        raise SystemExit(
            "this corpus loaded as a streaming IterableDataset, which cannot be "
            "fingerprinted without consuming it. Point --data_files at local json/jsonl "
            "files, which load map-style, or pass --streaming 0."
        )
    print(f"{args.label} rows={rows} fingerprint={digest}")


if __name__ == "__main__":
    main()
