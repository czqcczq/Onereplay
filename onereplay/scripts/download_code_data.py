"""Fetch HumanEval and MBPP into the layout the code metrics expect.

Run this on a login node: the compute nodes set HF_HUB_OFFLINE=1 because this
site hangs on outbound network, so nothing can be downloaded from inside a job.

The two metrics want different on-disk shapes, which is why this is not one
call: humaneval.py does load_dataset("parquet", data_files=...) so it needs a
single parquet file, while mbpp.py does load_from_disk() so it needs a saved
dataset directory with the splits intact.

    python -m onereplay.scripts.download_code_data --out_dir /path/datasets/code
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--humaneval_repo", type=str, default="openai/openai_humaneval")
    parser.add_argument("--mbpp_repo", type=str, default="google-research-datasets/mbpp")
    parser.add_argument(
        "--mbpp_config",
        type=str,
        default="full",
        help="'full' keeps the 500-problem test split the benchmark is defined on; "
        "'sanitized' is a different, smaller set and is not comparable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or None

    humaneval = load_dataset(args.humaneval_repo, split="test", cache_dir=cache)
    humaneval_path = out_dir / "humaneval_test.parquet"
    humaneval.to_parquet(str(humaneval_path))
    print(f"HumanEval: {len(humaneval)} 条 -> {humaneval_path}")

    mbpp = load_dataset(args.mbpp_repo, args.mbpp_config, cache_dir=cache)
    mbpp_path = out_dir / f"mbpp_{args.mbpp_config}"
    mbpp.save_to_disk(str(mbpp_path))
    sizes = ", ".join(f"{name}={len(split)}" for name, split in mbpp.items())
    print(f"MBPP ({args.mbpp_config}): {sizes} -> {mbpp_path}")
    print(
        "\n注意：评测要用 --dataset_split test（500 题）。evaluate.py 的默认值是\n"
        "validation，那只有 90 题，配对可分辨下限会差三倍。"
    )


if __name__ == "__main__":
    main()
