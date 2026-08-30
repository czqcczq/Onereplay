"""Fetch APPS into the layout the apps metric expects, and report its layout.

Run this on a login node: the compute nodes set HF_HUB_OFFLINE=1 because this
site hangs on outbound network, so nothing can be downloaded from inside a job.

    python -m onereplay.scripts.download_apps_data --out_dir /path/datasets/code

Produces apps_train.parquet (5000) and apps_test.parquet (5000). The metric
reads a single parquet via load_dataset("parquet", ...), matching humaneval.

The report at the end is the point of running this before anything else. APPS is
ordered by problem_id and the difficulty tiers are contiguous blocks, not
shuffled, so "the first N problems of the test split" is a difficulty selection
whether you meant it to be or not. It also counts call-based problems, which the
apps metric drops by default (they feed arguments to a named function rather
than stdin, so the stdin runner cannot score them).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--apps_repo", type=str, default="codeparrot/apps")
    parser.add_argument(
        "--head",
        type=int,
        default=500,
        help="Report the difficulty mix of the first N test problems.",
    )
    return parser.parse_args()


def load_split(repo: str, split: str, cache: str | None):
    from datasets import load_dataset

    try:
        return load_dataset(repo, split=split, cache_dir=cache)
    except Exception:
        # Older revisions of codeparrot/apps ship a loading script.
        return load_dataset(repo, split=split, cache_dir=cache, trust_remote_code=True)


def report(dataset, head: int) -> None:
    difficulties = list(dataset["difficulty"])
    print(f"\n  total: {len(difficulties)}")
    print(f"  difficulty mix: {dict(Counter(difficulties))}")

    print("  contiguous index range per difficulty:")
    for name in ("introductory", "interview", "competition"):
        indices = [i for i, value in enumerate(difficulties) if value == name]
        if indices:
            print(f"    {name:<13} n={len(indices):<5} index {indices[0]}..{indices[-1]}")

    if head > 0:
        print(f"  first {head} problems: {dict(Counter(difficulties[:head]))}")

    call_based = 0
    no_tests = 0
    for raw in dataset["input_output"]:
        try:
            spec = json.loads(raw or "{}")
        except (json.JSONDecodeError, TypeError):
            no_tests += 1
            continue
        if spec.get("fn_name"):
            call_based += 1
        if not (spec.get("inputs") and spec.get("outputs")):
            no_tests += 1
    print(f"  call-based (dropped by the stdin judge): {call_based}")
    print(f"  missing or unparseable test cases      : {no_tests}")


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or None

    for split in ("train", "test"):
        dataset = load_split(args.apps_repo, split, cache)
        path = out_dir / f"apps_{split}.parquet"
        dataset.to_parquet(str(path))
        print(f"\nAPPS {split}: {len(dataset)} 条 -> {path}")
        report(dataset, args.head if split == "test" else 0)

    print(
        "\n下一步：先验收判分器，再看任何模型分数。\n"
        f"  python -m onereplay.scripts.check_apps_judge \\\n"
        f"    --apps_data_file {out_dir / 'apps_test.parquet'} \\\n"
        "    --apps_difficulties introductory --limit 100\n"
        "gold solution 的 strict pass 就是这套 harness 的天花板；它偏低就先修判分器。"
    )


if __name__ == "__main__":
    main()
