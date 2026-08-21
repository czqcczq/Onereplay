"""Download safety-eval assets on a networked (login) node.

Run this ON THE RWTH LOGIN NODE (compute nodes are offline). It fetches and
normalizes everything the safety pipeline needs into stable file names/columns
so scripts/prepare_safety_data.py works with its defaults:

  - HarmBench standard text behaviors   -> <out_dir>/harmbench_behaviors_text_all.csv
  - In-The-Wild jailbreak prompts (HF)  -> <out_dir>/in_the_wild_jailbreak.csv
  - XSTest prompts (HF)                 -> <out_dir>/xstest.csv
  - StrongREJECT forbidden prompts (raw)-> <out_dir>/strongreject_dataset.csv
  - WildGuard judge model (gated HF)    -> <model_dir>/<model_name>/
  - StrongREJECT grader model (HF)      -> <model_dir>/<sr_model_name>/

WildGuard is gated: accept the license at
https://huggingface.co/allenai/wildguard and authenticate first
(`huggingface-cli login` or `export HF_TOKEN=...`) before using --with_model.
The StrongREJECT grader (qylu4156/strongreject-15k-v1) derives from gated Gemma;
accept https://huggingface.co/google/gemma-2b and authenticate the same way.

Do NOT set HF_HUB_OFFLINE / HF_DATASETS_OFFLINE when running this; it needs net.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

# Original StrongREJECT dataset CSV (columns: category, source, forbidden_prompt).
# Raw GitHub, ungated -- unlike the walledai HF mirror which requires accepting
# terms. strongreject_dataset.csv has 313 prompts; strongreject_small.csv has 60.
STRONGREJECT_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)


def parse_args() -> argparse.Namespace:
    """Parse output locations and which assets to fetch."""

    parser = argparse.ArgumentParser(description="Download safety-eval assets (login node).")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/hpcwork/xsz96350/Chen_logs/onereplay/datasets/safety",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/hpcwork/xsz96350/Chen_logs/checkpoints",
    )
    parser.add_argument("--model_name", type=str, default="wildguard")
    parser.add_argument("--judge_repo", type=str, default="allenai/wildguard")
    parser.add_argument("--sr_model_name", type=str, default="strongreject-grader")
    parser.add_argument("--sr_judge_repo", type=str, default="qylu4156/strongreject-15k-v1")
    parser.add_argument("--strongreject_url", type=str, default=STRONGREJECT_URL)
    parser.add_argument("--itw_repo", type=str, default="TrustAIRLab/in-the-wild-jailbreak-prompts")
    parser.add_argument("--itw_config", type=str, default="jailbreak_2023_12_25")
    parser.add_argument("--itw_split", type=str, default="train")
    parser.add_argument("--xstest_repo", type=str, default="walledai/XSTest")
    parser.add_argument("--xstest_split", type=str, default="test")
    parser.add_argument("--with_data", type=int, default=1)
    parser.add_argument("--with_model", type=int, default=1)
    # Convenience switch: only fetch the StrongREJECT dataset + grader, skipping
    # HarmBench / In-The-Wild / XSTest / WildGuard (already downloaded before).
    parser.add_argument("--only_sr", type=int, default=0)
    return parser.parse_args()


def warn_if_offline() -> None:
    """Fail fast if HF offline flags are set; this script needs network."""

    offline = [
        name
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
        if os.environ.get(name) == "1"
    ]
    if offline:
        raise SystemExit(
            f"{', '.join(offline)} set to 1; unset them and run on a networked node."
        )


def download_harmbench(out_dir: Path) -> None:
    """Fetch the HarmBench standard-behaviors CSV from GitHub raw."""

    target = out_dir / "harmbench_behaviors_text_all.csv"
    print(f"[harmbench] {HARMBENCH_URL} -> {target}")
    urllib.request.urlretrieve(HARMBENCH_URL, target)
    print(f"[harmbench] wrote {target} ({target.stat().st_size} bytes)")


def download_strongreject(args: argparse.Namespace, out_dir: Path) -> None:
    """Fetch the StrongREJECT forbidden-prompts CSV from GitHub raw."""

    target = out_dir / "strongreject_dataset.csv"
    print(f"[strongreject] {args.strongreject_url} -> {target}")
    urllib.request.urlretrieve(args.strongreject_url, target)
    print(f"[strongreject] wrote {target} ({target.stat().st_size} bytes)")


def download_itw(args: argparse.Namespace, out_dir: Path) -> None:
    """Fetch In-The-Wild jailbreak prompts and export as CSV."""

    from datasets import load_dataset

    print(f"[in-the-wild] load {args.itw_repo}:{args.itw_config}[{args.itw_split}]")
    dataset = load_dataset(args.itw_repo, args.itw_config, split=args.itw_split)
    target = out_dir / "in_the_wild_jailbreak.csv"
    dataset.to_csv(str(target))
    print(f"[in-the-wild] wrote {target} ({len(dataset)} rows, columns={dataset.column_names})")


def download_xstest(args: argparse.Namespace, out_dir: Path) -> None:
    """Fetch XSTest prompts and export as CSV."""

    from datasets import load_dataset

    print(f"[xstest] load {args.xstest_repo}[{args.xstest_split}]")
    dataset = load_dataset(args.xstest_repo, split=args.xstest_split)
    target = out_dir / "xstest.csv"
    dataset.to_csv(str(target))
    print(f"[xstest] wrote {target} ({len(dataset)} rows, columns={dataset.column_names})")


def download_model(args: argparse.Namespace) -> None:
    """Snapshot the gated WildGuard judge model into <model_dir>/<model_name>."""

    from huggingface_hub import snapshot_download

    target = Path(args.model_dir) / args.model_name
    target.mkdir(parents=True, exist_ok=True)
    print(f"[judge] snapshot {args.judge_repo} -> {target}")
    try:
        snapshot_download(repo_id=args.judge_repo, local_dir=str(target))
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            f"[judge] failed: {error}\n"
            "WildGuard is gated: accept the license at "
            f"https://huggingface.co/{args.judge_repo} and run "
            "`huggingface-cli login` (or export HF_TOKEN) first."
        )
    print(f"[judge] done -> {target}")


def download_sr_model(args: argparse.Namespace) -> None:
    """Snapshot the StrongREJECT grader into <model_dir>/<sr_model_name>."""

    from huggingface_hub import snapshot_download

    target = Path(args.model_dir) / args.sr_model_name
    target.mkdir(parents=True, exist_ok=True)
    print(f"[sr-grader] snapshot {args.sr_judge_repo} -> {target}")
    try:
        snapshot_download(repo_id=args.sr_judge_repo, local_dir=str(target))
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            f"[sr-grader] failed: {error}\n"
            "The StrongREJECT grader derives from gated Gemma: accept "
            "https://huggingface.co/google/gemma-2b and run "
            "`huggingface-cli login` (or export HF_TOKEN) first."
        )
    print(f"[sr-grader] done -> {target}")


def main() -> None:
    """Download the requested safety assets and report where they landed."""

    args = parse_args()
    warn_if_offline()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only_sr:
        download_strongreject(args, out_dir)
        download_sr_model(args)
        print("done (only_sr). StrongREJECT dataset + grader fetched.")
        sys.stdout.flush()
        return

    if args.with_data:
        download_harmbench(out_dir)
        download_itw(args, out_dir)
        download_xstest(args, out_dir)
        download_strongreject(args, out_dir)
    if args.with_model:
        download_model(args)
        download_sr_model(args)

    print("done. next: sanity-check columns with scripts/prepare_safety_data.py")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
