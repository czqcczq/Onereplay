"""Fetch Magicoder OSS-Instruct into the layout prepare_magicoder_ccode expects.

Run this on a login node: the compute nodes set HF_HUB_OFFLINE=1 because this
site hangs on outbound network, so nothing can be downloaded from inside a job.

Saved as parquet rather than save_to_disk because prepare_magicoder_ccode reads
it through the same Dataset.from_parquet path prepare_metamath_cmath uses, which
avoids the packaged-builder Hub round-trip that hangs offline.

    python -m onereplay.scripts.download_magicoder_data \\
        --out_dir /path/datasets/code_replay
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument(
        "--magicoder_repo",
        type=str,
        default="ise-uiuc/Magicoder-OSS-Instruct-75K",
        help="The 75K release with {problem, solution, lang}. The "
        "'-Instruction-Response' sibling names the same fields "
        "{instruction, response}; prepare_magicoder_ccode accepts either.",
    )
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.magicoder_repo, split=args.split, cache_dir=args.cache_dir or None
    )
    out_path = out_dir / "magicoder_oss_instruct_75k.parquet"
    dataset.to_parquet(str(out_path))
    print(f"Magicoder OSS-Instruct: {len(dataset)} 条 -> {out_path}")
    print(f"columns: {dataset.column_names}")

    if "lang" in dataset.column_names:
        counts: dict[str, int] = {}
        for value in dataset["lang"]:
            key = str(value).lower()
            counts[key] = counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda item: -item[1])
        print("lang 分布: " + ", ".join(f"{name}={count}" for name, count in top))
        print(
            "\n注意：`lang` 标的是 **seed 代码片段** 的语言，不是生成出来的问题的语言。"
            "\nMagicoder 论文 4.1 节明确说 OSS-Instruct 可能产出与 seed 不同语言的代码，"
            "\n作者自己统计 Python 用的判据是生成内容里是否含 ```python（约 43K），而按"
            "\nlang=='python' 过滤是另一个数（约 38K）。prepare_magicoder_ccode 默认取"
            "\n两者的合取，避免把非 Python 内容混进 C_code。"
        )


if __name__ == "__main__":
    main()
