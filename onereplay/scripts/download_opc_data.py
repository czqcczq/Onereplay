"""Fetch OpenCoder opc-sft-stage2 into the layout prepare_opc_ccode expects.

Run this on a login node: the compute nodes set HF_HUB_OFFLINE=1 because this
site hangs on outbound network, so nothing can be downloaded from inside a job.

Saved as parquet rather than save_to_disk for the same reason as the Magicoder
downloader: prepare_opc_ccode reads it through Dataset.from_parquet, which
avoids the packaged-builder Hub round-trip that hangs offline.

    python -m onereplay.scripts.download_opc_data \\
        --out_dir /path/datasets/code_replay

Only educational_instruct is pulled by default. It is the subset whose schema
lines up with the eval sets -- (instruction, code, entry_point, testcase),
against MBPP's (text, code, test_list) -- and the one the OpenCoder card
describes as "(instruction, code, test case) triples generated from the
algorithmic corpus, validated through a Python compiler". The other three
subsets are not wanted here: evol_instruct is Magicoder-Evol-Instruct-110k
(the pool this replaces), mceval_instruct is multilingual, and
package_instruct is pydoc API questions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--opc_repo", type=str, default="OpenCoder-LLM/opc-sft-stage2")
    parser.add_argument("--opc_config", type=str, default="educational_instruct")
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.opc_repo, args.opc_config, split=args.split, cache_dir=args.cache_dir or None
    )
    out_path = out_dir / f"opc_{args.opc_config}.parquet"
    dataset.to_parquet(str(out_path))
    print(f"{args.opc_repo}:{args.opc_config}: {len(dataset)} 条 -> {out_path}")
    print(f"columns: {dataset.column_names}")

    required = {"instruction", "output", "code", "entry_point", "testcase"}
    missing = required - set(dataset.column_names)
    if missing:
        print(f"!! 缺少 prepare_opc_ccode 需要的列: {sorted(missing)}")
        return

    # The two numbers that decide how much of the pool can become a
    # HumanEval-style stub, reported here so the建池 job does not have to be the
    # first place they show up.
    no_entry_point = sum(1 for value in dataset["entry_point"] if not str(value).strip())
    test_counts = Counter(len(tests or []) for tests in dataset["testcase"])
    with_tests = sum(count for size, count in test_counts.items() if size > 0)
    print(f"缺 entry_point: {no_entry_point} 条")
    print(f"带测试用例: {with_tests} 条（用例数中位档 {test_counts.most_common(3)}）")

    lengths = sorted(len(str(text)) for text in dataset["instruction"])
    if lengths:
        p50 = lengths[len(lengths) // 2]
        p99 = lengths[min(len(lengths) - 1, int(0.99 * len(lengths)))]
        print(f"instruction 字符数: P50={p50} P99={p99} max={lengths[-1]}")

    print(
        "\n下一步（计算节点）: qsub onereplay/pbs/50_opc_code_data_cov.pbs"
        "\n注意 testcase 不会进 prompt，只在建池时用来校验 HumanEval 式改造是否保持语义。"
    )


if __name__ == "__main__":
    main()
