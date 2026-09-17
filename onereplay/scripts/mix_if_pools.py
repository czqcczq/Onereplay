"""Concatenate several single-turn SFT pools into one training pool.

Why this is a separate step instead of a flag on prepare_flan_v2 /
prepare_conifer: the IF line has two capability sources, they teach different
halves of what IFEval measures, and both halves are measured. On
Qwen2.5-Math-1.5B, 1 epoch each, scored against the same base:

  FLAN v2 100k    answer p50 6 words    -> response p50 32 words
                  constraints that only RESTRICT output   27.3% -> 40.3%
                  constraints that need EXTRA output      35.3% -> 13.4%
                  IFEval prompt-strict                    20.7% -> 15.9%
  Conifer 38.9k   answer p50 144 words  -> response p50 189 words
                  restrict  27.3% -> 33.5%    produce  35.3% -> 31.5%
                  IFEval prompt-strict                    20.7% -> 22.4%

Conifer is the only arm that beat base overall, and the two sources own
disjoint constraint types rather than one dominating:

  Conifer only    number_bullet_lists   6.5% -> 32.3%   (FLAN drives it to 0.0%)
                  multiple_sections    14.3% -> 35.7%   (FLAN drives it to 0.0%)
                  json_format           0.0% -> 41.2%   (FLAN reaches 5.9%)
  FLAN only       punctuation:no_comma 34.8% -> 71.2%   (Conifer: 30.3%)
                  forbidden_words      36.7% -> 69.4%   (Conifer: 36.7%, unmoved)

Mixing is an attempt at both halves at once.

The ratio is a length decision, not a row-count decision
--------------------------------------------------------
Across both runs the model's output length tracked its pool's answer length
closely (6 -> 32, 144 -> 189). A pool that is half short-answer FLAN will pull
generation length back down and give up the produce-type constraints again, so
the long source has to stay in the majority. FLAN's contribution should also be
drawn from its long tail -- prepare_flan_v2's --answer_word_buckets with a
zero weight on the short bucket does exactly that -- because its 35% of one-
and two-word rows are what collapsed the length in the first place.

Worth knowing when reading the result: part of FLAN's no_comma and
forbidden_words advantage is a by-product of answering in 32 words rather than
an independent skill, so a mix that restores length should not be expected to
keep those numbers intact.

Columns
-------
Only {instruction, input, output} survive. That is all build_loader reads, and
the two sources carry different extra columns, so concatenation needs the
intersection. A pool_source column is added instead, which survives into
save_to_disk and lets a later analysis slice by origin.

Usage
-----
    python -m onereplay.scripts.mix_if_pools \\
        --pools /scratch/.../conifer_if_full,/scratch/.../flan_v2_if_long \\
        --rows 0,20000 \\
        --out_dir /scratch/.../if_mix \\
        --tokenizer_path /scratch/.../models/Qwen2.5-Math-1.5B
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

SFT_COLUMNS = ("instruction", "input", "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate single-turn SFT pools into one training pool."
    )
    parser.add_argument(
        "--pools",
        type=str,
        required=True,
        help="Comma-separated save_to_disk directories. Put the long-answer "
        "source first so the report reads in the order the mix was reasoned "
        "about; order does not affect the output, which is shuffled.",
    )
    parser.add_argument(
        "--rows",
        type=str,
        default="",
        help="Rows to take from each pool, comma-separated and aligned with "
        "--pools; 0 = the whole pool, empty = all of everything. Each pool was "
        "already shuffled by the script that built it, so a prefix is a "
        "reproducible random sample. Asking for more than a pool holds is an "
        "error rather than a silent short draw -- a mix that quietly came out "
        "20k rows light would be read as a ratio it never had.",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Model dir for the token-length report. Empty reports words only, "
        "which is enough to pick a ratio but not to pick --max_len.",
    )
    parser.add_argument(
        "--length_sample",
        type=int,
        default=5000,
        help="Rows measured per pool for the token report; 0 = all.",
    )
    parser.add_argument(
        "--jsonl_dir",
        type=str,
        default="",
        help="Where the manifest goes. Defaults to out_dir's parent, kept "
        "outside out_dir so save_to_disk owns that directory alone.",
    )
    return parser.parse_args()


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile of an already sorted list."""

    if not values:
        return 0
    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def word_stats(answers: list[str]) -> dict[str, Any]:
    words = sorted(len((answer or "").split()) for answer in answers)
    return {
        "rows": len(words),
        "mean": round(sum(words) / len(words), 1) if words else 0.0,
        **{f"p{q}": percentile(words, q) for q in (10, 25, 50, 75, 90, 99)},
        "max": words[-1] if words else 0,
    }


def load_pool(path: Path, rows: int) -> tuple[Any, int, list[str]]:
    """One pool, cut to `rows`, reduced to the three columns build_loader reads."""

    from datasets import load_from_disk

    if not path.exists():
        raise SystemExit(f"池子不存在: {path}")
    dataset = load_from_disk(str(path))
    if hasattr(dataset, "keys"):
        split = "train" if "train" in dataset else sorted(dataset.keys())[0]
        dataset = dataset[split]

    missing = [name for name in SFT_COLUMNS if name not in dataset.column_names]
    if missing:
        raise SystemExit(
            f"{path} 缺 {missing}；现有列 {dataset.column_names}。"
            "这个脚本只接 {instruction, input, output} 形态的池子。"
        )
    dropped = [name for name in dataset.column_names if name not in SFT_COLUMNS]
    if dropped:
        dataset = dataset.remove_columns(dropped)

    available = len(dataset)
    if rows > 0:
        if rows > available:
            raise SystemExit(
                f"{path} 只有 {available} 行，要不到 {rows} 行。\n"
                "   调小 --rows，或者放宽建这个池时的长度门槛"
                "（prepare_flan_v2 的 --answer_word_buckets 边界往下挪）。"
            )
        dataset = dataset.select(range(rows))
    return dataset, available, dropped


def token_report(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """Token lengths through the same apply_chat_template the trainer uses."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_source[row["pool_source"]].append(row)

    report: dict[str, Any] = {"tokenizer": args.tokenizer_path, "per_source": {}}
    full_all: list[int] = []
    answer_all: list[int] = []
    for label, subset in sorted(by_source.items()):
        sample = subset if args.length_sample <= 0 else subset[: args.length_sample]
        full: list[int] = []
        answer: list[int] = []
        for row in sample:
            user = row["instruction"]
            if row["input"] and row["input"].strip():
                user = f"{user}\n\nInput:\n{row['input'].strip()}"
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["output"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            full.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
            answer.append(
                len(tokenizer(row["output"], add_special_tokens=False)["input_ids"])
            )
        full.sort()
        answer.sort()
        full_all.extend(full)
        answer_all.extend(answer)
        report["per_source"][label] = {
            "rows_measured": len(sample),
            "full": {f"p{q}": percentile(full, q) for q in (50, 90, 99)}
            | {"max": full[-1] if full else 0},
            "answer": {f"p{q}": percentile(answer, q) for q in (50, 90, 99)}
            | {"max": answer[-1] if answer else 0},
        }

    full_all.sort()
    answer_all.sort()
    report["full"] = {f"p{q}": percentile(full_all, q) for q in (50, 90, 95, 99)} | {
        "max": full_all[-1] if full_all else 0
    }
    report["answer"] = {f"p{q}": percentile(answer_all, q) for q in (50, 90, 99)} | {
        "max": answer_all[-1] if answer_all else 0
    }
    # Truncation keeps the END of the sequence. That is benign on FLAN (the
    # question sits last) and destructive on Conifer (the constraints sit
    # first), so a mixed pool needs the budget the Conifer half requires.
    report["truncation_by_max_len"] = {
        budget: {
            "rows_truncated": sum(value > budget for value in full_all),
            "fraction_truncated": round(
                sum(value > budget for value in full_all) / len(full_all), 4
            )
            if full_all
            else 0.0,
        }
        for budget in (1024, 1536, 2048, 3072)
    }
    return report


def main() -> None:
    args = parse_args()

    paths = [Path(piece.strip()) for piece in args.pools.split(",") if piece.strip()]
    if not paths:
        raise SystemExit("--pools 是空的")
    if args.rows.strip():
        counts = [int(piece) for piece in args.rows.split(",") if piece.strip()]
        if len(counts) != len(paths):
            raise SystemExit(
                f"--pools 有 {len(paths)} 个，--rows 给了 {len(counts)} 个，要一一对应"
            )
    else:
        counts = [0] * len(paths)

    out_dir = Path(args.out_dir)
    print("==== 混合 IF 池 ====")

    from datasets import Dataset, DatasetDict, concatenate_datasets

    pieces = []
    source_info: dict[str, Any] = {}
    for path, want in zip(paths, counts):
        dataset, available, dropped = load_pool(path, want)
        label = path.name
        if label in source_info:
            raise SystemExit(f"两个池子的目录名都叫 {label}，没法区分来源")
        dataset = dataset.add_column("pool_source", [label] * len(dataset))
        pieces.append(dataset)
        source_info[label] = {
            "path": str(path),
            "rows_available": available,
            "rows_taken": len(dataset),
            "columns_dropped": dropped,
            "answer_words": word_stats(list(dataset["output"])),
        }
        stats = source_info[label]["answer_words"]
        print(
            f"  {label:<24} 取 {len(dataset):>6,} / 可用 {available:>6,}   "
            f"答案词数 p50={stats['p50']:>4} p90={stats['p90']:>4}"
            + (f"   丢列 {dropped}" if dropped else "")
        )

    merged = concatenate_datasets(pieces)
    rows: list[dict[str, Any]] = list(merged)
    # Shuffle across sources, otherwise a --max_train_samples cut or an
    # interrupted epoch sees one source only, and train.py's first steps would
    # be pure Conifer.
    random.Random(args.seed).shuffle(rows)

    total = len(rows)
    mixed_stats = word_stats([row["output"] for row in rows])
    print(f"\n  合并 {total:,} 行")
    for label, info in source_info.items():
        print(f"    {label:<24} {info['rows_taken']:>6,}  {info['rows_taken']/total:>6.1%}")
    print(
        f"  混合后答案词数  p10={mixed_stats['p10']} p25={mixed_stats['p25']} "
        f"p50={mixed_stats['p50']} p75={mixed_stats['p75']} p90={mixed_stats['p90']} "
        f"max={mixed_stats['max']}"
    )
    print(
        "  参照：两轮实验里模型输出长度紧跟池子的答案长度"
        "（FLAN 6 词 -> 输出 32 词，Conifer 144 词 -> 输出 189 词）。"
        "\n        上面这个 p50 就是对这次训练后输出长度的预测。"
    )

    # The token report runs before anything is written, and a failure in it is
    # not fatal: the pool itself is already valid without it, and letting a
    # tokenizer problem abort the job would leave a pool on disk with no
    # manifest next to it while `set -e` reported the data stage as failed.
    length_report: dict[str, Any] | None = None
    length_error = ""
    if args.tokenizer_path:
        try:
            length_report = token_report(rows, args)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            length_error = f"{type(exc).__name__}: {exc}"

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(out_dir))
    print(f"\n写出混合池 {out_dir}  ({total} 行)")

    manifest: dict[str, Any] = {
        "out_dir": str(out_dir),
        "num_rows": total,
        "seed": args.seed,
        "columns": [*SFT_COLUMNS, "pool_source"],
        "columns_note": "only the three build_loader reads, plus pool_source; "
        "the sources' own extra columns cannot survive concatenation because "
        "they differ between sources",
        "sources": source_info,
        "answer_words_mixed": mixed_stats,
        "ratio_note": "the mix ratio is a length decision: measured on "
        "Qwen2.5-Math-1.5B, response p50 tracked pool answer p50 (FLAN 6->32, "
        "Conifer 144->189), so a short-answer majority gives back the "
        "produce-type IFEval constraints",
    }
    if length_report is not None:
        manifest["length"] = report = length_report
        print("\n==== token 长度（同 apply_chat_template 口径）====")
        print(
            f"  full   p50={report['full']['p50']} p90={report['full']['p90']} "
            f"p95={report['full']['p95']} max={report['full']['max']}"
        )
        print(
            f"  answer p50={report['answer']['p50']} p90={report['answer']['p90']} "
            f"max={report['answer']['max']}"
        )
        print("  截断率（保留序列末尾，Conifer 的约束在开头，被切掉是静默的）:")
        for budget, entry in report["truncation_by_max_len"].items():
            print(
                f"    max_len={budget:<5} {entry['rows_truncated']:>6,} 行  "
                f"{entry['fraction_truncated']:>6.2%}"
            )
    elif length_error:
        manifest["length_error"] = length_error
        print(f"\n!! token 长度报告失败了：{length_error}")
        print("   混合池本身没问题，缺的只是截断表，所以定不了 MAXLEN。")
        print("   要么修 tokenizer 再单独重跑这一步，要么接受当前 MAXLEN 的风险：")
        print("   截断保留序列末尾，而 Conifer 的约束写在 user turn 开头。")
    else:
        print("\n（没传 --tokenizer_path，所以没有 token 长度和截断表，定不了 MAXLEN）")

    jsonl_dir = Path(args.jsonl_dir) if args.jsonl_dir else out_dir.parent
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = jsonl_dir / f"{out_dir.name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"写出 manifest {manifest_path}")


if __name__ == "__main__":
    main()
