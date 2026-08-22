"""Measure how the sampled FLAN subset is distributed across FLAN's tasks.

collect_cov estimates C = E[x x^T] over a subset of the local FLAN dump chosen by
a full shuffle at --sample_seed followed by select(range(--max_samples)). That
selection is uniform over *rows*, so it reproduces the raw corpus mixture, in
which a task's share is whatever fraction of the jsonl lines it happens to own.

FLAN's own pipeline deliberately does not train on that mixture. Wei et al. cap
each dataset at 30k examples and then sample with T5 examples-proportional
mixing at a mixing rate maximum of 3k, so every task holding at least 3k rows
gets equal weight and only smaller ones are down-weighted. Longpre et al. later
ablated this and found removing mixture balancing costs ~5 points of MMLU and
~11 of BBH. The Muennighoff/flan dump is the per-task data from *before* that
mixing policy is applied, so the policy is not baked into the files.

Whether a task mixture that matters for a training loss also matters for a
second moment of hidden states is a different question, and this script does not
answer it. It only produces the numbers that question needs:

  1. per-task row share of the whole corpus,
  2. per-task row share of the sampled subset, against FLAN's capped-proportional
     target share,
  3. per-task *token* share of the sampled subset.

(3) is the share that actually weights C. The covariance hook accumulates over
every non-padding token and divides by the total token count, so a summarization
task contributes far more than a sentence-classification task at equal rows.

Sampling is not reimplemented here: the corpus load and the shuffle/select come
from onereplay.data.old_knowledge, the same functions collect_cov calls, so the
rows counted below are the rows C saw. The printed fingerprint ties this run to a
specific covariance file's metadata.

No GPU and no model weights, only the tokenizer. Usage:

    python -m onereplay.scripts.analyze_flan_task_balance \
      --data_files "/scratch/.../datasets/flan/train/*.jsonl" \
      --cache_dir /scratch/.../results/dataset_cache \
      --tokenizer_path /scratch/.../models/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.data.old_knowledge import (  # noqa: E402
    example_to_model_text,
    fingerprint_pool,
    limit_dataset,
    load_old_knowledge_dataset,
)


def parse_args() -> argparse.Namespace:
    """Mirror collect_cov's dataset-selection flags so the same rows are counted."""

    parser = argparse.ArgumentParser(description="Analyze FLAN task balance in the C corpus.")

    # --- dataset selection: must match the collect_cov invocation being audited ---
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="Muennighoff/flan")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--data_files", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--streaming", type=int, default=0)
    parser.add_argument("--text_column", type=str, default="")
    parser.add_argument("--input_column", type=str, default="inputs")
    parser.add_argument("--target_column", type=str, default="targets")
    parser.add_argument("--task_column", type=str, default="task")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--sample_shuffle", type=int, default=1)
    parser.add_argument("--sample_seed", type=int, default=1)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--require_target", type=int, default=0)

    # --- rendering: must match collect_cov so the token counts are the real ones ---
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--use_chat_template", type=int, default=1)
    parser.add_argument("--include_target_in_chat", type=int, default=1)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--enable_thinking", type=int, default=0)
    parser.add_argument("--concat_prompt_target", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--truncation_side", type=str, default="")

    # --- FLAN's balancing policy, quoted here only to compute a target share ---
    parser.add_argument(
        "--mixing_rate_max",
        type=int,
        default=3000,
        help="FLAN's examples-proportional mixing rate maximum. Target share for a "
        "task is min(rows, this) normalized. FLAN's separate 30k per-dataset cap "
        "does not change the share because it is far above this value.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many head and tail tasks to print. 0 prints every task.",
    )
    parser.add_argument(
        "--scan_files",
        type=int,
        default=1,
        help="1 also counts lines per jsonl file, which localizes a skewed or "
        "incompletely transferred corpus to a specific file.",
    )
    parser.add_argument("--out_json", type=str, default="")
    return parser.parse_args()


def scan_source_files(pattern: str) -> None:
    """Report line counts per jsonl file before touching the datasets library.

    Useful on its own: the file names in Muennighoff/flan carry the task, so a
    task that is missing or truncated shows up here without any dependency on the
    arrow cache being warm or the `task` column existing.
    """

    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"scan_files: no files matched {pattern}")
        return

    counts: list[tuple[str, int, int]] = []
    for path in paths:
        lines = 0
        with open(path, "rb") as handle:
            for _ in handle:
                lines += 1
        counts.append((os.path.basename(path), lines, os.path.getsize(path)))

    total_lines = sum(item[1] for item in counts)
    total_bytes = sum(item[2] for item in counts)
    print(f"==== source files: {len(counts)} matched {pattern} ====")
    print(f"{'file':<52} {'lines':>12} {'bytes':>16} {'line share':>11}")
    for name, lines, size in sorted(counts, key=lambda item: -item[1]):
        share = lines / total_lines if total_lines else 0.0
        print(f"{name:<52} {lines:>12,} {size:>16,} {share:>10.4f}")
    print(f"{'TOTAL':<52} {total_lines:>12,} {total_bytes:>16,}")
    print()


def column_counter(dataset, column: str, chunk: int = 100_000) -> Counter:
    """Count values of one column without materializing the whole column at once."""

    counts: Counter = Counter()
    total = len(dataset)
    for start in range(0, total, chunk):
        counts.update(dataset[start : min(start + chunk, total)][column])
    return counts


def entropy_effective_count(shares: list[float]) -> float:
    """exp of the Shannon entropy: the number of equally-weighted tasks this matches.

    Reported instead of raw entropy because it is directly comparable to the task
    count: 66 tasks with an effective count of 20 means the mixture behaves like
    20 evenly weighted tasks, whatever the long tail nominally contributes.
    """

    total = 0.0
    for share in shares:
        if share > 0:
            total -= share * math.log(share)
    return math.exp(total)


def total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    """Half the L1 distance between two share dictionaries over the union of keys."""

    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def count_tokens_per_task(dataset, tokenizer, args: argparse.Namespace) -> Counter:
    """Count the non-padding tokens each task contributes to C.

    Renders and truncates exactly as build_collate_fn does, including
    add_special_tokens=False under a chat template, so these are the token counts
    the covariance hook would accumulate. Padding is excluded by construction
    because each row is tokenized on its own rather than padded into a batch, and
    the hook drops padded positions via the attention mask anyway.
    """

    if args.truncation_side:
        tokenizer.truncation_side = args.truncation_side

    tokens: Counter = Counter()
    total = len(dataset)
    chunk = 512
    for start in range(0, total, chunk):
        rows = dataset[start : min(start + chunk, total)]
        keys = list(rows.keys())
        size = len(rows[keys[0]])
        examples = [{key: rows[key][index] for key in keys} for index in range(size)]
        texts = [example_to_model_text(example, tokenizer, args) for example in examples]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_len,
            add_special_tokens=args.use_chat_template != 1,
        )
        for example, ids in zip(examples, encoded["input_ids"]):
            tokens[str(example.get(args.task_column, "<unknown>"))] += len(ids)
        if start and start % 5120 == 0:
            print(f"  tokenized {start}/{total} rows")
    return tokens


def report_table(rows: list[dict], args: argparse.Namespace, with_tokens: bool) -> None:
    """Print the per-task table, head and tail only unless --top 0."""

    header = (
        f"{'task':<44} {'corpus':>10} {'corpus%':>8} {'flan%':>7} "
        f"{'sample':>7} {'sample%':>8}"
    )
    if with_tokens:
        header += f" {'tokens':>10} {'token%':>7} {'tok/row':>8}"
    print(header)

    def emit(row: dict) -> None:
        line = (
            f"{row['task']:<44} {row['corpus_rows']:>10,} {row['corpus_share']:>8.4f} "
            f"{row['flan_target_share']:>7.4f} {row['sample_rows']:>7,} "
            f"{row['sample_share']:>8.4f}"
        )
        if with_tokens:
            line += (
                f" {row['sample_tokens']:>10,} {row['token_share']:>7.4f} "
                f"{row['tokens_per_row']:>8.1f}"
            )
        print(line)

    if args.top <= 0 or len(rows) <= 2 * args.top:
        for row in rows:
            emit(row)
        return

    for row in rows[: args.top]:
        emit(row)
    print(f"{'...':<44} ({len(rows) - 2 * args.top} tasks omitted, pass --top 0 for all)")
    for row in rows[-args.top :]:
        emit(row)


def main() -> None:
    args = parse_args()

    if args.scan_files == 1 and args.data_files:
        scan_source_files(args.data_files)

    corpus = load_old_knowledge_dataset(args)
    if not hasattr(corpus, "select"):
        raise SystemExit(
            "this corpus loaded as a streaming IterableDataset, which cannot be counted "
            "without consuming it. Point --data_files at local json/jsonl files, which "
            "load map-style, or pass --streaming 0."
        )
    if args.task_column not in corpus.column_names:
        raise SystemExit(
            f"no --task_column {args.task_column!r} in this corpus. Available columns: "
            f"{corpus.column_names}. Without a task field only the per-file scan above "
            "describes the mixture."
        )

    corpus_counts = column_counter(corpus, args.task_column)
    corpus_total = sum(corpus_counts.values())
    corpus_share = {task: count / corpus_total for task, count in corpus_counts.items()}

    # FLAN's target: weight proportional to min(rows, mixing_rate_max), so every
    # task at or above the cap is equally weighted and only smaller ones fall off.
    capped = {task: min(count, args.mixing_rate_max) for task, count in corpus_counts.items()}
    capped_total = sum(capped.values())
    flan_share = {task: value / capped_total for task, value in capped.items()}

    sample = limit_dataset(corpus, args)
    sample_counts = Counter(sample[args.task_column])
    sample_total = sum(sample_counts.values())
    sample_share = {task: count / sample_total for task, count in sample_counts.items()}

    rows_hashed, digest = fingerprint_pool(sample, args)

    token_counts: Counter = Counter()
    with_tokens = bool(args.tokenizer_path)
    if with_tokens:
        from transformers import AutoTokenizer

        print(f"loading tokenizer from {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        token_counts = count_tokens_per_task(sample, tokenizer, args)
    token_total = sum(token_counts.values())
    token_share = {task: count / token_total for task, count in token_counts.items()}

    missing = sorted(set(corpus_counts) - set(sample_counts))
    below_cap = [task for task, count in corpus_counts.items() if count < args.mixing_rate_max]

    print(f"==== corpus: {corpus_total:,} rows over {len(corpus_counts)} tasks ====")
    print(f"effective #tasks (exp entropy) : {entropy_effective_count(list(corpus_share.values())):.1f}")
    print(f"largest task share             : {max(corpus_share.values()):.4f}")
    print(f"smallest task share            : {min(corpus_share.values()):.6f}")
    print(f"tasks below mixing_rate_max={args.mixing_rate_max}: {len(below_cap)}")
    print(f"TVD(corpus, FLAN target)       : {total_variation(corpus_share, flan_share):.4f}")
    print()

    print(
        f"==== sample: {sample_total:,} rows "
        f"(seed={args.sample_seed} shuffle={args.sample_shuffle}) ===="
    )
    print(f"fingerprint                    : {digest}  ({rows_hashed} rows hashed)")
    print(f"tasks present                  : {len(sample_counts)} / {len(corpus_counts)}")
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        print(f"tasks absent from the sample   : {len(missing)}  [{preview}{suffix}]")
    print(f"effective #tasks (exp entropy) : {entropy_effective_count(list(sample_share.values())):.1f}")
    print(f"largest task share             : {max(sample_share.values()):.4f}")
    print(
        "TVD(sample, corpus)            : "
        f"{total_variation(sample_share, corpus_share):.4f}   "
        "<- 随机抽样是否忠实复现语料分布（应接近 0）"
    )
    print(
        "TVD(sample, FLAN target)       : "
        f"{total_variation(sample_share, flan_share):.4f}   "
        "<- 与 FLAN 平衡方案的实际差距"
    )
    if with_tokens:
        print(
            f"effective #tasks by token      : "
            f"{entropy_effective_count(list(token_share.values())):.1f}"
        )
        print(
            "TVD(token share, sample rows)  : "
            f"{total_variation(token_share, sample_share):.4f}   "
            "<- 长文本任务把行份额进一步放大了多少"
        )
        print(f"total tokens entering C        : {token_total:,}")
    print()

    table = []
    for task in corpus_counts:
        sample_rows = sample_counts.get(task, 0)
        tokens = token_counts.get(task, 0)
        table.append(
            {
                "task": task,
                "corpus_rows": corpus_counts[task],
                "corpus_share": corpus_share[task],
                "flan_target_share": flan_share[task],
                "sample_rows": sample_rows,
                "sample_share": sample_share.get(task, 0.0),
                "sample_tokens": tokens,
                "token_share": token_share.get(task, 0.0),
                "tokens_per_row": (tokens / sample_rows) if sample_rows else 0.0,
            }
        )
    sort_key = "token_share" if with_tokens else "sample_share"
    table.sort(key=lambda row: -row[sort_key])

    print(f"==== per-task table (sorted by {sort_key}) ====")
    report_table(table, args, with_tokens)

    if args.out_json:
        payload = {
            "corpus_rows": corpus_total,
            "sample_rows": sample_total,
            "sample_fingerprint": digest,
            "sample_seed": args.sample_seed,
            "max_samples": args.max_samples,
            "mixing_rate_max": args.mixing_rate_max,
            "max_len": args.max_len,
            "total_tokens": token_total,
            "tasks": table,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
