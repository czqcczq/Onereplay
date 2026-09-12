"""Normalize Commonsense170k into the {instruction, input, output} SFT schema.

Commonsense170k (LLM-Adapters, arXiv 2304.01933) is the training halves of
eight commonsense datasets -- BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e,
ARC-c, OBQA -- rendered into one instruction template per dataset. It already
ships in the schema train.py wants:

    instruction  the full rendered prompt, ending in "Answer format: a/b/c"
    input        always empty
    output       "the correct answer is <label>"
    answer       "<label>"

So conversion is nearly a no-op, and this script exists for the three things
that are not: the task tag, the length report, and the replay pool.

--------------------------------------------------------------------------
What this pool is, and what it is not
--------------------------------------------------------------------------
The eight test sets that metrics/commonsense_qa.py scores are the held-out
halves of these same eight datasets. Training here and evaluating there is
strictly in-domain, which is exactly what a Part 1 sanity check wants -- if
this pool cannot move BoolQ and PIQA, nothing will -- but it is not a general
or "open domain" capability, and calling it one would not survive review. The
capability this line establishes is multiple-choice commonsense QA in a fixed
template.

--------------------------------------------------------------------------
The number that makes this line different: supervised tokens per row
--------------------------------------------------------------------------
Every answer here is "the correct answer is <label>", which is 7-8 tokens, and
that is the entire supervised span -- the instruction is masked to -100. The
other two Part 1 lines carry roughly 330 (Conifer) and 3767 (OpenR1) supervised
tokens per row. Two consequences:

  * The optimal learning rate has no reason to match either of them. Per
    optimizer update this line back-propagates about one fiftieth of the IF
    line's supervision, so run MODE=lr_sweep rather than reusing 5e-5.
  * Part 2's replay budget cannot be quoted in rows and in supervised tokens
    interchangeably. With this line in the chain the two units differ by more
    than two orders of magnitude, so the choice decides the result. The length
    report prints both.

--------------------------------------------------------------------------
Truncation, and why 512 needs checking rather than inheriting
--------------------------------------------------------------------------
tokenizer_to_ids truncates with full_input_ids[-max_length:], keeping the END.
Since the answer is 7 tokens, a truncated row never loses its answer and never
loses the "Answer format: a/b/c" tail -- but it does lose the head of the
instruction, which is the task description and the start of the question stem.
Such a row teaches "choose among these options without having read the
question". HellaSwag contexts and ARC/OBQA stems are the long tail here, so
read the report before keeping --max_len at the 512 the LoRA line used.

--------------------------------------------------------------------------
Task tags are exact for four datasets and pooled for four
--------------------------------------------------------------------------
The label family identifies the source dataset for BoolQ (true/false), PIQA
(solution1/2), WinoGrande (option1/2) and HellaSwag (ending1-4). SIQA, ARC-e,
ARC-c and OBQA all render as "answer1..answerN" under the same
"Please choose the correct answer to the question:" prefix, so they are not
separable from the row alone; they are tagged answer_mc with num_options kept
alongside. Guessing between them would put a wrong label in the manifest, and
nothing downstream needs it: evaluation reads per-task test files whose task
name comes from the directory, and replay only needs the tag to balance.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

ANSWER_FORMAT_RE = re.compile(r"answer format:\s*([A-Za-z0-9/ ]+)", re.IGNORECASE)

# Exact label set -> (source dataset, family), where the mapping is one-to-one.
# The family cannot be derived for true/false the way it can for the numbered
# ones, so all three are spelled out.
LABELS_TO_TASK = {
    ("true", "false"): ("boolq", "true_false"),
    ("solution1", "solution2"): ("piqa", "solution"),
    ("option1", "option2"): ("winogrande", "option"),
}

# Budgets the truncation table reports on. 512 is what the LoRA line used.
CANDIDATE_MAX_LENS = (256, 384, 512, 768, 1024, 1536)


def parse_args() -> argparse.Namespace:
    """Parse the Commonsense170k location, pool mode, and output settings."""

    parser = argparse.ArgumentParser(
        description="Build the Commonsense170k SFT training pool."
    )
    parser.add_argument(
        "--commonsense_path",
        type=str,
        default="",
        help="commonsense_170k.json, a directory holding it, a save_to_disk "
        "dir, or a parquet file. Empty falls back to --commonsense_repo.",
    )
    parser.add_argument("--commonsense_repo", type=str, default="zwhe99/commonsense_170k")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="save_to_disk target; pass this as train.py --dataset_path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "sampled"],
        default="full",
        help="full keeps every row. sampled draws --pool_size rows for Part 2's "
        "replay pool.",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        choices=["balanced", "random"],
        default="balanced",
        help="balanced draws equally from each task tag, so the replay pool is "
        "not dominated by HellaSwag and ARC, which together are over half the "
        "pool. random draws uniformly over rows.",
    )
    parser.add_argument("--pool_size", type=int, default=20000)
    parser.add_argument(
        "--drop_unparsable",
        type=int,
        default=1,
        help="1 drops rows whose instruction has no 'Answer format: a/b/c' tail "
        "or whose gold answer is not in it. Such a row cannot be scored by "
        "ranking, and if it reaches the replay pool it teaches a label the "
        "metric will never propose.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Truncate the pool after conversion; 0 keeps all ~170k.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="Budget the truncation report is headlined against; does not itself truncate.",
    )
    parser.add_argument(
        "--length_sample",
        type=int,
        default=0,
        help="Rows measured for the length report; 0 = all. These sequences are "
        "short, so measuring all of them is affordable and removes the sampling "
        "question from the --max_len decision.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Model dir for the length report. Empty skips the report, which "
        "also skips the only check on --max_len.",
    )
    parser.add_argument(
        "--jsonl_dir",
        type=str,
        default="",
        help="Where the manifest goes. Defaults to out_dir's parent.",
    )
    return parser.parse_args()


def load_commonsense(args: argparse.Namespace):
    """Load Commonsense170k from json, parquet, a save_to_disk dir, or the hub."""

    from datasets import Dataset, load_dataset, load_from_disk

    if not args.commonsense_path:
        return load_dataset(args.commonsense_repo, split="train")

    source = Path(args.commonsense_path)
    if source.is_file():
        if source.suffix == ".parquet":
            return Dataset.from_parquet(str(source))
        return Dataset.from_list(json.loads(source.read_text(encoding="utf-8")))

    if not source.is_dir():
        raise SystemExit(f"--commonsense_path does not exist: {source}")

    if (source / "dataset_info.json").exists() or (source / "dataset_dict.json").exists():
        loaded = load_from_disk(str(source))
        if hasattr(loaded, "keys"):
            if "train" in loaded:
                return loaded["train"]
            raise SystemExit(f"No train split among {sorted(loaded)} in {source}")
        return loaded

    for pattern in ("commonsense_170k.json", "*.json", "*.parquet"):
        matches = sorted(path for path in source.glob(pattern) if path.is_file())
        if matches:
            if matches[0].suffix == ".parquet":
                return Dataset.from_parquet([str(path) for path in matches])
            return Dataset.from_list(json.loads(matches[0].read_text(encoding="utf-8")))
    raise SystemExit(f"No commonsense json/parquet found under {source}")


def parse_labels(instruction: str) -> list[str]:
    """Read the candidate labels off the instruction's 'Answer format:' tail.

    Kept byte-identical in behavior to metrics/commonsense_qa.parse_labels; the
    pool and the grader must agree on what a row's candidate set is. Duplicated
    rather than imported so this script runs on a login node with no torch.
    """

    match = ANSWER_FORMAT_RE.search(instruction)
    if not match:
        return []
    labels = [part.strip().lower() for part in match.group(1).split("/")]
    labels = [label for label in labels if label]
    return labels if len(labels) >= 2 else []


def tag_task(labels: list[str]) -> tuple[str, str]:
    """Return (task, answer_family) for a row's label set."""

    key = tuple(labels)
    if key in LABELS_TO_TASK:
        return LABELS_TO_TASK[key]
    family = labels[0].rstrip("0123456789")
    if family == "ending":
        return "hellaswag", family
    if family == "answer":
        # SIQA / ARC-e / ARC-c / OBQA are indistinguishable from the row.
        return "answer_mc", family
    return "other", family or "unknown"


def convert(dataset, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Tag every row with its task and drop the ones no grader can score."""

    for column in ("instruction", "output"):
        if column not in dataset.column_names:
            raise SystemExit(
                f"Commonsense170k column '{column}' not found; got {dataset.column_names}"
            )
    has_answer = "answer" in dataset.column_names

    rows: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "dropped_no_instruction": 0,
        "dropped_no_output": 0,
        "dropped_no_answer_format": 0,
        "dropped_gold_off_menu": 0,
    }

    for index in range(len(dataset)):
        record = dataset[index]
        stats["total"] += 1

        instruction = str(record.get("instruction") or "").strip()
        output = str(record.get("output") or "").strip()
        if not instruction:
            stats["dropped_no_instruction"] += 1
            continue
        if not output:
            stats["dropped_no_output"] += 1
            continue

        labels = parse_labels(instruction)
        # The gold label is normally its own column; recover it from the answer
        # sentence when the source dump omits it.
        answer = str(record.get("answer") or "").strip().lower() if has_answer else ""
        if not answer:
            answer = output.lower().rsplit(" ", 1)[-1] if " " in output else ""

        if not labels:
            if args.drop_unparsable == 1:
                stats["dropped_no_answer_format"] += 1
                continue
        elif answer not in labels:
            if args.drop_unparsable == 1:
                stats["dropped_gold_off_menu"] += 1
                continue

        task, family = tag_task(labels) if labels else ("other", "unknown")
        rows.append(
            {
                "instruction": instruction,
                "input": "",
                "output": output,
                "answer": answer,
                "task": task,
                "answer_family": family,
                "num_options": len(labels),
                "source_index": index,
            }
        )

    return rows, stats


def sample_pool(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Draw the Part 2 replay pool, balanced across task tags by default."""

    rng = random.Random(args.seed)
    if args.pool_size <= 0 or args.pool_size >= len(rows):
        return rows
    if args.sample_strategy == "random":
        return rng.sample(rows, args.pool_size)

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    for bucket in by_task.values():
        rng.shuffle(bucket)

    # Round-robin rather than a fixed per-task quota, so a small task running
    # out gives its share to the others instead of shrinking the pool.
    picked: list[dict[str, Any]] = []
    cursors = {task: 0 for task in by_task}
    while len(picked) < args.pool_size:
        progressed = False
        for task in sorted(by_task):
            if len(picked) >= args.pool_size:
                break
            cursor = cursors[task]
            if cursor < len(by_task[task]):
                picked.append(by_task[task][cursor])
                cursors[task] = cursor + 1
                progressed = True
        if not progressed:
            break
    return picked


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile of an already sorted list."""

    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def length_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Measure real training-time token lengths and the cost of each budget."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    sample = rows if args.length_sample <= 0 else rows[: args.length_sample]

    full_lengths: list[int] = []
    instruction_lengths: list[int] = []
    answer_lengths: list[int] = []
    for row in sample:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        full_lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        instruction_lengths.append(
            len(tokenizer(row["instruction"], add_special_tokens=False)["input_ids"])
        )
        answer_lengths.append(
            len(tokenizer(row["output"], add_special_tokens=False)["input_ids"])
        )
    full_lengths.sort()
    instruction_lengths.sort()
    answer_lengths.sort()

    truncation = {}
    for budget in sorted({*CANDIDATE_MAX_LENS, args.max_len}):
        over = sum(length > budget for length in full_lengths)
        lost = sum(max(length - budget, 0) for length in full_lengths)
        truncation[budget] = {
            "rows_truncated": over,
            "fraction_truncated": over / len(full_lengths),
            "tokens_dropped_mean_over_truncated": lost / over if over else 0.0,
            # Left truncation keeps the tail, and the tail here is the answer
            # plus the "Answer format" line. What a truncated row loses is the
            # task description and the head of the question stem.
            "rows_losing_question_head": over,
            "tokens_kept_total": sum(min(length, budget) for length in full_lengths),
        }

    mean_answer = sum(answer_lengths) / len(answer_lengths)
    report = {
        "tokenizer": args.tokenizer_path,
        "rows_measured": len(sample),
        "full_length": {f"p{q}": percentile(full_lengths, q) for q in (50, 90, 95, 99)}
        | {"max": full_lengths[-1]},
        "instruction_length": {
            f"p{q}": percentile(instruction_lengths, q) for q in (50, 90, 99)
        }
        | {"max": instruction_lengths[-1]},
        "answer_length": {f"p{q}": percentile(answer_lengths, q) for q in (50, 90, 99)}
        | {"max": answer_lengths[-1], "mean": mean_answer},
        "supervised_tokens_per_epoch": mean_answer * len(rows),
        "truncation_by_max_len": truncation,
    }

    print("==== Commonsense170k training-length report ====")
    print(
        f"n={len(sample)}  full: P50={report['full_length']['p50']} "
        f"P90={report['full_length']['p90']} P95={report['full_length']['p95']} "
        f"P99={report['full_length']['p99']} max={report['full_length']['max']}"
    )
    print(
        f"instruction only: P50={report['instruction_length']['p50']} "
        f"P90={report['instruction_length']['p90']} "
        f"max={report['instruction_length']['max']}"
    )
    print(
        f"answer only: P50={report['answer_length']['p50']} "
        f"max={report['answer_length']['max']} mean={mean_answer:.1f}"
    )
    print(
        f"{'max_len':>9}{'truncated':>11}{'share':>8}"
        f"{'mean lost':>12}{'question head gone':>20}{'tok/epoch':>13}"
    )
    for budget, entry in truncation.items():
        marker = "  <- --max_len" if budget == args.max_len else ""
        per_epoch = entry["tokens_kept_total"] / len(sample) * len(rows)
        print(
            f"{budget:>9}{entry['rows_truncated']:>11}"
            f"{entry['fraction_truncated']:>7.1%}"
            f"{entry['tokens_dropped_mean_over_truncated']:>12.0f}"
            f"{entry['rows_losing_question_head']:>20}"
            f"{per_epoch / 1e6:>11.1f}M{marker}"
        )
    print(
        "Truncation keeps the END, so a truncated row keeps its answer and its "
        '"Answer format" line but loses the task description and the head of the '
        "question -- it teaches picking an option without having read the stem."
    )
    print(
        f"Supervised tokens/epoch: {report['supervised_tokens_per_epoch'] / 1e6:.2f}M "
        f"({mean_answer:.1f} per row). The IF line carries about 330 per row and "
        "the Math line about 3767, so per optimizer update this line "
        "back-propagates one to two orders of magnitude less supervision. Do not "
        "reuse either line's learning rate, and fix Part 2's replay budget unit "
        "before quoting any number."
    )
    return report


def main() -> None:
    """Write the converted pool as save_to_disk plus a manifest."""

    args = parse_args()

    dataset = load_commonsense(args)
    print(f"loaded {len(dataset)} rows from {args.commonsense_path or args.commonsense_repo}")

    rows, stats = convert(dataset, args)
    if not rows:
        raise SystemExit("Commonsense170k converted to 0 usable rows; check the columns.")

    dropped = {key: value for key, value in stats.items() if key.startswith("dropped_") and value}
    if dropped:
        print("  dropped: " + ", ".join(f"{k[8:]}={v}" for k, v in dropped.items()))

    if args.max_train_samples > 0 and len(rows) > args.max_train_samples:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_train_samples]
        print(f"  truncated to --max_train_samples {len(rows)} rows")

    if args.mode == "sampled":
        before = len(rows)
        rows = sample_pool(rows, args)
        print(
            f"  mode=sampled strategy={args.sample_strategy}: "
            f"{before} -> {len(rows)} rows"
        )

    task_counts = Counter(row["task"] for row in rows)
    option_counts = Counter(row["num_options"] for row in rows)
    print(f"{len(rows)} rows; task mix:")
    for task, count in task_counts.most_common():
        print(f"    {task:<12} {count:>7}  {count / len(rows):>6.1%}")
    print(
        "  options per row: "
        + ", ".join(f"{k}->{v}" for k, v in sorted(option_counts.items()))
    )

    from datasets import Dataset, DatasetDict

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(out_dir))
    print(f"wrote pool to {out_dir}")

    manifest: dict[str, Any] = {
        "source": args.commonsense_path or args.commonsense_repo,
        "mode": args.mode,
        "sample_strategy": args.sample_strategy if args.mode == "sampled" else "",
        "pool_size": args.pool_size if args.mode == "sampled" else 0,
        "drop_unparsable": args.drop_unparsable,
        "max_train_samples": args.max_train_samples,
        "seed": args.seed,
        "train": {"path": str(out_dir), "num_rows": len(rows)},
        "counts": stats,
        "task_mix": dict(task_counts),
        "options_histogram": {str(k): v for k, v in sorted(option_counts.items())},
        "note": "train.py splits validation out of this pool with "
        "--val_fraction, and the split is by row, so it is not a clean held-out "
        "set for these eight datasets -- the accuracy conclusion comes from "
        "metrics/commonsense_qa.py on the LLM-Adapters test files, not from "
        "val_loss. task / answer_family / num_options / source_index are "
        "metadata; build_loader drops every column except the tokenized three "
        "before collating. answer_mc pools SIQA, ARC-e, ARC-c and OBQA, which "
        "share one template and are not separable from the row.",
    }
    if args.tokenizer_path:
        manifest["length"] = length_report(rows, args)
    else:
        print("skipping length report: no --tokenizer_path")

    jsonl_dir = Path(args.jsonl_dir) if args.jsonl_dir else out_dir.parent
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = jsonl_dir / f"{out_dir.name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
