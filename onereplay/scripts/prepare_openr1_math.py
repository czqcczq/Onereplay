"""Convert LUFFY's Openr1-Math-46k-8192 into the {instruction, input, output} SFT schema.

This is the Math capability source for the continual learning chain
Base -> IF -> Math -> Third. The dataset ships verl's RL layout, not an SFT one:

    data_source   str                      e.g. "olympiads"
    prompt        [{role, content}]        a system turn + the user question
    target        [{role, content}]        one assistant turn, a long <think> CoT
    ability       str
    reward_model  {ground_truth, style}    the gold final answer
    extra_info    {index, split}

45792 rows. Flattening it is mostly mechanical -- take the user turn, take the
assistant turn -- but two decisions are not, and both of them can silently
invalidate the Math stage.

--------------------------------------------------------------------------
Decision 1: what to do with the <think> tags  (--think_handling)
--------------------------------------------------------------------------
onereplay/eval/generation.py renders every evaluation prompt with
enable_thinking=False, which makes Qwen3's chat template append an EMPTY
"<think>\\n\\n</think>\\n\\n" block before the model starts writing. At
inference the model is therefore told that thinking is already over and it
should emit the answer directly.

If training keeps the <think> tags, the model learns the opposite: to open a
think block and reason inside it. The capability would be real but unreachable
through the evaluation path, and the Math sanity check would read as "OpenR1
did not teach Math" when the truth is that the eval prompt closed the think
block before the model could use it.

Measured on Qwen3-1.7B-Base's own template, keeping the tags also breaks label
masking. tokenizer_to_ids masks labels with
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
which assumes prompt_ids prefixes full_ids. With a <think>-carrying target the
two diverge after 21 shared tokens while prompt_ids runs to 25, so the first 4
tokens of the reasoning ('>\\nLet me denote') are masked out of the loss. Four
tokens in a 2000-token CoT is negligible, but it is a symptom of the same
prompt/answer mismatch, not an independent bug.

  strip_tags (default) removes the <think> and </think> markers and keeps the
      reasoning text as the start of a normal answer. Training and evaluation
      then agree, and the result matches what the graders already ask for:
      math500.py prompts with "Reason step by step, then give the final answer
      as \\boxed{...}" and scores the last \\boxed{...} span. Nothing in the
      eval path needs to change.
  keep        preserves the markers, for the day the eval path grows a
      per-capability enable_thinking. Do not pair this with the current
      evaluation code and expect meaningful Math numbers.
  solution_only drops the reasoning entirely and keeps only the text after
      </think>. Useful as a deliberate control arm: same questions, same
      answers, no long CoT. Comparing it against strip_tags separates "learned
      Math" from "learned to emit thousands of reasoning tokens", which matters
      because the latter is the more likely cause of IF forgetting.

--------------------------------------------------------------------------
Decision 2: what the instruction looks like  (--instruction_style)
--------------------------------------------------------------------------
The dataset's own system turn is a 762-character R1-style directive demanding
"<think> {thoughts} </think>" and a \\boxed{} answer. apply_train_template has
no system slot -- it renders exactly one user turn and one assistant turn -- so
that text either gets folded into the user turn or dropped.

  eval_aligned (default) uses math500.py's own wording, so the training prompt
      is the evaluation prompt. This is the cheapest way to keep the Math
      sanity check from failing for prompt-format reasons.
  bare        the raw question, nothing added.
  openr1_system prepends the original system directive, preserving LUFFY's
      training condition. Coherent only with --think_handling keep, since the
      directive asks for the tags that the other modes strip.

reward_model.ground_truth is carried through as a `gold_answer` column. It is
not used in training -- build_loader drops every column except the tokenized
three -- but it is what a later filter or a held-out accuracy probe needs, and
re-deriving it from the parquet later is avoidable work.

Read the length report before choosing --max_len. These CoTs are long and
truncation keeps the END of the sequence, so an over-long row loses the
question and keeps the tail of the reasoning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# math500.py's build_prompt, duplicated rather than imported so this script
# stays runnable on a login node with no torch. Keep the two in sync: their
# whole point is to be identical.
EVAL_PROMPT_PREFIX = (
    "Solve the following math problem. Reason step by step, then give "
    "the final answer as \\boxed{...} at the end.\n\nProblem: "
)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Budgets the truncation table reports on. The pool is pre-filtered to 8192, so
# the interesting question is how much cheaper a smaller budget is.
CANDIDATE_MAX_LENS = (1024, 2048, 3072, 4096, 6144, 8192)


def parse_args() -> argparse.Namespace:
    """Parse the OpenR1 location, flattening decisions, and output settings."""

    parser = argparse.ArgumentParser(
        description="Build the OpenR1-Math SFT training pool for the Math stage."
    )
    parser.add_argument(
        "--openr1_path",
        type=str,
        default="",
        help="Local openr1.parquet, a directory holding it, or a save_to_disk dir. "
        "Empty falls back to --openr1_repo (needs network, ~483MB).",
    )
    parser.add_argument("--openr1_repo", type=str, default="Elliott/Openr1-Math-46k-8192")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="save_to_disk target; pass this as train.py --dataset_path.",
    )
    parser.add_argument(
        "--think_handling",
        type=str,
        choices=["strip_tags", "keep", "solution_only"],
        default="strip_tags",
        help="strip_tags removes the <think>/</think> markers but keeps the "
        "reasoning, which is the only mode consistent with the current eval "
        "path (it renders prompts with enable_thinking=False). keep preserves "
        "them. solution_only discards the reasoning, as a no-long-CoT control.",
    )
    parser.add_argument(
        "--instruction_style",
        type=str,
        choices=["eval_aligned", "bare", "openr1_system"],
        default="eval_aligned",
        help="eval_aligned reuses math500.py's prompt wording so training and "
        "evaluation match. bare uses the raw question. openr1_system prepends "
        "the dataset's R1 directive, which only makes sense with "
        "--think_handling keep.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Truncate the pool to this many rows after conversion; 0 keeps all "
        "45792. Use a few thousand for a pipeline smoke test before paying for "
        "a full run at 4k context.",
    )
    parser.add_argument(
        "--require_boxed",
        type=int,
        default=0,
        help="1 drops rows whose answer has no \\boxed{...} span. The graders "
        "score exactly that span, so a row without one teaches a response "
        "format the metric cannot read.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_len",
        type=int,
        default=4096,
        help="Budget the truncation report is headlined against; does not itself truncate.",
    )
    parser.add_argument(
        "--length_sample",
        type=int,
        default=3000,
        help="Rows measured for the length report; 0 = all. These are long "
        "sequences, so measuring all 45792 is slow and adds nothing.",
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


def load_openr1(args: argparse.Namespace):
    """Load OpenR1 from a parquet file, a directory, a save_to_disk dir, or the hub."""

    from datasets import Dataset, load_dataset, load_from_disk

    if not args.openr1_path:
        return load_dataset(args.openr1_repo, split="train")

    source = Path(args.openr1_path)
    if source.is_file():
        return Dataset.from_parquet(str(source))

    if not source.is_dir():
        raise SystemExit(f"--openr1_path does not exist: {source}")

    if (source / "dataset_info.json").exists() or (source / "dataset_dict.json").exists():
        loaded = load_from_disk(str(source))
        if hasattr(loaded, "keys"):
            if "train" in loaded:
                return loaded["train"]
            raise SystemExit(f"No train split among {sorted(loaded)} in {source}")
        return loaded

    matches = sorted(path for path in source.rglob("*.parquet") if path.is_file())
    if not matches:
        raise SystemExit(f"No .parquet files found under {source}")
    return Dataset.from_parquet([str(path) for path in matches])


def message_content(messages: Any, role: str) -> str:
    """Return the first message with this role, or empty string."""

    for message in messages or []:
        if message.get("role") == role:
            return message.get("content") or ""
    return ""


def split_think(text: str) -> tuple[str, str]:
    """Split an R1-style answer into (reasoning, solution).

    The opening tag is optional in the wild -- some rows begin reasoning
    immediately and only close with </think> -- so the close tag is what the
    split keys on.
    """

    if THINK_CLOSE not in text:
        return "", text
    head, _, tail = text.partition(THINK_CLOSE)
    reasoning = head.split(THINK_OPEN, 1)[-1] if THINK_OPEN in head else head
    return reasoning.strip(), tail.strip()


def build_output(raw_target: str, mode: str) -> str:
    """Apply the chosen <think> policy to one assistant turn."""

    if mode == "keep":
        return raw_target.strip()

    reasoning, solution = split_think(raw_target)
    if mode == "solution_only":
        # A row with no close tag has no separable solution; its whole text is
        # the answer, so returning it unchanged is the only honest option.
        return solution.strip()

    if not reasoning:
        return solution.strip()
    return f"{reasoning}\n\n{solution}".strip()


def build_instruction(question: str, system_prompt: str, style: str) -> str:
    """Render the user turn under the chosen instruction style."""

    question = question.strip()
    if style == "bare":
        return question
    if style == "openr1_system":
        system_prompt = system_prompt.strip()
        return f"{system_prompt}\n\n{question}" if system_prompt else question
    return f"{EVAL_PROMPT_PREFIX}{question}"


def has_boxed(text: str) -> bool:
    """True when the answer carries a \\boxed{...} or \\fbox{...} span."""

    return "\\boxed" in text or "\\fbox" in text


def convert(dataset, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Flatten the verl-style rows into single-turn SFT rows."""

    for column in ("prompt", "target"):
        if column not in dataset.column_names:
            raise SystemExit(
                f"OpenR1 column '{column}' not found; got {dataset.column_names}"
            )

    has_reward = "reward_model" in dataset.column_names
    has_source = "data_source" in dataset.column_names

    rows: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "dropped_no_question": 0,
        "dropped_no_target": 0,
        "dropped_empty_output": 0,
        "dropped_no_boxed": 0,
        "targets_with_think": 0,
        "targets_without_think": 0,
    }

    for index in range(len(dataset)):
        record = dataset[index]
        stats["total"] += 1

        question = message_content(record["prompt"], "user")
        system_prompt = message_content(record["prompt"], "system")
        raw_target = message_content(record["target"], "assistant")

        if not question.strip():
            stats["dropped_no_question"] += 1
            continue
        if not raw_target.strip():
            stats["dropped_no_target"] += 1
            continue

        if THINK_CLOSE in raw_target:
            stats["targets_with_think"] += 1
        else:
            stats["targets_without_think"] += 1

        output = build_output(raw_target, args.think_handling)
        if not output:
            stats["dropped_empty_output"] += 1
            continue
        if args.require_boxed == 1 and not has_boxed(output):
            stats["dropped_no_boxed"] += 1
            continue

        reward = record.get("reward_model") or {} if has_reward else {}
        rows.append(
            {
                "instruction": build_instruction(
                    question, system_prompt, args.instruction_style
                ),
                "input": "",
                "output": output,
                "gold_answer": str(reward.get("ground_truth", "") or ""),
                "data_source": str(record.get("data_source", "") or "") if has_source else "",
                "source_index": index,
            }
        )

    return rows, stats


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile of an already sorted list."""

    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def length_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Measure real training-time token lengths and the cost of each budget."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    sample = rows if args.length_sample <= 0 else rows[: args.length_sample]

    full_lengths: list[int] = []
    prompt_lengths: list[int] = []
    output_lengths: list[int] = []
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
        prompt_lengths.append(
            len(tokenizer(row["instruction"], add_special_tokens=False)["input_ids"])
        )
        output_lengths.append(
            len(tokenizer(row["output"], add_special_tokens=False)["input_ids"])
        )
    full_lengths.sort()
    prompt_lengths.sort()
    output_lengths.sort()

    truncation = {}
    for budget in sorted({*CANDIDATE_MAX_LENS, args.max_len}):
        over = sum(length > budget for length in full_lengths)
        lost = sum(max(length - budget, 0) for length in full_lengths)
        truncation[budget] = {
            "rows_truncated": over,
            "fraction_truncated": over / len(full_lengths),
            "tokens_dropped_mean_over_truncated": lost / over if over else 0.0,
            # Left truncation keeps the tail, so a row whose answer alone fills
            # the budget loses its question entirely.
            "rows_losing_whole_question": sum(
                length >= budget for length in output_lengths
            ),
            "tokens_kept_total": sum(min(length, budget) for length in full_lengths),
        }

    report = {
        "tokenizer": args.tokenizer_path,
        "rows_measured": len(sample),
        "full_length": {f"p{q}": percentile(full_lengths, q) for q in (50, 90, 95, 99)}
        | {"max": full_lengths[-1]},
        "question_length": {f"p{q}": percentile(prompt_lengths, q) for q in (50, 90, 99)}
        | {"max": prompt_lengths[-1]},
        "answer_length": {f"p{q}": percentile(output_lengths, q) for q in (50, 90, 99)}
        | {"max": output_lengths[-1]},
        "truncation_by_max_len": truncation,
    }

    print("==== OpenR1-Math training-length report ====")
    print(
        f"n={len(sample)}  full: P50={report['full_length']['p50']} "
        f"P90={report['full_length']['p90']} P95={report['full_length']['p95']} "
        f"P99={report['full_length']['p99']} max={report['full_length']['max']}"
    )
    print(
        f"question only: P50={report['question_length']['p50']} "
        f"P90={report['question_length']['p90']} max={report['question_length']['max']}"
    )
    print(
        f"answer only: P50={report['answer_length']['p50']} "
        f"P90={report['answer_length']['p90']} max={report['answer_length']['max']}"
    )
    print(
        f"{'max_len':>9}{'truncated':>11}{'share':>8}"
        f"{'mean lost':>12}{'question gone':>15}{'tok/epoch':>13}"
    )
    for budget, entry in truncation.items():
        marker = "  <- --max_len" if budget == args.max_len else ""
        per_epoch = entry["tokens_kept_total"] / len(sample) * len(rows)
        print(
            f"{budget:>9}{entry['rows_truncated']:>11}"
            f"{entry['fraction_truncated']:>7.1%}"
            f"{entry['tokens_dropped_mean_over_truncated']:>12.0f}"
            f"{entry['rows_losing_whole_question']:>15}"
            f"{per_epoch / 1e6:>11.1f}M{marker}"
        )
    print(
        "Truncation keeps the END of the sequence, so a truncated row loses its "
        "question and keeps the tail of the reasoning. The tok/epoch column is "
        "the training cost driver: at 4096 this pool is an order of magnitude "
        "more tokens per epoch than Commonsense170k at 512."
    )
    return report


def main() -> None:
    """Write the converted pool as save_to_disk plus a manifest."""

    args = parse_args()
    if args.instruction_style == "openr1_system" and args.think_handling != "keep":
        print(
            "WARNING: --instruction_style openr1_system asks the model for "
            f"<think> tags that --think_handling {args.think_handling} removes "
            "from the targets. The instruction and the answer disagree."
        )
    if args.think_handling == "keep":
        print(
            "WARNING: --think_handling keep is inconsistent with the current "
            "eval path, which renders prompts with enable_thinking=False and so "
            "closes the think block before generation starts. Math scores from "
            "this pool will understate what the model learned."
        )

    dataset = load_openr1(args)
    print(f"loaded {len(dataset)} rows from {args.openr1_path or args.openr1_repo}")

    rows, stats = convert(dataset, args)
    if not rows:
        raise SystemExit("OpenR1 converted to 0 usable rows; check the column contents.")

    print(
        f"think_handling={args.think_handling} "
        f"instruction_style={args.instruction_style} -> {len(rows)} rows"
    )
    print(
        f"  targets with <think>: {stats['targets_with_think']}, "
        f"without: {stats['targets_without_think']}"
    )
    dropped = {
        key: value
        for key, value in stats.items()
        if key.startswith("dropped_") and value
    }
    if dropped:
        print("  dropped: " + ", ".join(f"{k[8:]}={v}" for k, v in dropped.items()))

    if args.max_train_samples > 0 and len(rows) > args.max_train_samples:
        import random

        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_train_samples]
        print(f"  truncated to --max_train_samples {len(rows)} rows")

    boxed = sum(1 for row in rows if has_boxed(row["output"]))
    print(
        f"  answers carrying \\boxed: {boxed} ({boxed / len(rows):.1%}) "
        "-- the graders score exactly this span"
    )

    from datasets import Dataset, DatasetDict

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(out_dir))
    print(f"wrote pool to {out_dir}")

    manifest: dict[str, Any] = {
        "source": args.openr1_path or args.openr1_repo,
        "think_handling": args.think_handling,
        "instruction_style": args.instruction_style,
        "eval_prompt_prefix": EVAL_PROMPT_PREFIX
        if args.instruction_style == "eval_aligned"
        else "",
        "require_boxed": args.require_boxed,
        "max_train_samples": args.max_train_samples,
        "seed": args.seed,
        "train": {"path": str(out_dir), "num_rows": len(rows)},
        "counts": stats,
        "boxed_answer_rows": boxed,
        "note": "train.py splits validation out of this pool with --val_fraction. "
        "gold_answer / data_source / source_index are metadata; build_loader "
        "drops every column except the tokenized three before collating.",
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
