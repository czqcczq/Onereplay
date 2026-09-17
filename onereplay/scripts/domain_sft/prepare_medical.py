"""Build the Medical specialist pool from OctoMed/MedQA-5options.

The corpus is 10178 USMLE-style questions in the train split, each carrying 16
pre-generated teacher responses, so no teacher has to be run. Each response is
shaped::

    <think>
    {long reasoning}
    </think>

    {discussion of the options}

    <answer>E</answer>

Three properties of that shape drive this script.

Correctness is checkable, so it is checked
------------------------------------------
``<answer>X</answer>`` gives a single letter and ``answer`` gives the gold as
``"E: Nitrofurantoin"``. A response is kept only when the parsed letter equals
the gold letter. Sixteen samples per question means wrong ones are plentiful and
free to discard; keeping a wrong answer because the reasoning reads well would
be training the model to argue its way to the wrong option.

The ``<think>`` tags have to go
-------------------------------
Not a style preference -- a correctness requirement. ``tokenizer_to_ids`` masks
labels by prompt *length*, which assumes the prompt render is a token prefix of
the full render. Qwen3's template special-cases an assistant message containing
``</think>``: it splits the message and hoists the reasoning into the think
block, so the full render diverges from the prompt render right after
``<think>\\n`` and the mask lands in the wrong place. Stripping the markers
while keeping the reasoning text restores the prefix property, and matches what
the eval path does anyway (it renders with ``enable_thinking=False``).
common.serialize re-checks the invariant per row, so a regression here shows up
as a non-zero ``prompt_prefix_violations`` rather than as a quietly worse model.

The answer marker is normalized to ``\\boxed{}``
-----------------------------------------------
Math answers end in ``\\boxed{...}`` and ODA-Fin's do too. Rewriting
``<answer>E</answer>`` to ``\\boxed{E}`` lets one extractor grade all three
domains, which matters because these models are later evaluated against each
other for forgetting: a grader that is accidentally stricter on one domain would
look exactly like forgetting.

Trace selection
---------------
Sixteen correct responses to the same question are near-copies of each other --
same guideline, same elimination order, different wording. Training on all of
them would weight that question 16x and teach phrasing rather than medicine, so
each question contributes at most --max_traces_per_question.

Selection order deviates from the obvious "dedup, cap, then apply the length
limit" pipeline on purpose: rows are serialized and measured *first*, and
candidates that already fit in --max_tokens are preferred when filling a
question's quota. Capping first would let a question spend its four slots on
four over-long traces and then lose all of them, ending up with nothing while a
usable short trace sat unused. The counts for every stage are still reported
separately, so nothing is hidden by the reordering.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from onereplay.scripts.domain_sft.common import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    FilterLog,
    budget_scan,
    build_metadata,
    build_record,
    load_tokenizer,
    serialize,
    summarize,
    write_dataset_dir,
    write_jsonl,
    write_stats,
)

DOMAIN = "medical"
SOURCE = "OctoMed/MedQA-5options"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Deliberately parallel to prepare_openr1_math.EVAL_PROMPT_PREFIX: same two
# sentences, same "reason then box it" contract, only the task noun changes. The
# domains must not differ in how hard the prompt works.
INSTRUCTION_PREFIX = (
    "Answer the following multiple-choice medical question. Reason step by "
    "step, then give the final answer as \\boxed{...} at the end.\n\nQuestion: "
)

ANSWER_PATTERN = re.compile(r"<answer>\s*([A-E])\s*</answer>", re.IGNORECASE)
GOLD_PATTERN = re.compile(r"^\s*([A-E])\s*:")
VALID_LETTERS = frozenset("ABCDE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Medical SFT pool from OctoMed/MedQA-5options."
    )
    parser.add_argument(
        "--medqa_path",
        type=str,
        default="",
        help="Local parquet dir or save_to_disk dir. Takes precedence over --medqa_repo.",
    )
    parser.add_argument("--medqa_repo", type=str, default=SOURCE)
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default="",
        help="Commit hash of the source dataset, recorded in the stats file.",
    )
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument(
        "--out_jsonl", type=str, default="data/processed/medical_train.jsonl"
    )
    parser.add_argument("--out_stats", type=str, default="data/stats/medical_stats.json")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/medical_train",
        help="save_to_disk target, which is what training loads. Empty skips it.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Rows whose full serialized length exceeds this are dropped, never truncated.",
    )
    parser.add_argument(
        "--max_traces_per_question",
        type=int,
        default=4,
        help="Cap on correct traces kept per question. 3 or 4 per the plan; the "
        "chosen value and the resulting pool size are both recorded.",
    )
    parser.add_argument(
        "--near_dup_threshold",
        type=float,
        default=0.8,
        help="Word-shingle Jaccard above which two traces for the same question "
        "count as near-duplicates. 1.0 disables near-dup filtering.",
    )
    parser.add_argument("--shingle_size", type=int, default=5)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Must match what training is launched with: it contributes tokens "
        "that count against --max_tokens.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_questions",
        type=int,
        default=0,
        help="Debug aid: stop after this many questions. 0 processes all.",
    )
    return parser.parse_args()


def load_medqa(args: argparse.Namespace):
    """Load the train split from a local copy or the Hub."""

    from datasets import load_dataset, load_from_disk

    if args.medqa_path:
        path = Path(args.medqa_path)
        if (path / "dataset_dict.json").exists():
            return load_from_disk(str(path))["train"]
        if (path / "dataset_info.json").exists():
            return load_from_disk(str(path))
        shards = sorted(str(item) for item in path.glob("**/train-*.parquet"))
        if not shards:
            raise SystemExit(f"no train-*.parquet under {path}")
        return load_dataset("parquet", data_files={"train": shards})["train"]
    return load_dataset(args.medqa_repo, split="train")


def gold_letter(answer: str) -> str:
    """Extract the letter from a gold answer like ``"E: Nitrofurantoin"``.

    Falls back to a bare single letter, which is what a future revision of the
    corpus would most plausibly switch to.
    """

    match = GOLD_PATTERN.match(answer or "")
    if match:
        return match.group(1).upper()
    stripped = (answer or "").strip().upper()
    return stripped if stripped in VALID_LETTERS else ""


def build_question(question: str, options: list[str]) -> str:
    """Render the user turn: instruction, question stem, then the five options.

    The options already arrive as ``"A: Ampicillin"``, so they are emitted
    verbatim. Rewriting them would change the surface the model has to map its
    \\boxed letter onto, and the eval prompt would have to match exactly.
    """

    body = "\n".join(option.strip() for option in options if option and option.strip())
    return f"{INSTRUCTION_PREFIX}{question.strip()}\n\nOptions:\n{body}"


def convert_response(text: str) -> tuple[str, str]:
    """Return (cleaned response, predicted letter), or ("", "") if unusable.

    Rejects anything whose reasoning block or answer marker is malformed rather
    than trying to repair it: a truncated trace is exactly the kind of sample
    that teaches a model to stop mid-thought.
    """

    if not text or not text.strip():
        return "", ""

    matches = ANSWER_PATTERN.findall(text)
    if len(matches) != 1:
        return "", ""
    letter = matches[0].upper()

    opens, closes = text.count(THINK_OPEN), text.count(THINK_CLOSE)
    if opens != closes or opens > 1:
        return "", ""

    if closes == 1:
        reasoning, _, remainder = text.partition(THINK_CLOSE)
        reasoning = reasoning.replace(THINK_OPEN, "", 1).strip()
        if not reasoning:
            # A think block that opened and closed with nothing in it means the
            # trace lost its reasoning, which is the part being distilled.
            return "", ""
        body = f"{reasoning}\n\n{remainder.strip()}".strip()
    else:
        body = text.strip()

    body = ANSWER_PATTERN.sub(f"\\\\boxed{{{letter}}}", body)
    if THINK_OPEN in body or THINK_CLOSE in body:
        return "", ""
    if not body.strip():
        return "", ""
    return body, letter


def normalize(text: str) -> str:
    """Casefold and collapse whitespace, for exact-after-normalization dedup."""

    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str, size: int) -> frozenset[str]:
    """Word n-grams used for the within-question near-duplicate test."""

    words = normalize(text).split()
    if len(words) < size:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(
        " ".join(words[index : index + size]) for index in range(len(words) - size + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_traces_per_question:
        raise SystemExit("--max_traces_per_question must be at least 1")

    tokenizer = load_tokenizer(args.tokenizer_path, args.system_prompt)
    dataset = load_medqa(args)
    print(f"loaded {len(dataset)} questions from {args.medqa_path or args.medqa_repo}")

    log = FilterLog()
    rng = random.Random(args.seed)
    seen_global: set[str] = set()
    records: list[dict[str, Any]] = []
    prefix_violations = 0
    raw_responses = 0

    # Scan-wide token lengths, collected before the cutoff so budget_scan can
    # report what other budgets would have bought.
    scan_totals: list[int] = []
    scan_responses: list[int] = []

    questions_kept = 0
    questions_with_no_trace = 0
    limit = args.max_questions or len(dataset)

    for index in range(min(limit, len(dataset))):
        row = dataset[index]
        responses = row.get("responses") or []
        raw_responses += len(responses)

        gold = gold_letter(row.get("answer", ""))
        if not gold:
            log.bump("dropped_question_unparsable_gold")
            continue
        options = list(row.get("options") or [])
        if len(options) < 2:
            log.bump("dropped_question_missing_options")
            continue
        if not (row.get("question") or "").strip():
            log.bump("dropped_question_empty_stem")
            continue
        question = build_question(row["question"], options)

        # Stage 1: per-response validity, correctness and global exact dedup.
        candidates: list[dict[str, Any]] = []
        local_seen: set[str] = set()
        for response in responses:
            body, letter = convert_response(response)
            if not body:
                log.bump("dropped_response_malformed")
                continue
            if letter != gold:
                log.bump("dropped_response_wrong_answer")
                continue
            key = normalize(body)
            if key in local_seen:
                log.bump("dropped_response_exact_dup_in_question")
                continue
            if key in seen_global:
                log.bump("dropped_response_exact_dup_global")
                continue
            local_seen.add(key)
            candidates.append({"body": body, "letter": letter, "key": key})

        if not candidates:
            questions_with_no_trace += 1
            continue

        # Stage 2: measure every surviving candidate. Done before the quota is
        # filled so an over-long trace cannot occupy a slot a usable one needed.
        for candidate in candidates:
            measured = serialize(tokenizer, question, candidate["body"])
            candidate["serialized"] = measured
            scan_totals.append(measured.total_tokens)
            scan_responses.append(measured.response_tokens)
            if not measured.prompt_is_prefix:
                prefix_violations += 1

        # Stage 3: fill the quota. Shuffle under the run seed so the choice is
        # reproducible without being biased toward the teacher's sampling order,
        # then sort fitting candidates ahead of over-long ones -- a stable sort,
        # so the shuffled order survives within each group.
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item["serialized"].total_tokens > args.max_tokens)

        selected: list[dict[str, Any]] = []
        selected_shingles: list[frozenset[str]] = []
        for candidate in candidates:
            if len(selected) >= args.max_traces_per_question:
                log.bump("dropped_response_over_trace_quota")
                continue
            if candidate["serialized"].total_tokens > args.max_tokens:
                log.bump("dropped_response_over_max_tokens")
                continue
            if args.near_dup_threshold < 1.0:
                fingerprint = shingles(candidate["body"], args.shingle_size)
                if any(
                    jaccard(fingerprint, existing) >= args.near_dup_threshold
                    for existing in selected_shingles
                ):
                    log.bump("dropped_response_near_dup_in_question")
                    continue
                selected_shingles.append(fingerprint)
            selected.append(candidate)

        if not selected:
            questions_with_no_trace += 1
            continue
        questions_kept += 1

        for candidate in selected:
            seen_global.add(candidate["key"])
            records.append(
                build_record(
                    domain=DOMAIN,
                    index=len(records),
                    source=SOURCE,
                    question=question,
                    response=candidate["body"],
                    serialized=candidate["serialized"],
                    original_id=f"medqa-train-{index}",
                    gold_answer=gold,
                    response_answer=candidate["letter"],
                )
            )

    print(f"\nkept {len(records)} traces from {questions_kept} questions")
    print(f"questions that produced nothing: {questions_with_no_trace}")
    print("filter tally:")
    print(log.report())

    if prefix_violations:
        print(
            f"\nWARNING: {prefix_violations} rows failed the prompt-prefix check. "
            "Label masking is misaligned for those rows -- inspect before training."
        )

    write_jsonl(args.out_jsonl, records)
    print(f"\nwrote {len(records)} rows to {args.out_jsonl}")
    if args.out_dir:
        write_dataset_dir(args.out_dir, records)
        print(f"wrote trainable dataset to {args.out_dir}")

    stats = summarize(records)
    stats["traces_per_question"] = {
        "cap": args.max_traces_per_question,
        "questions_kept": questions_kept,
        "questions_with_no_trace": questions_with_no_trace,
        "mean_traces": len(records) / questions_kept if questions_kept else 0.0,
    }
    stats["budget_scan"] = budget_scan(scan_totals, scan_responses)
    stats["counts"] = log.as_dict()
    stats["metadata"] = build_metadata(
        domain=DOMAIN,
        dataset_name=SOURCE,
        dataset_revision=args.dataset_revision,
        tokenizer_path=args.tokenizer_path,
        max_tokens=args.max_tokens,
        seed=args.seed,
        raw_count=raw_responses,
        filtered_count=len(scan_totals),
        final_count=len(records),
        filter_rules={
            "split": "train",
            "correctness": "parsed <answer>X</answer> must equal gold letter",
            "think_handling": "strip_tags (markers removed, reasoning kept)",
            "answer_marker": "<answer>X</answer> rewritten to \\boxed{X}",
            "dedup": "normalized-exact globally and within question",
            "near_dup": {
                "metric": "word-shingle Jaccard",
                "shingle_size": args.shingle_size,
                "threshold": args.near_dup_threshold,
                "scope": "within question",
            },
            "max_traces_per_question": args.max_traces_per_question,
        },
        prompt_prefix_violations=prefix_violations,
        system_prompt=args.system_prompt,
    )
    write_stats(args.out_stats, stats)
    print(f"wrote stats to {args.out_stats}")
    print(
        f"\nnum_examples={stats['num_examples']} "
        f"total_tokens={stats['total_tokens']} "
        f"assistant_tokens={stats['total_assistant_tokens']}"
    )


if __name__ == "__main__":
    main()
