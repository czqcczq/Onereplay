"""Build the Law specialist pool from Lawyer-Instruct + LegalLAMA US-Terms.

Two corpora concatenated in their original order, 9241 + 5806 raw rows, which is
where the 61:39 split in the design comes from -- it is a property of the two
files, not a sampling target, so nothing here resamples to hit it.

    RAW=datasets/raw
    python -m onereplay.scripts.domain_sft.prepare_law \\
        --lawyer_json "${RAW}/Lawyer-Instruct/alpacmygavel.json" \\
        --us_terms_jsonl "${RAW}/legal_lama/us_terms.jsonl" \\
        --tokenizer_path models/Qwen2.5-1.5B-Instruct \\
        --system_prompt '' --max_tokens 1216 --seed 42

Lawyer-Instruct
---------------
Kept verbatim: `instruction` becomes the user turn and `output` the assistant
turn, with no filtering of any kind. Worth knowing what that means, because the
corpus is not the instruction/response legal QA its name suggests: it is a
multi-agent "Lawyer 1 / Lawyer 2 / Lawyer 3" case discussion cut into adjacent
turn pairs, so a large share of rows are bare agreement ("I agree. It's
important to...") and ~860 carry ReAct / Tree-of-Thought boilerplate verbatim
("generating reasoning traces and task-specific actions in an interleaved
manner"). Training on it as published is a deliberate choice; whether it buys
any legal ability is what the MMLU law delta over base is there to answer. If
that delta comes out flat, this is the first place to look, not the trainer.

`input` is empty on all 9241 rows, so there is no Input: section to append and
the rendered sequence is question + response only.

LegalLAMA US-Terms
------------------
One deterministic format conversion and nothing else: the masked sentence
becomes the question under a fixed instruction, and `obj_label` becomes the
answer as published. No chain of thought, no teacher, no relabeling, no
rewriting of the gold term -- including its case. The 145 distinct raw labels
collapse to 85 under casefold, and the difference is Title Case coming from
terms that sat in a document heading ("Limited Liability Live-Stock Contract");
lower-casing those would make the answer read wrong where it was lifted from,
so the label space is reported both ways instead of normalized.

Three properties of this corpus are measured and recorded rather than fixed,
because each one would otherwise be mistaken for legal ability later:

  * 24% of rows have the gold term appearing verbatim elsewhere in the same
    sentence (25% ignoring case). LegalLAMA is a cloze probe built from court
    documents, and a term recurs within one document, so for a quarter of the
    corpus the task is copying a span rather than recalling a term. Each row
    carries `answer_in_context` so that a fast-falling loss on this source can
    be attributed instead of celebrated.
  * 85 labels over 5806 rows is roughly 100 examples per term, so this half of
    the pool is a narrow classification task wearing a generation task's shape.
  * `legal_topic` is kept as provenance and deliberately kept *out* of the
    prompt. Putting it in would narrow the 85-way choice to within one of seven
    topics, which makes the task easier without making the model better.

Nothing is deduplicated (84 sentences repeat) and nothing is cleaned (2138 rows
carry case-reporter page artifacts like `*785`). ``strip()`` is the one
normalization applied, which is format, not content.

US-Terms is also a *benchmark* being used as a training pool: legal_lama.py
defines only a TEST split. It shares no rows with MMLU so the law evaluation is
unaffected, but the fact is recorded in the metadata so nobody later evaluates
on us_terms against a model trained on all of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from onereplay.scripts.domain_sft.common import (  # noqa: E402
    FilterLog,
    budget_scan,
    build_metadata,
    build_record,
    load_tokenizer,
    serialize_many,
    summarize,
    write_dataset_dir,
    write_jsonl,
    write_stats,
)

DOMAIN = "law"
LAWYER_DATASET = "Alignment-Lab-AI/Lawyer-Instruct"
US_TERMS_DATASET = "lexlms/legal_lama:us_terms"

# The literal the published file uses. Not a CLI knob: it describes the input,
# and if a re-download ever changes it every row fails the mask-count check
# below and says so, which is the outcome we want over a silent no-op replace.
SOURCE_MASK = "<mask>"

US_TERMS_INSTRUCTION = (
    "Complete the masked legal statement with the correct U.S. legal term:"
)

# What SOURCE_MASK is rewritten to. `[MASK]` reads as a placeholder rather than
# as a chat special token; no row in the corpus already contains it, so the
# substitution is unambiguous.
US_TERMS_MASK_TOKEN = "[MASK]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Law SFT pool from Lawyer-Instruct + LegalLAMA US-Terms."
    )
    parser.add_argument("--lawyer_json", type=str, required=True)
    parser.add_argument("--us_terms_jsonl", type=str, required=True)
    parser.add_argument("--dataset_revision", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, default="data/processed/law_train.jsonl")
    parser.add_argument("--out_stats", type=str, default="data/stats/law_stats.json")
    parser.add_argument("--out_dir", type=str, default="data/processed/law_train")
    parser.add_argument("--max_tokens", type=int, default=1216)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    # Both live on the command line so the question text is recorded in the
    # stats file rather than implied by whichever revision of this file ran.
    parser.add_argument("--us_terms_instruction", type=str, default=US_TERMS_INSTRUCTION)
    parser.add_argument("--us_terms_mask_token", type=str, default=US_TERMS_MASK_TOKEN)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Debug aid; 0 reads all. Applied per source, not to the total.",
    )
    return parser.parse_args()


def iter_json_records(path: str) -> Iterator[dict[str, Any]]:
    """Read a JSON array or JSON lines file, whichever it turns out to be."""

    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        for row in json.loads(stripped):
            yield row
        return
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def load_lawyer_instruct(path: str, log: FilterLog, max_rows: int) -> tuple[list[dict[str, Any]], int]:
    """Lawyer-Instruct rows, verbatim. Returns (usable, raw_count)."""

    usable: list[dict[str, Any]] = []
    raw_count = 0
    for index, row in enumerate(iter_json_records(path)):
        if max_rows and raw_count >= max_rows:
            break
        raw_count += 1
        question = str(row.get("instruction") or "").strip()
        response = str(row.get("output") or "").strip()
        if not question:
            log.bump("lawyer_dropped_empty_instruction")
            continue
        if not response:
            log.bump("lawyer_dropped_empty_output")
            continue
        usable.append(
            {
                "source": LAWYER_DATASET,
                "question": question,
                "response": response,
                "original_id": f"lawyer-{index}",
                "extra": {},
            }
        )
    return usable, raw_count


def load_us_terms(
    path: str,
    instruction: str,
    mask_token: str,
    log: FilterLog,
    max_rows: int,
) -> tuple[list[dict[str, Any]], int]:
    """US-Terms rows, format-converted. Returns (usable, raw_count).

    The mask-count and list-length checks drop the row and tally the reason
    rather than falling back to ``masked_sentences[0]`` or to a no-op replace.
    Every one of the 5806 published rows passes both, so a non-zero tally means
    the input file is not the one this was written against -- which is exactly
    the case that must not pass silently.
    """

    usable: list[dict[str, Any]] = []
    raw_count = 0
    for row in iter_json_records(path):
        if max_rows and raw_count >= max_rows:
            break
        raw_count += 1

        sentences = row.get("masked_sentences")
        if not isinstance(sentences, list) or len(sentences) != 1:
            log.bump("us_terms_dropped_bad_masked_sentences")
            continue
        sentence = str(sentences[0] or "").strip()
        if not sentence:
            log.bump("us_terms_dropped_empty_sentence")
            continue
        if sentence.count(SOURCE_MASK) != 1:
            log.bump("us_terms_dropped_mask_count")
            continue

        term = str(row.get("obj_label") or "").strip()
        if not term:
            log.bump("us_terms_dropped_empty_label")
            continue

        # Measured on the sentence with the blank removed, so the mask itself
        # cannot be mistaken for an occurrence of the term.
        context = sentence.replace(SOURCE_MASK, "")
        usable.append(
            {
                "source": US_TERMS_DATASET,
                "question": f"{instruction}\n\n{sentence.replace(SOURCE_MASK, mask_token)}",
                "response": term,
                "original_id": str(row.get("id") or ""),
                "gold_answer": term,
                "extra": {
                    "legal_topic": str(row.get("legal_topic") or ""),
                    "answer_in_context": term in context,
                    "answer_in_context_casefold": term.lower() in context.lower(),
                },
            }
        )
    return usable, raw_count


def us_terms_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The three US-Terms properties that must travel with the pool.

    Computed after the length cutoff so the numbers describe the rows that were
    actually trained on, not the rows that were read.
    """

    rows = [record for record in records if record["source"] == US_TERMS_DATASET]
    total = len(rows)
    if not total:
        return {"num_examples": 0}

    leaked = sum(1 for record in rows if record.get("answer_in_context"))
    leaked_cf = sum(1 for record in rows if record.get("answer_in_context_casefold"))
    labels = Counter(record["response"] for record in rows)
    return {
        "num_examples": total,
        # A quarter of this source is a copy task. Read any loss curve on it
        # against this number before calling it legal knowledge.
        "answer_in_context": leaked,
        "answer_in_context_fraction": leaked / total,
        "answer_in_context_casefold": leaked_cf,
        "answer_in_context_casefold_fraction": leaked_cf / total,
        # Both, because the gap between them is the Title Case variants that
        # were deliberately not normalized away.
        "distinct_labels": len(labels),
        "distinct_labels_casefold": len({label.lower() for label in labels}),
        "examples_per_label_mean": total / max(len(labels), 1),
        "by_legal_topic": dict(
            sorted(
                Counter(record.get("legal_topic", "") for record in rows).items(),
                key=lambda item: -item[1],
            )
        ),
    }


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path, args.system_prompt)

    log = FilterLog()
    # Concatenated in this order and never interleaved or shuffled: train.py
    # shuffles, and doing it here too would mean the pool a checkpoint was
    # trained on could not be reproduced from --seed alone.
    lawyer, lawyer_raw = load_lawyer_instruct(args.lawyer_json, log, args.max_rows)
    us_terms, us_terms_raw = load_us_terms(
        args.us_terms_jsonl,
        args.us_terms_instruction,
        args.us_terms_mask_token,
        log,
        args.max_rows,
    )
    usable = lawyer + us_terms
    raw_count = lawyer_raw + us_terms_raw

    print(f"loaded {lawyer_raw} rows from {args.lawyer_json} -> {len(lawyer)} usable")
    print(f"loaded {us_terms_raw} rows from {args.us_terms_jsonl} -> {len(us_terms)} usable")
    print(f"{len(usable)} rows usable in total; measuring lengths...")
    measured = serialize_many(
        tokenizer,
        [(item["question"], item["response"]) for item in usable],
        batch_size=args.batch_size,
        progress_every=5000,
    )
    scan_totals = [item.total_tokens for item in measured]
    scan_responses = [item.response_tokens for item in measured]
    prefix_violations = sum(1 for item in measured if not item.prompt_is_prefix)

    fitting: list[dict[str, Any]] = []
    for item, serialized in zip(usable, measured):
        if serialized.total_tokens > args.max_tokens:
            # Per-source because the two halves are dropped for different
            # reasons: US-Terms has a 14.5k-word tail of whole court opinions,
            # Lawyer-Instruct tops out around 233 words and loses nothing.
            log.bump(
                "lawyer_dropped_over_max_tokens"
                if item["source"] == LAWYER_DATASET
                else "us_terms_dropped_over_max_tokens"
            )
            continue
        item["serialized"] = serialized
        fitting.append(item)
    print(f"{len(fitting)} rows fit within {args.max_tokens} tokens")

    records = [
        build_record(
            domain=DOMAIN,
            index=position,
            source=item["source"],
            question=item["question"],
            response=item["response"],
            serialized=item["serialized"],
            original_id=item["original_id"],
            gold_answer=item.get("gold_answer", ""),
            extra=item["extra"],
        )
        for position, item in enumerate(fitting)
    ]

    print("\nfilter tally:")
    print(log.report())
    if prefix_violations:
        print(
            f"\nWARNING: {prefix_violations} rows failed the prompt-prefix check; "
            "label masking would be misaligned for those rows."
        )

    write_jsonl(args.out_jsonl, records)
    print(f"\nwrote {len(records)} rows to {args.out_jsonl}")
    if args.out_dir:
        # legal_topic / answer_in_context ride in the jsonl only:
        # write_dataset_dir's column list is shared with the other three
        # domains, and widening it for law alone would change their schema too.
        write_dataset_dir(args.out_dir, records)
        print(f"wrote trainable dataset to {args.out_dir}")

    stats = summarize(records)
    stats["budget_scan"] = budget_scan(scan_totals, scan_responses)
    stats["counts"] = log.as_dict()
    stats["us_terms"] = us_terms_report(records)
    stats["metadata"] = build_metadata(
        domain=DOMAIN,
        dataset_name=f"{LAWYER_DATASET} + {US_TERMS_DATASET}",
        dataset_revision=args.dataset_revision,
        tokenizer_path=args.tokenizer_path,
        max_tokens=args.max_tokens,
        seed=args.seed,
        raw_count=raw_count,
        filtered_count=len(fitting),
        final_count=len(records),
        filter_rules={
            "selection": "none -- both corpora in full",
            "order": "original concat: Lawyer-Instruct then US-Terms, never interleaved",
            "dedup": "none (84 US-Terms sentences repeat and are kept)",
            "cleaning": "strip() only; case-reporter artifacts and gold-term case are left alone",
            "length": "drop rows over max_total_tokens; never truncate",
            "lawyer_instruct": "verbatim instruction -> user, output -> assistant; input is empty on every row",
            "us_terms_conversion": (
                f"format only: '{SOURCE_MASK}' -> '{args.us_terms_mask_token}' under a fixed "
                "instruction; obj_label is the target as published. No CoT, no teacher, "
                "no relabeling, no answer rewriting."
            ),
            "us_terms_instruction": args.us_terms_instruction,
            "us_terms_mask_token": args.us_terms_mask_token,
            "us_terms_legal_topic": "provenance column only; deliberately not in the prompt",
            "us_terms_source_split": (
                "test -- the only split legal_lama defines. The whole probing benchmark is "
                "used as a training pool, so it can no longer serve as an evaluation set "
                "for models trained on this pool."
            ),
        },
        prompt_prefix_violations=prefix_violations,
        system_prompt=args.system_prompt,
    )
    write_stats(args.out_stats, stats)
    print(f"wrote stats to {args.out_stats}")

    by_source = stats["by_source"]
    print(
        f"\nnum_examples={stats['num_examples']} "
        f"total_tokens={stats['total_tokens']} "
        f"assistant_tokens={stats['total_assistant_tokens']}"
    )
    for name, bucket in by_source.items():
        print(f"  {name}: {bucket['count']} rows ({bucket['percentage']:.1f}%)")
    report = stats["us_terms"]
    if report.get("num_examples"):
        print(
            f"  US-Terms: {report['distinct_labels']} labels "
            f"({report['distinct_labels_casefold']} casefolded), "
            f"answer already in context on {report['answer_in_context_fraction'] * 100:.1f}% of rows"
        )


if __name__ == "__main__":
    main()
