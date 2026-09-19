"""Build the Medical specialist pool from Medical Meadow medical flashcards.

The corpus is 33,955 instruction/input/output rows answered in free-form prose:
no chain of thought, no \\boxed{}, no multiple choice. That shape is why nothing
here filters -- the whole corpus goes in. A row only disappears if it is empty
or does not fit --max_tokens, and at 1216 none do (the longest is ~430 tokens).

    MEADOW=datasets/raw/medical_meadow_medical_flashcards
    python -m onereplay.scripts.domain_sft.prepare_med_meadow \\
        --meadow_json "${MEADOW}/medical_meadow_wikidoc_medical_flashcards.json" \\
        --tokenizer_path models/Qwen2.5-1.5B-Instruct \\
        --system_prompt '' --max_tokens 1216 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
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

DOMAIN = "medical"
DATASET = "medalpaca/medical_meadow_medical_flashcards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Medical SFT pool from Medical Meadow flashcards (full corpus)."
    )
    parser.add_argument("--meadow_json", type=str, required=True)
    parser.add_argument("--dataset_revision", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, default="data/processed/medical_train.jsonl")
    parser.add_argument("--out_stats", type=str, default="data/stats/medical_stats.json")
    parser.add_argument("--out_dir", type=str, default="data/processed/medical_train")
    parser.add_argument("--max_tokens", type=int, default=1216)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_rows", type=int, default=0, help="Debug aid; 0 reads all.")
    return parser.parse_args()


def iter_records(path: str) -> Iterator[dict[str, Any]]:
    """Read the flashcards file, whether it is a JSON array or JSON lines."""

    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        for row in json.loads(stripped):
            yield row
        return
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path, args.system_prompt)

    log = FilterLog()
    usable: list[dict[str, Any]] = []
    raw_count = 0
    for index, row in enumerate(iter_records(args.meadow_json)):
        if args.max_rows and raw_count >= args.max_rows:
            break
        raw_count += 1
        instruction = str(row.get("instruction") or "").strip()
        question_text = str(row.get("input") or "").strip()
        response = str(row.get("output") or "").strip()
        if not question_text:
            log.bump("dropped_empty_input")
            continue
        if not response:
            log.bump("dropped_empty_output")
            continue

        # The upstream instruction ("Answer this question truthfully") is kept in
        # front of the question rather than dropped: it is part of the corpus as
        # published, and removing it would make the trained prompt differ from
        # the one every published Medical Meadow result uses.
        question = f"{instruction}\n\n{question_text}" if instruction else question_text
        usable.append(
            {
                "source": DATASET,
                "question": question,
                "response": response,
                "original_id": f"meadow-{index}",
            }
        )

    print(f"loaded {raw_count} rows from {args.meadow_json}")
    print(f"{len(usable)} rows usable; measuring lengths...")
    measured = serialize_many(
        tokenizer,
        [(item["question"], item["response"]) for item in usable],
        batch_size=args.batch_size,
        progress_every=20000,
    )
    scan_totals = [item.total_tokens for item in measured]
    scan_responses = [item.response_tokens for item in measured]
    prefix_violations = sum(1 for item in measured if not item.prompt_is_prefix)

    fitting: list[dict[str, Any]] = []
    for item, serialized in zip(usable, measured):
        if serialized.total_tokens > args.max_tokens:
            log.bump("dropped_over_max_tokens")
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
        write_dataset_dir(args.out_dir, records)
        print(f"wrote trainable dataset to {args.out_dir}")

    stats = summarize(records)
    stats["budget_scan"] = budget_scan(scan_totals, scan_responses)
    stats["counts"] = log.as_dict()
    stats["metadata"] = build_metadata(
        domain=DOMAIN,
        dataset_name=DATASET,
        dataset_revision=args.dataset_revision,
        tokenizer_path=args.tokenizer_path,
        max_tokens=args.max_tokens,
        seed=args.seed,
        raw_count=raw_count,
        filtered_count=len(fitting),
        final_count=len(records),
        filter_rules={
            "selection": "none -- full corpus",
            "dedup": "none",
            "length": "drop rows over max_total_tokens; never truncate",
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
