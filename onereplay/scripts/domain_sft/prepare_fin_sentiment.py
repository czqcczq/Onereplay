"""Build the Finance specialist pool from FinGPT sentiment-train (full corpus).

76,772 instruction/input/output rows whose answer is a sentiment label, not
prose. Two properties of the corpus decide the whole script:

  * The instruction is *in the data* -- three variants, one per upstream source
    ("this news" for the FPB-style rows, "this tweet" for the TFNS-style ones,
    and a nine-level variant for the FiQA-SA rows). It is kept verbatim and put
    in front of the sentence, so the eval prompts in eval/metrics/finance.py can
    be the same string and the trained model is asked what it was taught.
  * 16,184 of the rows answer on a nine-level scale (``strong positive`` ..
    ``strong negative``) because their instruction asks for one. Collapsing
    those to three classes would break the instruction/answer contract, so they
    are kept as-is and the graders map nine -> three instead.

Nothing is filtered. A row only disappears if it is empty or does not fit
--max_tokens, and at 1216 none do (p99 is ~136 tokens).

    python -m onereplay.scripts.domain_sft.prepare_fin_sentiment \\
        --fingpt_path datasets/raw/fingpt-sentiment-train \\
        --tokenizer_path models/Qwen2.5-1.5B-Instruct \\
        --system_prompt '' --max_tokens 1216 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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

DOMAIN = "finance"
DATASET = "FinGPT/fingpt-sentiment-train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Finance SFT pool from FinGPT sentiment-train (full corpus)."
    )
    parser.add_argument("--fingpt_path", type=str, default="")
    parser.add_argument("--fingpt_repo", type=str, default=DATASET)
    parser.add_argument("--dataset_revision", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, default="data/processed/finance_train.jsonl")
    parser.add_argument("--out_stats", type=str, default="data/stats/finance_stats.json")
    parser.add_argument("--out_dir", type=str, default="data/processed/finance_train")
    parser.add_argument("--max_tokens", type=int, default=1216)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--max_rows", type=int, default=0, help="Debug aid; 0 reads all.")
    return parser.parse_args()


def load_fingpt(args: argparse.Namespace):
    """Load FinGPT from a local parquet/save_to_disk copy, or the Hub."""

    from datasets import load_dataset, load_from_disk

    if args.fingpt_path:
        path = Path(args.fingpt_path)
        if path.is_file():
            return load_dataset(
                "parquet",
                data_files=str(path),
                split="train",
                cache_dir=args.cache_dir or None,
            )
        if (path / "dataset_dict.json").exists():
            return load_from_disk(str(path))[args.split]
        if (path / "dataset_info.json").exists():
            return load_from_disk(str(path))
        shards = sorted(str(item) for item in path.glob("**/*.parquet"))
        if not shards:
            raise SystemExit(f"no parquet under {path}")
        return load_dataset(
            "parquet", data_files=shards, split="train", cache_dir=args.cache_dir or None
        )
    return load_dataset(args.fingpt_repo, split=args.split, cache_dir=args.cache_dir or None)


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path, args.system_prompt)
    dataset = load_fingpt(args)
    total_rows = len(dataset) if not args.max_rows else min(args.max_rows, len(dataset))
    print(f"loaded {len(dataset)} rows from {args.fingpt_path or args.fingpt_repo}")
    print(f"columns: {list(dataset.column_names)}")

    log = FilterLog()
    usable: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()

    for index in range(total_rows):
        row = dataset[index]
        instruction = str(row.get("instruction") or "").strip()
        sentence = str(row.get("input") or "").strip()
        response = str(row.get("output") or "").strip()
        if not sentence:
            log.bump("dropped_empty_input")
            continue
        if not response:
            log.bump("dropped_empty_output")
            continue

        label_counts[response.lower()] += 1
        template_counts[instruction] += 1
        question = f"{instruction}\n{sentence}" if instruction else sentence
        usable.append(
            {
                "source": DATASET,
                "question": question,
                "response": response,
                "original_id": f"fingpt-{args.split}-{index}",
                "gold_answer": response,
            }
        )

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
            gold_answer=item["gold_answer"],
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
    # Both distributions are here to be read before the first eval: the label
    # mix says how much of the pool answers on the nine-level scale, and the
    # instruction mix is the list of prompts the eval metrics have to match.
    stats["label_distribution"] = dict(label_counts.most_common())
    stats["instruction_templates"] = dict(template_counts.most_common())
    stats["metadata"] = build_metadata(
        domain=DOMAIN,
        dataset_name=DATASET,
        dataset_revision=args.dataset_revision,
        tokenizer_path=args.tokenizer_path,
        max_tokens=args.max_tokens,
        seed=args.seed,
        raw_count=total_rows,
        filtered_count=len(fitting),
        final_count=len(records),
        filter_rules={
            "selection": "none -- full corpus",
            "dedup": "none",
            "label_normalization": "none -- nine-level answers kept verbatim",
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
    print("\ninstruction templates:")
    for template, count in template_counts.most_common(5):
        print(f"  {count:>6}  {template[:96]}")


if __name__ == "__main__":
    main()
