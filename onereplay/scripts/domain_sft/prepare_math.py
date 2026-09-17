"""Rebuild the Math specialist pool from NuminaMath-CoT under the shared rules.

The Math stage already trained once, from ``prepare_numina_math.py``. This
script is not a refinement of it -- it is the same corpus re-emitted under the
rules the three domains now share, which differ from that run in three ways that
each require the model to be retrained:

  * **No system turn.** The earlier run injected a math-specific system message.
    Keeping it would mean the Math specialist was prompted differently from the
    other two, so a later "Math forgot X" result could just as well be "Math was
    asked differently".
  * **4096, drop instead of truncate.** The earlier run left length to training,
    where ``tokenizer_to_ids`` truncates *from the left* -- keeping the answer and
    cutting the question's opening. Fewer than 1% of NuminaMath rows are affected
    (p99 is around 1462 tokens against a 3045 max), so this changes little here;
    it matters that the rule is the same one Medical and Finance obey.
  * **seed 42**, matching the other two.

Sampling stays what it was: proportional stratified, so a 100k draw preserves
the corpus's own source mix. cn_k12, synthetic_math and orca_math are ~70% of
NuminaMath and are grade-school to mid-difficulty, while olympiads, aops_forum,
amc_aime and math carry the competition problems. Fixed per-source quotas would
be an undeclared difficulty decision; proportional leaves it to the corpus.
``proportional_quotas`` is imported from the original script rather than
reimplemented so the two pools stay comparable.

Order of operations: filter, then measure, then sample. Quotas computed before
the length cutoff would be quotas over rows that do not all survive it, and each
source would land slightly under its share.
"""

from __future__ import annotations

import argparse
import random
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
    serialize_many,
    summarize,
    write_dataset_dir,
    write_jsonl,
    write_stats,
)
from onereplay.scripts.prepare_openr1_math import EVAL_PROMPT_PREFIX, has_boxed  # noqa: E402
from onereplay.scripts.prepare_numina_math import proportional_quotas  # noqa: E402

DOMAIN = "math"
DATASET = "AI-MO/NuminaMath-CoT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Math SFT pool from NuminaMath-CoT under the shared domain rules."
    )
    parser.add_argument(
        "--numina_path",
        type=str,
        default="",
        help="Directory of NuminaMath-CoT parquet shards, or a save_to_disk dir.",
    )
    parser.add_argument("--numina_repo", type=str, default=DATASET)
    parser.add_argument("--dataset_revision", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, default="data/processed/math_train.jsonl")
    parser.add_argument("--out_stats", type=str, default="data/stats/math_stats.json")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/math_train",
        help="save_to_disk target, which is what training loads. Empty skips it.",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=100000,
        help="Target pool size. 0 keeps every usable row.",
    )
    parser.add_argument(
        "--require_boxed",
        type=int,
        default=1,
        help="Keep only solutions with a \\boxed/\\fbox span. On by default: the "
        "graders read that span and nothing else.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Must match what training is launched with: it contributes tokens "
        "that count against --max_tokens. The default is 91_s1_train_math.pbs's.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_rows", type=int, default=0, help="Debug aid; 0 reads all.")
    return parser.parse_args()


def load_numina(args: argparse.Namespace):
    """Load the NuminaMath train split from a local copy or the Hub."""

    from datasets import load_dataset, load_from_disk

    if args.numina_path:
        path = Path(args.numina_path)
        if (path / "dataset_dict.json").exists():
            return load_from_disk(str(path))["train"]
        if (path / "dataset_info.json").exists():
            return load_from_disk(str(path))
        shards = sorted(str(item) for item in path.glob("**/train-*.parquet"))
        if not shards:
            raise SystemExit(f"no train-*.parquet under {path}")
        return load_dataset("parquet", data_files={"train": shards})["train"]
    return load_dataset(args.numina_repo, split="train")


def build_question(problem: str) -> str:
    """Render the user turn with the same prefix the earlier Math run used.

    Byte-identical to prepare_openr1_math's EVAL_PROMPT_PREFIX so this pool stays
    comparable with the Math runs that came before it, and so the eval prompt
    does not have to change.
    """

    return f"{EVAL_PROMPT_PREFIX}{problem.strip()}"


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path, args.system_prompt)
    dataset = load_numina(args)
    total_rows = len(dataset) if not args.max_rows else min(args.max_rows, len(dataset))
    print(f"loaded {len(dataset)} rows from {args.numina_path or args.numina_repo}")

    log = FilterLog()
    usable: list[dict[str, str]] = []

    for index in range(total_rows):
        row = dataset[index]
        problem = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        source = (row.get("source") or "").strip() or "(unknown)"

        if not problem:
            log.bump("dropped_empty_problem")
            continue
        if not solution:
            log.bump("dropped_empty_solution")
            continue
        if args.require_boxed and not has_boxed(solution):
            log.bump("dropped_no_boxed")
            continue
        usable.append(
            {
                "source": source,
                "question": build_question(problem),
                "response": solution,
                "original_id": f"numina-train-{index}",
            }
        )

    print(f"{len(usable)} rows usable before the length cutoff; measuring...")
    measured = serialize_many(
        tokenizer,
        [(item["question"], item["response"]) for item in usable],
        batch_size=args.batch_size,
        progress_every=50000,
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

    # Proportional stratified draw, computed on the rows that survived the cutoff.
    by_source: dict[str, list[int]] = {}
    for index, item in enumerate(fitting):
        by_source.setdefault(item["source"], []).append(index)
    available = {name: len(indices) for name, indices in by_source.items()}

    if args.sample_rows and args.sample_rows < len(fitting):
        quotas = proportional_quotas(available, args.sample_rows)
        rng = random.Random(args.seed)
        chosen: list[int] = []
        for name in sorted(by_source):
            indices = sorted(by_source[name])
            quota = quotas.get(name, 0)
            if quota < len(indices):
                log.bump("dropped_proportional_sampling", len(indices) - quota)
                indices = rng.sample(indices, quota)
            chosen.extend(indices)
        chosen.sort()
    else:
        quotas = dict(available)
        chosen = list(range(len(fitting)))

    records = [
        build_record(
            domain=DOMAIN,
            index=position,
            source=fitting[index]["source"],
            question=fitting[index]["question"],
            response=fitting[index]["response"],
            serialized=fitting[index]["serialized"],
            original_id=fitting[index]["original_id"],
        )
        for position, index in enumerate(chosen)
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
    stats["sampling"] = {
        "method": "proportional stratified, largest-remainder quotas",
        "seed": args.seed,
        "target_rows": args.sample_rows,
        "available_before_sampling": dict(
            sorted(available.items(), key=lambda item: -item[1])
        ),
        "quotas": dict(sorted(quotas.items(), key=lambda item: -item[1])),
    }
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
            "require_boxed": bool(args.require_boxed),
            "instruction_prefix": EVAL_PROMPT_PREFIX,
            "sampling": "proportional stratified over sources, after the length cutoff",
            "sample_rows": args.sample_rows,
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
