"""Normalize the three direct-safety benchmarks into one prompts JSONL.

Follows the NSPO direct safety evaluation protocol: only direct harmful
requests, no jailbreak shells and no over-refusal set. The three benchmarks
stay separate all the way to the final table -- this script only puts them in
one file so a single generation pass can serve all three under provably
identical decoding settings.

Raw inputs come from codedata_check/safe/check_safe_data.py:

  advbench/harmful_behaviors.csv                 520 rows, columns goal,target
  harmbench/harmbench_behaviors_text_test.csv    320 rows, standard/contextual/copyright
  sorry_bench/question.jsonl                     440 rows, prompt_style=base

Every output row is {bench, id, prompt, meta}. `meta` keeps whatever the judge
or a later per-category breakdown needs; nothing downstream re-reads the raw
files.

    python -m onereplay.scripts.prepare_direct_safety_data \\
        --data_dir data/direct_safety_benchmarks \\
        --out_dir data/direct_safety_prompts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# HarmBench ships context-dependent behaviors as a separate context string. The
# official evaluation glues the two together with this separator before showing
# them to the model; the judge still sees the bare behavior.
HARMBENCH_CONTEXT_SEPARATOR = "\n\n---\n\n"

EXPECTED_COUNTS = {"advbench": 520, "harmbench": 320, "sorrybench": 440}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize AdvBench / HarmBench / SORRY-Bench into one prompts JSONL."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/direct_safety_benchmarks",
        help="Directory written by check_safe_data.py.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/direct_safety_prompts",
        help="Where to write direct_safety_prompts.jsonl.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="direct_safety_prompts.jsonl",
    )
    parser.add_argument(
        "--limit_per_bench",
        type=int,
        default=0,
        help="Keep only the first N rows of each benchmark. 0 keeps everything; "
        "use a small value to smoke-test the generate -> judge chain.",
    )
    parser.add_argument(
        "--strict",
        type=int,
        default=1,
        help="1 fails when a benchmark row count differs from the published size.",
    )
    return parser.parse_args()


def load_advbench(path: Path) -> list[dict[str, Any]]:
    """AdvBench: the `goal` column is the request; `target` is an attack-time
    prefix ("Sure, here is ...") and must not be shown to the model."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as file:
        for index, row in enumerate(csv.DictReader(file)):
            goal = (row.get("goal") or "").strip()
            if not goal:
                continue
            rows.append(
                {
                    "bench": "advbench",
                    "id": f"advbench-{index:04d}",
                    "prompt": goal,
                    "meta": {"goal": goal},
                }
            )
    return rows


def load_harmbench(path: Path) -> list[dict[str, Any]]:
    """HarmBench text test split, all three functional categories.

    `contextual` rows only make sense with their context prepended, so the
    prompt differs from the behavior. Both are kept: `prompt` is what the model
    sees, meta["behavior"] is what the judge scores against.
    """

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as file:
        for index, row in enumerate(csv.DictReader(file)):
            behavior = (row.get("Behavior") or "").strip()
            if not behavior:
                continue
            context = (row.get("ContextString") or "").strip()
            prompt = f"{context}{HARMBENCH_CONTEXT_SEPARATOR}{behavior}" if context else behavior
            rows.append(
                {
                    "bench": "harmbench",
                    "id": (row.get("BehaviorID") or f"harmbench-{index:04d}").strip(),
                    "prompt": prompt,
                    "meta": {
                        "behavior": behavior,
                        "functional_category": (row.get("FunctionalCategory") or "").strip(),
                        "semantic_category": (row.get("SemanticCategory") or "").strip(),
                        "has_context": bool(context),
                    },
                }
            )
    return rows


def load_sorrybench(path: Path) -> list[dict[str, Any]]:
    """SORRY-Bench base split: 44 categories x 10, single turn each."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            turns = record.get("turns") or []
            if not turns:
                continue
            if len(turns) > 1:
                raise ValueError(
                    f"SORRY-Bench row {record.get('question_id')} has {len(turns)} turns; "
                    "this protocol only evaluates single-turn direct requests."
                )
            rows.append(
                {
                    "bench": "sorrybench",
                    "id": f"sorry-{record.get('question_id', index)}",
                    "prompt": str(turns[0]).strip(),
                    "meta": {
                        "category": str(record.get("category", "")),
                        "prompt_style": record.get("prompt_style", ""),
                    },
                }
            )
    return rows


def describe(bench: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{bench}: {len(rows)} rows")
    if bench == "harmbench":
        counts = Counter(row["meta"]["functional_category"] for row in rows)
        contextual = sum(1 for row in rows if row["meta"]["has_context"])
        print(f"  functional categories: {dict(counts)}")
        print(f"  rows with a context string: {contextual}")
    if bench == "sorrybench":
        counts = Counter(row["meta"]["category"] for row in rows)
        print(f"  categories: {len(counts)} (each {min(counts.values())}-{max(counts.values())})")
    lengths = [len(row["prompt"]) for row in rows]
    print(f"  prompt chars: min={min(lengths)} median={sorted(lengths)[len(lengths) // 2]} max={max(lengths)}")
    print(f"  first prompt: {rows[0]['prompt'][:160]!r}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    sources = {
        "advbench": (data_dir / "advbench" / "harmful_behaviors.csv", load_advbench),
        "harmbench": (
            data_dir / "harmbench" / "harmbench_behaviors_text_test.csv",
            load_harmbench,
        ),
        "sorrybench": (data_dir / "sorry_bench" / "question.jsonl", load_sorrybench),
    }

    missing = [str(path) for path, _ in sources.values() if not path.is_file()]
    if missing:
        print("missing raw benchmark files:", *missing, sep="\n  ")
        print("\nrun codedata_check/safe/check_safe_data.py first (SORRY-Bench needs HF_TOKEN).")
        sys.exit(1)

    all_rows: list[dict[str, Any]] = []
    for bench, (path, loader) in sources.items():
        rows = loader(path)
        expected = EXPECTED_COUNTS[bench]
        if len(rows) != expected:
            message = f"{bench}: expected {expected} rows, loaded {len(rows)} from {path}"
            if args.strict:
                raise SystemExit(message + " (pass --strict 0 to continue anyway)")
            print(f"warning: {message}")
        describe(bench, rows)
        if args.limit_per_bench > 0:
            rows = rows[: args.limit_per_bench]
        all_rows.extend(rows)

    identifiers = [row["id"] for row in all_rows]
    duplicates = [key for key, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        raise SystemExit(f"duplicate prompt ids would break judge joins: {duplicates[:5]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    with out_path.open("w", encoding="utf-8") as file:
        for row in all_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nwrote {out_path} ({len(all_rows)} prompts)")
    print("per-bench:", dict(Counter(row["bench"] for row in all_rows)))


if __name__ == "__main__":
    main()
