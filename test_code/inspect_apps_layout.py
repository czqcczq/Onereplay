"""Report the APPS test split's layout. Answers "what does 'the first N' select?"

    python test_code/inspect_apps_layout.py --apps_dir apps_ref/apps_data

Reads only the small columns, never the multi-megabyte test-case payloads, so it
finishes in seconds on the 1.2 GB test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_from_disk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps_dir", type=str, default="apps_ref/apps_data")
    parser.add_argument("--head", type=int, default=500)
    parser.add_argument("--sample_io", type=int, default=300)
    args = parser.parse_args()

    loaded = load_from_disk(args.apps_dir)
    print(f"splits: { {name: len(split) for name, split in loaded.items()} }")

    for split_name in loaded:
        split = loaded[split_name]
        print(f"\n==== {split_name}: {len(split)} problems ====")
        print(f"columns: {split.column_names}")

        difficulties = split["difficulty"]
        print(f"difficulty mix : {dict(Counter(difficulties))}")
        print("contiguous index range per difficulty:")
        for name in ("introductory", "interview", "competition"):
            indices = [i for i, value in enumerate(difficulties) if value == name]
            if indices:
                contiguous = indices == list(range(indices[0], indices[-1] + 1))
                print(
                    f"  {name:<13} n={len(indices):<5} index {indices[0]}..{indices[-1]}"
                    f"  contiguous={contiguous}"
                )

        if args.head > 0 and split_name == "test":
            head = dict(Counter(difficulties[: args.head]))
            print(f"first {args.head} problems -> {head}")

        # fn_name / test-case presence needs the big column, so sample instead of
        # scanning: this is a layout report, not a filter.
        take = min(args.sample_io, len(split))
        call_based = 0
        no_tests = 0
        num_tests: list[int] = []
        for index in range(take):
            try:
                spec = json.loads(split[index]["input_output"] or "{}")
            except (json.JSONDecodeError, TypeError):
                no_tests += 1
                continue
            if spec.get("fn_name"):
                call_based += 1
            inputs = spec.get("inputs") or []
            if not inputs or not (spec.get("outputs") or []):
                no_tests += 1
            else:
                num_tests.append(len(inputs))
        print(f"first {take} problems: call-based={call_based}, missing tests={no_tests}")
        if num_tests:
            ordered = sorted(num_tests)
            def pct(q: float) -> int:
                return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
            print(
                f"  test cases per problem: P50={pct(0.5)} P90={pct(0.9)} "
                f"max={ordered[-1]}  (--apps_max_tests caps this)"
            )

        lengths = [len(text) for text in split["question"]]
        ordered_len = sorted(lengths)
        def lpct(q: float) -> int:
            return ordered_len[min(len(ordered_len) - 1, int(q * (len(ordered_len) - 1)))]
        print(
            f"question chars: P50={lpct(0.5)} P90={lpct(0.9)} P99={lpct(0.99)} "
            f"max={ordered_len[-1]}  (~/3.5 for a rough token count)"
        )


if __name__ == "__main__":
    main()
