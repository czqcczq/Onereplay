"""Export HumanEval-like code old-knowledge examples for covariance collection.

HumanEval is a docstring/prompt-to-Python-function benchmark. To avoid using
the HumanEval eval tasks themselves as old-knowledge data, this script builds a
local jsonl mixture from:

  1. MBPP train examples: natural language task -> reference Python code.
  2. Filtered OpenCodeInstruct examples: instruction -> Python code, requiring
     non-empty unit tests and mostly passing recorded test execution status.

The output has two fields, input and output, so scripts/collect_cov.py can format
it with the chat template using --input_column input --target_column output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    """Parse data source, filtering, and output settings."""

    parser = argparse.ArgumentParser(description="Export HumanEval-like code data.")
    parser.add_argument("--cache_dir", type=str, default="/home/weiliu1/huggingface/datasets/cache")
    parser.add_argument("--output_path", type=str, default="mycode/onereplay/results/humaneval_like_code_old_knowledge.jsonl")
    parser.add_argument("--max_mbpp", type=int, default=0, help="0 means all available MBPP train examples")
    parser.add_argument("--max_opencode", type=int, default=20000)
    parser.add_argument("--min_test_score", type=float, default=0.8)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    """Convert a dataset field into compact string content."""

    return str(value or "").strip()


def parse_test_status(value: Any) -> list[str]:
    """Return OpenCodeInstruct test execution statuses as a list of strings."""

    if isinstance(value, list):
        return [str(item).lower() for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).lower() for item in parsed]
        except json.JSONDecodeError:
            return [item.strip().lower() for item in value.split(",") if item.strip()]
    return []


def has_majority_pass(statuses: list[str]) -> bool:
    """Require that more than half of recorded unit tests passed."""

    if not statuses:
        return False
    passed = sum(status == "pass" for status in statuses)
    return passed / len(statuses) >= 0.5


def is_python_solution(text: str) -> bool:
    """Heuristic for HumanEval-like Python code completions."""

    lowered = text.lower()
    return "```python" in lowered or "\ndef " in text or text.lstrip().startswith("def ")


def export_mbpp(file, max_mbpp: int, cache_dir: str) -> int:
    """Write MBPP train examples into the output jsonl."""

    dataset = load_dataset(
        "google-research-datasets/mbpp",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    count = 0
    for example in dataset:
        input_text = clean_text(example.get("text"))
        output_text = clean_text(example.get("code"))
        if not input_text or not output_text:
            continue
        record = {
            "input": input_text,
            "output": output_text,
            "source": "mbpp_train",
            "task_id": str(example.get("task_id", "")),
        }
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
        if max_mbpp > 0 and count >= max_mbpp:
            break
    return count


def export_filtered_opencode(file, max_opencode: int, min_test_score: float, cache_dir: str) -> int:
    """Write filtered OpenCodeInstruct examples into the output jsonl."""

    dataset = load_dataset(
        "nvidia/OpenCodeInstruct",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    count = 0
    seen = 0
    for example in dataset:
        seen += 1
        input_text = clean_text(example.get("input"))
        output_text = clean_text(example.get("output"))
        unit_tests = clean_text(example.get("unit_tests"))
        statuses = parse_test_status(example.get("tests_execution_status"))
        try:
            test_score = float(example.get("average_test_score", 0.0))
        except (TypeError, ValueError):
            test_score = 0.0

        if not input_text or not output_text:
            continue
        if not unit_tests or not statuses:
            continue
        if test_score < min_test_score or not has_majority_pass(statuses):
            continue
        if not is_python_solution(output_text):
            continue

        record = {
            "input": input_text,
            "output": output_text,
            "source": "opencodeinstruct_python_unit_tests",
            "id": str(example.get("id", "")),
            "average_test_score": test_score,
        }
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
        if count % 1000 == 0:
            print(f"filtered OpenCodeInstruct: kept {count}, scanned {seen}")
        if max_opencode > 0 and count >= max_opencode:
            break
    return count


def main() -> None:
    """Export the local HumanEval-like jsonl mixture."""

    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        mbpp_count = export_mbpp(file, args.max_mbpp, args.cache_dir)
        opencode_count = export_filtered_opencode(
            file,
            args.max_opencode,
            args.min_test_score,
            args.cache_dir,
        )

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "mbpp_count": mbpp_count,
                "opencode_count": opencode_count,
                "total": mbpp_count + opencode_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
