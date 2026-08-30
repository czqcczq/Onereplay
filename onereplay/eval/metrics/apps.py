"""APPS pass@1 + pass_ratio metric (code retention probe).

Reported numbers
----------------
pass_at_1        strict: every evaluated test case matched. Same shape as the
                 MBPP/HumanEval columns, so the three are readable side by side.
mean_pass_ratio  average fraction of test cases passed. Continuous per problem,
                 so a paired test on it resolves far smaller differences than
                 McNemar on strict pass -- which matters because the strict
                 metric on 500 problems cannot resolve the effect size this
                 experiment expects (see 39_code_eval_base_vanilla.pbs).

Both come from the same execution pass; the ratio is free.

Absolute scores are not comparable to published APPS results: judging is
whitespace-normalised exact match (stricter than the official harness), test
cases may be truncated (--apps_max_tests), and call-based problems are dropped
by default. Run onereplay.scripts.check_apps_judge to get this harness's
ceiling on gold solutions before reading any model number.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk

from onereplay.eval.apps_exec import (
    evaluate_stdin_program,
    extract_apps_code,
    looks_truncated,
)
from onereplay.eval.generation import generate_response

PROMPT_TEMPLATE = """You are a helpful assistant that generates Python code to solve programming problems.

Problem:
{question}

Please think step by step and generate code to solve this problem.

Important Requirements:
- Your solution MUST read input using input() and write output using print().
- Do NOT hardcode or generate inputs yourself.
- First think about how many and what type of inputs you need.
- Then generate the code to solve the problem.
- Finally print the result.

Respond in the format:
**Code:**
```python
# your code here
```"""


def _as_text(value: Any) -> str:
    """APPS stores a test case either as a string or as a list of lines."""

    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _normalise_cell(value: Any) -> str:
    text = _as_text(value)
    if not text.endswith("\n"):
        text += "\n"
    return text


def load_apps(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load APPS problems, filtered down to what this harness can score.

    --apps_data_file accepts either a parquet file (what download_apps_data
    writes, matching how humaneval is loaded) or a datasets.save_to_disk
    directory, in which case --apps_split picks the split.

    Returns the usable rows plus a count of what was dropped and why, because
    "how many problems am I actually scoring" is not obvious from the split name
    and has to go into the summary.
    """

    data_file = cfg.get("apps_data_file", "")
    if not data_file:
        raise ValueError("apps metric needs --apps_data_file (see download_apps_data)")

    if Path(data_file).is_dir():
        split = cfg.get("apps_split", "test")
        loaded = load_from_disk(data_file)
        dataset = loaded[split] if split in loaded else loaded
    else:
        dataset = load_dataset(
            "parquet",
            data_files=data_file,
            split="train",
            cache_dir=cfg.get("cache_dir") or None,
        )

    wanted = [
        name.strip()
        for name in str(cfg.get("apps_difficulties", "")).split(",")
        if name.strip()
    ]
    stdin_only = int(cfg.get("apps_stdin_only", 1))

    # The difficulty tiers are contiguous index blocks (test: interview 0-2999,
    # competition 3000-3999, introductory 4000-4999), so selecting a tier means
    # skipping thousands of rows. Filter on the difficulty column first -- it is
    # cheap -- and only then materialise rows, because each carries its full
    # test-case payload and the test split is 1.2 GB.
    candidates = range(len(dataset))
    if wanted:
        difficulties = dataset["difficulty"]
        candidates = [i for i, value in enumerate(difficulties) if value in wanted]

    limit = int(cfg.get("limit", 0))
    dropped = {"call_based": 0, "no_tests": 0, "bad_json": 0, "scanned": 0}
    rows: list[dict[str, Any]] = []
    for index in candidates:
        if limit > 0 and len(rows) >= limit:
            break
        dropped["scanned"] += 1
        example = dict(dataset[index])
        try:
            spec = json.loads(example.get("input_output") or "{}")
        except (json.JSONDecodeError, TypeError):
            dropped["bad_json"] += 1
            continue
        # Call-based problems pass arguments to a named function instead of
        # feeding stdin. Scoring them needs a second execution path; until then
        # they are dropped rather than silently mangled into unpassable stdin.
        if stdin_only and spec.get("fn_name"):
            dropped["call_based"] += 1
            continue
        inputs = spec.get("inputs") or []
        outputs = spec.get("outputs") or []
        if not inputs or not outputs:
            dropped["no_tests"] += 1
            continue
        rows.append(
            {
                "problem_id": example.get("id", example.get("problem_id")),
                "difficulty": example.get("difficulty"),
                "question": (example.get("question") or "").strip(),
                "solutions": example.get("solutions") or "[]",
                "test_inputs": [_normalise_cell(item) for item in inputs],
                "test_outputs": [_normalise_cell(item) for item in outputs],
            }
        )

    return rows, dropped


def build_apps_prompt(example: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(question=example["question"])


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-problem records into the reported numbers."""

    count = max(len(records), 1)
    passed = sum(1 for row in records if row["passed"])
    error_buckets: dict[str, int] = {}
    for row in records:
        if row["passed"]:
            continue
        label = row.get("error") or "unknown"
        # Keep the exception class, drop the message: 'IndexError: list index out
        # of range' and 'IndexError: ...' are the same bucket.
        bucket = label.split(":", 1)[0].strip() if ":" in label else label
        error_buckets[bucket] = error_buckets.get(bucket, 0) + 1
    return {
        "num_examples": len(records),
        "passed": passed,
        "pass_at_1": passed / count,
        "mean_pass_ratio": sum(row["pass_ratio"] for row in records) / count,
        "num_truncated": sum(1 for row in records if row.get("truncated")),
        "num_no_code_block": sum(1 for row in records if row.get("no_code_block")),
        "error_buckets": error_buckets,
    }


class APPSMetric:
    name = "apps"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        max_new_tokens = int(
            cfg.get("apps_max_new_tokens", cfg.get("code_max_new_tokens", 1024))
        )
        timeout = float(cfg.get("timeout", 10.0))
        max_tests = int(cfg.get("apps_max_tests", 10))
        tmp_root = os.environ.get("TMPDIR", "")
        run_name = cfg.get("run_name", "base")

        rows, dropped = load_apps(cfg)
        print(
            f"apps: {len(rows)} problems to score out of {dropped['scanned']} scanned; "
            f"dropped {dropped['call_based']} call-based, "
            f"{dropped['no_tests']} without tests, {dropped['bad_json']} unparseable"
        )

        records: list[dict[str, Any]] = []
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for index, example in enumerate(rows, start=1):
                raw = generate_response(
                    model, tokenizer, build_apps_prompt(example), device, max_new_tokens
                )
                completion = extract_apps_code(raw)
                result = evaluate_stdin_program(
                    completion,
                    example["test_inputs"],
                    example["test_outputs"],
                    timeout,
                    tmp_root=tmp_root,
                    max_tests=max_tests,
                )
                record = {
                    "problem_id": example["problem_id"],
                    "difficulty": example["difficulty"],
                    "passed": result["passed"],
                    "pass_ratio": result["pass_ratio"],
                    "num_tests": result["num_tests"],
                    "num_passed": result["num_passed"],
                    "error": result["error"],
                    "truncated": looks_truncated(raw),
                    "no_code_block": "```" not in raw,
                    "completion": completion,
                }
                records.append(record)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % 10 == 0:
                    print(f"apps generated/tested {index}/{len(rows)}")

        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            **summarise(records),
            "max_tests": max_tests,
            "max_new_tokens": max_new_tokens,
            "difficulties": cfg.get("apps_difficulties", "all"),
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "apps_summary.csv"
        flat = {
            key: (json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value)
            for key, value in summary.items()
        }
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(flat.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(flat)
        return summary
