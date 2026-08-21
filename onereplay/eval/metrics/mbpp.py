"""MBPP pass@1 metric (code retention probe; not in the main IFEval flow)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk

from onereplay.eval.code_exec import cleanup_completion, evaluate_assert_program
from onereplay.eval.generation import generate_response


def load_mbpp(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load MBPP from a local datasets directory or HuggingFace."""

    dataset_path = cfg.get("mbpp_dataset_path", "")
    split = cfg.get("dataset_split", "validation")
    if dataset_path:
        dataset_dict = load_from_disk(dataset_path)
        dataset = dataset_dict[split] if split in dataset_dict else dataset_dict
    else:
        config = cfg.get("dataset_config", "full") or None
        dataset = load_dataset(
            cfg.get("dataset_name", "google-research-datasets/mbpp"),
            config,
            split=split,
            cache_dir=cfg.get("cache_dir", ""),
        )
    rows = [dict(dataset[i]) for i in range(len(dataset))]
    limit = int(cfg.get("limit", 0))
    if limit > 0:
        rows = rows[:limit]
    return rows


def build_mbpp_prompt(example: dict[str, Any]) -> str:
    """Format one MBPP task as a code-generation user prompt."""

    tests = example.get("test_list") or example.get("test") or []
    tests_text = tests if isinstance(tests, str) else "\n".join(str(test) for test in tests)
    return (
        "Write a Python solution for the following programming task. "
        "Return only valid Python code; do not include markdown fences or explanations.\n\n"
        f"Task:\n{example.get('text', example.get('prompt', ''))}\n\n"
        f"The solution must pass these tests:\n{tests_text}"
    )


class MBPPMetric:
    name = "mbpp"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        max_new_tokens = int(cfg.get("code_max_new_tokens", cfg.get("max_new_tokens", 512)))
        timeout = float(cfg.get("timeout", 5.0))
        run_name = cfg.get("run_name", "base")

        rows = load_mbpp(cfg)
        passed = 0
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for idx, example in enumerate(rows, start=1):
                raw = generate_response(
                    model, tokenizer, build_mbpp_prompt(example), device, max_new_tokens
                )
                completion = cleanup_completion(raw)
                tests = example.get("test_list") or example.get("tests") or []
                if isinstance(tests, str):
                    tests = [tests]
                ok, error = evaluate_assert_program(completion, list(tests), timeout)
                passed += int(ok)
                file.write(
                    json.dumps(
                        {
                            "task_id": example.get("task_id"),
                            "passed": ok,
                            "error": error,
                            "completion": completion,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if idx % 10 == 0:
                    print(f"mbpp generated/tested {idx}/{len(rows)}")

        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "num_examples": len(rows),
            "passed": passed,
            "pass_at_1": passed / max(len(rows), 1),
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "mbpp_summary.csv"
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary
