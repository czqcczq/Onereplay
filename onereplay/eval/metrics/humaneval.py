"""HumanEval pass@1 metric (code retention probe; not in the main IFEval flow)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from onereplay.eval.code_exec import cleanup_completion, evaluate_entry_point_program
from onereplay.eval.generation import generate_response


def load_humaneval(data_file: str, cache_dir: str, limit: int) -> list[dict[str, Any]]:
    """Load downloaded HumanEval parquet data from disk."""

    dataset = load_dataset("parquet", data_files=data_file, split="train", cache_dir=cache_dir)
    rows = [dict(dataset[i]) for i in range(len(dataset))]
    if limit > 0:
        rows = rows[:limit]
    return rows


class HumanEvalMetric:
    name = "humaneval"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        data_file = cfg.get("humaneval_data_file", "")
        cache_dir = cfg.get("cache_dir", "")
        limit = int(cfg.get("limit", 0))
        max_new_tokens = int(cfg.get("code_max_new_tokens", cfg.get("max_new_tokens", 384)))
        timeout = float(cfg.get("timeout", 5.0))
        run_name = cfg.get("run_name", "base")

        rows = load_humaneval(data_file, cache_dir, limit)
        passed = 0
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for idx, example in enumerate(rows, start=1):
                prompt = example["prompt"]
                user_prompt = (
                    "Complete the following Python function. "
                    "Return only valid Python code for the function body or continuation; "
                    "do not include markdown fences.\n\n"
                    f"{prompt}"
                )
                raw = generate_response(model, tokenizer, user_prompt, device, max_new_tokens)
                completion = cleanup_completion(raw)
                program = prompt + completion
                ok, error = evaluate_entry_point_program(
                    program, example["entry_point"], example["test"], timeout
                )
                passed += int(ok)
                file.write(
                    json.dumps(
                        {
                            "task_id": example.get("task_id"),
                            "entry_point": example.get("entry_point"),
                            "passed": ok,
                            "error": error,
                            "completion": completion,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if idx % 10 == 0:
                    print(f"humaneval generated/tested {idx}/{len(rows)}")

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
        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "humaneval_summary.csv"
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary
