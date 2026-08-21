"""GSM8K metric (math retention probe; not in the main IFEval flow)."""

from __future__ import annotations

import csv
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from onereplay.eval.generation import generate_response

NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+")


def normalize_number(text: str) -> str | None:
    matches = NUMBER_RE.findall(text.replace("$", ""))
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    return format(value.normalize(), "f")


def gold_answer(answer: str) -> str | None:
    if "####" in answer:
        answer = answer.split("####")[-1]
    return normalize_number(answer)


def predicted_answer(response: str) -> str | None:
    if "####" in response:
        response = response.split("####")[-1]
    return normalize_number(response)


class GSM8KMetric:
    name = "gsm8k"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = cfg.get("gsm8k_data_path", cfg.get("data_path", ""))
        limit = int(cfg.get("limit", 0))
        max_new_tokens = int(cfg.get("math_max_new_tokens", cfg.get("max_new_tokens", 512)))
        run_name = cfg.get("run_name", "base")

        examples: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                examples.append(json.loads(line))
                if limit > 0 and len(examples) >= limit:
                    break

        correct = 0
        scored = 0
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for idx, example in enumerate(examples, start=1):
                question = example["question"]
                prompt = (
                    "Solve the following grade-school math problem. "
                    "Reason briefly, then end your answer with '#### ' followed by the final number.\n\n"
                    f"Problem: {question}"
                )
                response = generate_response(model, tokenizer, prompt, device, max_new_tokens)
                gold = gold_answer(example.get("answer", ""))
                pred = predicted_answer(response)
                is_correct = gold is not None and pred == gold
                correct += int(is_correct)
                scored += int(gold is not None)
                file.write(
                    json.dumps(
                        {
                            "question": question,
                            "gold": gold,
                            "prediction": pred,
                            "correct": is_correct,
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if idx % 25 == 0:
                    print(f"gsm8k generated {idx}/{len(examples)}")

        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "data_path": data_path,
            "num_examples": len(examples),
            "num_scored": scored,
            "correct": correct,
            "accuracy": correct / max(scored, 1),
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "gsm8k_summary.csv"
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary
