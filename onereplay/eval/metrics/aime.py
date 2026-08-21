"""AIME integer-answer metric (math retention probe; not in the main IFEval flow)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from onereplay.eval.generation import generate_response

INTEGER_RE = re.compile(r"-?\d+")
QUESTION_KEYS = ("problem", "question", "prompt", "input")
ANSWER_KEYS = ("answer", "final_answer", "solution_answer", "target", "output")


def load_json_records(path: str) -> list[dict[str, Any]]:
    """Load AIME examples from jsonl or json."""

    data_path = Path(path)
    if data_path.suffix.lower() == ".jsonl":
        records = []
        with data_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        return records

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("test", "validation", "eval", "train", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def first_existing_key(example: dict[str, Any], preferred: str, candidates: tuple[str, ...]) -> str:
    """Return the configured key or the first available candidate key."""

    if preferred:
        if preferred not in example:
            raise KeyError(f"Field {preferred!r} not found. Available keys: {sorted(example)}")
        return preferred
    for key in candidates:
        if key in example:
            return key
    raise KeyError(f"None of {candidates} found. Available keys: {sorted(example)}")


def normalize_integer(text: str) -> str | None:
    """Extract the last integer and remove leading zeros."""

    matches = INTEGER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    return str(int(matches[-1]))


def _strip_boxed(text: str) -> str:
    if "####" in text:
        text = text.split("####")[-1]
    if "\\boxed" in text:
        boxed_tail = text.split("\\boxed")[-1]
        boxed_match = re.search(r"\{([^{}]+)\}", boxed_tail)
        if boxed_match:
            text = boxed_match.group(1)
    return text


def gold_answer(answer: str) -> str | None:
    """Extract the gold AIME integer from common answer formats."""

    return normalize_integer(_strip_boxed(answer))


def predicted_answer(response: str) -> str | None:
    """Extract the model's final integer, preferring text after #### or boxed."""

    return normalize_integer(_strip_boxed(response))


class AIMEMetric:
    name = "aime"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = cfg.get("aime_data_path", cfg.get("data_path", ""))
        limit = int(cfg.get("limit", 0))
        max_new_tokens = int(cfg.get("math_max_new_tokens", cfg.get("max_new_tokens", 1024)))
        run_name = cfg.get("run_name", "base")
        question_field = cfg.get("question_field", "")
        answer_field = cfg.get("answer_field", "")

        records = load_json_records(data_path)
        examples: list[dict[str, str]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            question_key = first_existing_key(record, question_field, QUESTION_KEYS)
            answer_key = first_existing_key(record, answer_field, ANSWER_KEYS)
            question = str(record[question_key]).strip()
            answer = str(record[answer_key]).strip()
            if question and answer:
                examples.append({"question": question, "answer": answer})
            if limit > 0 and len(examples) >= limit:
                break

        correct = 0
        scored = 0
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for idx, example in enumerate(examples, start=1):
                prompt = (
                    "Solve the following AIME-style math problem. The final answer is an "
                    "integer from 0 to 999. Reason carefully, then end your response with "
                    "'#### ' followed by only the final integer.\n\n"
                    f"Problem: {example['question']}"
                )
                response = generate_response(model, tokenizer, prompt, device, max_new_tokens)
                gold = gold_answer(example["answer"])
                pred = predicted_answer(response)
                is_correct = gold is not None and pred == gold
                correct += int(is_correct)
                scored += int(gold is not None)
                file.write(
                    json.dumps(
                        {
                            "question": example["question"],
                            "gold": gold,
                            "prediction": pred,
                            "correct": is_correct,
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if idx % 10 == 0:
                    print(f"aime generated {idx}/{len(examples)}")

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
        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "aime_summary.csv"
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary
