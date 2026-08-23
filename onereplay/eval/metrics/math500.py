"""MATH-500 metric with the Hendrycks MATH answer-equivalence check.

Unlike aime.py (integer equality), MATH answers are LaTeX -- fractions, roots,
units, matrices -- so scoring needs the official Hendrycks `is_equiv` string
normalization (fractions, sqrt, \\left/\\right, units, etc.). Using integer
equality here would systematically under-count correct answers and make the
retention numbers meaningless.

The predicted answer is the last \\boxed{...} in the model response; the gold
answer is the dataset's `answer` field (already extracted) or the boxed span in
its `solution`. Both are normalized and compared with `is_equiv`.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from onereplay.eval.generation import generate_response

QUESTION_KEYS = ("problem", "question", "prompt", "input")
ANSWER_KEYS = ("answer", "final_answer", "solution", "target", "output")


# --- Hendrycks MATH answer normalization (verbatim logic from the MATH repo) ---
def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int, b_int = int(a), int(b)
        if string != f"{a_int}/{b_int}":
            return string
        return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
    except ValueError:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1: str | None, str2: str | None) -> bool:
    """Hendrycks MATH answer equivalence via string normalization."""

    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:  # noqa: BLE001
        return str1 == str2


def last_boxed_only_string(string: str) -> str | None:
    """Return the last \\boxed{...} / \\fbox{...} span with balanced braces."""

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    depth = 0
    right = None
    for i in range(idx, len(string)):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
    return string[idx : right + 1] if right is not None else None


def remove_boxed(span: str | None) -> str | None:
    """Strip the \\boxed{...} wrapper, returning the inner answer text."""

    if span is None:
        return None
    for prefix in ("\\boxed{", "\\fbox{"):
        if span.startswith(prefix) and span.endswith("}"):
            return span[len(prefix) : -1]
    if span.startswith("\\boxed "):
        return span[len("\\boxed ") :]
    return span


def extract_answer(text: str) -> str | None:
    """Pull the final boxed answer from a solution or a model response."""

    return remove_boxed(last_boxed_only_string(text))


def build_prompt(question: str) -> str:
    """Render the eval prompt. Shared so budget probes measure the real thing."""

    return (
        "Solve the following math problem. Reason step by step, then give "
        "the final answer as \\boxed{...} at the end.\n\n"
        f"Problem: {question}"
    )


def load_json_records(path: str) -> list[dict[str, Any]]:
    """Load MATH-500 examples from jsonl or json."""

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
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def first_existing_key(example: dict, preferred: str, candidates: tuple[str, ...]) -> str:
    """Return the configured key or the first available candidate key."""

    if preferred:
        if preferred not in example:
            raise KeyError(f"Field {preferred!r} not found. Keys: {sorted(example)}")
        return preferred
    for key in candidates:
        if key in example:
            return key
    raise KeyError(f"None of {candidates} found. Keys: {sorted(example)}")


class MATH500Metric:
    name = "math500"
    # Config keys checked in order for this metric's eval file. Subclasses (AMC)
    # point at their own path so several boxed-answer sets can run in one job
    # without clobbering each other's output_dir / summary csv.
    data_path_keys = ("math500_data_path", "data_path")

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = ""
        for key in self.data_path_keys:
            if cfg.get(key):
                data_path = cfg[key]
                break
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
            raw_answer = str(record[answer_key]).strip()
            # If the gold field is a full solution, pull its boxed answer.
            gold = extract_answer(raw_answer) or raw_answer
            if question and gold:
                examples.append({"question": question, "answer": gold})
            if limit > 0 and len(examples) >= limit:
                break

        correct = 0
        scored = 0
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for idx, example in enumerate(examples, start=1):
                prompt = build_prompt(example["question"])
                response = generate_response(model, tokenizer, prompt, device, max_new_tokens)
                gold = example["answer"]
                pred = extract_answer(response)
                is_correct = is_equiv(pred, gold)
                correct += int(is_correct)
                scored += 1
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
                if idx % 25 == 0:
                    print(f"math500 generated {idx}/{len(examples)}")

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
        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / f"{self.name}_summary.csv"
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary


class AMCMetric(MATH500Metric):
    """AMC competition set, scored with the same boxed-answer / is_equiv logic."""

    name = "amc"
    data_path_keys = ("amc_data_path", "data_path")
