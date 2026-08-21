"""Evaluate instruction-following ability after LoRA fine-tuning.

This script is intentionally lightweight and fully automatic. It does not try
to judge whether an answer is factually correct. Instead, it checks whether the
model follows explicit output constraints, such as:

  - answer with exactly one word
  - return valid JSON with required keys
  - produce exactly N bullet points
  - avoid a forbidden word

These tests are useful for OneReplay because your method should ideally reduce
old-knowledge damage without making the model worse at following instructions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import load_causal_lm_and_tokenizer, set_seed  # noqa: E402


@dataclass
class InstructionCase:
    """One automatically checkable instruction-following example."""

    case_id: str
    prompt: str
    check_type: str
    params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse model, adapter, generation, and output settings."""

    parser = argparse.ArgumentParser(description="Evaluate instruction following.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="Optional LoRA adapter path. Empty means evaluate base model.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_jsonl", type=str, default="mycode/onereplay/results/if_eval.jsonl")
    parser.add_argument("--summary_csv", type=str, default="mycode/onereplay/results/if_eval_summary.csv")
    parser.add_argument("--run_name", type=str, default="")
    return parser.parse_args()


def build_instruction_cases() -> list[InstructionCase]:
    """Create a small deterministic benchmark of instruction constraints.

    The prompts are deliberately simple. The score should reflect whether the
    model obeys the requested format, not whether it has specialized knowledge.
    """

    return [
        InstructionCase(
            "one_word_yes",
            "Answer with exactly one word: yes. Do not add punctuation.",
            "exact",
            {"text": "yes"},
        ),
        InstructionCase(
            "one_word_no",
            "Answer with exactly one word: no. Do not add punctuation.",
            "exact",
            {"text": "no"},
        ),
        InstructionCase(
            "uppercase_token",
            "Output exactly this token in uppercase and nothing else: onereplay",
            "exact",
            {"text": "ONEREPLAY"},
        ),
        InstructionCase(
            "lowercase_token",
            "Output exactly this token in lowercase and nothing else: QWEN",
            "exact",
            {"text": "qwen"},
        ),
        InstructionCase(
            "json_two_keys",
            'Return valid JSON only, with exactly two keys: "answer" and "confidence". '
            'Set "answer" to "True" and "confidence" to 0.7.',
            "json_exact_keys",
            {"keys": ["answer", "confidence"], "values": {"answer": "True", "confidence": 0.7}},
        ),
        InstructionCase(
            "json_list",
            'Return valid JSON only: {"items":["red","blue","green"]}.',
            "json_value",
            {"value": {"items": ["red", "blue", "green"]}},
        ),
        InstructionCase(
            "three_bullets",
            "Give exactly three bullet points about safe model evaluation. Each bullet must start with '- '.",
            "bullet_count",
            {"count": 3},
        ),
        InstructionCase(
            "two_numbered",
            "Give exactly two numbered steps for saving an experiment log. Use the format '1. ' and '2. '.",
            "numbered_count",
            {"count": 2},
        ),
        InstructionCase(
            "forbidden_word",
            "Describe LoRA in one short sentence. Do not use the word adapter.",
            "forbidden_word",
            {"word": "adapter", "max_words": 30},
        ),
        InstructionCase(
            "required_words",
            "Write one sentence that contains all three words: matrix, hidden, regularizer.",
            "contains_all",
            {"words": ["matrix", "hidden", "regularizer"]},
        ),
        InstructionCase(
            "max_words",
            "Explain gradient accumulation in at most 12 words.",
            "max_words",
            {"max_words": 12},
        ),
        InstructionCase(
            "single_letter",
            "Choose option A. Output only the single letter.",
            "exact",
            {"text": "A"},
        ),
        InstructionCase(
            "no_markdown",
            "Say that the experiment is complete in one sentence. Do not use Markdown.",
            "no_markdown",
            {},
        ),
        InstructionCase(
            "csv_format",
            "Return exactly one CSV row with three fields: model,onereplay,pass",
            "csv_row",
            {"fields": ["model", "onereplay", "pass"]},
        ),
        InstructionCase(
            "bracketed",
            "Output the word stable inside square brackets and nothing else.",
            "exact",
            {"text": "[stable]"},
        ),
    ]


def normalize_text(text: str) -> str:
    """Strip chat artifacts and surrounding whitespace before scoring."""

    text = text.strip()
    text = re.sub(r"^```(?:json|csv|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def count_words(text: str) -> int:
    """Count simple word tokens for length-constrained instructions."""

    return len(re.findall(r"[A-Za-z0-9_'-]+", text))


def parse_json_response(text: str) -> Any:
    """Parse JSON, allowing only a surrounding fenced block to be removed."""

    return json.loads(normalize_text(text))


def check_case(case: InstructionCase, response: str) -> tuple[bool, str]:
    """Return pass/fail and a short diagnostic for one generated response."""

    text = normalize_text(response)
    params = case.params

    if case.check_type == "exact":
        expected = params["text"]
        return text == expected, f"expected exact {expected!r}, got {text!r}"

    if case.check_type == "json_exact_keys":
        try:
            obj = parse_json_response(text)
        except Exception as exc:
            return False, f"invalid JSON: {exc}"
        expected_keys = params["keys"]
        if sorted(obj.keys()) != sorted(expected_keys):
            return False, f"expected keys {expected_keys}, got {sorted(obj.keys())}"
        for key, expected_value in params["values"].items():
            if obj.get(key) != expected_value:
                return False, f"key {key!r}: expected {expected_value!r}, got {obj.get(key)!r}"
        return True, "valid JSON with exact keys and values"

    if case.check_type == "json_value":
        try:
            obj = parse_json_response(text)
        except Exception as exc:
            return False, f"invalid JSON: {exc}"
        expected = params["value"]
        return obj == expected, f"expected JSON {expected!r}, got {obj!r}"

    if case.check_type == "bullet_count":
        lines = [line for line in text.splitlines() if line.strip()]
        ok = len(lines) == params["count"] and all(line.startswith("- ") for line in lines)
        return ok, f"expected {params['count']} '- ' bullets, got {len(lines)} lines"

    if case.check_type == "numbered_count":
        lines = [line for line in text.splitlines() if line.strip()]
        expected = [f"{idx}. " for idx in range(1, params["count"] + 1)]
        ok = len(lines) == params["count"] and all(
            line.startswith(prefix) for line, prefix in zip(lines, expected)
        )
        return ok, f"expected numbered prefixes {expected}, got {lines!r}"

    if case.check_type == "forbidden_word":
        word = params["word"].lower()
        has_forbidden = re.search(rf"\b{re.escape(word)}\b", text.lower()) is not None
        within_limit = count_words(text) <= params["max_words"]
        return (not has_forbidden and within_limit), (
            f"forbidden={has_forbidden}, words={count_words(text)}"
        )

    if case.check_type == "contains_all":
        lower = text.lower()
        missing = [word for word in params["words"] if word.lower() not in lower]
        return not missing, f"missing words: {missing}"

    if case.check_type == "max_words":
        words = count_words(text)
        return words <= params["max_words"], f"words={words}, max={params['max_words']}"

    if case.check_type == "no_markdown":
        has_markdown = any(marker in text for marker in ["```", "#", "- ", "* ", "|"])
        return not has_markdown and len(text) > 0, f"has_markdown={has_markdown}"

    if case.check_type == "csv_row":
        fields = [field.strip() for field in text.split(",")]
        return fields == params["fields"], f"expected fields {params['fields']}, got {fields}"

    raise ValueError(f"Unknown check_type: {case.check_type}")


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Format the evaluation prompt as a chat conversation for Qwen-style models."""

    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_response(model, tokenizer, prompt: str, device, max_new_tokens: int) -> str:
    """Generate one deterministic response for an instruction case."""

    chat_text = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model_for_eval(args: argparse.Namespace):
    """Load the base model and optionally attach a trained LoRA adapter."""

    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    return model, tokenizer


def write_summary_csv(path: str, summary: dict[str, Any]) -> None:
    """Append a one-line summary so multiple runs are easy to compare."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or (Path(args.adapter_path).name if args.adapter_path else "base")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_for_eval(args)
    model.to(device)
    model.eval()

    cases = build_instruction_cases()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    passed = 0
    records = []
    for case in cases:
        response = generate_response(
            model,
            tokenizer,
            case.prompt,
            device,
            args.max_new_tokens,
        )
        ok, diagnostic = check_case(case, response)
        passed += int(ok)
        record = {
            "run_name": run_name,
            "adapter_path": args.adapter_path,
            "case_id": case.case_id,
            "check_type": case.check_type,
            "passed": ok,
            "diagnostic": diagnostic,
            "prompt": case.prompt,
            "response": response,
        }
        records.append(record)
        print(f"{case.case_id}: {'PASS' if ok else 'FAIL'} | {diagnostic}")

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "run_name": run_name,
        "adapter_path": args.adapter_path,
        "num_cases": len(cases),
        "passed": passed,
        "pass_rate": passed / len(cases),
        "output_jsonl": str(output_path),
    }
    write_summary_csv(args.summary_csv, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

