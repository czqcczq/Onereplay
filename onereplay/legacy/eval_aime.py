"""Evaluate AIME-style math accuracy for a base model or saved LoRA adapter.

AIME problems have integer answers, usually in [0, 999]. This script generates
one deterministic response per problem and scores exact match on the final
integer. The prompt asks the model to end with "#### <answer>", matching the
format used by the GSM8K evaluator but with AIME-specific wording.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import load_causal_lm_and_tokenizer, set_seed  # noqa: E402


INTEGER_RE = re.compile(r"-?\d+")
QUESTION_KEYS = ("problem", "question", "prompt", "input")
ANSWER_KEYS = ("answer", "final_answer", "solution_answer", "target", "output")


def parse_args() -> argparse.Namespace:
    """Parse model, AIME data, and output settings."""

    parser = argparse.ArgumentParser(description="Run AIME integer-answer evaluation.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--data_path", type=str, default="/home/weiliu1/huggingface/datasets/aime_eval.jsonl")
    parser.add_argument("--question_field", type=str, default="")
    parser.add_argument("--answer_field", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="mycode/onereplay/results/aime/base")
    parser.add_argument("--summary_csv", type=str, default="mycode/onereplay/results/aime/aime_summary.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()


def load_json_records(path: str) -> list[dict[str, Any]]:
    """Load AIME examples from jsonl or json.

    The json path may contain either a list of examples or a dictionary with a
    common split key such as "test" or "train".
    """

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
    raise ValueError(f"Unsupported AIME data format: {path}")


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


def load_examples(args: argparse.Namespace) -> list[dict[str, str]]:
    """Read AIME examples and normalize them to question/answer strings."""

    raw_records = load_json_records(args.data_path)
    examples = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        question_key = first_existing_key(record, args.question_field, QUESTION_KEYS)
        answer_key = first_existing_key(record, args.answer_field, ANSWER_KEYS)
        question = str(record[question_key]).strip()
        answer = str(record[answer_key]).strip()
        if question and answer:
            examples.append({"question": question, "answer": answer})
        if args.limit > 0 and len(examples) >= args.limit:
            break
    return examples


def normalize_integer(text: str) -> str | None:
    """Extract the last integer and remove leading zeros."""

    matches = INTEGER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    return str(int(matches[-1]))


def gold_answer(answer: str) -> str | None:
    """Extract the gold AIME integer from common answer formats."""

    if "####" in answer:
        answer = answer.split("####")[-1]
    if "\\boxed" in answer:
        boxed_tail = answer.split("\\boxed")[-1]
        boxed_match = re.search(r"\{([^{}]+)\}", boxed_tail)
        if boxed_match:
            answer = boxed_match.group(1)
    return normalize_integer(answer)


def predicted_answer(response: str) -> str | None:
    """Extract the model's final integer, preferring text after #### or boxed."""

    if "####" in response:
        response = response.split("####")[-1]
    if "\\boxed" in response:
        boxed_tail = response.split("\\boxed")[-1]
        boxed_match = re.search(r"\{([^{}]+)\}", boxed_tail)
        if boxed_match:
            response = boxed_match.group(1)
    return normalize_integer(response)


def apply_chat_template(tokenizer, question: str) -> str:
    """Format an AIME problem as a chat prompt."""

    prompt = (
        "Solve the following AIME-style math problem. The final answer is an "
        "integer from 0 to 999. Reason carefully, then end your response with "
        "'#### ' followed by only the final integer.\n\n"
        f"Problem: {question}"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(args: argparse.Namespace):
    """Load the base model and optionally attach one LoRA adapter."""

    model, tokenizer = load_causal_lm_and_tokenizer(args.model_dir, args.model_name, args.use_bf16, args)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    return model, tokenizer


def generate(model, tokenizer, question: str, device, max_new_tokens: int) -> str:
    """Generate one deterministic AIME response."""

    text = apply_chat_template(tokenizer, question)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
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


def main() -> None:
    """Run generation, score exact integer match, and write JSONL/CSV outputs."""

    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (Path(args.adapter_path).name if args.adapter_path else "base")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args)
    model.to(device)
    model.eval()

    examples = load_examples(args)
    correct = 0
    scored = 0
    response_path = output_dir / "responses.jsonl"
    with response_path.open("w", encoding="utf-8") as file:
        for idx, example in enumerate(examples, start=1):
            response = generate(model, tokenizer, example["question"], device, args.max_new_tokens)
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
                print(f"generated {idx}/{len(examples)}")

    summary = {
        "run_name": run_name,
        "adapter_path": args.adapter_path,
        "data_path": args.data_path,
        "num_examples": len(examples),
        "num_scored": scored,
        "correct": correct,
        "accuracy": correct / max(scored, 1),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
