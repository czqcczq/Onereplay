"""Evaluate GSM8K math accuracy for a base model or a saved LoRA adapter.

The script generates one deterministic answer per question and compares the
last numeric answer in the response with the GSM8K gold answer after "####".
It is intended as a lightweight math-retention check alongside IFEval.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import load_causal_lm_and_tokenizer, set_seed  # noqa: E402


NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+")


def parse_args() -> argparse.Namespace:
    """Parse model, data, and output settings."""

    parser = argparse.ArgumentParser(description="Run GSM8K numeric-answer evaluation.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--data_path", type=str, default="/home/weiliu1/huggingface/datasets/gsm8k_test_public.jsonl")
    parser.add_argument("--output_dir", type=str, default="mycode/onereplay/results/gsm8k/base")
    parser.add_argument("--summary_csv", type=str, default="mycode/onereplay/results/gsm8k/gsm8k_summary.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def apply_chat_template(tokenizer, question: str) -> str:
    """Format a GSM8K question as a chat prompt with a final-answer request."""

    prompt = (
        "Solve the following grade-school math problem. "
        "Reason briefly, then end your answer with '#### ' followed by the final number.\n\n"
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


def load_examples(path: str, limit: int) -> list[dict[str, Any]]:
    """Read GSM8K jsonl examples and optionally keep only the first limit."""

    examples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            examples.append(json.loads(line))
            if limit > 0 and len(examples) >= limit:
                break
    return examples


def normalize_number(text: str) -> str | None:
    """Return the last number in text as a normalized Decimal string."""

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
    """Extract GSM8K gold final answer from the text after ####."""

    if "####" in answer:
        answer = answer.split("####")[-1]
    return normalize_number(answer)


def predicted_answer(response: str) -> str | None:
    """Extract the model's final numeric answer, preferring text after ####."""

    if "####" in response:
        response = response.split("####")[-1]
    return normalize_number(response)


def load_model(args: argparse.Namespace):
    """Load the base model and optionally attach one LoRA adapter."""

    model, tokenizer = load_causal_lm_and_tokenizer(args.model_dir, args.model_name, args.use_bf16, args)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    return model, tokenizer


def generate(model, tokenizer, question: str, device, max_new_tokens: int) -> str:
    """Generate one deterministic GSM8K response."""

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
    """Run generation, score numeric exact match, and write JSONL/CSV outputs."""

    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (Path(args.adapter_path).name if args.adapter_path else "base")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args)
    model.to(device)
    model.eval()

    examples = load_examples(args.data_path, args.limit)
    correct = 0
    scored = 0
    response_path = output_dir / "responses.jsonl"
    with response_path.open("w", encoding="utf-8") as file:
        for idx, example in enumerate(examples, start=1):
            response = generate(model, tokenizer, example["question"], device, args.max_new_tokens)
            gold = gold_answer(example.get("answer", ""))
            pred = predicted_answer(response)
            is_correct = gold is not None and pred == gold
            correct += int(is_correct)
            scored += int(gold is not None)
            record = {
                "question": example["question"],
                "gold": gold,
                "prediction": pred,
                "correct": is_correct,
                "response": response,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if idx % 25 == 0:
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
