"""Evaluate HumanEval pass@1 for a base model or a saved LoRA adapter.

The script uses deterministic generation and executes each generated solution
against the HumanEval test function in a short-lived subprocess with a timeout.
This is a lightweight code-retention check; generated code is also filtered for
obvious dangerous filesystem/process operations before execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import load_causal_lm_and_tokenizer, set_seed  # noqa: E402


DANGEROUS_PATTERNS = (
    "import os",
    "import sys",
    "import subprocess",
    "from os",
    "from sys",
    "from subprocess",
    "open(",
    "exec(",
    "eval(",
    "__import__",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "pathlib",
    "pickle",
)


def parse_args() -> argparse.Namespace:
    """Parse model, HumanEval, execution, and output settings."""

    parser = argparse.ArgumentParser(description="Run HumanEval pass@1.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--data_file",
        type=str,
        default="/home/weiliu1/huggingface/datasets/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet",
    )
    parser.add_argument("--cache_dir", type=str, default="/home/weiliu1/huggingface/datasets/cache")
    parser.add_argument("--output_dir", type=str, default="mycode/onereplay/results/humaneval/base")
    parser.add_argument("--summary_csv", type=str, default="mycode/onereplay/results/humaneval/humaneval_summary.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Ask the model to complete the HumanEval Python function only."""

    user_prompt = (
        "Complete the following Python function. "
        "Return only valid Python code for the function body or continuation; "
        "do not include markdown fences.\n\n"
        f"{prompt}"
    )
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_humaneval(data_file: str, cache_dir: str, limit: int) -> list[dict[str, Any]]:
    """Load downloaded HumanEval parquet data from disk."""

    dataset = load_dataset("parquet", data_files=data_file, split="train", cache_dir=cache_dir)
    rows = [dict(dataset[i]) for i in range(len(dataset))]
    if limit > 0:
        rows = rows[:limit]
    return rows


def load_model(args: argparse.Namespace):
    """Load the base model and optionally attach one LoRA adapter."""

    model, tokenizer = load_causal_lm_and_tokenizer(args.model_dir, args.model_name, args.use_bf16, args)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    return model, tokenizer


def cleanup_completion(text: str) -> str:
    """Remove chatty wrappers and stop at the next top-level definition."""

    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = max(parts, key=len).replace("python", "", 1).strip()
    stop_markers = ["\nclass ", "\ndef ", "\nif __name__", "\n# Example", "\nprint("]
    for marker in stop_markers:
        index = text.find(marker)
        if index >= 0:
            text = text[:index].rstrip()
    return text.rstrip() + "\n"


def generate_completion(model, tokenizer, prompt: str, device, max_new_tokens: int) -> str:
    """Generate one deterministic HumanEval completion."""

    text = apply_chat_template(tokenizer, prompt)
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
    return cleanup_completion(tokenizer.decode(new_tokens, skip_special_tokens=True))


def has_dangerous_code(code: str) -> bool:
    """Reject completions that try to access files, processes, network, or eval."""

    lowered = code.lower()
    return any(pattern in lowered for pattern in DANGEROUS_PATTERNS)


def run_one_test(program: str, entry_point: str, test_code: str, queue: mp.Queue) -> None:
    """Execute one HumanEval test program inside a subprocess."""

    try:
        namespace: dict[str, Any] = {}
        exec(program, namespace)
        exec(test_code, namespace)
        namespace["check"](namespace[entry_point])
        queue.put({"passed": True, "error": ""})
    except BaseException as exc:  # noqa: BLE001 - report model/runtime failures.
        queue.put({"passed": False, "error": repr(exc)})


def evaluate_program(program: str, entry_point: str, test_code: str, timeout: float) -> tuple[bool, str]:
    """Run one generated program with timeout protection."""

    if has_dangerous_code(program):
        return False, "blocked dangerous code pattern"
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=run_one_test, args=(program, entry_point, test_code, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return False, "timeout"
    if queue.empty():
        return False, "no result"
    result = queue.get()
    return bool(result["passed"]), str(result["error"])


def main() -> None:
    """Generate completions, execute tests, and write HumanEval pass@1."""

    args = parse_args()
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (Path(args.adapter_path).name if args.adapter_path else "base")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args)
    model.to(device)
    model.eval()

    examples = load_humaneval(args.data_file, args.cache_dir, args.limit)
    passed = 0
    response_path = output_dir / "responses.jsonl"
    with response_path.open("w", encoding="utf-8") as file:
        for idx, example in enumerate(examples, start=1):
            completion = generate_completion(model, tokenizer, example["prompt"], device, args.max_new_tokens)
            program = example["prompt"] + completion
            ok, error = evaluate_program(
                program,
                example["entry_point"],
                example["test"],
                timeout=args.timeout,
            )
            passed += int(ok)
            record = {
                "task_id": example["task_id"],
                "entry_point": example["entry_point"],
                "passed": ok,
                "error": error,
                "completion": completion,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if idx % 10 == 0:
                print(f"generated/tested {idx}/{len(examples)}")

    summary = {
        "run_name": run_name,
        "adapter_path": args.adapter_path,
        "num_examples": len(examples),
        "passed": passed,
        "pass_at_1": passed / max(len(examples), 1),
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
