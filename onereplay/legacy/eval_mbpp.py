"""Evaluate MBPP pass@1 for a base model or saved LoRA adapter.

MBPP examples provide a natural-language programming task plus unit-test style
assertions. This script asks the model for one deterministic Python solution,
executes it with the provided tests in a short-lived subprocess, and reports
pass@1 on the selected split.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, load_from_disk
from peft import PeftModel

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import load_causal_lm_and_tokenizer, set_seed  # noqa: E402
from onereplay.legacy.eval_humaneval import DANGEROUS_PATTERNS, cleanup_completion  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse model, MBPP data, execution, and output settings."""

    parser = argparse.ArgumentParser(description="Run MBPP pass@1.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="google-research-datasets/mbpp")
    parser.add_argument("--dataset_config", type=str, default="full")
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--cache_dir", type=str, default="/home/weiliu1/huggingface/datasets/cache")
    parser.add_argument("--output_dir", type=str, default="mycode/onereplay/results/mbpp/base")
    parser.add_argument("--summary_csv", type=str, default="mycode/onereplay/results/mbpp/mbpp_summary.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def load_mbpp(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load MBPP from a local datasets directory or HuggingFace."""

    if args.dataset_path:
        dataset_dict = load_from_disk(args.dataset_path)
        dataset = dataset_dict[args.dataset_split] if args.dataset_split in dataset_dict else dataset_dict
    else:
        config = args.dataset_config if args.dataset_config else None
        dataset = load_dataset(
            args.dataset_name,
            config,
            split=args.dataset_split,
            cache_dir=args.cache_dir,
        )
    rows = [dict(dataset[i]) for i in range(len(dataset))]
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def apply_chat_template(tokenizer, example: dict[str, Any]) -> str:
    """Format one MBPP task as a code-generation chat prompt."""

    tests = example.get("test_list") or example.get("test") or []
    if isinstance(tests, str):
        tests_text = tests
    else:
        tests_text = "\n".join(str(test) for test in tests)
    prompt = (
        "Write a Python solution for the following programming task. "
        "Return only valid Python code; do not include markdown fences or explanations.\n\n"
        f"Task:\n{example.get('text', example.get('prompt', ''))}\n\n"
        f"The solution must pass these tests:\n{tests_text}"
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


def generate_solution(model, tokenizer, example: dict[str, Any], device, max_new_tokens: int) -> str:
    """Generate one deterministic MBPP solution."""

    text = apply_chat_template(tokenizer, example)
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


def run_one_test(program: str, tests: list[str], queue: mp.Queue) -> None:
    """Execute one MBPP program and its asserts inside a subprocess."""

    try:
        namespace: dict[str, Any] = {}
        exec(program, namespace)
        for test in tests:
            exec(str(test), namespace)
        queue.put({"passed": True, "error": ""})
    except BaseException as exc:  # noqa: BLE001 - report model/runtime failures.
        queue.put({"passed": False, "error": repr(exc)})


def evaluate_program(program: str, tests: list[str], timeout: float) -> tuple[bool, str]:
    """Run one generated program with timeout protection."""

    if has_dangerous_code(program):
        return False, "blocked dangerous code pattern"
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=run_one_test, args=(program, tests, queue))
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
    """Generate solutions, execute MBPP tests, and write pass@1 outputs."""

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

    examples = load_mbpp(args)
    passed = 0
    response_path = output_dir / "responses.jsonl"
    with response_path.open("w", encoding="utf-8") as file:
        for idx, example in enumerate(examples, start=1):
            completion = generate_solution(model, tokenizer, example, device, args.max_new_tokens)
            tests = example.get("test_list") or example.get("test") or []
            if isinstance(tests, str):
                tests = [tests]
            ok, error = evaluate_program(completion, [str(test) for test in tests], timeout=args.timeout)
            passed += int(ok)
            record = {
                "task_id": example.get("task_id", example.get("id", idx)),
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
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
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
