"""Generate and score Google IFEval for base or LoRA-adapted models.

This script uses the official Google Research IFEval checker vendored under:

  mycode/onereplay/third_party/instruction_following_eval

Outputs:
  - responses jsonl: prompt/response pairs
  - eval_results_strict.jsonl and eval_results_loose.jsonl
  - summary json/csv with prompt-level and instruction-level accuracies
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import load_causal_lm_and_tokenizer, set_seed  # noqa: E402

THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from instruction_following_eval import evaluation_lib  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse generation and evaluation settings."""

    parser = argparse.ArgumentParser(description="Run official IFEval.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--input_data",
        type=str,
        default="mycode/onereplay/third_party/instruction_following_eval/data/input_data.jsonl",
    )
    parser.add_argument("--output_dir", type=str, default="mycode/onereplay/results/ifeval/base")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Format IFEval prompt as a chat user turn."""

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
    """Generate a deterministic response for one IFEval prompt."""

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
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model(args: argparse.Namespace):
    """Load base model and optionally attach a saved LoRA adapter."""

    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    return model, tokenizer


def summarize_outputs(outputs) -> dict[str, Any]:
    """Compute prompt-level and instruction-level IFEval accuracies."""

    prompt_total = len(outputs)
    prompt_correct = sum(o.follow_all_instructions for o in outputs)
    instruction_total = sum(len(o.follow_instruction_list) for o in outputs)
    instruction_correct = sum(sum(o.follow_instruction_list) for o in outputs)
    return {
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "prompt_accuracy": prompt_correct / max(prompt_total, 1),
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
        "instruction_accuracy": instruction_correct / max(instruction_total, 1),
    }


def write_summary_csv(path: Path, row: dict[str, Any]) -> None:
    """Append one run summary to a CSV file."""

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (Path(args.adapter_path).name if args.adapter_path else "base")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args)
    model.to(device)
    model.eval()

    inputs = evaluation_lib.read_prompt_list(args.input_data)
    if args.limit > 0:
        inputs = inputs[: args.limit]

    response_path = output_dir / "responses.jsonl"
    prompt_to_response = {}
    with response_path.open("w", encoding="utf-8") as file:
        for idx, inp in enumerate(inputs, start=1):
            response = generate_response(
                model,
                tokenizer,
                inp.prompt,
                device,
                args.max_new_tokens,
            )
            prompt_to_response[inp.prompt] = response
            file.write(json.dumps({"prompt": inp.prompt, "response": response}, ensure_ascii=False) + "\n")
            if idx % 25 == 0:
                print(f"generated {idx}/{len(inputs)}")

    strict_outputs = [
        evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        for inp in inputs
    ]
    loose_outputs = [
        evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
        for inp in inputs
    ]
    evaluation_lib.write_outputs(str(output_dir / "eval_results_strict.jsonl"), strict_outputs)
    evaluation_lib.write_outputs(str(output_dir / "eval_results_loose.jsonl"), loose_outputs)

    strict_summary = summarize_outputs(strict_outputs)
    loose_summary = summarize_outputs(loose_outputs)
    summary = {
        "run_name": run_name,
        "adapter_path": args.adapter_path,
        "num_prompts": len(inputs),
        "strict_prompt_accuracy": strict_summary["prompt_accuracy"],
        "strict_instruction_accuracy": strict_summary["instruction_accuracy"],
        "loose_prompt_accuracy": loose_summary["prompt_accuracy"],
        "loose_instruction_accuracy": loose_summary["instruction_accuracy"],
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(output_dir.parent / "ifeval_summary.csv", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

