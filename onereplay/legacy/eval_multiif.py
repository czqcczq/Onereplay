"""Generate and score Multi-IF (English subset by default) for base or LoRA models.

Multi-IF (facebook/Multi-IF) extends the single-turn IFEval into 3-turn
cumulative conversations across multiple languages. This script:

  - loads the Multi-IF CSV (multiIF_20241018.csv),
  - keeps only one language (English by default) to keep the retention signal
    clean on small models,
  - runs cumulative multi-turn greedy generation, feeding the model's own
    previous responses back as assistant turns,
  - checks each turn's response with the vendored IFEval verifiable checkers
    (no LLM judge), and
  - reports per-turn strict/loose prompt- and instruction-level accuracy.

The per-turn instruction checking reuses the same vendored library used by
eval_ifeval.py:

  mycode/onereplay/third_party/instruction_following_eval

Multi-IF stores, for each conversation row and each turn i:
  turn_i_prompt              JSON message dict, e.g. {"role": "user", ...}
  turn_i_instruction_id_list JSON list of IFEval instruction ids
  turn_i_kwargs              JSON list whose elements are JSON-encoded dicts
  language                   language name, e.g. "English"
  key                        unique conversation id
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

from instruction_following_eval import evaluation_lib, instructions_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse generation and evaluation settings."""

    parser = argparse.ArgumentParser(description="Run Multi-IF (English) instruction following.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="mycode/onereplay/third_party/Multi-IF/multiIF_20241018.csv",
        help="Path to the Multi-IF CSV downloaded from facebook/Multi-IF.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Keep only this language; use 'all' to keep every language.",
    )
    parser.add_argument("--output_dir", type=str, default="mycode/onereplay/results/multiif/base")
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all rows.")
    return parser.parse_args()


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


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render a full conversation into text with an open assistant turn."""

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


def generate_response(model, tokenizer, messages, device, max_new_tokens: int) -> str:
    """Greedy-decode one assistant response for the current conversation."""

    text = render_chat(tokenizer, messages)
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


def _load_json_maybe(value: Any) -> Any:
    """json.loads a value only if it is a string; pass through otherwise."""

    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_turn(row: dict[str, Any], turn: int) -> dict[str, Any] | None:
    """Extract one turn's user text, instruction ids, and kwargs.

    Returns None when the turn is absent (Multi-IF marks missing turns with the
    literal string "None" or an empty cell).
    """

    prompt_field = row.get(f"turn_{turn}_prompt", "")
    if prompt_field in ("", "None", None):
        return None

    message = _load_json_maybe(prompt_field)
    if isinstance(message, list):
        message = message[0] if message else {}
    user_content = str(message.get("content", "")).strip()
    if not user_content:
        return None

    instruction_id_list = _load_json_maybe(row.get(f"turn_{turn}_instruction_id_list", "[]"))
    raw_kwargs = _load_json_maybe(row.get(f"turn_{turn}_kwargs", "[]"))
    kwargs = [_load_json_maybe(kwarg) for kwarg in raw_kwargs]

    return {
        "user_content": user_content,
        "instruction_id_list": list(instruction_id_list),
        "kwargs": kwargs,
    }


def check_turn(turn_data: dict[str, Any], response: str) -> dict[str, Any] | None:
    """Score one turn's response with the vendored strict/loose IFEval checkers.

    Any instruction id missing from the vendored registry is reported so a
    checker-version mismatch is visible instead of silently crashing. Turns that
    contain an unknown id are skipped from scoring and counted separately.
    """

    instruction_id_list = turn_data["instruction_id_list"]
    unsupported = [
        instruction_id
        for instruction_id in instruction_id_list
        if instruction_id not in instructions_registry.INSTRUCTION_DICT
    ]
    if unsupported:
        return {"unsupported": unsupported}

    inp = evaluation_lib.InputExample(
        key=0,
        instruction_id_list=instruction_id_list,
        prompt=turn_data["user_content"],
        kwargs=turn_data["kwargs"],
    )
    # A fresh single-entry map avoids key collisions between turns/rows that
    # happen to share identical user text.
    prompt_to_response = {turn_data["user_content"]: response}
    strict = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    loose = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
    return {
        "strict_follow_list": strict.follow_instruction_list,
        "loose_follow_list": loose.follow_instruction_list,
    }


def read_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Read the Multi-IF CSV, optionally filtering to one language."""

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Multi-IF CSV not found: {input_path}. Download facebook/Multi-IF on a "
            "networked node first, e.g. `git clone https://huggingface.co/datasets/"
            "facebook/Multi-IF`."
        )

    rows: list[dict[str, Any]] = []
    # keep_default_na semantics: read everything as raw strings so the literal
    # "None" turn marker and empty cells are preserved exactly like the official
    # Multi-IF pipeline (pandas read_csv(keep_default_na=False)).
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            if args.language != "all" and record.get("language") != args.language:
                continue
            rows.append(record)
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def new_turn_counter() -> dict[str, int]:
    """Per-turn accumulators for strict/loose prompt- and instruction-level hits."""

    return {
        "prompt_total": 0,
        "prompt_strict": 0,
        "prompt_loose": 0,
        "inst_total": 0,
        "inst_strict": 0,
        "inst_loose": 0,
    }


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

    rows = read_rows(args)
    print(f"Multi-IF: scoring {len(rows)} conversations (language={args.language})")

    per_turn = {turn: new_turn_counter() for turn in range(1, args.max_turns + 1)}
    unsupported_seen: dict[str, int] = {}
    response_path = output_dir / "responses.jsonl"

    with response_path.open("w", encoding="utf-8") as response_file:
        for idx, row in enumerate(rows, start=1):
            messages: list[dict[str, str]] = []
            turn_records: list[dict[str, Any]] = []

            for turn in range(1, args.max_turns + 1):
                turn_data = parse_turn(row, turn)
                if turn_data is None:
                    break

                messages.append({"role": "user", "content": turn_data["user_content"]})
                response = generate_response(
                    model, tokenizer, messages, device, args.max_new_tokens
                )
                messages.append({"role": "assistant", "content": response})

                result = check_turn(turn_data, response)
                record = {
                    "turn": turn,
                    "prompt": turn_data["user_content"],
                    "response": response,
                    "instruction_id_list": turn_data["instruction_id_list"],
                }

                if result is None or "unsupported" in (result or {}):
                    for instruction_id in (result or {}).get("unsupported", []):
                        unsupported_seen[instruction_id] = unsupported_seen.get(instruction_id, 0) + 1
                    record["scored"] = False
                else:
                    strict_list = result["strict_follow_list"]
                    loose_list = result["loose_follow_list"]
                    counter = per_turn[turn]
                    counter["prompt_total"] += 1
                    counter["prompt_strict"] += int(all(strict_list))
                    counter["prompt_loose"] += int(all(loose_list))
                    counter["inst_total"] += len(strict_list)
                    counter["inst_strict"] += sum(strict_list)
                    counter["inst_loose"] += sum(loose_list)
                    record["scored"] = True
                    record["strict_follow_list"] = strict_list
                    record["loose_follow_list"] = loose_list

                turn_records.append(record)

            response_file.write(
                json.dumps(
                    {"key": row.get("key"), "language": row.get("language"), "turns": turn_records},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if idx % 25 == 0:
                print(f"generated {idx}/{len(rows)}")

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    turn_summaries: dict[str, Any] = {}
    strict_prompt_values: list[float] = []
    strict_inst_values: list[float] = []
    loose_prompt_values: list[float] = []
    loose_inst_values: list[float] = []
    for turn in range(1, args.max_turns + 1):
        counter = per_turn[turn]
        strict_prompt = ratio(counter["prompt_strict"], counter["prompt_total"])
        strict_inst = ratio(counter["inst_strict"], counter["inst_total"])
        loose_prompt = ratio(counter["prompt_loose"], counter["prompt_total"])
        loose_inst = ratio(counter["inst_loose"], counter["inst_total"])
        turn_summaries[f"turn_{turn}"] = {
            "num_prompts": counter["prompt_total"],
            "strict_prompt_accuracy": strict_prompt,
            "strict_instruction_accuracy": strict_inst,
            "loose_prompt_accuracy": loose_prompt,
            "loose_instruction_accuracy": loose_inst,
        }
        if counter["prompt_total"]:
            strict_prompt_values.append(strict_prompt)
            strict_inst_values.append(strict_inst)
            loose_prompt_values.append(loose_prompt)
            loose_inst_values.append(loose_inst)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "run_name": run_name,
        "adapter_path": args.adapter_path,
        "language": args.language,
        "num_conversations": len(rows),
        "max_turns": args.max_turns,
        "avg_strict_prompt_accuracy": mean(strict_prompt_values),
        "avg_strict_instruction_accuracy": mean(strict_inst_values),
        "avg_loose_prompt_accuracy": mean(loose_prompt_values),
        "avg_loose_instruction_accuracy": mean(loose_inst_values),
        "per_turn": turn_summaries,
        "unsupported_instructions": unsupported_seen,
        "output_dir": str(output_dir),
    }

    if unsupported_seen:
        print(
            "WARNING: some instruction ids are missing from the vendored IFEval "
            f"registry and were skipped: {unsupported_seen}. Consider syncing the "
            "checker with the Multi-IF version."
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    flat_row = {
        "run_name": run_name,
        "adapter_path": args.adapter_path,
        "language": args.language,
        "num_conversations": len(rows),
        "avg_strict_prompt_accuracy": summary["avg_strict_prompt_accuracy"],
        "avg_strict_instruction_accuracy": summary["avg_strict_instruction_accuracy"],
        "avg_loose_prompt_accuracy": summary["avg_loose_prompt_accuracy"],
        "avg_loose_instruction_accuracy": summary["avg_loose_instruction_accuracy"],
    }
    for turn in range(1, args.max_turns + 1):
        turn_summary = turn_summaries[f"turn_{turn}"]
        flat_row[f"turn_{turn}_strict_prompt_accuracy"] = turn_summary["strict_prompt_accuracy"]
        flat_row[f"turn_{turn}_strict_instruction_accuracy"] = turn_summary["strict_instruction_accuracy"]

    summary_csv = output_dir.parent / "multiif_summary.csv"
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat_row)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
