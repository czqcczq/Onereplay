"""Multi-IF English metric (multi-turn cumulative IFEval-style checks)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from onereplay.eval.generation import generate_from_messages

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from instruction_following_eval import evaluation_lib, instructions_registry  # noqa: E402


def _load_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_turn(row: dict[str, Any], turn: int) -> dict[str, Any] | None:
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


def check_turn(turn_data: dict[str, Any], response: str) -> dict[str, Any]:
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
    prompt_to_response = {turn_data["user_content"]: response}
    strict = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    loose = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
    return {
        "strict_follow_list": strict.follow_instruction_list,
        "loose_follow_list": loose.follow_instruction_list,
    }


class MultiIFMetric:
    name = "multiif"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = cfg.get("run_name", "base")
        input_csv = cfg.get(
            "multiif_input",
            str(THIRD_PARTY / "Multi-IF" / "multiIF_20241018.csv"),
        )
        language = cfg.get("multiif_language", "English")
        max_turns = int(cfg.get("max_turns", 3))
        max_new_tokens = int(cfg.get("multiif_max_new_tokens", cfg.get("max_new_tokens", 1024)))
        limit = int(cfg.get("multiif_limit", cfg.get("limit", 0)))

        input_path = Path(input_csv)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Multi-IF CSV not found: {input_path}. Download facebook/Multi-IF first."
            )

        rows: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for record in reader:
                if language != "all" and record.get("language") != language:
                    continue
                rows.append(record)
        if limit > 0:
            rows = rows[:limit]

        per_turn = {
            turn: {
                "prompt_total": 0,
                "prompt_strict": 0,
                "prompt_loose": 0,
                "inst_total": 0,
                "inst_strict": 0,
                "inst_loose": 0,
            }
            for turn in range(1, max_turns + 1)
        }
        unsupported_seen: dict[str, int] = {}
        response_path = output_dir / "responses.jsonl"

        with response_path.open("w", encoding="utf-8") as response_file:
            for idx, row in enumerate(rows, start=1):
                messages: list[dict[str, str]] = []
                turn_records: list[dict[str, Any]] = []
                for turn in range(1, max_turns + 1):
                    turn_data = parse_turn(row, turn)
                    if turn_data is None:
                        break
                    messages.append({"role": "user", "content": turn_data["user_content"]})
                    response = generate_from_messages(
                        model, tokenizer, messages, device, max_new_tokens
                    )
                    messages.append({"role": "assistant", "content": response})
                    result = check_turn(turn_data, response)
                    record = {
                        "turn": turn,
                        "prompt": turn_data["user_content"],
                        "response": response,
                        "instruction_id_list": turn_data["instruction_id_list"],
                    }
                    if "unsupported" in result:
                        for instruction_id in result["unsupported"]:
                            unsupported_seen[instruction_id] = (
                                unsupported_seen.get(instruction_id, 0) + 1
                            )
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
                        {
                            "key": row.get("key"),
                            "language": row.get("language"),
                            "turns": turn_records,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if idx % 25 == 0:
                    print(f"multiif generated {idx}/{len(rows)}")

        def ratio(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0

        turn_summaries: dict[str, Any] = {}
        strict_prompt_values: list[float] = []
        strict_inst_values: list[float] = []
        for turn in range(1, max_turns + 1):
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

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "language": language,
            "num_conversations": len(rows),
            "max_turns": max_turns,
            "avg_strict_prompt_accuracy": mean(strict_prompt_values),
            "avg_strict_instruction_accuracy": mean(strict_inst_values),
            "per_turn": turn_summaries,
            "unsupported_instructions": unsupported_seen,
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        flat_row = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "language": language,
            "num_conversations": len(rows),
            "avg_strict_prompt_accuracy": summary["avg_strict_prompt_accuracy"],
            "avg_strict_instruction_accuracy": summary["avg_strict_instruction_accuracy"],
        }
        for turn in range(1, max_turns + 1):
            turn_summary = turn_summaries[f"turn_{turn}"]
            flat_row[f"turn_{turn}_strict_prompt_accuracy"] = turn_summary[
                "strict_prompt_accuracy"
            ]
            flat_row[f"turn_{turn}_strict_instruction_accuracy"] = turn_summary[
                "strict_instruction_accuracy"
            ]
        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "multiif_summary.csv"
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(flat_row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(flat_row)
        return summary
