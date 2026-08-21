"""IFEval metric using the vendored Google checker."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from onereplay.eval.generation import generate_response

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from instruction_following_eval import evaluation_lib  # noqa: E402


class IFEvalMetric:
    name = "ifeval"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = cfg.get("run_name", "base")
        input_data = cfg.get(
            "ifeval_input",
            str(THIRD_PARTY / "instruction_following_eval" / "data" / "input_data.jsonl"),
        )
        max_new_tokens = int(cfg.get("max_new_tokens", 768))
        limit = int(cfg.get("limit", 0))

        inputs = evaluation_lib.read_prompt_list(input_data)
        if limit > 0:
            inputs = inputs[:limit]

        response_path = output_dir / "responses.jsonl"
        prompt_to_response = {}
        with response_path.open("w", encoding="utf-8") as file:
            for idx, inp in enumerate(inputs, start=1):
                response = generate_response(
                    model, tokenizer, inp.prompt, device, max_new_tokens
                )
                prompt_to_response[inp.prompt] = response
                file.write(
                    json.dumps({"prompt": inp.prompt, "response": response}, ensure_ascii=False)
                    + "\n"
                )
                if idx % 25 == 0:
                    print(f"ifeval generated {idx}/{len(inputs)}")

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

        def summarize(outputs):
            prompt_total = len(outputs)
            prompt_correct = sum(o.follow_all_instructions for o in outputs)
            instruction_total = sum(len(o.follow_instruction_list) for o in outputs)
            instruction_correct = sum(sum(o.follow_instruction_list) for o in outputs)
            return {
                "prompt_accuracy": prompt_correct / max(prompt_total, 1),
                "instruction_accuracy": instruction_correct / max(instruction_total, 1),
            }

        strict_summary = summarize(strict_outputs)
        loose_summary = summarize(loose_outputs)
        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "num_prompts": len(inputs),
            "strict_prompt_accuracy": strict_summary["prompt_accuracy"],
            "strict_instruction_accuracy": strict_summary["instruction_accuracy"],
            "loose_prompt_accuracy": loose_summary["prompt_accuracy"],
            "loose_instruction_accuracy": loose_summary["instruction_accuracy"],
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "ifeval_summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary
