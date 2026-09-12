"""DialogSum ROUGE: did the new task actually get learned?

Every other metric here measures retention -- how much of the base model's old
ability survived fine-tuning. This one measures the other end of the trade-off.
Both are needed to read a lambda sweep, because a regularizer can always win on
retention by refusing to learn anything at all; without a new-task score there
is no way to tell "protected the old knowledge" from "barely trained".

The prompt is not rebuilt here. apply_train_template is the same function
train.py feeds the model, so the prompt half is byte-identical to training and
a base-vs-SFT gap cannot come from a template mismatch.

One reference per dialogue: prepare_dialogsum.py emits the SFT schema and
refuses the official test variant that ships summary1/2/3. Scoring against
three references (max over refs, as the DialogSum paper does) would need that
variant preserved upstream, so the numbers here are comparable across our runs
but sit below published three-reference scores.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from onereplay.data.chat import apply_train_template
from onereplay.eval.generation import batched_generate_from_texts, resolve_batch_size

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")

_MISSING_ROUGE = (
    "rouge_score is not installed, so the DialogSum metric cannot score.\n"
    "Install it on the LOGIN node (compute nodes are offline):\n"
    "    pip install rouge-score\n"
    "Or drop 'dialogsum' from --metrics."
)


def load_rows(path: Path, limit: int) -> list[dict[str, str]]:
    """Read the held-out JSONL written by prepare_dialogsum.py."""

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("input") and row.get("output"):
                rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


class DialogSumMetric:
    name = "dialogsum"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        try:
            from rouge_score import rouge_scorer
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(_MISSING_ROUGE) from exc

        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = cfg.get("run_name", "base")

        data_path = Path(cfg["dialogsum_input"])
        if not data_path.is_file():
            raise FileNotFoundError(
                f"DialogSum held-out file not found: {data_path}. "
                "prepare_dialogsum.py writes <out_dir>_test.jsonl next to the "
                "train pool; pass --dialogsum_input to point elsewhere."
            )

        max_new_tokens = int(cfg.get("dialogsum_max_new_tokens", 256))
        limit = int(cfg.get("dialogsum_limit", 0) or cfg.get("limit", 0))
        rows = load_rows(data_path, limit)
        if not rows:
            raise ValueError(f"No usable rows in {data_path}")

        # Reuse the training template so the prompt is byte-identical; take only
        # its prompt half and hand it to the raw generator, which does not
        # re-apply a chat template on top.
        prompts = [
            apply_train_template(tokenizer, row["instruction"], row["input"], row["output"])[1]
            for row in rows
        ]
        references = [row["output"] for row in rows]

        predictions = batched_generate_from_texts(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens,
            batch_size=resolve_batch_size(cfg, "dialogsum_batch_size"),
            log_label="dialogsum",
        )

        scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
        totals = {rouge_type: 0.0 for rouge_type in ROUGE_TYPES}
        empty = 0

        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for row, reference, prediction in zip(rows, references, predictions):
                if not prediction.strip():
                    empty += 1
                scores = scorer.score(reference, prediction)
                per_row = {
                    rouge_type: scores[rouge_type].fmeasure for rouge_type in ROUGE_TYPES
                }
                for rouge_type, value in per_row.items():
                    totals[rouge_type] += value
                file.write(
                    json.dumps(
                        {
                            # 'prompt' keeps the dialogue only: the full rendered
                            # prompt would bloat the file and the length
                            # diagnostic only reads 'response'.
                            "prompt": row["input"],
                            "response": prediction,
                            "reference": reference,
                            **{f"{k}_f": v for k, v in per_row.items()},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        count = len(rows)
        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "num_samples": count,
            "rouge1_f": totals["rouge1"] / count,
            "rouge2_f": totals["rouge2"] / count,
            "rougeL_f": totals["rougeL"] / count,
            "empty_responses": empty,
            "empty_rate": empty / count,
            "max_new_tokens": max_new_tokens,
            "data_path": str(data_path),
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "dialogsum_summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary
