"""DialogSum ROUGE + BERTScore: did the new task actually get learned?

Every other metric here measures retention -- how much of the base model's old
ability survived fine-tuning. This one measures the other end of the trade-off.
Both are needed to read a lambda sweep, because a regularizer can always win on
retention by refusing to learn anything at all; without a new-task score there
is no way to tell "protected the old knowledge" from "barely trained".

It is also the check val_loss cannot do. Loss falling monotonically is
consistent with the model drifting into paraphrasing the dialogue instead of
summarising it; ROUGE against real references is not, and unlike loss it is an
absolute number that can be put next to published ones.

Direction is the opposite of every other metric in this package: the base model
is the floor, and fine-tuning is supposed to push these scores *up*. A run whose
ROUGE sits at base level did not learn the task, whatever its retention says.

The prompt is not rebuilt here. apply_train_template is the same function
train.py feeds the model, so the prompt half is byte-identical to training and
a base-vs-SFT gap cannot come from a template mismatch.

Multi-reference: prepare_dialogsum.py emits one row per dialogue with every
reference under `outputs`, and both scorers take the max over them, which is the
DialogSum paper's convention. Rows carrying a single reference still work; they
just score lower, so a validation-split number and a test-split number are not
interchangeable.

BERTScore is off by default because it needs a second model on an offline node.
Enable it by pointing --dialogsum_bertscore_model at a local roberta-large and
leave --dialogsum_bertscore_layers at 17: bert-score normally looks the layer
count up by model *name*, and a filesystem path is not in that table. Baseline
rescaling stays off for the same reason (the baseline file is keyed by name), so
the absolute values sit in the high 0.8s and only differences between runs mean
anything.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from onereplay.data.chat import apply_train_template
from onereplay.eval.generation import batched_generate_from_texts, resolve_batch_size

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")

# bert-score's own default layer for roberta-large. Passed explicitly because
# model2layers is keyed by hub name and we load from a path.
DEFAULT_BERTSCORE_LAYERS = 17

_MISSING_ROUGE = (
    "rouge_score is not installed, so the DialogSum metric cannot score.\n"
    "Install it on the LOGIN node (compute nodes are offline):\n"
    "    pip install rouge-score\n"
    "Or drop 'dialogsum' from --metrics."
)

_MISSING_BERTSCORE = (
    "bert_score is not installed, so DialogSum BERTScore cannot run.\n"
    "Install it on the LOGIN node (compute nodes are offline):\n"
    "    pip install bert-score\n"
    "Or pass --dialogsum_bertscore 0 to score ROUGE only."
)

_MISSING_BERTSCORE_MODEL = (
    "DialogSum BERTScore is enabled but --dialogsum_bertscore_model is empty.\n"
    "Compute nodes run with HF_HUB_OFFLINE=1, so the scoring model has to be on\n"
    "disk already. Fetch it once on a LOGIN node:\n"
    "    python -c \"from transformers import AutoModel, AutoTokenizer; \\\n"
    "      AutoModel.from_pretrained('roberta-large').save_pretrained('<dir>'); \\\n"
    "      AutoTokenizer.from_pretrained('roberta-large').save_pretrained('<dir>')\"\n"
    "then pass --dialogsum_bertscore_model <dir>, or --dialogsum_bertscore 0."
)


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read the held-out JSONL written by prepare_dialogsum.py.

    `outputs` is the multi-reference field; `output` is the single-reference
    fallback, which is what a JSONL written before the grouping existed has.
    """

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            references = row.get("outputs") or [row.get("output", "")]
            references = [text for text in (str(ref).strip() for ref in references) if text]
            if not row.get("input") or not references:
                continue
            row["outputs"] = references
            # Only the prompt half of apply_train_template is used, but it still
            # wants an output argument, and a pre-grouping JSONL may lack one.
            row["output"] = references[0]
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def score_bertscore(
    predictions: list[str],
    reference_groups: list[list[str]],
    device,
    cfg: dict[str, Any],
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    """Per-row BERTScore P/R/F1, max over each row's references, plus provenance.

    The scorer is dropped before returning: it holds a second model on the same
    card, and the runner reports peak memory per metric, so leaving it resident
    would show up in whichever metric happens to run next.
    """

    try:
        from bert_score import BERTScorer
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(_MISSING_BERTSCORE) from exc

    model_type = str(cfg.get("dialogsum_bertscore_model") or "").strip()
    if not model_type:
        raise RuntimeError(_MISSING_BERTSCORE_MODEL)
    if not Path(model_type).is_dir():
        raise FileNotFoundError(
            f"--dialogsum_bertscore_model is not a directory: {model_type}\n"
            + _MISSING_BERTSCORE_MODEL
        )

    num_layers = int(cfg.get("dialogsum_bertscore_layers") or DEFAULT_BERTSCORE_LAYERS)
    batch_size = int(cfg.get("dialogsum_bertscore_batch_size") or 64)
    scorer = BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        device=str(device),
        rescale_with_baseline=False,
    )
    # refs as a list of lists: bert-score expands each group, scores every pair
    # and keeps the max, matching how ROUGE is aggregated above.
    precision, recall, f1 = scorer.score(predictions, reference_groups, batch_size=batch_size)
    meta = {
        "bertscore_model": model_type,
        "bertscore_layers": num_layers,
        "bertscore_rescaled": False,
    }
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return precision.tolist(), recall.tolist(), f1.tolist(), meta


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
        reference_groups = [row["outputs"] for row in rows]

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
        per_row_rouge: list[dict[str, float]] = []
        empty = 0
        for references, prediction in zip(reference_groups, predictions):
            if not prediction.strip():
                empty += 1
            per_reference = [scorer.score(reference, prediction) for reference in references]
            best = {
                rouge_type: max(scores[rouge_type].fmeasure for scores in per_reference)
                for rouge_type in ROUGE_TYPES
            }
            for rouge_type, value in best.items():
                totals[rouge_type] += value
            per_row_rouge.append(best)

        # Keys are present whether or not BERTScore ran: the summary dict is the
        # header of an appended CSV, so a run that skipped it must not write a
        # narrower row than one that did.
        use_bertscore = int(cfg.get("dialogsum_bertscore", 0) or 0) == 1
        bert_meta: dict[str, Any] = {
            "bertscore_model": "",
            "bertscore_layers": 0,
            "bertscore_rescaled": False,
        }
        if use_bertscore:
            bert_p, bert_r, bert_f1, bert_meta = score_bertscore(
                predictions, reference_groups, device, cfg
            )
        else:
            bert_p = bert_r = bert_f1 = [float("nan")] * len(rows)

        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for row, references, prediction, best, f1 in zip(
                rows, reference_groups, predictions, per_row_rouge, bert_f1
            ):
                file.write(
                    json.dumps(
                        {
                            "id": row.get("id", ""),
                            # 'prompt' keeps the dialogue only: the full rendered
                            # prompt would bloat the file and the length
                            # diagnostic only reads 'response'.
                            "prompt": row["input"],
                            "response": prediction,
                            "references": references,
                            **{f"{k}_f": v for k, v in best.items()},
                            "bertscore_f1": None if f1 != f1 else f1,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        count = len(rows)

        def mean(values: list[float]) -> float | None:
            usable = [value for value in values if value == value]
            return sum(usable) / len(usable) if usable else None

        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "num_samples": count,
            "num_references": sum(len(group) for group in reference_groups),
            "rouge1_f": totals["rouge1"] / count,
            "rouge2_f": totals["rouge2"] / count,
            "rougeL_f": totals["rougeL"] / count,
            "bertscore_p": mean(bert_p),
            "bertscore_r": mean(bert_r),
            "bertscore_f1": mean(bert_f1),
            **bert_meta,
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
