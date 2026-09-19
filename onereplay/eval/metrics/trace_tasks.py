"""The eight TRACE task metrics: did each task get learned, and then forgotten?

One class per task, all sharing ``TraceTaskMetric``, registered as
``trace_cstance`` ... ``trace_20minuten``. Each scores the official test split of
one task, so running "every task seen so far" after each training stage fills in
the upper triangle of TRACE's forgetting matrix; summarize_trace_matrix.py turns
those summary.json files into the table.

Direction differs by column. The task just trained on is supposed to be high --
that is the "did it learn" reading. Every earlier task is supposed to *hold*, and
a drop there is the forgetting this experiment exists to measure. Neither reading
works without the other: a run that refuses to learn anything forgets nothing.

Prompts are not rebuilt here. ``apply_train_template`` is the same function
train.py feeds the model, and prepare_trace.py already truncated each row's body
so the rendered sequence fits, so the prompt half is byte-identical to the one
seen during training and a base-vs-stage gap cannot come from formatting.

Each task reports two accuracies where accuracy applies. ``accuracy_strict`` is
TRACE's own full-string equality; ``accuracy`` extracts the first standalone
label or number first. The pair is what separates "forgot the task" from "stopped
emitting the bare label", and only the second is recoverable by prompting, so
reading the drop without both invites the wrong conclusion.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from onereplay.data.chat import apply_train_template
from onereplay.eval.generation import batched_generate_from_texts, resolve_batch_size
from onereplay.eval.trace_scoring import (
    bleu_score,
    corpus_mean,
    extract_choice,
    extract_number,
    fuzz_ratio,
    numbers_equal,
    py150_postprocess,
    rouge_l,
    sari_score,
    split_science_qa,
    strict_accuracy,
)
from onereplay.scripts.prepare_trace import SPEC_BY_KEY, TaskSpec

# Per-task decoding budgets. TRACE decodes 512 tokens for every task, which on
# C-STANCE is 511 tokens spent to read one character: the per-stage evaluation
# runs up to 8 task sets after every one of 8 stages, so a shared budget is the
# difference between minutes and hours. Sized from the gold answers measured in
# prepare_trace.py (p99 plus headroom), not guessed.
DEFAULT_MAX_NEW_TOKENS = {
    "cstance": 16,
    "fomc": 16,
    "meetingbank": 320,
    "py150": 96,
    "scienceqa": 640,
    "numglue_cm": 48,
    "numglue_ds": 48,
    "20minuten": 320,
}

CHOICE_LABELS = ("A", "B", "C")


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read one prepare_trace.py test JSONL, preserving file order.

    Order matters: TRACE's scorers zip predictions against golds positionally
    (see the warning at the top of its metrics.py), and so does this module.
    """

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("instruction") or not str(row.get("output", "")).strip():
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


class TraceTaskMetric:
    """Generate on one TRACE test split and score it the way that task is scored."""

    key = ""

    @property
    def name(self) -> str:
        return f"trace_{self.key}"

    @property
    def spec(self) -> TaskSpec:
        return SPEC_BY_KEY[self.key]

    def score(
        self,
        rows: list[dict[str, Any]],
        predictions: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return (summary fields, per-row fields). Overridden per task."""

        raise NotImplementedError

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = cfg.get("run_name", "base")

        data_dir = Path(cfg.get("trace_data_dir") or "data/processed")
        data_path = Path(
            cfg.get(f"{self.name}_input") or data_dir / f"trace_{self.key}_test.jsonl"
        )
        if not data_path.is_file():
            raise FileNotFoundError(
                f"TRACE test file not found: {data_path}. Build it with\n"
                f"    python -m onereplay.scripts.prepare_trace --trace_dir <LLM-CL-Benchmark_5000> "
                f"--tokenizer_path <model>\n"
                f"or point --trace_data_dir at the directory holding "
                f"trace_{self.key}_test.jsonl."
            )

        max_new_tokens = int(
            cfg.get(f"{self.name}_max_new_tokens")
            or cfg.get("trace_max_new_tokens")
            or DEFAULT_MAX_NEW_TOKENS[self.key]
        )
        limit = int(cfg.get("trace_limit", 0) or cfg.get("limit", 0))
        rows = load_rows(data_path, limit)
        if not rows:
            raise ValueError(f"No usable rows in {data_path}")

        prompts = [
            apply_train_template(tokenizer, row["instruction"], "", row["output"])[1]
            for row in rows
        ]
        predictions = batched_generate_from_texts(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens,
            batch_size=resolve_batch_size(cfg, "trace_batch_size"),
            log_label=self.name,
        )

        scores, per_row = self.score(rows, predictions)
        empty = sum(1 for prediction in predictions if not prediction.strip())

        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as handle:
            for row, prediction, extra in zip(rows, predictions, per_row):
                handle.write(
                    json.dumps(
                        {
                            "id": row.get("id", ""),
                            "response": prediction,
                            "gold": row["output"],
                            **extra,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "task": self.spec.directory,
            "task_key": self.key,
            "num_samples": len(rows),
            **scores,
            "empty_responses": empty,
            "empty_rate": empty / len(rows),
            "max_new_tokens": max_new_tokens,
            "data_path": str(data_path),
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / f"{self.name}_summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary


class _ChoiceMetric(TraceTaskMetric):
    """Three-way A/B/C classification, scored by accuracy. C-STANCE and FOMC."""

    def score(self, rows, predictions):
        golds = [row["output"].strip() for row in rows]
        stripped = [prediction.strip() for prediction in predictions]

        lenient_hits = 0
        per_row: list[dict[str, Any]] = []
        for prediction, gold in zip(stripped, golds):
            extracted = extract_choice(prediction, CHOICE_LABELS)
            hit = bool(extracted) and extracted == gold.upper()
            lenient_hits += int(hit)
            per_row.append({"extracted": extracted, "correct": hit})

        return {
            "accuracy": lenient_hits / len(rows),
            "accuracy_strict": strict_accuracy(stripped, golds),
            "parse_rate": sum(1 for item in per_row if item["extracted"]) / len(rows),
            "primary_metric": "accuracy",
        }, per_row


class _NumberMetric(TraceTaskMetric):
    """Short numeric answers, scored by accuracy. NumGLUE-cm and NumGLUE-ds."""

    def score(self, rows, predictions):
        golds = [row["output"].strip() for row in rows]
        stripped = [prediction.strip() for prediction in predictions]

        lenient_hits = 0
        per_row: list[dict[str, Any]] = []
        for prediction, gold in zip(stripped, golds):
            extracted = extract_number(prediction)
            hit = numbers_equal(extracted, extract_number(gold) or gold)
            lenient_hits += int(hit)
            per_row.append({"extracted": extracted, "correct": hit})

        return {
            "accuracy": lenient_hits / len(rows),
            "accuracy_strict": strict_accuracy(stripped, golds),
            "parse_rate": sum(1 for item in per_row if item["extracted"]) / len(rows),
            "primary_metric": "accuracy",
        }, per_row


class _GenerationMetric(TraceTaskMetric):
    """BLEU-1/4 and ROUGE-L over free text. MeetingBank's full metric set."""

    def score(self, rows, predictions):
        golds = [row["output"].strip() for row in rows]
        stripped = [prediction.strip() for prediction in predictions]

        bleu1: list[float] = []
        bleu4: list[float] = []
        rouge: list[float] = []
        per_row: list[dict[str, Any]] = []
        for prediction, gold in zip(stripped, golds):
            if not prediction or not gold:
                per_row.append({"bleu_1": 0.0, "bleu_4": 0.0, "rouge_l": 0.0})
                continue
            row_bleu1 = bleu_score(gold, prediction, 1)
            row_bleu4 = bleu_score(gold, prediction, 4)
            row_rouge = rouge_l(gold, prediction)
            bleu1.append(row_bleu1)
            bleu4.append(row_bleu4)
            rouge.append(row_rouge)
            per_row.append({"bleu_1": row_bleu1, "bleu_4": row_bleu4, "rouge_l": row_rouge})

        count = len(rows)
        return {
            "bleu_1": corpus_mean(bleu1, count),
            "bleu_4": corpus_mean(bleu4, count),
            "rouge_l": corpus_mean(rouge, count),
            "primary_metric": "rouge_l",
        }, per_row


class TraceCStanceMetric(_ChoiceMetric):
    key = "cstance"


class TraceFOMCMetric(_ChoiceMetric):
    key = "fomc"


class TraceNumGLUECmMetric(_NumberMetric):
    key = "numglue_cm"


class TraceNumGLUEDsMetric(_NumberMetric):
    key = "numglue_ds"


class TraceMeetingBankMetric(_GenerationMetric):
    key = "meetingbank"


class TracePy150Metric(TraceTaskMetric):
    """Code continuation, scored by fuzzy string similarity after restoring literals.

    The literal restoration is not cosmetic: the gold answers carry
    ``<STR_LIT:foo>`` placeholders, and a model that has learned the task emits
    them too, so scoring before restoration would compare two encodings of the
    same code and reward matching the placeholder syntax rather than the code.
    """

    key = "py150"

    def score(self, rows, predictions):
        similarities: list[float] = []
        exact = 0
        per_row: list[dict[str, Any]] = []
        for row, prediction in zip(rows, predictions):
            gold = py150_postprocess(row["output"].strip())
            # Only the first line is the continuation; anything past it is the
            # model carrying on writing the file, which the gold never contains.
            candidate = py150_postprocess(prediction.strip().split("\n")[0].strip())
            similarity = fuzz_ratio(candidate, gold)
            similarities.append(similarity)
            exact += int(candidate == gold and bool(gold))
            per_row.append({"similarity": similarity, "prediction_clean": candidate})

        count = len(rows)
        return {
            "similarity": corpus_mean(similarities, count),
            "exact_match": exact / count,
            "primary_metric": "similarity",
        }, per_row


class TraceScienceQAMetric(TraceTaskMetric):
    """Choice plus free-form reasoning, scored separately.

    The two halves can move independently -- a model can keep picking the right
    letter while its explanations decay into the format of whatever task it was
    trained on most recently -- so collapsing them into one number would hide
    exactly the kind of partial forgetting this is looking for.
    """

    key = "scienceqa"

    def score(self, rows, predictions):
        answer_hits = 0
        lenient_hits = 0
        bleu1: list[float] = []
        bleu4: list[float] = []
        rouge: list[float] = []
        per_row: list[dict[str, Any]] = []

        for row, prediction in zip(rows, predictions):
            gold_answer, gold_reasoning = split_science_qa(row["output"])
            predicted_answer, predicted_reasoning = split_science_qa(prediction)

            answer_hits += int(bool(gold_answer) and predicted_answer == gold_answer)
            extracted = extract_choice(prediction, CHOICE_LABELS)
            lenient_hits += int(bool(extracted) and extracted == gold_answer.upper())

            row_bleu1 = row_bleu4 = row_rouge = 0.0
            if predicted_reasoning.strip() and gold_reasoning.strip():
                row_bleu1 = bleu_score(gold_reasoning, predicted_reasoning, 1)
                row_bleu4 = bleu_score(gold_reasoning, predicted_reasoning, 4)
                row_rouge = rouge_l(gold_reasoning, predicted_reasoning)
                bleu1.append(row_bleu1)
                bleu4.append(row_bleu4)
                rouge.append(row_rouge)
            per_row.append(
                {
                    "extracted": extracted,
                    "answer_correct": bool(gold_answer) and predicted_answer == gold_answer,
                    "bleu_1": row_bleu1,
                    "bleu_4": row_bleu4,
                    "rouge_l": row_rouge,
                }
            )

        count = len(rows)
        return {
            "accuracy": lenient_hits / count,
            "accuracy_strict": answer_hits / count,
            "bleu_1": corpus_mean(bleu1, count),
            "bleu_4": corpus_mean(bleu4, count),
            "rouge_l": corpus_mean(rouge, count),
            "primary_metric": "accuracy",
        }, per_row


class Trace20MinutenMetric(TraceTaskMetric):
    """German text simplification: BLEU, ROUGE-L, and SARI against the source.

    SARI is the metric that actually measures simplification -- it rewards
    keeping, deleting and adding the right words relative to the input -- so it
    is the primary here even though BLEU and ROUGE are reported alongside. It
    needs the source paragraph, which prepare_trace.py carries in
    ``source_text`` rather than leaving this module to re-peel the prompt's
    affixes and risk getting the boundaries subtly different.
    """

    key = "20minuten"

    def score(self, rows, predictions):
        golds = [row["output"].strip() for row in rows]
        stripped = [prediction.strip() for prediction in predictions]
        sources = [row.get("source_text") or row["instruction"] for row in rows]

        bleu1: list[float] = []
        bleu4: list[float] = []
        rouge: list[float] = []
        per_row: list[dict[str, Any]] = []
        for prediction, gold in zip(stripped, golds):
            if not prediction or not gold:
                per_row.append({"bleu_1": 0.0, "bleu_4": 0.0, "rouge_l": 0.0})
                continue
            row_bleu1 = bleu_score(gold, prediction, 1)
            row_bleu4 = bleu_score(gold, prediction, 4)
            row_rouge = rouge_l(gold, prediction)
            bleu1.append(row_bleu1)
            bleu4.append(row_bleu4)
            rouge.append(row_rouge)
            per_row.append({"bleu_1": row_bleu1, "bleu_4": row_bleu4, "rouge_l": row_rouge})

        count = len(rows)
        return {
            "sari": sari_score(sources, stripped, golds),
            "bleu_1": corpus_mean(bleu1, count),
            "bleu_4": corpus_mean(bleu4, count),
            "rouge_l": corpus_mean(rouge, count),
            "primary_metric": "sari",
        }, per_row


__all__ = [
    "Trace20MinutenMetric",
    "TraceCStanceMetric",
    "TraceFOMCMetric",
    "TraceMeetingBankMetric",
    "TraceNumGLUECmMetric",
    "TraceNumGLUEDsMetric",
    "TracePy150Metric",
    "TraceScienceQAMetric",
    "TraceTaskMetric",
]
