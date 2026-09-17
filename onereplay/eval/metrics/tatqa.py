"""TAT-QA metric, scored with the official TAT-QA evaluator.

TAT-QA asks questions over a table plus its surrounding paragraphs, and its
answers come in four shapes -- span, multi-span, arithmetic, count -- each
carrying a separate ``scale`` field ("", thousand, million, billion, percent).
The official score is exact match and a DROP-style F1 computed *after* answer and
scale are folded into one string, so the scoring logic below is transplanted from
the TAT-QA repo (tatqa_metric.py / tatqa_utils.py) rather than reimplemented.

Why transplanted and not rewritten
----------------------------------
The fold is not the obvious one. ``get_answer_str`` computes
``'%.4f' % (round(value, 2) * scale_to_num(scale))``, rounding to two decimals
*before* applying the scale, so a gold of ``23.42`` with ``scale="percent"``
becomes the string ``"0.2342"`` -- while a model that answers ``0.2342``
directly would be rounded to ``"0.2300"`` and marked wrong. The official
evaluator patches exactly that case in ``add_percent_pred``, which appends an
un-rounded candidate. A reimplementation that looked correct would silently
disagree with every published TAT-QA number on precisely the question type the
benchmark is built around.

For the same reason ``to_number`` here is TAT-QA's, **not** the one in
math500.py. This one resolves written scales and percent signs as part of
parsing: "1.5 million" parses to 1500000.0 and "23.42%" to 0.2342. Mixing the
two would undo the alignment this module exists to preserve.

Scale is the hard part for a generative model
---------------------------------------------
The official baseline predicts answer and scale with two separate heads. A
language model has to write the unit into its answer, and a model that computes
1.5 correctly from a table captioned "in millions" but writes ``\\boxed{1.5}``
scores zero under the official fold. That is the benchmark's own contract, so it
is what ``accuracy`` reports, and the prompt asks for the unit explicitly. The
size of that effect is worth knowing rather than guessing, so a second,
scale-blind number is reported next to it -- see ``em_scale_agnostic``. It is a
diagnostic, not the headline.
"""

from __future__ import annotations

import csv
import json
import re
import string
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from onereplay.eval.generation import batched_generate, resolve_batch_size
from onereplay.eval.metrics.finance import render_table
from onereplay.eval.metrics.math500 import extract_answer

INSTRUCTION_PREFIX = (
    "Answer the following question using the table and text below. Reason step "
    "by step, then give the final answer as \\boxed{...} at the end. If the "
    "answer carries a unit or scale (thousand, million, billion, percent), "
    "include it inside the box."
)

# --- tatqa_utils.py, transplanted verbatim in behavior --------------------------
EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"


def scale_to_num(scale: str) -> float:
    scale = (scale or "").lower()
    if "hundred" in scale:
        return 100.0
    if "thousand" in scale:
        return 1000.0
    if "million" in scale:
        return 1000000.0
    if "billion" in scale:
        return 1000000000.0
    if "percent" in scale:
        return 0.01
    return 1.0


def _clean_num(text: str) -> str:
    return "".join(ch for ch in str(text) if ch not in EXCLUDE_IN_NUM)


def extract_one_num_from_str(text: str) -> float | int | None:
    text = _clean_num(text)
    groups = re.findall(r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)", text)
    if not groups:
        return None
    number = groups[0][0]
    if number == "":
        return None
    return float(number) if "." in number else int(number)


def is_number(text: str) -> bool:
    try:
        words = " ".join(_clean_num(word) for word in str(text).split()).split()
        if not words:
            return False
        value = float(words[0])
        if np.isnan(value):
            return False
        if len(words) >= 2 and scale_to_num(words[1]) == 1.0:
            return False
        return True
    except ValueError:
        return False


def negative_num_handle(text: str) -> float:
    """"(134)" is accounting notation for -134."""

    return -1.0 if re.findall(r"(\([\d.\s]+\))", str(text).strip()) else 1.0


def percent_num_handle(text: str) -> float:
    return 0.01 if re.findall(r"([\d.\s]+%)", str(text).strip()) else 1.0


def word_scale_handle(text: str) -> float:
    for match in re.finditer(r"([\d.]+\s?[a-zA-Z]+)", str(text)):
        return scale_to_num(match.group(0).lower())
    return 1.0


def to_number(text: str) -> float | None:
    """TAT-QA's numeric parse: resolves written scale, percent and parentheses."""

    number = extract_one_num_from_str(text)
    if number is None:
        return None
    return round(
        number * word_scale_handle(text) * negative_num_handle(text) * percent_num_handle(text),
        4,
    )


def _remove_articles(text: str) -> str:
    return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


_EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    return text if is_number(text) else "".join(ch for ch in text if ch not in _EXCLUDE)


def _normalize_number(text: str) -> str:
    return str(to_number(text)) if is_number(text) else text


def normalize_answer(text: str) -> str:
    """Lower, drop punctuation and articles, canonicalize numbers."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(token.lower()))))
        for token in re.split(" ", text)
    ]
    return " ".join(part for part in parts if part.strip()).strip()


# --- tatqa_metric.py, transplanted ---------------------------------------------
def _answer_to_bags(answer: Any) -> tuple[list[str], list[set[str]]]:
    raw_spans = answer if isinstance(answer, (list, tuple)) else [answer]
    normalized_spans = [normalize_answer(str(span)) for span in raw_spans]
    return normalized_spans, [set(span.split()) for span in normalized_spans]


def _compute_f1(predicted_bag: set[str], gold_bag: set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = 1.0 if not predicted_bag else intersection / float(len(predicted_bag))
    recall = 1.0 if not gold_bag else intersection / float(len(gold_bag))
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _align_bags(predicted: list[set[str]], gold: list[set[str]]) -> np.ndarray:
    """Best 1-1 assignment between predicted and gold spans (multi-span answers)."""

    from scipy.optimize import linear_sum_assignment

    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def get_metrics(predicted: Any, gold: Any) -> tuple[float, float]:
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    exact_match = float(
        set(predicted_bags[0]) == set(gold_bags[0])
        and len(predicted_bags[0]) == len(gold_bags[0])
    )
    return exact_match, round(float(np.mean(_align_bags(predicted_bags[1], gold_bags[1]))), 2)


def get_answer_str(answers: Sequence[Any], scale: str) -> list[str]:
    """Fold answers and their scale into the single string the scorer compares.

    The ``round(value, 2)`` before the scale multiply is the official behavior and
    the reason add_percent_pred exists; see the module docstring.
    """

    parts = []
    for answer in sorted(answers, key=str):
        text = str(answer)
        if is_number(text):
            number = to_number(text)
            if number is None:
                text = f"{text} {scale}" if scale else text
            elif "%" in text:
                text = "%.4f" % number
            else:
                text = "%.4f" % (round(number, 2) * scale_to_num(scale))
        elif scale:
            text = f"{text} {scale}"
        parts.append(text)
    return [" ".join(parts)]


def add_percent_pred(prediction_strings: list[str], pred_scale: str, pred: Sequence[Any]) -> list[str]:
    """Append an un-rounded candidate so 0.2342 can match a gold of 23.42 percent."""

    if len(pred) > 1:
        return prediction_strings
    text = str(pred[0])
    if not pred_scale and "%" not in text and is_number(text):
        number = to_number(text)
        if number is not None:
            prediction_strings.append("%.4f" % number)
    return prediction_strings


def extract_gold_answers(annotation: dict[str, Any]) -> tuple[str, list[str], str]:
    answer_type = annotation.get("answer_type", "")
    scale = annotation.get("scale", "") or ""
    content = annotation.get("answer")
    if answer_type in ("multi-span", "span") and isinstance(content, list):
        return answer_type, [str(item) for item in content], scale
    if answer_type == "count":
        try:
            return answer_type, [str(int(content))], scale
        except (TypeError, ValueError):
            return answer_type, [str(content)], scale
    return answer_type, [str(content)], scale


def metric_max_over_ground_truths(
    predictions: Sequence[str], ground_truths: Sequence[str]
) -> tuple[float, float]:
    scores = [
        get_metrics(prediction, truth)
        for prediction in predictions
        for truth in ground_truths
    ]
    return max(scores) if scores else (0.0, 0.0)


# --- the metric ----------------------------------------------------------------
_SCALE_WORDS = ("thousand", "million", "billion", "percent")


def split_prediction(text: str) -> tuple[str, str]:
    """Split a boxed answer into (answer text, scale).

    A written scale is pulled out into the scale slot so it reaches the scorer the
    way the official evaluator expects. A bare "%" is left in place: to_number
    already folds it, and moving it would double-apply the 0.01.
    """

    cleaned = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", text or "").strip()
    cleaned = cleaned.replace("\\%", "%").replace("\\$", "").replace("$", "")
    lowered = cleaned.lower()
    for scale in _SCALE_WORDS:
        if re.search(rf"\b{scale}s?\b", lowered):
            stripped = re.sub(rf"\b{scale}s?\b", "", cleaned, flags=re.IGNORECASE)
            return _white_space_fix(stripped).strip(" ,"), scale
    return cleaned, ""


def render_paragraphs(paragraphs: Any) -> str:
    if not isinstance(paragraphs, (list, tuple)):
        return ""
    ordered = sorted(
        (item for item in paragraphs if isinstance(item, dict)),
        key=lambda item: item.get("order", 0),
    )
    return "\n".join(str(item.get("text", "")).strip() for item in ordered if item.get("text"))


class TATQAMetric:
    """TAT-QA dev split (the test split is blind, so dev is what gets reported)."""

    name = "tatqa"
    data_path_keys = ("tatqa_data_path", "data_path")

    def data_path(self, cfg: dict[str, Any]) -> str:
        for key in self.data_path_keys:
            if cfg.get(key):
                return str(cfg[key])
        return ""

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        payload = json.loads(Path(self.data_path(cfg)).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data", [])

        examples: list[dict[str, Any]] = []
        for block in payload:
            if not isinstance(block, dict):
                continue
            table = block.get("table")
            table_rows = table.get("table") if isinstance(table, dict) else table
            document = render_table(table_rows)
            paragraphs = render_paragraphs(block.get("paragraphs"))
            context = "\n\n".join(
                part for part in (f"Table:\n{document}" if document else "", paragraphs) if part
            )
            for question in block.get("questions") or []:
                if not isinstance(question, dict):
                    continue
                stem = str(question.get("question", "")).strip()
                if not stem or question.get("answer") is None:
                    continue
                examples.append(
                    {
                        "id": question.get("uid", ""),
                        "question": stem,
                        "prompt": f"{INSTRUCTION_PREFIX}\n\n{context}\n\nQuestion: {stem}",
                        "annotation": question,
                    }
                )
        return examples

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        max_new_tokens = int(
            cfg.get("finance_max_new_tokens", cfg.get("max_new_tokens", 1024))
        )
        run_name = cfg.get("run_name", "base")

        examples = self.load_examples(cfg)
        limit = int(cfg.get("limit", 0))
        if limit > 0:
            examples = examples[:limit]

        responses = batched_generate(
            model,
            tokenizer,
            [example["prompt"] for example in examples],
            device,
            max_new_tokens,
            resolve_batch_size(cfg, "finance_batch_size"),
            log_label=self.name,
        )

        total_em = 0.0
        total_f1 = 0.0
        scale_em = 0.0
        agnostic_em = 0.0
        parsed = 0
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for example, response in zip(examples, responses):
                annotation = example["annotation"]
                gold_type, gold_answers, gold_scale = extract_gold_answers(annotation)
                boxed = extract_answer(response)
                prediction, pred_scale = split_prediction(boxed or "")
                parsed += int(bool(boxed))

                if not prediction or not gold_answers:
                    exact_match, f1_score = 0.0, 0.0
                else:
                    gold_strings = get_answer_str(gold_answers, gold_scale)
                    pred_strings = get_answer_str([prediction], pred_scale)
                    pred_strings = add_percent_pred(pred_strings, pred_scale, [prediction])
                    exact_match, f1_score = metric_max_over_ground_truths(
                        pred_strings, gold_strings
                    )
                    # The official evaluator collapses F1 onto EM for answer types
                    # where a partial token overlap is not partial credit.
                    if gold_type in ("arithmetic", "count"):
                        f1_score = exact_match

                scale_hit = float(pred_scale == gold_scale)
                # Diagnostic only: same comparison with both scales blanked, so a
                # right number reported without its unit still registers.
                blind_em, _ = (
                    metric_max_over_ground_truths(
                        get_answer_str([prediction], ""), get_answer_str(gold_answers, "")
                    )
                    if prediction and gold_answers
                    else (0.0, 0.0)
                )

                total_em += exact_match
                total_f1 += f1_score
                scale_em += scale_hit
                agnostic_em += blind_em
                file.write(
                    json.dumps(
                        {
                            "id": example["id"],
                            "question": example["question"],
                            "answer_type": gold_type,
                            "gold": gold_answers,
                            "gold_scale": gold_scale,
                            "prediction": prediction,
                            "pred_scale": pred_scale,
                            "em": exact_match,
                            "f1": f1_score,
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        total = max(len(examples), 1)
        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "data_path": self.data_path(cfg),
            "num_examples": len(examples),
            "exact_match": total_em / total,
            "f1": total_f1 / total,
            # How often the unit itself was right. A low value next to a high
            # em_scale_agnostic means the model can read the table but does not
            # report the scale, which is a formatting gap, not a reasoning one.
            "scale_accuracy": scale_em / total,
            "em_scale_agnostic": agnostic_em / total,
            "parse_rate": parsed / total,
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / f"{self.name}_summary.csv"
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary
