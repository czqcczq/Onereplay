"""Finance evaluation: FPB, FiQA-SA, TFNS -- three-way sentiment classification.

The Finance specialist trains on FinGPT sentiment-train, whose answer is a single
label word, so these three sets are graded the way that corpus is written: the
prompt is the instruction the corpus itself carries, and the answer is one of
negative / neutral / positive.

Two things the graders have to tolerate, both consequences of the training data:

  * 16,184 training rows answer on a nine-level scale ("mildly positive"), so a
    trained model will sometimes answer that way here. Those collapse onto the
    three classes rather than counting as unparsed.
  * TFNS spells its classes Bearish / Bullish / Neutral. Those are the same three
    classes under different names and are mapped, on both the gold and the
    prediction side.

macro_f1 is reported next to accuracy because all three sets are unbalanced --
FPB is roughly 60% neutral -- so a model that answers neutral to everything
posts a respectable accuracy and a poor macro F1.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from onereplay.eval.generation import batched_generate, resolve_batch_size
from onereplay.eval.metrics.math500 import extract_answer

LABELS = ("negative", "neutral", "positive")

# Byte-identical to the two instructions FinGPT sentiment-train carries, so the
# model is asked at eval time exactly what it was taught. The nine-level variant
# in that corpus is deliberately not reused: the gold here is three-way, and
# asking for nine levels would make every answer need a mapping instead of only
# the ones that volunteer it.
NEWS_INSTRUCTION = (
    "What is the sentiment of this news? Please choose an answer from "
    "{negative/neutral/positive}."
)
TWEET_INSTRUCTION = (
    "What is the sentiment of this tweet? Please choose an answer from "
    "{negative/neutral/positive}."
)

_SENTIMENT_WORD = re.compile(r"\b(positive|negative|neutral|bullish|bearish)\b", re.IGNORECASE)


def normalize_label(value: Any) -> str:
    """Map any spelling of the three classes onto one of LABELS, or "".

    Substring matching is what collapses the nine-level scale ("moderately
    negative" -> negative) and TFNS's Bearish/Bullish onto the three classes.
    """

    lowered = str(value or "").strip().lower()
    if not lowered:
        return ""
    if "positive" in lowered or "bullish" in lowered:
        return "positive"
    if "negative" in lowered or "bearish" in lowered:
        return "negative"
    if "neutral" in lowered:
        return "neutral"
    return ""


def extract_sentiment(response: str) -> str:
    """Pull one sentiment label out of a response, or "" if there is none.

    The first line is searched before the whole response: a trained model answers
    with the bare label, while a base model writes a paragraph that may mention
    other classes on the way ("this is not negative, it is positive"). Within a
    scope the last mention wins, which is what resolves that construction.
    """

    boxed = extract_answer(response)
    text = (boxed or response).strip()
    if not text:
        return ""
    for scope in (text.splitlines()[0], text):
        hits = _SENTIMENT_WORD.findall(scope)
        if hits:
            return normalize_label(hits[-1])
    return ""


def load_records(path: str) -> list[dict[str, Any]]:
    """Read a finance eval file, json array / jsonl / {split: [...]} alike.

    jsonl matters because the HuggingFace mirrors ship parquet, and the download
    step converts parquet to jsonl rather than to a json array.
    """

    data_path = Path(path)
    if data_path.suffix.lower() in (".jsonl", ".ndjson"):
        records = []
        with data_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
        return records

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("data", "test", "dev", "validation", "train"):
            if isinstance(payload.get(key), list):
                return [record for record in payload[key] if isinstance(record, dict)]
    return []


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Unweighted mean of the per-class F1 scores.

    Averaged over the classes the *gold* actually uses, not over all three:
    FiQA-SA's polarity scores are almost never exactly zero, so its gold set is
    effectively two-class, and including an absent class would cap the score at
    2/3 for reasons that have nothing to do with the model.

    Unparsed predictions are "" and therefore count as a false negative for
    their gold class and nothing else, which is the same convention accuracy
    uses: an answer the grader cannot find was not given.
    """

    present = [label for label in LABELS if any(gold == label for gold, _ in pairs)]
    if not present:
        return 0.0

    scores = []
    for label in present:
        true_positive = sum(1 for gold, pred in pairs if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in pairs if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores)


class _SentimentMetric:
    """Shared body for the three sentiment sets: render, decode, extract, score.

    Subclasses only declare where the sentence and the gold label live, because
    every mirror spells those differently while the task is identical.
    """

    name = ""
    data_path_keys: tuple[str, ...] = ()
    instruction = NEWS_INSTRUCTION
    # query last: on the TheFinAI mirrors it holds a fully rendered prompt, which
    # would put a second instruction inside ours.
    text_keys: tuple[str, ...] = ("text", "sentence", "tweet", "input", "query")
    label_keys: tuple[str, ...] = ("answer", "label", "sentiment", "gold", "output")
    # Only consulted when the label arrives as a bare integer and the row carries
    # no `choices` column to resolve it against.
    int_labels: dict[int, str] = {}

    def data_path(self, cfg: dict[str, Any]) -> str:
        for key in self.data_path_keys:
            if cfg.get(key):
                return str(cfg[key])
        return ""

    def sentence(self, record: dict[str, Any]) -> str:
        for key in self.text_keys:
            value = str(record.get(key) or "").strip()
            if value:
                return value
        return ""

    def gold(self, record: dict[str, Any]) -> str:
        # choices[gold] first: the FinBen-style mirrors ship the label names in
        # the row itself, which beats any mapping table we could hardcode.
        choices = record.get("choices")
        if isinstance(choices, (list, tuple)) and choices:
            for key in ("gold", "label", "answer"):
                value = record.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(choices):
                    label = normalize_label(choices[value])
                    if label:
                        return label

        for key in self.label_keys:
            if key not in record:
                continue
            value = record[key]
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                label = normalize_label(self.int_labels.get(value, ""))
            elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
                label = normalize_label(self.int_labels.get(int(value.strip()), ""))
            else:
                label = normalize_label(value)
            if label:
                return label

        # FiQA-SA publishes a polarity score in [-1, 1] instead of a class. Sign
        # is the dataset's own definition of the class boundary; any other
        # threshold would be one we invented.
        score = record.get("score")
        if isinstance(score, str):
            try:
                score = float(score.strip())
            except ValueError:
                score = None
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            if score > 0:
                return "positive"
            if score < 0:
                return "negative"
            return "neutral"
        return ""

    def build_prompt(self, sentence: str) -> str:
        # Same shape as the training rows: instruction, newline, sentence.
        return f"{self.instruction}\n{sentence}"

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for index, record in enumerate(load_records(self.data_path(cfg))):
            sentence = self.sentence(record)
            gold = self.gold(record)
            if not sentence or not gold:
                continue
            examples.append(
                {
                    "id": record.get("id", record.get("_id", index)),
                    "sentence": sentence,
                    "prompt": self.build_prompt(sentence),
                    "gold": gold,
                }
            )
        return examples

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = self.data_path(cfg)
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

        correct = 0
        parsed = 0
        pairs: list[tuple[str, str]] = []
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for example, response in zip(examples, responses):
                prediction = extract_sentiment(response)
                is_correct = bool(prediction) and prediction == example["gold"]
                correct += int(is_correct)
                parsed += int(bool(prediction))
                pairs.append((example["gold"], prediction))
                file.write(
                    json.dumps(
                        {
                            "id": example.get("id", ""),
                            "sentence": example["sentence"],
                            "gold": example["gold"],
                            "prediction": prediction,
                            "correct": is_correct,
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
            "data_path": data_path,
            "num_examples": len(examples),
            "correct": correct,
            "accuracy": correct / total,
            "macro_f1": macro_f1(pairs),
            # Rows where no label could be found. Counted as wrong in accuracy,
            # so a base model that refuses the format has to be read through this
            # number rather than through accuracy alone.
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


class FPBMetric(_SentimentMetric):
    """Financial PhraseBank test split (970 sentences, ~60% neutral)."""

    name = "fpb"
    data_path_keys = ("fpb_data_path", "data_path")
    instruction = NEWS_INSTRUCTION
    int_labels = {0: "negative", 1: "neutral", 2: "positive"}


class FiQASAMetric(_SentimentMetric):
    """FiQA-2018 Task 1 sentiment test split (235 headlines and posts).

    The release grades a continuous polarity score; the three-way class comes
    from its sign. See ``_SentimentMetric.gold``.
    """

    name = "fiqasa"
    data_path_keys = ("fiqasa_data_path", "data_path")
    instruction = NEWS_INSTRUCTION
    int_labels = {0: "negative", 1: "neutral", 2: "positive"}


class TFNSMetric(_SentimentMetric):
    """Twitter Financial News Sentiment validation split (2388 tweets).

    The validation split, not test: the released test split has no labels. Its
    classes are Bearish / Bullish / Neutral, mapped in ``normalize_label``.
    """

    name = "tfns"
    data_path_keys = ("tfns_data_path", "data_path")
    instruction = TWEET_INSTRUCTION
    int_labels = {0: "negative", 1: "positive", 2: "neutral"}
