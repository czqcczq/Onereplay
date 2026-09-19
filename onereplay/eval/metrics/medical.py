"""Medical evaluation: MedQA, PubMedQA, MedMCQA, CareQA.

Four sets, one harness: decode once, extract one token, score. Three are option
letters and PubMedQA is yes/no/maybe.

The prompts do not ask for \\boxed{} any more. The Medical specialist now trains
on Medical Meadow flashcards, which answer in plain prose with no CoT and no
boxed span, so a boxed contract would be a format the trained model was never
taught -- and the base model would fail it for reasons that have nothing to do
with medicine. The prompts instead ask for the bare answer, and extraction still
accepts a boxed one so an old checkpoint remains gradable.

parse_rate is reported next to accuracy because unparsed rows count as wrong: an
accuracy that looks bad has to be checked against it before it is believed.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from onereplay.eval.generation import batched_generate, resolve_batch_size
from onereplay.eval.metrics.math500 import extract_answer, load_json_records

# Anchored on an explicit answer statement: a looser "any capital letter near the
# end" would fire on "option D is wrong because ..." and award credit the model
# never claimed. The last match wins, since models restate after reconsidering.
_ANSWER_STATEMENT = re.compile(
    r"(?:final\s+answer|answer|choice|option)\s*(?:is|:)\s*\**\s*\(?([A-J])\)?\b",
    re.IGNORECASE,
)
# A response that opens with the bare answer -- "C", "C.", "(C) Nitrofurantoin",
# "**C**". The trailing delimiter is required so "A is wrong, ..." does not match A.
_LEADING_LETTER = re.compile(r"^\s*\**\s*\(?([A-J])\)?\**\s*(?:[.:,)\]]|$|\n)")
_TRAILING_LETTER = re.compile(r"\b([A-J])\b[\s.)\]*]*$")
_DECISION_STATEMENT = re.compile(
    r"(?:final\s+answer|answer|decision)\s*(?:is|:)\s*\**\s*(yes|no|maybe)\b",
    re.IGNORECASE,
)
_LEADING_DECISION = re.compile(r"^\s*\**\s*(yes|no|maybe)\b", re.IGNORECASE)
_TRAILING_DECISION = re.compile(r"\b(yes|no|maybe)\b[\s.!)\]*]*$", re.IGNORECASE)


def _unwrap_boxed(boxed: str) -> str:
    """Strip the LaTeX wrappers a boxed answer arrives in."""

    return re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", boxed)


def extract_choice(response: str, valid: str) -> str:
    """Pull a single option letter out of a response, or "" if there is none.

    ``valid`` is the live option set ("ABCD" for MedMCQA, "ABCDE" for MedQA); a
    letter outside it is treated as no answer rather than as a wrong one, so a
    model writing prose containing a stray "F" on a 4-option question is not
    recorded as having answered F.
    """

    boxed = extract_answer(response)
    if boxed:
        letter = _unwrap_boxed(boxed).strip().strip("()$ ").upper()[:1]
        if letter in valid:
            return letter

    matches = _ANSWER_STATEMENT.findall(response)
    for candidate in reversed(matches):
        if candidate.upper() in valid:
            return candidate.upper()

    for pattern in (_LEADING_LETTER, _TRAILING_LETTER):
        hit = pattern.search(response.strip())
        if hit and hit.group(1).upper() in valid:
            return hit.group(1).upper()
    return ""


def extract_decision(response: str) -> str:
    """Pull yes/no/maybe out of a PubMedQA response, or "" if there is none."""

    boxed = extract_answer(response)
    if boxed:
        token = _unwrap_boxed(boxed).strip().strip("().$ ").lower()
        if token in ("yes", "no", "maybe"):
            return token

    matches = _DECISION_STATEMENT.findall(response)
    if matches:
        return matches[-1].lower()
    for pattern in (_LEADING_DECISION, _TRAILING_DECISION):
        hit = pattern.search(response.strip())
        if hit:
            return hit.group(1).lower()
    return ""


def field(record: dict[str, Any], *names: str, default: Any = "") -> Any:
    """Read the first present field, trying an exact match then case-insensitively.

    PubMedQA is the reason this exists: the official ori_pqal.json spells its
    fields ``QUESTION`` and ``CONTEXTS`` while the HuggingFace mirror spells them
    ``question`` and ``context``. Reading only one convention does not raise --
    it yields empty questions, every row is skipped, and the metric reports a
    clean 0.0 that looks like a model result.
    """

    for name in names:
        if name in record:
            return record[name]
    lowered = {str(key).lower(): value for key, value in record.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return default


def normalize_options(raw: Any) -> list[str]:
    """Render an option set as ``["A: text", ...]`` regardless of its shape.

    MedQA ships under several field conventions depending on the mirror -- a
    list of "A: text" strings, a {letter: text} dict, or a list of {key, value}
    dicts -- and which one lands on the cluster is not worth a separate flag.
    """

    if isinstance(raw, dict):
        return [f"{key}: {value}" for key, value in sorted(raw.items())]
    if not isinstance(raw, (list, tuple)):
        return []

    options: list[str] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            key = item.get("key") or item.get("label") or chr(ord("A") + index)
            value = item.get("value") or item.get("text") or ""
            options.append(f"{key}: {value}")
        else:
            text = str(item).strip()
            options.append(text if re.match(r"^[A-J]\s*[:.)]", text) else
                           f"{chr(ord('A') + index)}: {text}")
    return options


# MedMCQA spells its options opa/opb/opc/opd, CareQA op1..op4. Both are read by
# pattern rather than by a per-dataset field list, because the two mirrors differ
# only in that suffix and a missing spelling yields an empty option set, which
# skips every row and reports a clean 0.0.
_OPTION_KEY = re.compile(r"^op(?:tion)?[_\s]?([a-e]|[1-5])$", re.IGNORECASE)
_GOLD_INDEX_KEYS = ("cop", "correct_option", "answer_index", "answer_idx")


def spread_options(record: dict[str, Any]) -> list[str]:
    """Collect ``op*`` columns into ``["A: text", ...]`` in their natural order."""

    found: list[tuple[str, str]] = []
    for key, value in record.items():
        match = _OPTION_KEY.match(str(key))
        text = str(value or "").strip()
        if match and text:
            found.append((match.group(1).lower(), text))
    found.sort(key=lambda item: item[0])
    return [f"{chr(ord('A') + index)}: {text}" for index, (_, text) in enumerate(found)]


def gold_index(record: dict[str, Any]) -> int | None:
    """The gold option as an integer, under whichever column carries it."""

    for key in _GOLD_INDEX_KEYS:
        value = record.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def index_base(records: list[dict[str, Any]]) -> int:
    """Whether the gold index column counts from 0 or from 1.

    Decided once from the whole column, not per row: MedMCQA's ``cop`` is
    0-based and CareQA's is 1-based, and guessing per row would silently shift
    every answer by one on one of them. A 0 anywhere in the column settles it --
    a 1-based column cannot contain one.
    """

    values = [value for value in (gold_index(record) for record in records) if value is not None]
    return 0 if values and min(values) == 0 else 1


def resolve_gold_letter(record: dict[str, Any], options: list[str]) -> str:
    """Find the gold option letter under whichever field the mirror used."""

    for key in ("answer_idx", "label", "gold", "correct_answer"):
        value = str(record.get(key, "")).strip().upper()
        if len(value) == 1 and value.isalpha():
            return value

    answer = str(record.get("answer", "")).strip()
    head = re.match(r"^([A-J])\s*[:.)]", answer.upper())
    if head:
        return head.group(1)
    if len(answer) == 1 and answer.upper().isalpha():
        return answer.upper()
    # Gold given as the option *text* ("Nitrofurantoin"): match it back to a letter.
    if answer:
        for option in options:
            prefix, _, body = option.partition(":")
            if body.strip().lower() == answer.lower():
                return prefix.strip().upper()[:1]
    return ""


def render_mcq(stem: str, options: list[str]) -> str:
    """The one multiple-choice prompt all three MCQ sets use.

    Shared so a difference between two medical numbers cannot come from the
    prompt. No \\boxed{} instruction: see the module docstring.
    """

    body = "\n".join(option.strip() for option in options if option and option.strip())
    return (
        "Answer the following multiple-choice medical question.\n\n"
        f"Question: {stem.strip()}\n\nOptions:\n{body}\n\n"
        "Answer with the option letter only."
    )


class _ChoiceMetric:
    """Shared body for the four medical sets: decode once, extract, score.

    Subclasses supply the data loader and the prompt; everything below the
    prompt -- decoding, extraction, the responses.jsonl, the summary, the
    appended csv row -- is identical on purpose, so a difference between two
    medical numbers cannot come from the harness.
    """

    name = ""
    data_path_keys: tuple[str, ...] = ()
    valid_letters = "ABCDE"

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def build_prompt(self, example: dict[str, Any]) -> str:
        return example["prompt"]

    def extract(self, response: str, example: dict[str, Any]) -> str:
        # Per-example valid set: CareQA rows do not all carry the same number of
        # options, and a letter outside a row's own set is not an answer.
        return extract_choice(response, example.get("valid") or self.valid_letters)

    def data_path(self, cfg: dict[str, Any]) -> str:
        for key in self.data_path_keys:
            if cfg.get(key):
                return str(cfg[key])
        return ""

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = self.data_path(cfg)
        max_new_tokens = int(
            cfg.get("medical_max_new_tokens", cfg.get("max_new_tokens", 1024))
        )
        run_name = cfg.get("run_name", "base")

        examples = self.load_examples(cfg)
        limit = int(cfg.get("limit", 0))
        if limit > 0:
            examples = examples[:limit]

        responses = batched_generate(
            model,
            tokenizer,
            [self.build_prompt(example) for example in examples],
            device,
            max_new_tokens,
            resolve_batch_size(cfg, "medical_batch_size"),
            log_label=self.name,
        )

        correct = 0
        parsed = 0
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for example, response in zip(examples, responses):
                prediction = self.extract(response, example)
                is_correct = bool(prediction) and prediction == example["gold"]
                correct += int(is_correct)
                parsed += int(bool(prediction))
                file.write(
                    json.dumps(
                        {
                            "id": example.get("id", ""),
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
            # Unparsed rows count as wrong in `accuracy`. That is the right
            # default -- an answer the grader cannot find was not given -- but it
            # makes a format-noncompliant base model look ignorant, so the
            # denominator is published alongside it.
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


class MedQAMetric(_ChoiceMetric):
    """MedQA 5-option test split (1273 USMLE questions)."""

    name = "medqa"
    data_path_keys = ("medqa_data_path", "data_path")
    valid_letters = "ABCDE"

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for index, record in enumerate(load_json_records(self.data_path(cfg))):
            if not isinstance(record, dict):
                continue
            stem = str(record.get("question", "")).strip()
            options = normalize_options(record.get("options"))
            gold = resolve_gold_letter(record, options)
            if not stem or len(options) < 2 or not gold:
                continue
            examples.append(
                {
                    "id": record.get("id", record.get("qid", index)),
                    "prompt": render_mcq(stem, options),
                    "gold": gold,
                    "valid": "".join(chr(ord("A") + i) for i in range(len(options))),
                }
            )
        return examples


class _IndexedMCQMetric(_ChoiceMetric):
    """MedMCQA and CareQA: options in ``op*`` columns, gold as an integer index."""

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        records = [
            record for record in load_json_records(self.data_path(cfg)) if isinstance(record, dict)
        ]
        base = index_base(records)
        examples: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            stem = str(field(record, "question", "Question")).strip()
            options = spread_options(record) or normalize_options(
                field(record, "options", "choices", default=None)
            )
            if not stem or len(options) < 2:
                continue

            position = gold_index(record)
            if position is not None:
                position -= base
                gold = chr(ord("A") + position) if 0 <= position < len(options) else ""
            else:
                gold = resolve_gold_letter(record, options)
            if not gold:
                continue
            examples.append(
                {
                    "id": field(record, "id", "qid", "unique_id", default=index),
                    "prompt": render_mcq(stem, options),
                    "gold": gold,
                    "valid": "".join(chr(ord("A") + i) for i in range(len(options))),
                }
            )
        return examples


class MedMCQAMetric(_IndexedMCQMetric):
    """MedMCQA validation split (4183 AIIMS/NEET-PG questions, 4 options).

    The validation split, not test: the released test split has no labels.
    """

    name = "medmcqa"
    data_path_keys = ("medmcqa_data_path", "data_path")
    valid_letters = "ABCD"


class CareQAMetric(_IndexedMCQMetric):
    """CareQA English split (Spanish MIR-style healthcare exams, 4 options)."""

    name = "careqa"
    data_path_keys = ("careqa_data_path", "data_path")
    valid_letters = "ABCD"


class PubMedQAMetric(_ChoiceMetric):
    """PubMedQA PQA-L (1000 expert-labeled): yes/no/maybe over an abstract.

    The label distribution is skewed toward "yes", so a model that answers yes to
    everything lands around 55%. Read this set as a floor check, not as a ranking.
    """

    name = "pubmedqa"
    data_path_keys = ("pubmedqa_data_path", "data_path")

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        records = load_json_records(self.data_path(cfg))
        # The official release is a {pmid: record} object, which load_json_records
        # cannot flatten (its list-valued keys are the context, not the rows).
        if not records:
            payload = json.loads(Path(self.data_path(cfg)).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                records = [
                    {**value, "id": key}
                    for key, value in payload.items()
                    if isinstance(value, dict)
                ]

        examples: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            question = str(field(record, "question", "QUESTION")).strip()
            gold = str(field(record, "final_decision", "answer")).strip().lower()
            if not question or gold not in ("yes", "no", "maybe"):
                continue
            examples.append(
                {
                    "id": field(record, "id", "pubid", default=index),
                    "prompt": self._render(question, field(record, "context", "CONTEXTS")),
                    "gold": gold,
                }
            )
        return examples

    @staticmethod
    def _render(question: str, context: Any) -> str:
        # dict -> HuggingFace mirror ({"contexts": [...]}), list -> official CONTEXTS.
        if isinstance(context, dict):
            body = "\n".join(str(part) for part in context.get("contexts", []))
        elif isinstance(context, (list, tuple)):
            body = "\n".join(str(part) for part in context)
        else:
            body = str(context or "")
        return (
            "Answer the following biomedical research question using the abstract "
            f"below.\n\nAbstract:\n{body.strip()}\n\nQuestion: {question}\n\n"
            "Answer yes, no, or maybe."
        )

    def extract(self, response: str, example: dict[str, Any]) -> str:
        return extract_decision(response)
