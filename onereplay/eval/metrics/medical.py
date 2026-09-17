"""Medical evaluation: MedQA, PubMedQA, MedXpertQA.

All three are multiple choice and all three are graded by pulling one token out
of ``\\boxed{...}``, which is the contract the Medical specialist was trained on
(prepare_medical rewrites ``<answer>E</answer>`` to ``\\boxed{E}`` for exactly
this reason). Math and Finance end in a boxed answer too, so one extractor
grades all three domains -- necessary because these models are evaluated against
each other for forgetting, and a grader that happened to be stricter on one
domain would be indistinguishable from a model that forgot it.

What each set is for
--------------------
MedQA is the **in-domain** set: the specialist trains on the train split of this
same corpus, so its prompt is imported from prepare_medical rather than rebuilt
here. A reworded eval prompt would measure how well the model tolerates a new
surface form, not how much medicine it knows, and the forgetting curves would
inherit that confound.

PubMedQA is out-of-domain: yes/no/maybe over a PubMed abstract, so it tests
whether the medical ability generalizes past USMLE-style vignettes.

MedXpertQA is a **difficulty ruler, not a forgetting metric**. Ten options, so
chance is 10%, and the paper's own table has Qwen2.5-32B at 15.06% -- a 4B model
sits close enough to chance that a drop of a few points says nothing about what
it forgot. It is reported because "how far is this model from expert level" is
worth knowing; it must not be read as retention.

Why parse_rate is reported next to accuracy
-------------------------------------------
The base model is scored on these sets too, and a base model that answers
correctly in prose but never writes ``\\boxed{}`` would score zero. That is not
a measurement of medical knowledge, it is a measurement of format compliance,
and it would inflate every "training helped" delta and corrupt the forgetting
baseline. So extraction falls back to a few explicit answer-statement patterns,
and the fraction of rows that yielded any answer is reported. A run whose
accuracy looks bad should be checked against its parse_rate before it is
believed.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from onereplay.eval.generation import batched_generate, resolve_batch_size
from onereplay.eval.metrics.math500 import extract_answer, load_json_records
from onereplay.scripts.domain_sft.prepare_medical import build_question, gold_letter

# Fallbacks for a response that never boxed anything. Deliberately anchored on an
# explicit answer statement: a looser rule ("any capital letter near the end")
# would fire on "option D is wrong because ..." and award credit the model never
# claimed. Ordered most-specific first; the last match in the text wins, since
# models restate the answer after reconsidering.
_ANSWER_STATEMENT = re.compile(
    r"(?:final\s+answer|answer|choice|option)\s*(?:is|:)\s*\**\s*\(?([A-J])\)?\b",
    re.IGNORECASE,
)
_TRAILING_LETTER = re.compile(r"\b([A-J])\b[\s.)\]]*$")
_DECISION_STATEMENT = re.compile(
    r"(?:final\s+answer|answer|decision)\s*(?:is|:)\s*\**\s*(yes|no|maybe)\b",
    re.IGNORECASE,
)
_TRAILING_DECISION = re.compile(r"\b(yes|no|maybe)\b[\s.!)\]]*$", re.IGNORECASE)


def extract_choice(response: str, valid: str) -> str:
    """Pull a single option letter out of a response, or "" if there is none.

    ``valid`` is the live option set ("ABCDE" for MedQA, "ABCDEFGHIJ" for
    MedXpertQA); a letter outside it is treated as no answer rather than as a
    wrong one, so a model that writes prose containing a stray "F" on a 5-option
    question is not recorded as having answered F.
    """

    boxed = extract_answer(response)
    if boxed:
        # \boxed{E}, \boxed{\text{E}}, \boxed{E: Nitrofurantoin} all appear.
        cleaned = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", boxed)
        letter = cleaned.strip().strip("()$ ").upper()[:1]
        if letter in valid:
            return letter

    matches = _ANSWER_STATEMENT.findall(response)
    for candidate in reversed(matches):
        if candidate.upper() in valid:
            return candidate.upper()

    tail = _TRAILING_LETTER.search(response.strip())
    if tail and tail.group(1).upper() in valid:
        return tail.group(1).upper()
    return ""


def extract_decision(response: str) -> str:
    """Pull yes/no/maybe out of a PubMedQA response, or "" if there is none."""

    boxed = extract_answer(response)
    if boxed:
        cleaned = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", boxed)
        token = cleaned.strip().strip("().$ ").lower()
        if token in ("yes", "no", "maybe"):
            return token

    matches = _DECISION_STATEMENT.findall(response)
    if matches:
        return matches[-1].lower()
    tail = _TRAILING_DECISION.search(response.strip())
    return tail.group(1).lower() if tail else ""


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
            # Already "A: text"? Leave it; prepare_medical emits options verbatim
            # and the training prompt has to be reproduced character for character.
            options.append(text if re.match(r"^[A-J]\s*[:.)]", text) else
                           f"{chr(ord('A') + index)}: {text}")
    return options


def resolve_gold_letter(record: dict[str, Any], options: list[str]) -> str:
    """Find the gold option letter under whichever field the mirror used."""

    for key in ("answer_idx", "label", "gold", "correct_answer"):
        value = str(record.get(key, "")).strip().upper()
        if len(value) == 1 and value.isalpha():
            return value

    answer = str(record.get("answer", "")).strip()
    letter = gold_letter(answer)
    if letter:
        return letter
    # Gold given as the option *text* ("Nitrofurantoin"): match it back to a letter.
    if answer:
        for option in options:
            head, _, body = option.partition(":")
            if body.strip().lower() == answer.lower():
                return head.strip().upper()[:1]
    return ""


class _ChoiceMetric:
    """Shared body for the three medical sets: decode once, extract, score.

    Subclasses supply the data loader and the prompt; everything below the
    prompt -- decoding, extraction, the responses.jsonl, the summary, the
    appended csv row -- is identical on purpose, so a difference between two
    medical numbers cannot come from the harness.
    """

    name = ""
    data_path_keys: tuple[str, ...] = ()
    valid_letters = "ABCDE"
    # Set on MedXpertQA. Carried into the summary so a reader who finds the row
    # in a forgetting table knows the number does not belong there.
    ruler_only = False

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def build_prompt(self, example: dict[str, Any]) -> str:
        raise NotImplementedError

    def extract(self, response: str) -> str:
        return extract_choice(response, self.valid_letters)

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
                prediction = self.extract(response)
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
        if self.ruler_only:
            summary["ruler_only"] = 1
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
    """MedQA 5-option test split (1273 USMLE questions). In-domain.

    The prompt is prepare_medical.build_question, imported rather than copied:
    the specialist saw that exact string on every one of its training rows, and
    two copies of a prompt drift the moment either side is edited.
    """

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
                    "prompt": build_question(stem, options),
                    "gold": gold,
                }
            )
        return examples

    def build_prompt(self, example: dict[str, Any]) -> str:
        return example["prompt"]


class MedXpertQAMetric(_ChoiceMetric):
    """MedXpertQA Text (2450 questions, options A-J). Difficulty ruler only.

    Chance is 10% and Qwen2.5-32B scores 15.06% in the source paper, so for a 4B
    model the spread between "knows some medicine" and "guessing" is a couple of
    points. Reported for the ceiling it marks, excluded from retention claims --
    see ``ruler_only`` in the summary.
    """

    name = "medxpertqa"
    data_path_keys = ("medxpertqa_data_path", "data_path")
    valid_letters = "ABCDEFGHIJ"
    ruler_only = True

    # The dataset's `question` field already ends with the option block, so
    # re-emitting `options` under it would show the model every choice twice.
    _HAS_OPTIONS = re.compile(r"^\s*A[\s.:)]", re.MULTILINE)

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for index, record in enumerate(load_json_records(self.data_path(cfg))):
            if not isinstance(record, dict):
                continue
            stem = str(record.get("question", "")).strip()
            options = normalize_options(record.get("options"))
            gold = resolve_gold_letter(record, options)
            if not stem or not gold:
                continue
            if self._HAS_OPTIONS.search(stem):
                body = stem
            elif options:
                body = f"{stem}\n\nOptions:\n" + "\n".join(options)
            else:
                continue
            examples.append(
                {
                    "id": record.get("id", index),
                    "prompt": (
                        "Answer the following multiple-choice medical question. "
                        "Reason step by step, then give the final answer as "
                        "\\boxed{...} at the end.\n\nQuestion: " + body
                    ),
                    "gold": gold,
                }
            )
        return examples

    def build_prompt(self, example: dict[str, Any]) -> str:
        return example["prompt"]


class PubMedQAMetric(_ChoiceMetric):
    """PubMedQA PQA-L (1000 expert-labeled): yes/no/maybe over an abstract.

    Graded through the same boxed contract as the other two so the specialist is
    not asked to produce a format it was never trained on -- the answer token is
    a word instead of a letter, which is all that changes.

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
            "below. Reason step by step, then give the final answer as "
            "\\boxed{yes}, \\boxed{no}, or \\boxed{maybe} at the end.\n\n"
            f"Abstract:\n{body.strip()}\n\nQuestion: {question}"
        )

    def build_prompt(self, example: dict[str, Any]) -> str:
        return example["prompt"]

    def extract(self, response: str) -> str:
        return extract_decision(response)
