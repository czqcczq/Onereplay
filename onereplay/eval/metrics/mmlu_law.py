"""MMLU law evaluation: Professional Law, International Law, Jurisprudence.

The three law subjects of MMLU, run 0-shot on the same harness as the four
medical sets: decode once, extract one option letter, score. Reusing
``_ChoiceMetric`` rather than writing a parallel loop is the point -- decoding,
extraction, responses.jsonl, summary.json and the appended csv row are then
byte-identical across domains, so a gap between a law number and a medical
number cannot have come from the harness.

0-shot rather than the 5-shot MMLU standard, for two reasons. The arms being
compared are all instruction-tuned chat checkpoints that answer a direct
question without demonstrations, and Professional Law stems run long enough that
five of them would dominate the context -- a retention experiment does not
benefit from measuring how well each arm survives a 4k prompt. The tradeoff is
that these numbers are not comparable to published 5-shot MMLU; they are only
comparable to each other, which is all the before/after needs.

parse_rate matters more here than elsewhere and is the first thing to read: a
1.5B model asked for a bare letter 0-shot may write prose instead, and an
accuracy that looks like ignorance can be a formatting miss. Unparsed rows count
as wrong, so accuracy without parse_rate next to it is not interpretable.
"""

from __future__ import annotations

from typing import Any

from onereplay.eval.metrics.math500 import load_json_records
from onereplay.eval.metrics.medical import _ChoiceMetric, field

# Every law subject is 4-option single-answer, unlike the medical sets which mix
# 4 and 5. A stray "E" in prose is therefore not an answer.
LAW_LETTERS = "ABCD"


def render_law_mcq(stem: str, options: list[str]) -> str:
    """The one prompt all three law subjects use.

    Deliberately the same skeleton as medical's ``render_mcq``, down to the
    closing "Answer with the option letter only." -- only the domain word
    differs. Nothing asks for \\boxed{} or for a rationale: the Law specialist
    trains on free-form legal prose and short gold terms, so a format contract
    it was never taught would be measurement, not knowledge.
    """

    body = "\n".join(option.strip() for option in options if option and option.strip())
    return (
        "Answer the following multiple-choice law question.\n\n"
        f"Question: {stem.strip()}\n\nOptions:\n{body}\n\n"
        "Answer with the option letter only."
    )


class _MMLULawMetric(_ChoiceMetric):
    """Shared loader for the three subjects; they differ only by file and name.

    The gold column arrives as a 0-based index into ``choices`` (that is what
    download_mmlu_law_data.py writes and what cais/mmlu publishes), and the
    letter is derived here. Deriving it at scoring time rather than storing it
    means the option ordering the model saw and the letter the grader expects
    cannot drift apart.
    """

    valid_letters = LAW_LETTERS
    max_new_tokens_key = "law_max_new_tokens"
    batch_size_key = "law_batch_size"

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for index, record in enumerate(load_json_records(self.data_path(cfg))):
            if not isinstance(record, dict):
                continue
            stem = str(field(record, "question", "Question", default="")).strip()
            raw_choices = field(record, "choices", "options", default=None) or []
            choices = [str(choice).strip() for choice in raw_choices if str(choice).strip()]
            gold_index = field(record, "answer", "Answer", "label", default=None)
            if not stem or len(choices) < 2:
                continue
            try:
                gold_index = int(gold_index)
            except (TypeError, ValueError):
                continue
            if not 0 <= gold_index < len(choices):
                continue
            letters = "".join(chr(ord("A") + position) for position in range(len(choices)))
            examples.append(
                {
                    "id": record.get("id", index),
                    "prompt": render_law_mcq(
                        stem,
                        [f"{letters[position]}: {text}" for position, text in enumerate(choices)],
                    ),
                    "gold": letters[gold_index],
                    "valid": letters,
                }
            )
        return examples


class MMLUProfessionalLawMetric(_MMLULawMetric):
    """MMLU Professional Law test split (~1534 bar-exam-style questions)."""

    name = "mmlu_professional_law"
    data_path_keys = ("mmlu_professional_law_data_path", "data_path")


class MMLUInternationalLawMetric(_MMLULawMetric):
    """MMLU International Law test split (~121 questions).

    Small enough that one question is ~0.8 points, so a difference of a couple
    of points here is noise. Read it alongside Professional Law, not alone.
    """

    name = "mmlu_international_law"
    data_path_keys = ("mmlu_international_law_data_path", "data_path")


class MMLUJurisprudenceMetric(_MMLULawMetric):
    """MMLU Jurisprudence test split (~108 questions). Same caveat as above."""

    name = "mmlu_jurisprudence"
    data_path_keys = ("mmlu_jurisprudence_data_path", "data_path")
