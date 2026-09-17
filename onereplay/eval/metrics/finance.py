"""Finance evaluation: FinQA and ConvFinQA.

Both sets ask a numerical-reasoning question about one filing page -- some
paragraphs of text and one table -- and both grade on the executed numeric
answer, so they share the document rendering, the answer extraction and the
scorer here. ConvFinQA is FinQA's corpus turned into multi-turn dialogues; the
only real difference is that a question may depend on earlier turns.

The prompt ends in the same "reason step by step, then \\boxed{...}" contract as
Math and Medical. That is not cosmetic: these three specialists are compared
against each other for forgetting, so the answer must be found the same way in
all three domains, or grader strictness shows up as a domain effect.

Percent scale is the trap in this benchmark
-------------------------------------------
FinQA stores two versions of the same answer. ``exe_ans`` is what the annotated
program evaluates to, usually a ratio like ``0.1111``, while ``answer`` is the
human-written string, ``"11.1%"``. A model asked for a percentage change will
write either one and be right both times. Comparing against ``exe_ans`` alone
would mark every ``\\boxed{11.1\\%}`` wrong, which is a structural zero on a
whole question type -- the same failure mode AMC had when integer golds arrived
as "142.0" and string equality scored the entire set wrong.

So a mismatch is retried at 100x and 1/100x, but **only when a percent sign
appears on one of the two sides**. Gating on the percent sign is what keeps this
from becoming a blanket "answers within two orders of magnitude are correct":
without it, a model that misread thousands for millions would be given credit.

Numbers are compared with a relative tolerance because the golds are rounded
quantities: the annotator's "11.1%" and a correct 11.1349% are the same answer.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from onereplay.eval.generation import batched_generate, resolve_batch_size
from onereplay.eval.metrics.math500 import (
    extract_answer,
    is_equiv,
    numbers_close,
    to_number,
)

INSTRUCTION_PREFIX = (
    "Answer the following financial question using the report excerpt below. "
    "Reason step by step, then give the final answer as \\boxed{...} at the end."
)


def render_table(table: Any) -> str:
    """Flatten a filing table to one pipe-separated row per line.

    Plain enough that the tokenizer does not spend the budget on markdown rules,
    structured enough that column alignment survives -- which is the entire task
    in TAT-QA-style table reasoning.
    """

    if not isinstance(table, (list, tuple)):
        return ""
    lines = []
    for row in table:
        if isinstance(row, (list, tuple)):
            lines.append(" | ".join(str(cell).strip() for cell in row))
        else:
            lines.append(str(row).strip())
    return "\n".join(line for line in lines if line)


def render_document(record: dict[str, Any]) -> str:
    """Assemble pre_text / table / post_text into the prompt's context block."""

    def join(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return " ".join(str(part).strip() for part in value if str(part).strip())
        return str(value or "").strip()

    parts = []
    pre_text = join(record.get("pre_text"))
    if pre_text:
        parts.append(pre_text)
    table = render_table(record.get("table") or record.get("table_ori"))
    if table:
        parts.append("Table:\n" + table)
    post_text = join(record.get("post_text"))
    if post_text:
        parts.append(post_text)
    return "\n\n".join(parts)


def build_prompt(document: str, question: str, history: str = "") -> str:
    """Render the eval prompt. Module level so probes measure the real thing."""

    blocks = [INSTRUCTION_PREFIX, f"Report:\n{document}"]
    if history:
        blocks.append(f"Previous questions and answers:\n{history}")
    blocks.append(f"Question: {question}")
    return "\n\n".join(blocks)


def _percent_involved(*texts: str) -> bool:
    return any("%" in (text or "") for text in texts)


def answer_matches(
    prediction: str | None, gold_text: str, gold_value: float | None, rel_tol: float
) -> bool:
    """Numeric comparison with a percent-scale retry, falling back to strings.

    Non-numeric answers do exist (a handful of FinQA rows answer "yes"/"no"), so
    anything that will not parse as a number is handed to the MATH string
    equivalence check -- conservative, in that it can only withhold credit.
    """

    if not prediction:
        return False
    predicted_value = to_number(prediction)
    if gold_value is None:
        gold_value = to_number(gold_text)

    if predicted_value is not None and gold_value is not None:
        if numbers_close(predicted_value, gold_value, rel_tol):
            return True
        # See the module docstring: only a percent sign on one side licenses the
        # rescale, so this cannot rescue an answer that is merely off by 100x.
        if _percent_involved(prediction, gold_text):
            for scaled in (predicted_value / 100.0, predicted_value * 100.0):
                if numbers_close(scaled, gold_value, rel_tol):
                    return True
        return False
    return is_equiv(prediction, gold_text)


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


class _FinanceMetric:
    """Shared decode/score/report body. Subclasses only build the example list."""

    name = ""
    data_path_keys: tuple[str, ...] = ()

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

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
            cfg.get("finance_max_new_tokens", cfg.get("max_new_tokens", 1024))
        )
        rel_tol = float(cfg.get("finance_rel_tol", 0.01) or 0.01)
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
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for example, response in zip(examples, responses):
                prediction = extract_answer(response)
                is_correct = answer_matches(
                    prediction, example["gold_text"], example["gold_value"], rel_tol
                )
                correct += int(is_correct)
                parsed += int(bool(prediction))
                file.write(
                    json.dumps(
                        {
                            "id": example.get("id", ""),
                            "question": example["question"],
                            "gold": example["gold_text"],
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
            # Rows where nothing was boxed. A base model that reasons in prose
            # and never boxes would score zero here for reasons that have nothing
            # to do with finance, so the two numbers have to be read together.
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


class FinQAMetric(_FinanceMetric):
    """FinQA test split (1147 questions over S&P 500 filing pages)."""

    name = "finqa"
    data_path_keys = ("finqa_data_path", "data_path")

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for index, record in enumerate(load_records(self.data_path(cfg))):
            document = render_document(record)
            # The qa block is "qa" in the original release but "qa_0"/"qa_1" on
            # pages carrying two questions. The HuggingFace mirror
            # (dreamerdeo/finqa) instead flattens question/answers to the top
            # level and drops exe_ans entirely, so a flat row is its own block.
            blocks = [record[key] for key in ("qa", "qa_0", "qa_1") if isinstance(record.get(key), dict)]
            if not blocks and isinstance(record.get("question"), str):
                blocks = [record]
            for block_index, block in enumerate(blocks):
                question = str(block.get("question", "")).strip()
                # "answers" (plural) is the mirror's spelling. Reading only
                # "answer" would leave every gold empty and score a clean 0.
                gold_text = str(block.get("answer") or block.get("answers") or "").strip()
                gold_value = block.get("exe_ans")
                gold_value = float(gold_value) if isinstance(gold_value, (int, float)) else None
                if not question or (not gold_text and gold_value is None):
                    continue
                examples.append(
                    {
                        "id": f"{record.get('id', index)}#{block_index}",
                        "question": question,
                        "prompt": build_prompt(document, question),
                        "gold_text": gold_text,
                        "gold_value": gold_value,
                    }
                )
        return examples


class ConvFinQAMetric(_FinanceMetric):
    """ConvFinQA dev: FinQA pages turned into multi-turn dialogues.

    Every turn is scored as its own row, and the turns before it are supplied
    with their **gold** answers rather than the model's. Feeding the model its own
    earlier answers would let one early arithmetic slip corrupt the rest of the
    conversation, so a single mistake would cost several rows and the score would
    partly measure conversation length. Teacher forcing keeps each turn an
    independent question, which is how the set is normally reported.

    Two file layouts are accepted, because the release ships both and which one
    gets downloaded is a coin flip:

    * ``dev_turn.json`` (1490 rows) is already one row per turn, carrying
      ``cur_dial`` (the questions up to and including this one) and ``exe_ans``.
      Preferred -- the turn boundaries are the authors', not ours.
    * ``dev.json`` (421 conversations) needs expanding through
      ``annotation.dialogue_break`` and ``annotation.exe_ans_list``.
    """

    name = "convfinqa"
    data_path_keys = ("convfinqa_data_path", "data_path")

    def load_examples(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for index, record in enumerate(load_records(self.data_path(cfg))):
            document = render_document(record)
            annotation = record.get("annotation")
            annotation = annotation if isinstance(annotation, dict) else {}
            # In the released dev_turn.json the turn fields sit under
            # "annotation", not at the top level. The top-level qa block there is
            # the conversation's *source* FinQA question and is byte-identical on
            # every turn of that conversation, so reading it would score all 1490
            # rows against only 421 distinct golds -- and silently, since every
            # field it needs is present and well-formed.
            dialogue = record.get("cur_dial") or annotation.get("cur_dial")
            if isinstance(dialogue, (list, tuple)) and dialogue:
                gold = record.get("exe_ans")
                if gold is None:
                    gold = annotation.get("exe_ans")
                turn = record.get("turn_ind", annotation.get("turn_ind", 0))
                example = self._from_turn_row(record, document, index, dialogue, gold, turn)
                if example:
                    examples.append(example)
                continue
            examples.extend(self._from_conversation_row(record, document, index))
        return examples

    @staticmethod
    def _from_turn_row(
        record: dict[str, Any],
        document: str,
        index: int,
        dialogue: Any,
        gold: Any,
        turn: Any,
    ) -> dict[str, Any] | None:
        question = str(dialogue[-1]).strip()
        if not question or gold is None:
            return None
        # cur_dial holds this turn's question as its last element; the rest is
        # context. Their gold answers are not on a turn row, so history is the
        # question thread alone -- enough to resolve "and in 2018?" style
        # references, which is what the earlier turns are there for.
        history = "\n".join(f"Q: {str(part).strip()}" for part in dialogue[:-1])
        # A handful of turns give exe_ans as a string ("yes", "12.5%").
        value = float(gold) if isinstance(gold, (int, float)) else to_number(str(gold))
        return {
            "id": f"{record.get('id', index)}#turn{turn}",
            "question": question,
            "prompt": build_prompt(document, question, history),
            "gold_text": str(gold),
            "gold_value": value,
        }

    @staticmethod
    def _from_conversation_row(
        record: dict[str, Any], document: str, index: int
    ) -> list[dict[str, Any]]:
        annotation = record.get("annotation")
        if not isinstance(annotation, dict):
            return []
        questions = annotation.get("dialogue_break") or []
        golds = annotation.get("exe_ans_list") or []
        if not questions or len(golds) < len(questions):
            return []

        rows: list[dict[str, Any]] = []
        history: list[str] = []
        for turn, raw_question in enumerate(questions):
            question = str(raw_question).strip()
            gold = golds[turn]
            gold_text = str(gold)
            if question:
                rows.append(
                    {
                        "id": f"{record.get('id', index)}#turn{turn}",
                        "question": question,
                        "prompt": build_prompt(document, question, "\n".join(history)),
                        "gold_text": gold_text,
                        "gold_value": float(gold) if isinstance(gold, (int, float)) else None,
                    }
                )
            history.append(f"Q: {question}\nA: {gold_text}")
        return rows
