"""Shared serialization, length accounting and statistics for the three domains.

Everything in here exists because it must not be reimplemented per domain. The
three specialists are only comparable if a row means the same thing in all
three corpora, and the two places that silently break are the chat rendering
and the token count.

Length accounting
-----------------
The 4096 rule is "drop the row", not "cut the row", so the number it is applied
to has to be the number of tokens training will actually see -- not an estimate
of it. This module therefore renders with the *training* helper,
``onereplay.data.chat.apply_train_template``, and tokenizes exactly the way
``tokenizer_to_ids`` does (``add_special_tokens=False`` on both the full text
and the prompt text). Two details make a hand-rolled estimate come out wrong:

  * the training helper ``rstrip()``s the rendered text and appends ``eos`` when
    the template did not, which the manifest-style length reports in
    ``prepare_openr1_math.length_report`` do not do. On a *Base* checkpoint this
    fires on every single row: Qwen3-Base's ``eos_token`` is ``<|endoftext|>``
    while the template closes with ``<|im_end|>``, so every sequence really ends
    ``<|im_end|><|endoftext|>`` and is one token longer than the template says.
  * the chat template itself contributes role markers, and on Qwen3 an empty
    ``<think></think>`` block. Rendering "question + response" without the
    template undercounts by tens of tokens.

Because rows are dropped rather than truncated, a pool built at 4096 and
trained at ``--max_len 4096`` never reaches the left-truncation path in
``tokenizer_to_ids``. That is the point: no sample is ever shown to the model
with its question chopped off the front.

Prompt-prefix invariant
-----------------------
``tokenizer_to_ids`` masks labels by *length*::

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

which is only correct when ``prompt_ids`` is a genuine token prefix of
``full_ids``. Reading the Qwen3 template, that holds for a plain response and
breaks for one carrying its own ``<think>`` tags:

  * ``prompt_text`` is rendered with ``enable_thinking=False``, which appends
    ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``.
  * ``full_text`` hits the template's final-assistant branch, which emits
    ``<|im_start|>assistant\\n<think>\\n`` + reasoning + ``\\n</think>\\n\\n`` +
    content. With no ``</think>`` in the response the reasoning is empty, the
    two renders coincide exactly, and the prefix is genuine.
  * With ``</think>`` in the response the template *splits* it, hoisting the
    reasoning into that block. The renders then diverge right after
    ``<think>\\n``, and the length-based mask silently labels the opening of the
    reasoning as prompt.

So this is not a theoretical guard: a corpus shipping ``<think>`` tags (ODA-Fin
does) must have them stripped, and the check below is what proves it happened.

System message
--------------
All three domains carry the same system turn, and it is not empty. Qwen3-*Base*
renders a system block only when one is supplied -- unlike Qwen2.5-Math, whose
template injects the \\boxed{} instruction on its own. Leaving it empty would
mean an untrained base model never writes \\boxed{}, scores near zero because
the grader finds no answer span, while the fine-tuned model scores normally
because it picked up the format from the data. That gap is measurement, not
learning, and it would inflate every before/after number in the study.

DEFAULT_SYSTEM_PROMPT is therefore byte-identical to what 91_s1_train_math.pbs
already used, and it applies cleanly to all three domains because all three
were normalized to answer in \\boxed{}. The prompt must be identical at
preparation and at training time: it contributes tokens, and those tokens count
against the 4096 cutoff. Passing a different --system_prompt to training than to
these scripts silently invalidates the length guarantee, which is why the value
in use is recorded in each stats file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from onereplay.core.chat_policy import configure_system_prompt
from onereplay.data.chat import apply_train_template

# Reported by every domain so the cost of the cutoff is visible rather than
# assumed. 4096 is the rule; the neighbours are what the rule is traded against.
CANDIDATE_MAX_LENS = (1024, 2048, 3072, 4096, 6144, 8192)

# Byte-identical to 91_s1_train_math.pbs's SYSTEM_PROMPT_PRESET=auto expansion.
# Changing it means the Math specialist is no longer prompted the way the
# already-trained one was, and means re-running every baseline it is compared to.
DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# The record schema all three domains write. Provenance columns are kept out of
# training (the loader drops everything except the tokenized three) but decide
# whether a finished pool can still be audited, so they are not optional.
REQUIRED_FIELDS = (
    "id",
    "domain",
    "source",
    "question",
    "response",
    "input_tokens",
    "response_tokens",
    "total_tokens",
)


@dataclass
class SerializedExample:
    """One row rendered exactly as training will render it."""

    full_text: str
    prompt_text: str
    total_tokens: int
    input_tokens: int
    response_tokens: int
    prompt_is_prefix: bool


def load_tokenizer(tokenizer_path: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """Load the tokenizer and pin the run-wide system turn in one place.

    configure_system_prompt is global state that apply_train_template reads, so
    setting it is not optional: get it wrong and nothing raises, the rendered
    text simply gains or loses a system block, every length shifts, and the pool
    stops matching what training will build.
    """

    from transformers import AutoTokenizer

    configure_system_prompt(system_prompt)
    return AutoTokenizer.from_pretrained(tokenizer_path)


def serialize(tokenizer, question: str, response: str) -> SerializedExample:
    """Render one row the way training does and count its tokens.

    input_tokens is the masked prefix and response_tokens is what the loss is
    computed on, defined the same way ``tokenizer_to_ids`` defines them, so
    response_tokens equals the number of label positions that are not -100.
    """

    full_text, prompt_text = apply_train_template(tokenizer, question, "", response)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # min() mirrors tokenizer_to_ids: a prompt longer than the full render (only
    # possible when the two diverge) must not produce a negative count.
    input_tokens = min(len(prompt_ids), len(full_ids))
    return SerializedExample(
        full_text=full_text,
        prompt_text=prompt_text,
        total_tokens=len(full_ids),
        input_tokens=input_tokens,
        response_tokens=len(full_ids) - input_tokens,
        prompt_is_prefix=full_ids[: len(prompt_ids)] == prompt_ids,
    )


def serialize_many(
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    batch_size: int = 512,
    keep_text: bool = False,
    progress_every: int = 0,
) -> list[SerializedExample]:
    """Batched ``serialize`` for corpora too large to measure row by row.

    NuminaMath is 859k rows and every row costs a Jinja render plus two
    tokenizer calls. A fast tokenizer batches those calls internally, which is
    worth an order of magnitude; the Jinja render stays per-row because the
    template only takes one conversation.

    Batching cannot change the result: no padding and no truncation are
    requested, and ``add_special_tokens=False`` makes each sequence independent
    of its neighbours, so a batched encode is elementwise equal to the
    single-row one. tests/ pins that equality rather than trusting the claim.

    keep_text defaults to False because holding the rendered strings for a
    corpus this size doubles its memory footprint for no benefit -- callers
    measuring lengths in bulk only need the counts.
    """

    results: list[SerializedExample] = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        rendered = [
            apply_train_template(tokenizer, question, "", response)
            for question, response in chunk
        ]
        full_ids_batch = tokenizer(
            [item[0] for item in rendered], add_special_tokens=False
        )["input_ids"]
        prompt_ids_batch = tokenizer(
            [item[1] for item in rendered], add_special_tokens=False
        )["input_ids"]

        for (full_text, prompt_text), full_ids, prompt_ids in zip(
            rendered, full_ids_batch, prompt_ids_batch
        ):
            input_tokens = min(len(prompt_ids), len(full_ids))
            results.append(
                SerializedExample(
                    full_text=full_text if keep_text else "",
                    prompt_text=prompt_text if keep_text else "",
                    total_tokens=len(full_ids),
                    input_tokens=input_tokens,
                    response_tokens=len(full_ids) - input_tokens,
                    prompt_is_prefix=full_ids[: len(prompt_ids)] == prompt_ids,
                )
            )
        if progress_every and len(results) % progress_every < batch_size:
            print(f"  measured {len(results)}/{len(pairs)} rows")
    return results


def build_record(
    *,
    domain: str,
    index: int,
    source: str,
    question: str,
    response: str,
    serialized: SerializedExample,
    original_id: str = "",
    gold_answer: str = "",
    response_answer: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one output record in the shared schema.

    The id is positional (``math-0000123``) rather than a content hash: the
    pipeline is seeded, so position is reproducible, and a readable id makes a
    row traceable back to a line number during debugging. original_id keeps the
    upstream identifier when the corpus has one.
    """

    record = {
        "id": f"{domain}-{index:07d}",
        "domain": domain,
        "source": source,
        "question": question,
        "response": response,
        "input_tokens": serialized.input_tokens,
        "response_tokens": serialized.response_tokens,
        "total_tokens": serialized.total_tokens,
    }
    if original_id:
        record["original_id"] = original_id
    if gold_answer:
        record["gold_answer"] = gold_answer
    if response_answer:
        record["response_answer"] = response_answer
    if extra:
        record.update(extra)
    return record


def percentile(sorted_values: Sequence[int], fraction: float) -> int:
    """Nearest-rank percentile of an already sorted sequence.

    Nearest-rank rather than interpolated: these are token counts fed into a
    budget decision, and an interpolated 4095.5 is not a length any row has.
    """

    if not sorted_values:
        return 0
    rank = max(1, min(len(sorted_values), int(round(fraction * len(sorted_values)))))
    return int(sorted_values[rank - 1])


def length_summary(values: Sequence[int]) -> dict[str, Any]:
    """Distribution of one length column."""

    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "total": 0, "mean": 0.0, "median": 0, "p90": 0, "p95": 0, "max": 0}
    total = sum(ordered)
    return {
        "count": len(ordered),
        "total": total,
        "mean": total / len(ordered),
        "median": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def budget_scan(
    totals: Sequence[int],
    responses: Sequence[int],
    budgets: Iterable[int] = CANDIDATE_MAX_LENS,
) -> dict[str, Any]:
    """What each candidate cutoff would keep, measured rather than guessed.

    Reported before the cutoff is applied, so the row loss of the chosen budget
    can be compared against its neighbours. ``response_tokens_kept`` is the
    column that matters: it is the actual supervision each budget buys, and it
    is what a matched-token ablation would later have to equalize.
    """

    scan: dict[str, Any] = {}
    for budget in sorted(set(budgets)):
        kept = [index for index, total in enumerate(totals) if total <= budget]
        scan[str(budget)] = {
            "rows_kept": len(kept),
            "rows_dropped": len(totals) - len(kept),
            "fraction_kept": len(kept) / len(totals) if totals else 0.0,
            "total_tokens_kept": sum(totals[index] for index in kept),
            "response_tokens_kept": sum(responses[index] for index in kept),
        }
    return scan


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The per-domain statistics block required of all three pools."""

    totals = [record["total_tokens"] for record in records]
    inputs = [record["input_tokens"] for record in records]
    responses = [record["response_tokens"] for record in records]

    by_source: dict[str, dict[str, Any]] = {}
    for record in records:
        bucket = by_source.setdefault(
            record["source"], {"count": 0, "total_tokens": 0, "response_tokens": 0}
        )
        bucket["count"] += 1
        bucket["total_tokens"] += record["total_tokens"]
        bucket["response_tokens"] += record["response_tokens"]
    for bucket in by_source.values():
        bucket["percentage"] = bucket["count"] / len(records) * 100 if records else 0.0
        bucket["average_tokens"] = bucket["total_tokens"] / bucket["count"] if bucket["count"] else 0.0

    return {
        "num_examples": len(records),
        "total_tokens": sum(totals),
        "total_input_tokens": sum(inputs),
        "total_assistant_tokens": sum(responses),
        "mean_total_tokens": sum(totals) / len(totals) if totals else 0.0,
        "median_total_tokens": percentile(sorted(totals), 0.50),
        "p90_total_tokens": percentile(sorted(totals), 0.90),
        "p95_total_tokens": percentile(sorted(totals), 0.95),
        "max_total_tokens": max(totals) if totals else 0,
        "total_tokens_distribution": length_summary(totals),
        "input_tokens_distribution": length_summary(inputs),
        # Called out separately because the SFT loss only lands here, so this is
        # the column to compare across domains before claiming they got
        # comparable amounts of supervision.
        "assistant_tokens_distribution": length_summary(responses),
        "by_source": dict(sorted(by_source.items(), key=lambda item: -item[1]["count"])),
    }


@dataclass
class FilterLog:
    """Ordered tally of why rows disappeared.

    Insertion-ordered so the printed report reads as the pipeline ran, which is
    what makes a surprising final count diagnosable instead of merely visible.
    """

    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, reason: str, amount: int = 1) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + amount

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)

    def report(self) -> str:
        width = max((len(name) for name in self.counts), default=0)
        return "\n".join(f"  {name:<{width}} {count}" for name, count in self.counts.items())


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write the training pool, one JSON object per line."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as sink:
        for record in records:
            missing = [name for name in REQUIRED_FIELDS if name not in record]
            if missing:
                raise ValueError(f"record {record.get('id')} is missing {missing}")
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def write_dataset_dir(path: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    """Save the pool in the shape the training entry point actually loads.

    The JSONL is the auditable artifact; this is the trainable one.
    ``load_and_prepare_dataset`` calls ``load_from_disk`` and then maps
    ``build_sft_tokenize_fn``, which reads ``instruction`` / ``input`` /
    ``output`` -- so the schema is renamed here rather than at training time,
    where a per-domain rename would be one more thing that could differ between
    the three runs.

    ``input`` is empty for every row, which matters: ``apply_train_template``
    appends an "Input:" section when it is not, and that section was not part of
    the text measured against the 4096 cutoff. Empty keeps the trained sequence
    byte-identical to the measured one.

    Provenance columns ride along. ``build_loader`` drops every column except
    the tokenized three before collating, so they cost nothing at training time
    and keep the saved pool self-describing.
    """

    from datasets import Dataset, DatasetDict

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list[Any]] = {
        "instruction": [record["question"] for record in records],
        "input": ["" for _ in records],
        "output": [record["response"] for record in records],
    }
    for name in (
        "id",
        "domain",
        "source",
        "original_id",
        "gold_answer",
        "response_answer",
        "input_tokens",
        "response_tokens",
        "total_tokens",
    ):
        default: Any = 0 if name.endswith("_tokens") else ""
        columns[name] = [record.get(name, default) for record in records]

    DatasetDict({"train": Dataset.from_dict(columns)}).save_to_disk(str(target))
    return target


def write_stats(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write the statistics/metadata block next to the pool."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def build_metadata(
    *,
    domain: str,
    dataset_name: str,
    dataset_revision: str,
    tokenizer_path: str,
    max_tokens: int,
    seed: int,
    raw_count: int,
    filtered_count: int,
    final_count: int,
    filter_rules: dict[str, Any],
    prompt_prefix_violations: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Provenance block: enough to rebuild this pool from scratch.

    prompt_prefix_violations is in here rather than in a log line because a
    non-zero value invalidates the label masking for those rows, and that is a
    property of the pool that has to travel with it.
    """

    return {
        "domain": domain,
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "tokenizer": tokenizer_path,
        "max_total_tokens": max_tokens,
        "length_rule": "drop rows over max_total_tokens; never truncate",
        # Training must be launched with exactly this string, or the lengths
        # measured here do not describe the sequences it builds.
        "system_prompt": system_prompt,
        "serialization": "onereplay.data.chat.apply_train_template (system/user/assistant)",
        "seed": seed,
        "raw_count": raw_count,
        "filtered_count": filtered_count,
        "final_count": final_count,
        "filter_rules": filter_rules,
        "prompt_prefix_violations": prompt_prefix_violations,
    }
