"""Build the Finance specialist pool from OpenDataArena/ODA-Fin-SFT-318k.

318599 samples aggregated from 25+ finance corpora, with CoT distilled from
Qwen3-235B-A22B-Thinking and verified. Each row is::

    {"id", "source", "instruction", "input", "output", "answer", "process"}

where ``output`` is ``<think>{CoT}</think><answer>{final}</answer>`` and the
final segment normally ends in ``\\boxed{...}``. ``input`` is empty throughout.

``answer`` is optional and whole sources omit it
------------------------------------------------
Three of the largest constituents -- Agentar-DeepFinance-100K (98694 rows),
DianJin-R1-Data (36560) and Finance_R1-Distill_Data (2375) -- ship only
``{id, instruction, input, output, source}``. Not "answer is blank sometimes":
the key is absent on every row of those sources. Requiring it therefore selects
on which upstream corpus bothered to denormalize its final answer into a
separate column, which has nothing to do with whether the row is usable.

That matters more than it sounds: those three are the reasoning-heavy part of
the aggregate, and dropping them left the pool ~40% sentiment classification.
So ``answer`` is provenance only -- it is written to ``gold_answer`` for
traceability and never read by training, which consumes ``instruction`` and
``response``. When the field is missing, the final ``\\boxed{}`` span is used
instead, and a row with neither is still kept. Completeness is enforced where it
actually lives: strip_tags rejects damaged ``<think>``/``<answer>`` structure.
``--require_answer_field 1`` restores the old behaviour.

Loading
-------
The corpus ships as one ``train.json``. ``load_dataset("json", ...)`` cannot
infer a schema across it -- the Hub's own viewer fails with ``KeyError:
'answer'`` -- and pinning explicit features would break on the rows where
``answer`` is a number rather than a string. So this reads the file with an
incremental decoder that yields one object at a time and tolerates missing keys,
which also keeps a multi-gigabyte array from being materialized at once.

Why the tags are stripped
-------------------------
The same mask-alignment reason as Medical: Qwen3's chat template splits an
assistant message that contains ``</think>`` and re-emits the reasoning inside
its own think block, which desynchronizes the prompt render from the full render
and makes the length-based label mask wrong. ``<answer>`` tags go too -- they are
this corpus's convention, not something the other two domains share, and
``\\boxed{}`` already carries the final answer.

Ordering: decontamination comes first
-------------------------------------
"Eval first" is a real constraint, not a preference. Deciding the benchmark after
sampling the training data means any contamination found later can only be fixed
by rebuilding the pool, and a pool that was already trained on cannot be fixed at
all. With FinQA, TAT-QA and ConvFinQA as the Finance benchmarks, rows whose
``source`` names them (or a derivative) are removed before anything else, so the
sampling stage never sees them and no downstream quota silently reintroduces one.
``--eval_questions_jsonl`` additionally removes rows matching eval questions at
the text level, for the case where a benchmark leaked into a differently named
source.

Off-task sources are excluded outright
--------------------------------------
Separate from decontamination, and for a different reason. Roughly 40% of the
aggregate is single-label work -- headline sentiment, hawkish/dovish, relation
extraction -- where the target is one token and the distilled CoT is filler.
All three benchmarks (FinQA, ConvFinQA, TAT-QA) ask for multi-step numeric
reasoning over financial tables, so those rows spend the budget without moving
the thing being measured. See DEFAULT_EXCLUDE_SOURCE_PATTERNS.

The pool is therefore deliberately smaller than Math's 100k. Matching example
counts across domains was never the goal; a Finance model trained on 60k
reasoning traces is a better instrument than one trained on 100k where 40k are
tweet labels, and Medical lands near 30k for the same kind of reason.

Source balancing
----------------
The aggregate is dominated by a handful of large sources, so a plain shuffle-
and-take would hand most of the pool to whichever corpus happened to be biggest
-- which would make "the Finance specialist" mean "the Finance-Instruct-500k
specialist". Every source is capped at --max_source_ratio of the final pool. The
cap is solved by iteration rather than applied once, because capping shrinks the
pool, which shrinks the cap: see solve_source_cap.

Nothing here forces the pool to 100k. The length rule and the quality filters
decide the count, and the count is reported.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from onereplay.scripts.domain_sft.common import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    FilterLog,
    budget_scan,
    build_metadata,
    build_record,
    load_tokenizer,
    serialize_many,
    summarize,
    write_dataset_dir,
    write_jsonl,
    write_stats,
)

DOMAIN = "finance"
DATASET = "OpenDataArena/ODA-Fin-SFT-318k"

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"
BOXED_OPEN = re.compile(r"\\boxed\s*\{")

# Substrings matched case-insensitively against the `source` field. FinQA,
# TAT-QA and ConvFinQA are the chosen Finance benchmarks, so their training
# splits and anything derived from them must not be trained on. "finqa" also
# matches "convfinqa", which is intended.
DEFAULT_EVAL_SOURCE_PATTERNS = ("finqa", "tat-qa", "tatqa", "tat_qa", "convfinqa")

# Single-label tasks: tag a headline as positive/negative, a FOMC line as
# hawkish/dovish, a sentence pair as a relation type. They are finance text, but
# the target is one token and the CoT distilled onto them is filler -- nothing
# here trains multi-step numeric reasoning, which is what all three benchmarks
# ask for. Nine sources, ~48k rows, 40% of the pre-filter pool. Dropping them is
# why the pool can be smaller than Math's and still be worth more.
DEFAULT_EXCLUDE_SOURCE_PATTERNS = (
    "sentiment",  # financial-tweets-sentiment, fingpt-sentiment-train, twitter-financial-news-sentiment
    "phrasebank",  # takala/financial_phrasebank
    "en-fpb",  # TheFinAI/en-fpb, the same corpus relabelled
    "hawkish",  # gtfintechlab/fomc-hawkish-dovish
    "financial-classification",  # nickmuchi/financial-classification
    "finentity",  # yixuantt/FinEntity, span tagging
    "finred",  # FinGPT/fingpt-finred, relation extraction
)

CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Finance SFT pool from OpenDataArena/ODA-Fin-SFT-318k."
    )
    parser.add_argument(
        "--fin_json",
        type=str,
        required=True,
        help="Path to train.json from the dataset repo.",
    )
    parser.add_argument("--dataset_revision", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument(
        "--out_jsonl", type=str, default="data/processed/finance_train.jsonl"
    )
    parser.add_argument("--out_stats", type=str, default="data/stats/finance_stats.json")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/finance_train",
        help="save_to_disk target, which is what training loads. Empty skips it.",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "any"],
        default="en",
        help="en drops rows containing CJK characters. The corpus is bilingual "
        "while Math and Medical are English-only, and a bilingual Finance "
        "specialist would differ from the other two in more than its domain.",
    )
    parser.add_argument(
        "--eval_source_patterns",
        type=str,
        nargs="*",
        default=list(DEFAULT_EVAL_SOURCE_PATTERNS),
        help="Case-insensitive substrings of `source` to remove as benchmark "
        "contamination. Pass an empty list to disable (not recommended).",
    )
    parser.add_argument(
        "--eval_questions_jsonl",
        type=str,
        default="",
        help="Optional JSONL of held-out eval questions (field `question` or "
        "`instruction`) removed by normalized text match.",
    )
    parser.add_argument(
        "--max_source_ratio",
        type=float,
        default=0.15,
        help="No single source may exceed this share of the final pool.",
    )
    parser.add_argument(
        "--deprioritize_sources",
        type=str,
        nargs="*",
        default=[],
        help="Substrings of `source` given a halved cap, for low-reasoning data "
        "such as plain sentiment classification. Set after reading the "
        "by_source table (count / percentage / average_tokens) from a first run.",
    )
    parser.add_argument(
        "--exclude_sources",
        type=str,
        nargs="*",
        default=list(DEFAULT_EXCLUDE_SOURCE_PATTERNS),
        help="Case-insensitive substrings of `source` dropped as off-task. "
        "Distinct from --eval_source_patterns: that one prevents cheating, this "
        "one is a statement about what the Finance specialist is for. Pass an "
        "empty list to keep every source.",
    )
    parser.add_argument(
        "--require_boxed",
        type=int,
        default=0,
        help="1 keeps only responses carrying a \\boxed span. Off by default: "
        "long-form financial analysis legitimately ends without one.",
    )
    parser.add_argument(
        "--require_answer_field",
        type=int,
        default=0,
        help="1 drops rows without a top-level `answer`. Off by default because "
        "three whole sources omit the key -- including the two largest "
        "reasoning-heavy ones -- so requiring it selects on upstream schema, "
        "not on data quality. See the module docstring.",
    )
    parser.add_argument("--target_rows", type=int, default=0, help="0 keeps everything the filters allow.")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Must match what training is launched with: it contributes tokens "
        "that count against --max_tokens.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=0, help="Debug aid; 0 reads all.")
    return parser.parse_args()


def iter_json_records(path: str, chunk_size: int = 1 << 22) -> Iterator[dict[str, Any]]:
    """Yield objects from a JSON array or a JSONL file without loading it whole.

    Written by hand rather than delegated to `datasets` because this specific
    file defeats schema inference (a missing `answer` on some rows, and an
    `answer` that is sometimes a number and sometimes a string). raw_decode
    consumes one object per call and reports where it stopped, so the buffer only
    ever holds the tail that has not been parsed yet.
    """

    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as handle:
        probe = handle.read(1)
        while probe and probe.isspace():
            probe = handle.read(1)
        if not probe:
            return

        if probe != "[":
            handle.seek(0)
            for line in handle:
                line = line.strip().rstrip(",")
                if line and line not in ("[", "]"):
                    yield json.loads(line)
            return

        buffer = ""
        exhausted = False
        while True:
            if not exhausted:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    exhausted = True

            cursor = 0
            while True:
                while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
                    cursor += 1
                if cursor >= len(buffer):
                    break
                if buffer[cursor] == "]":
                    return
                try:
                    obj, cursor = decoder.raw_decode(buffer, cursor)
                except ValueError:
                    # Truncated object: wait for the next chunk.
                    break
                yield obj
            buffer = buffer[cursor:]

            if exhausted and not buffer.strip(" \t\r\n,"):
                return
            if exhausted and buffer.strip(" \t\r\n,") and buffer.strip()[0] != "]":
                raise ValueError(f"trailing unparsed data in {path}: {buffer[:120]!r}")
            if exhausted:
                return


def strip_tags(text: str) -> str:
    """Flatten ``<think>CoT</think><answer>final</answer>`` into plain text.

    Returns "" when the structure is malformed. Structural damage is the signal
    the dataset card's own verification could not catch -- an unclosed think
    block means the trace was cut off, and training on it teaches truncation.
    """

    if not text or not text.strip():
        return ""

    n_think_open, n_think_close = text.count(THINK_OPEN), text.count(THINK_CLOSE)
    n_ans_open, n_ans_close = text.count(ANSWER_OPEN), text.count(ANSWER_CLOSE)
    if n_think_open != n_think_close or n_think_open > 1:
        return ""
    if n_ans_open != n_ans_close or n_ans_open > 1:
        return ""

    reasoning = ""
    remainder = text
    if n_think_close == 1:
        reasoning, _, remainder = text.partition(THINK_CLOSE)
        reasoning = reasoning.replace(THINK_OPEN, "", 1).strip()
        if not reasoning:
            return ""

    if n_ans_close == 1:
        _, _, tail = remainder.partition(ANSWER_OPEN)
        final, _, trailing = tail.partition(ANSWER_CLOSE)
        if trailing.strip():
            # Content after </answer> means the tags do not delimit what they
            # claim to; flattening would reorder the response.
            return ""
        remainder = final

    body = f"{reasoning}\n\n{remainder.strip()}".strip() if reasoning else remainder.strip()
    if not body:
        return ""
    if any(tag in body for tag in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE)):
        return ""
    return body


def extract_boxed(text: str) -> str:
    """Contents of the last ``\\boxed{...}`` span, or "" if there is none.

    Brace-matched rather than regex-matched: finance answers carry ``\\text{}``
    and ``\\frac{}{}`` often enough that a non-greedy ``\\{([^}]*)\\}`` would cut
    at the first inner close brace and return a fragment. The last span wins
    because a trace that boxes an intermediate result still boxes the final
    answer last.
    """

    last = ""
    for match in BOXED_OPEN.finditer(text):
        index, depth = match.end(), 1
        while index < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            index += 1
        if depth == 0:
            last = text[match.end() : index - 1].strip()
    return last


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def load_eval_questions(path: str) -> set[str]:
    """Normalized eval questions for text-level decontamination."""

    if not path:
        return set()
    questions: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("question") or row.get("instruction") or ""
            if text.strip():
                questions.add(normalize(text))
    return questions


def solve_source_cap(counts: dict[str, int], max_ratio: float) -> tuple[int, dict[str, int]]:
    """Largest pool size where no source exceeds max_ratio, with per-source caps.

    Capping a source shrinks the pool, which shrinks the cap in absolute terms,
    which may cap a source that was previously under the line. Iterating to a
    fixed point is the cheap way to solve that; the total is non-increasing and
    bounded below, so it terminates.

    Caller beware: the fixed point is only useful when ``len(counts) >= 1 /
    max_ratio``. Below that no distribution can satisfy the constraint and the
    iteration collapses toward one row per source -- two sources at ratio 0.15
    converge to a total of 2. main() refuses to run in that regime rather than
    letting this return a technically-correct answer that destroys the pool.
    """

    total = sum(counts.values())
    if max_ratio >= 1.0 or not counts:
        return total, dict(counts)

    for _ in range(1000):
        cap = max(1, int(max_ratio * total))
        capped = {name: min(count, cap) for name, count in counts.items()}
        new_total = sum(capped.values())
        if new_total == total:
            return total, capped
        total = new_total
    return total, {name: min(count, max(1, int(max_ratio * total))) for name, count in counts.items()}


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path, args.system_prompt)
    eval_questions = load_eval_questions(args.eval_questions_jsonl)
    patterns = [pattern.lower() for pattern in args.eval_source_patterns]
    if not patterns:
        print("WARNING: no --eval_source_patterns; benchmark contamination is NOT removed.")
    excluded = [pattern.lower() for pattern in args.exclude_sources]

    log = FilterLog()
    raw_count = 0
    seen_instructions: set[str] = set()
    raw_source_counts: dict[str, int] = {}
    # Rows that passed every content filter. Measured in bulk afterwards rather
    # than one at a time inside the loop: the tokenizer batches, and at this
    # scale that is the difference between minutes and half an hour.
    candidates: list[dict[str, Any]] = []
    prefix_violations = 0
    scan_totals: list[int] = []
    scan_responses: list[int] = []

    for row in iter_json_records(args.fin_json):
        raw_count += 1
        if args.max_rows and raw_count > args.max_rows:
            raw_count -= 1
            break
        if raw_count % 25000 == 0:
            print(f"  read {raw_count} rows, kept {len(candidates)}")

        source = str(row.get("source") or "").strip()
        raw_source_counts[source or "(unknown)"] = raw_source_counts.get(source or "(unknown)", 0) + 1

        # Decontamination first, before any quality filter can "rescue" a row.
        lowered = source.lower()
        if any(pattern in lowered for pattern in patterns):
            log.bump("dropped_eval_contaminated_source")
            continue
        if any(pattern in lowered for pattern in excluded):
            log.bump("dropped_off_task_source")
            continue

        instruction = str(row.get("instruction") or "").strip()
        output = str(row.get("output") or "").strip()

        if not instruction:
            log.bump("dropped_empty_instruction")
            continue
        if not output:
            log.bump("dropped_empty_output")
            continue

        if eval_questions and normalize(instruction) in eval_questions:
            log.bump("dropped_eval_contaminated_text")
            continue

        if args.language == "en" and (
            CJK_PATTERN.search(instruction) or CJK_PATTERN.search(output)
        ):
            log.bump("dropped_non_english")
            continue

        body = strip_tags(output)
        if not body:
            log.bump("dropped_malformed_reasoning_structure")
            continue
        if args.require_boxed and "\\boxed" not in body:
            log.bump("dropped_no_boxed")
            continue

        # Provenance only -- training never reads this. Resolved after strip_tags
        # so the fallback searches the same text that becomes the response.
        answer = row.get("answer")
        answer = "" if answer is None else str(answer).strip()
        if not answer:
            answer = extract_boxed(body)
            log.bump("answer_from_boxed" if answer else "answer_unavailable")
        if args.require_answer_field and not answer:
            log.bump("dropped_empty_answer")
            continue

        key = normalize(instruction)
        if key in seen_instructions:
            log.bump("dropped_duplicate_instruction")
            continue
        seen_instructions.add(key)

        candidates.append(
            {
                "source": source or "(unknown)",
                "instruction": instruction,
                "response": body,
                "answer": answer,
                "original_id": str(row.get("id") or ""),
            }
        )

    print(f"\nread {raw_count} raw rows; measuring {len(candidates)} survivors...")
    measured_all = serialize_many(
        tokenizer,
        [(item["instruction"], item["response"]) for item in candidates],
        progress_every=25000,
    )

    survivors: list[dict[str, Any]] = []
    for item, measured in zip(candidates, measured_all):
        scan_totals.append(measured.total_tokens)
        scan_responses.append(measured.response_tokens)
        if not measured.prompt_is_prefix:
            prefix_violations += 1
        if measured.total_tokens > args.max_tokens:
            log.bump("dropped_over_max_tokens")
            continue
        item["serialized"] = measured
        survivors.append(item)

    print(f"{len(survivors)} rows fit within {args.max_tokens} tokens")
    print("filter tally:")
    print(log.report())
    if prefix_violations:
        print(
            f"\nWARNING: {prefix_violations} rows failed the prompt-prefix check; "
            "label masking would be misaligned for those rows."
        )

    # Source balancing, on the rows that already passed every filter.
    by_source: dict[str, list[int]] = {}
    for index, item in enumerate(survivors):
        by_source.setdefault(item["source"], []).append(index)
    available = {name: len(indices) for name, indices in by_source.items()}

    if args.max_source_ratio < 1.0:
        needed = math.ceil(1.0 / args.max_source_ratio)
        if len(available) < needed:
            # Not a warning: solve_source_cap would "succeed" here and return one
            # row per source, silently turning the pool into a handful of rows.
            raise SystemExit(
                f"{len(available)} sources survived filtering, but "
                f"--max_source_ratio {args.max_source_ratio} needs at least "
                f"{needed} to be satisfiable. Raise --max_source_ratio (>= "
                f"{1 / len(available):.3f}) or relax the filters."
            )

    _, caps = solve_source_cap(available, args.max_source_ratio)
    for pattern in args.deprioritize_sources:
        for name in caps:
            if pattern.lower() in name.lower():
                caps[name] = max(1, caps[name] // 2)

    rng = random.Random(args.seed)
    chosen: list[int] = []
    for name in sorted(by_source):
        indices = sorted(by_source[name])
        cap = caps.get(name, len(indices))
        if len(indices) > cap:
            log.bump("dropped_source_balancing", len(indices) - cap)
            indices = rng.sample(indices, cap)
        chosen.extend(indices)

    if args.target_rows and len(chosen) > args.target_rows:
        log.bump("dropped_target_rows_sampling", len(chosen) - args.target_rows)
        chosen = rng.sample(chosen, args.target_rows)

    chosen.sort()
    records: list[dict[str, Any]] = []
    for position, index in enumerate(chosen):
        item = survivors[index]
        records.append(
            build_record(
                domain=DOMAIN,
                index=position,
                source=item["source"],
                question=item["instruction"],
                response=item["response"],
                serialized=item["serialized"],
                original_id=item["original_id"],
                gold_answer=item["answer"],
            )
        )

    write_jsonl(args.out_jsonl, records)
    print(f"\nwrote {len(records)} rows to {args.out_jsonl}")
    if args.out_dir:
        write_dataset_dir(args.out_dir, records)
        print(f"wrote trainable dataset to {args.out_dir}")

    stats = summarize(records)
    stats["budget_scan"] = budget_scan(scan_totals, scan_responses)
    stats["counts"] = log.as_dict()
    stats["source_balancing"] = {
        "max_source_ratio": args.max_source_ratio,
        "num_sources_raw": len(raw_source_counts),
        "num_sources_final": len({record["source"] for record in records}),
        "available_before_balancing": dict(
            sorted(available.items(), key=lambda item: -item[1])
        ),
        "caps": dict(sorted(caps.items(), key=lambda item: -item[1])),
        "deprioritized": list(args.deprioritize_sources),
    }
    stats["raw_source_counts"] = dict(
        sorted(raw_source_counts.items(), key=lambda item: -item[1])
    )
    stats["metadata"] = build_metadata(
        domain=DOMAIN,
        dataset_name=DATASET,
        dataset_revision=args.dataset_revision,
        tokenizer_path=args.tokenizer_path,
        max_tokens=args.max_tokens,
        seed=args.seed,
        raw_count=raw_count,
        filtered_count=len(survivors),
        final_count=len(records),
        filter_rules={
            "decontamination_order": "eval sources removed before any sampling",
            "eval_source_patterns": list(args.eval_source_patterns),
            "exclude_sources": list(args.exclude_sources),
            "eval_questions_jsonl": args.eval_questions_jsonl or "(none)",
            "language": args.language,
            "tag_handling": "<think>/<answer> markers stripped, text kept",
            "required_fields": ["instruction", "output"],
            "answer_field": (
                "required"
                if args.require_answer_field
                else "provenance only; falls back to the last \\boxed span"
            ),
            "dedup": "normalized-exact on instruction",
            "require_boxed": bool(args.require_boxed),
            "max_source_ratio": args.max_source_ratio,
            "target_rows": args.target_rows,
        },
        prompt_prefix_violations=prefix_violations,
        system_prompt=args.system_prompt,
    )
    write_stats(args.out_stats, stats)
    print(f"wrote stats to {args.out_stats}")
    print(
        f"\nnum_examples={stats['num_examples']} "
        f"total_tokens={stats['total_tokens']} "
        f"assistant_tokens={stats['total_assistant_tokens']}"
    )
    top = list(stats["by_source"].items())[:10]
    print("\ntop sources in final pool:")
    for name, bucket in top:
        print(f"  {name:<50} {bucket['count']:>7}  {bucket['percentage']:5.1f}%  "
              f"avg_tokens={bucket['average_tokens']:.0f}")


if __name__ == "__main__":
    main()
