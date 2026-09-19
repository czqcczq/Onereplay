"""Convert the TRACE benchmark into the {instruction, input, output} SFT schema.

TRACE (Wang et al., "TRACE: A Comprehensive Benchmark for Continual Learning in
Large Language Models") ships nine directories of ``[{"prompt", "answer"}]``
JSON. Eight of them form the official continual-learning sequence; this script
turns each into one save_to_disk pool that ``train.py --dataset_path`` loads
directly, plus one held-out JSONL that the trace_* metrics score against.

Two deliberate departures from the official pipeline, both forced by what the
experiment is measuring.

**Chat template instead of raw concatenation.** TRACE builds ``BOS + prompt +
answer + EOS`` with no chat template at all (utils/data/data_collator.py). We
render through ``apply_train_template`` instead, because the point of this run
is whether GSM8K / IFEval collapse after the eight tasks, and those are decoded
through the model's chat template. Training on raw concatenation destroys the
chat format itself, which would zero every downstream benchmark for a reason
that has nothing to do with forgetting. Same template on both sides means a
base-vs-stage-8 gap cannot be an artifact of prompt formatting. The cost is that
absolute TRACE scores no longer line up with the paper's tables; every
conclusion here is a within-pipeline comparison, so that does not matter.

**Truncation happens here, on the task body, not in the tokenizer.**
``tokenizer_to_ids`` truncates with ``full_input_ids[-max_length:]`` -- it keeps
the *last* max_length tokens. On raw concatenation that costs you the head of a
transcript. On a chat-templated sequence it eats ``<|begin_of_text|>`` and the
``user`` header, handing the model a decapitated example. MeetingBank makes this
unavoidable rather than theoretical: its prompts run to 373k characters (p90 is
40k). So this script shrinks the body until the *rendered* sequence fits the
budget, and training never truncates anything.

Every TRACE prompt is ``instruction_prefix + body + answer_cue``, e.g.

    Write a summary of the following meeting transcripts.
    Meeting transcripts:
    <...transcript...>
    Summary:

Both affixes have to survive: the prefix is the only statement of the task, and
the cue is what tells the model to start answering. Only the body between them
is cut. Note that TRACE's own left-truncation deletes its instruction prefix on
exactly the rows that are long enough to need it -- that is a defect of the
implementation, not a design choice, so it is not reproduced.

Py150 is the exception and has no instruction: its prompts start straight into
code (verified -- 1875 distinct first lines in 2000 rows, versus 1 for every
other task). It is also the one task where the *end* of the body is what the
answer continues from, so it keeps its tail and drops its head.

Lima is not part of the eight. TRACE uses it only as a replay buffer, and its
test split ships 300 rows whose answers are all empty strings, so it cannot be
scored. ``--include_lima`` builds a train pool for it and nothing else.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.core.chat_policy import configure_system_prompt  # noqa: E402
from onereplay.data.chat import apply_train_template  # noqa: E402
from onereplay.scripts.domain_sft.common import (  # noqa: E402
    FilterLog,
    SerializedExample,
    build_record,
    length_summary,
    summarize,
    write_dataset_dir,
    write_jsonl,
    write_stats,
)


@dataclass(frozen=True)
class TaskSpec:
    """One TRACE task: where it lives, how its prompt is built, how to cut it.

    ``prefix`` / ``suffix`` are asserted against the data rather than derived
    from it. They were measured over 1500 rows per split and are identical in
    train and test; encoding them as constants turns a silent upstream change
    into a loud one. A row that fails the assertion is still usable -- it just
    gets treated as all-body, so nothing is specially protected.
    """

    key: str
    directory: str
    prefix: str
    suffix: str
    keep: str  # "head" keeps the start of the body, "tail" keeps the end

    @property
    def dataset_prefix(self) -> str:
        return f"trace_{self.key}"


# Declaration order is the official training order (README and every
# scripts/train_seq_*.sh): C-STANCE, FOMC, MeetingBank, Py150, ScienceQA,
# NumGLUE-cm, NumGLUE-ds, 20Minuten. The epoch schedule 5,3,7,5,3,5,5,7 is
# positional against this list, so reordering it silently rewires the schedule.
TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        key="cstance",
        directory="C-STANCE",
        prefix="判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n文本：\n",
        suffix="\n态度：",
        keep="head",
    ),
    TaskSpec(
        key="fomc",
        directory="FOMC",
        prefix=(
            "What is the monetary policy stance for the following text? "
            "A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\nText:\n"
        ),
        suffix="\nStance:",
        keep="head",
    ),
    TaskSpec(
        key="meetingbank",
        directory="MeetingBank",
        prefix="Write a summary of the following meeting transcripts.\nMeeting transcripts:\n",
        suffix="\nSummary:",
        keep="head",
    ),
    # No instruction prefix, and the answer continues from the final line, so
    # the tail is the half worth keeping. Its prompts do end in " <EOL>", but
    # that is preserved for free by keeping the tail.
    TaskSpec(key="py150", directory="Py150", prefix="", suffix="", keep="tail"),
    TaskSpec(
        key="scienceqa",
        directory="ScienceQA",
        prefix="Choose an answer for the following question and give your reasons.\n\nQuestion:\n",
        suffix="\n\nAnswer:",
        keep="head",
    ),
    TaskSpec(
        key="numglue_cm",
        directory="NumGLUE-cm",
        prefix="Solve the following math problem.\nQuestion:\n",
        suffix="\nAnswer:",
        keep="head",
    ),
    TaskSpec(
        key="numglue_ds",
        directory="NumGLUE-ds",
        prefix="Solve the following math problem.\nQuestion:\n",
        suffix="\nAnswer:",
        keep="head",
    ),
    TaskSpec(
        key="20minuten",
        directory="20Minuten",
        prefix="Provide a simplified version of the following paragraph in German.\n\nParagraph:\n",
        suffix="\n\nSimplification:",
        keep="head",
    ),
)

LIMA_SPEC = TaskSpec(key="lima", directory="Lima", prefix="", suffix="", keep="head")

SPEC_BY_KEY = {spec.key: spec for spec in TASK_SPECS}

# Official per-task epoch counts, positional against TASK_SPECS. Recorded in the
# manifest so the PBS schedule and the data provenance cannot drift apart.
OFFICIAL_EPOCHS = (5, 3, 7, 5, 3, 5, 5, 7)

# Extra tokens the shrink loop cuts beyond the measured overflow. Slicing body
# token ids and decoding back to text does not round-trip exactly, so the
# re-rendered sequence can land a token or two above the arithmetic prediction.
SHRINK_MARGIN = 8

MAX_SHRINK_ROUNDS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TRACE continual-learning SFT pools and held-out test files."
    )
    parser.add_argument(
        "--trace_dir",
        type=str,
        required=True,
        help="Directory holding the nine task folders, i.e. LLM-CL-Benchmark_5000.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="The model that will be trained. Lengths are meaningless under any other.",
    )
    parser.add_argument("--out_root", type=str, default="data/processed")
    parser.add_argument("--stats_root", type=str, default="data/stats")
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated subset of task keys. Empty builds all eight.",
    )
    # TRACE's own max_prompt_len. Applied to the rendered prompt half, so the
    # chat scaffold counts against it -- a handful of tokens tighter than TRACE,
    # which measured raw text.
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    # Must equal the --max_len training is launched with. Rows are shrunk until
    # the full rendered sequence fits, which is what keeps tokenizer_to_ids from
    # ever reaching its left-truncation branch.
    parser.add_argument("--max_len", type=int, default=2048)
    # Must equal the --system_prompt training is launched with: it changes the
    # rendered text, hence every length measured here.
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--include_lima", type=int, default=0)
    parser.add_argument(
        "--build_mix",
        type=int,
        default=1,
        help="Also write trace_mix_train, the eight pools shuffled into one. Not used "
        "by the sequential run; it is the pool for treating TRACE as a single "
        "ordinary training set.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug: keep only the first N rows of each split.",
    )
    parser.add_argument(
        "--report_only",
        type=int,
        default=0,
        help="Measure and print, write nothing. Use it to pick --max_len.",
    )
    # A row is unfittable when its answer alone leaves no room for a body, which
    # means --max_len is too small for this task. Dropping such rows quietly
    # would shrink a training set by an amount that depends on the budget, so
    # the default is to stop and make you raise --max_len instead.
    parser.add_argument("--allow_unfittable_drop", type=int, default=0)
    return parser.parse_args()


def load_split(
    trace_dir: Path, spec: TaskSpec, split: str, limit: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Read one task/split, dropping rows that cannot be answered either way.

    Two filters. Missing halves are obvious. The interesting one is an empty
    *body*: MeetingBank ships one train row and one test row whose prompt is the
    instruction and the "Summary:" cue with no transcript between them, yet
    carries a real gold summary. Training on those teaches the model to invent a
    summary from nothing, and scoring on them is a guaranteed zero for every arm,
    so both would add noise to the forgetting signal without adding information.
    """

    path = trace_dir / spec.directory / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{spec.directory}/{split}.json not found under {trace_dir}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a JSON list")

    kept: list[dict[str, str]] = []
    dropped = {"empty_half": 0, "empty_body": 0}
    for row in rows:
        prompt = str(row.get("prompt") or "")
        answer = str(row.get("answer") or "")
        if not prompt.strip() or not answer.strip():
            dropped["empty_half"] += 1
            continue
        _prefix, body, _suffix, matched = split_affixes(prompt, spec)
        if matched and not body.strip():
            dropped["empty_body"] += 1
            continue
        kept.append({"prompt": prompt, "answer": answer})
        if limit > 0 and len(kept) >= limit:
            break
    return kept, dropped


def split_affixes(prompt: str, spec: TaskSpec) -> tuple[str, str, str, bool]:
    """Peel the instruction prefix and answer cue off a prompt.

    Returns (prefix, body, suffix, matched). A prompt that does not carry the
    declared affixes is returned whole as the body, so truncation still works --
    it just has nothing to protect.
    """

    if spec.keep == "tail" or (not spec.prefix and not spec.suffix):
        return "", prompt, "", True

    matched = prompt.startswith(spec.prefix) and prompt.endswith(spec.suffix)
    if not matched:
        return "", prompt, "", False

    body = prompt[len(spec.prefix) : len(prompt) - len(spec.suffix)] if spec.suffix else prompt[len(spec.prefix) :]
    return spec.prefix, body, spec.suffix, True


@dataclass
class Rendered:
    """One row after truncation, with the measurements that justified it."""

    question: str
    body: str
    prompt_tokens: int
    total_tokens: int
    rounds: int
    body_tokens_before: int
    body_tokens_after: int
    prompt_budget: int
    fits: bool


def fit_row(
    tokenizer,
    spec: TaskSpec,
    prompt: str,
    answer: str,
    max_prompt_tokens: int,
    max_len: int,
) -> Rendered:
    """Shrink the body until the rendered sequence fits both budgets.

    Two budgets, because they bind on different rows. ``max_prompt_tokens`` is
    TRACE's prompt cap and binds on MeetingBank and Py150. ``max_len`` is the
    training window and binds where a short prompt meets a long answer, which in
    this data means only the longest ScienceQA reasonings. Enforcing both by
    construction is what guarantees ``tokenizer_to_ids`` never truncates, and
    therefore that the chat scaffold always survives.

    The loop re-measures instead of trusting arithmetic: the body is cut in token
    space and then decoded back to text, and that round trip can shift the
    re-rendered length slightly.
    """

    prefix, body, suffix, _matched = split_affixes(prompt, spec)
    body_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
    body_tokens_before = len(body_ids)
    keep = body_tokens_before

    for round_index in range(MAX_SHRINK_ROUNDS):
        sliced = body_ids[:keep] if spec.keep == "head" else body_ids[len(body_ids) - keep :]
        current_body = tokenizer.decode(sliced, skip_special_tokens=False) if keep else ""
        question = f"{prefix}{current_body}{suffix}"
        full_text, prompt_text = apply_train_template(tokenizer, question, "", answer)
        prompt_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        total_tokens = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])

        prompt_budget = min(max_prompt_tokens, max_len - (total_tokens - prompt_tokens))
        over = max(prompt_tokens - max_prompt_tokens, total_tokens - max_len)
        if over <= 0 or keep == 0:
            return Rendered(
                question=question,
                body=current_body,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                rounds=round_index,
                body_tokens_before=body_tokens_before,
                body_tokens_after=keep,
                prompt_budget=prompt_budget,
                fits=over <= 0,
            )
        keep = max(0, keep - over - SHRINK_MARGIN)

    return Rendered(
        question=question,
        body=current_body,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        rounds=MAX_SHRINK_ROUNDS,
        body_tokens_before=body_tokens_before,
        body_tokens_after=keep,
        prompt_budget=min(max_prompt_tokens, max_len),
        fits=False,
    )


def build_task(
    tokenizer,
    trace_dir: Path,
    spec: TaskSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build one task's train records, test rows, and measurement block."""

    log = FilterLog()
    result: dict[str, Any] = {"spec": spec}

    for split in ("train", "test"):
        rows, source_dropped = load_split(trace_dir, spec, split, args.limit)
        log.bump(f"{split}_kept", len(rows))
        for reason, count in source_dropped.items():
            if count:
                log.bump(f"{split}_dropped_{reason}", count)

        records: list[dict[str, Any]] = []
        test_rows: list[dict[str, Any]] = []
        truncated = 0
        answer_bound = 0
        affix_violations = 0
        dropped_unfittable = 0
        body_tokens_dropped = 0
        totals: list[int] = []
        prompts: list[int] = []

        for index, row in enumerate(rows):
            _prefix, _body, _suffix, matched = split_affixes(row["prompt"], spec)
            if not matched:
                affix_violations += 1

            fitted = fit_row(
                tokenizer,
                spec,
                row["prompt"],
                row["answer"],
                args.max_prompt_tokens,
                args.max_len,
            )
            if not fitted.fits:
                dropped_unfittable += 1
                log.bump(f"{split}_dropped_unfittable")
                continue

            if fitted.body_tokens_after < fitted.body_tokens_before:
                truncated += 1
                body_tokens_dropped += fitted.body_tokens_before - fitted.body_tokens_after
            # The prompt cap did not bind; the training window did. Tracked
            # separately because it is the one place a test prompt's length
            # depends on its gold answer, and that is worth being able to audit.
            if fitted.prompt_budget < args.max_prompt_tokens:
                answer_bound += 1

            totals.append(fitted.total_tokens)
            prompts.append(fitted.prompt_tokens)

            if split == "train":
                serialized = SerializedExample(
                    full_text="",
                    prompt_text="",
                    total_tokens=fitted.total_tokens,
                    input_tokens=fitted.prompt_tokens,
                    response_tokens=fitted.total_tokens - fitted.prompt_tokens,
                    prompt_is_prefix=True,
                )
                records.append(
                    build_record(
                        domain=spec.dataset_prefix,
                        index=index,
                        source=spec.directory,
                        question=fitted.question,
                        response=row["answer"],
                        serialized=serialized,
                        extra={"trace_task": spec.directory},
                    )
                )
            else:
                test_rows.append(
                    {
                        "id": f"{spec.dataset_prefix}-test-{index:06d}",
                        "task": spec.directory,
                        "instruction": fitted.question,
                        "input": "",
                        "output": row["answer"],
                        # SARI needs the source text it simplified, and the
                        # Py150 scorer wants the code prefix without the cue.
                        # Carrying the body spares every metric from having to
                        # re-peel the affixes and get it subtly different.
                        "source_text": fitted.body,
                    }
                )

        block = {
            "rows_in": len(rows),
            "rows_out": len(records) if split == "train" else len(test_rows),
            "dropped_at_load": source_dropped,
            "rows_truncated": truncated,
            "truncated_fraction": truncated / len(rows) if rows else 0.0,
            "body_tokens_dropped": body_tokens_dropped,
            "rows_bound_by_max_len": answer_bound,
            "affix_violations": affix_violations,
            "dropped_unfittable": dropped_unfittable,
            "total_tokens_distribution": length_summary(totals),
            "prompt_tokens_distribution": length_summary(prompts),
        }
        result[split] = {"records": records, "test_rows": test_rows, "block": block}

    result["log"] = log.as_dict()
    return result


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    out_root = Path(args.out_root)
    stats_root = Path(args.stats_root)

    selected: list[TaskSpec]
    if args.tasks.strip():
        keys = [key.strip() for key in args.tasks.split(",") if key.strip()]
        unknown = [key for key in keys if key not in SPEC_BY_KEY]
        if unknown:
            raise SystemExit(f"unknown task keys {unknown}; choose from {sorted(SPEC_BY_KEY)}")
        selected = [SPEC_BY_KEY[key] for key in keys]
    else:
        selected = list(TASK_SPECS)
    if args.include_lima == 1:
        selected.append(LIMA_SPEC)

    from transformers import AutoTokenizer

    # Same global the trainer sets from --system_prompt, read by
    # apply_train_template. Setting it wrong raises nothing; it just silently
    # adds or drops a system block and invalidates every length below.
    configure_system_prompt(args.system_prompt)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"tokenizer: {args.tokenizer_path}")
    print(f"system_prompt: {args.system_prompt!r}")
    print(f"budgets: max_prompt_tokens={args.max_prompt_tokens} max_len={args.max_len}")
    print()

    mix_records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "trace_dir": str(trace_dir),
        "tokenizer": args.tokenizer_path,
        "system_prompt": args.system_prompt,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_len": args.max_len,
        "seed": args.seed,
        "serialization": "onereplay.data.chat.apply_train_template (system/user/assistant)",
        "length_rule": (
            "shrink the task body until the rendered prompt fits max_prompt_tokens "
            "and the rendered prompt+answer fits max_len; instruction prefix and "
            "answer cue are always preserved"
        ),
        "training_order": [spec.key for spec in TASK_SPECS],
        "official_epochs": list(OFFICIAL_EPOCHS),
        "tasks": {},
    }

    for spec in selected:
        built = build_task(tokenizer, trace_dir, spec, args)
        train_records = built["train"]["records"]
        test_rows = built["test"]["test_rows"]
        train_block = built["train"]["block"]
        test_block = built["test"]["block"]

        print(f"=== {spec.directory} ({spec.key}) keep={spec.keep}")
        for name, block in (("train", train_block), ("test", test_block)):
            print(
                f"  {name}: {block['rows_out']}/{block['rows_in']} rows, "
                f"truncated {block['rows_truncated']} "
                f"({block['truncated_fraction'] * 100:.1f}%), "
                f"max_len-bound {block['rows_bound_by_max_len']}, "
                f"affix violations {block['affix_violations']}, "
                f"dropped {block['dropped_unfittable']}"
            )
            distribution = block["total_tokens_distribution"]
            print(
                f"    total tokens: median {distribution['median']} "
                f"p95 {distribution['p95']} max {distribution['max']}"
            )
            if block["dropped_unfittable"] and args.allow_unfittable_drop == 0:
                raise SystemExit(
                    f"\n{spec.directory}/{name}: {block['dropped_unfittable']} rows do not fit "
                    f"--max_len {args.max_len} even with an empty body, meaning their answer "
                    f"alone exceeds the window.\nRaise --max_len (and pass the same value to "
                    f"train.py --max_len), or pass --allow_unfittable_drop 1 to drop them on "
                    f"purpose."
                )

        stats = {
            "task": spec.directory,
            "key": spec.key,
            "keep": spec.keep,
            "train": train_block,
            "test": test_block,
            "train_pool": summarize(train_records) if train_records else {},
            "metadata": {
                key: value for key, value in manifest.items() if key != "tasks"
            },
        }
        manifest["tasks"][spec.key] = {
            "directory": spec.directory,
            "train_rows": train_block["rows_out"],
            "test_rows": test_block["rows_out"],
            "train_truncated_fraction": train_block["truncated_fraction"],
            "test_truncated_fraction": test_block["truncated_fraction"],
        }

        if args.report_only == 1:
            continue

        dataset_dir = out_root / f"{spec.dataset_prefix}_train"
        write_dataset_dir(dataset_dir, train_records)
        write_jsonl(out_root / f"{spec.dataset_prefix}_train.jsonl", train_records)
        test_path = out_root / f"{spec.dataset_prefix}_test.jsonl"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with test_path.open("w", encoding="utf-8") as sink:
            for row in test_rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_stats(stats_root / f"{spec.dataset_prefix}_stats.json", stats)
        print(f"  wrote {dataset_dir} and {test_path}")

        if spec.key in SPEC_BY_KEY:
            mix_records.extend(train_records)

    if args.build_mix == 1 and args.report_only == 0 and mix_records:
        random.Random(args.seed).shuffle(mix_records)
        mix_dir = out_root / "trace_mix_train"
        write_dataset_dir(mix_dir, mix_records)
        write_jsonl(out_root / "trace_mix_train.jsonl", mix_records)
        write_stats(
            stats_root / "trace_mix_stats.json",
            {
                "task": "trace_mix",
                "key": "mix",
                "num_examples": len(mix_records),
                "train_pool": summarize(mix_records),
                "metadata": {key: value for key, value in manifest.items() if key != "tasks"},
            },
        )
        print(f"\nmix pool: {len(mix_records)} rows -> {mix_dir}")

    if args.report_only == 0:
        write_stats(stats_root / "trace_manifest.json", manifest)
        print(f"manifest -> {stats_root / 'trace_manifest.json'}")


if __name__ == "__main__":
    main()
