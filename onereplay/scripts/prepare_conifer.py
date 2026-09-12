"""Convert Conifer into the {instruction, input, output} single-turn SFT schema.

Conifer is the IF (instruction-following) capability source for the continual
learning chain Base -> IF -> Math -> Third. It ships 13606 multi-turn
conversations as {prompt, messages, type}, while train.py's pipeline consumes
one save_to_disk directory of single-turn rows, so the conversations have to be
flattened. The two `type` subsets need different flattening rules:

  Easy-to-Hard (10302 convs, 1-5 turn pairs)
      Each turn restates the whole question with one more constraint bolted on,
      and -- verified on the real data -- every user turn is self-contained
      rather than referring back. Turn k is therefore a valid standalone sample
      whose difficulty is k, and flattening all turns yields ~35.6k rows that
      already span the difficulty ladder. That is the same order of magnitude
      as the Math stage's 45.8k, which keeps the two stages comparable.

  Process Feedback (3304 convs, 1-2 turn pairs)
      Its second user turn is a critique of the assistant's attempt ("No, the
      answer does not follow all of the constraints in the question. Here's
      why: ..."), not a new question, so enumerating turns the way Easy-to-Hard
      allows would produce rows whose "instruction" is a critique with no
      context. Instead each conversation contributes exactly one row pairing
      the FIRST user message with the LAST assistant message:

          4 messages -> (messages[0], messages[3])
          2 messages -> (messages[0], messages[1])

      For the 4-message case that answer is the one revised after the critique,
      which on 1950 of those 2176 conversations differs from the first attempt,
      so the pairing keeps the corrected answer and discards the rejected one
      along with the critique itself. The result is a clean single-turn row.

      Worth recording for later interpretation: these answers were written
      under criticism rather than in one shot, so their distribution is not
      identical to Easy-to-Hard's direct answers. --include_process_feedback 0
      drops the subset if that ever needs to be ruled out as a confound.

Two output modes, because the same corpus plays two roles in the experiment:

  --mode full       every Easy-to-Hard turn, for Stage 1 where the point is to
                    acquire IF from as much of the difficulty ladder as exists.
  --mode sampled    at most one row per conversation, for the replay / protection
                    pool, where re-training on five paraphrases of one seed
                    spends token budget without adding information.

Orthogonal to the mode, --format_constraints_only 1 restricts the pool to turns
whose constraints IFEval and IFBench can actually check (15.6% of rows). The
other 84.4% are content-coverage constraints, which are instruction-following
but are not what the benchmarks score. Having both pools available turns the
planned FLAN-vs-Conifer comparison into a three-point alignment gradient --
generic instructions, mixed constraints, verifiable constraints -- which is a
sharper test of "the more aligned the replay data, the better the protection"
than a binary contrast can be.

Difficulty metadata (conv_id / conifer_type / turn_index / n_turns) is written
alongside the three SFT columns. It survives into the save_to_disk directory but
never reaches the model: build_loader removes every column except input_ids,
labels and attention_mask before collating. Keeping it means a later
difficulty-aware sampler can select on it without re-deriving it from the
parquet.

Read the length report before picking --max_len. tokenizer_to_ids truncates
with full_input_ids[-max_length:], which keeps the END of the sequence, so an
over-long Conifer row loses the beginning of the user turn -- the constraints
themselves -- while keeping the answer. That failure is silent and it trains
the model to produce constrained-looking answers to unconstrained questions,
which is precisely the capability this stage is supposed to build.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

# Conifer's two subsets, which this script flattens by different rules.
EASY_TO_HARD = "Easy-to-Hard"
PROCESS_FEEDBACK = "Process Feedback"

REQUIRED_COLUMNS = ("messages", "type")

# Budgets the truncation table reports on, so --max_len can be read off
# measured truncation rather than inherited from Commonsense170k's 512.
CANDIDATE_MAX_LENS = (512, 768, 1024, 1536, 2048, 3072)

# Surface forms of the constraint families IFEval and IFBench can actually
# verify programmatically (length, casing, punctuation, list/markdown
# structure, placeholders, postscripts). Conifer's other constraints are
# content-coverage requirements -- "focus on Private Equity", "address
# Managing Directors" -- which are just as much instruction-following but
# which no automatic checker scores, so they contribute to IFEval only
# indirectly. Measured on the flattened Easy-to-Hard pool, this pattern
# matches 15.6% of rows (5544 of 35603), rising from 12.5% at difficulty 1 to
# 20.8% at difficulty 4.
#
# This is a lexical approximation of IFEval's 25 verifiable instruction types,
# not a reimplementation of its checkers: it over-matches words like "section"
# or "bold" used descriptively and misses paraphrased constraints. Good enough
# to build a deliberately IFEval-aligned subset, not good enough to report as
# a property of Conifer.
FORMAT_CONSTRAINT_PATTERN = re.compile(
    r"\b(bullet point|bulleted|numbered list|ordered list|in list format|json|markdown|"
    r"no more than \d+ word|at least \d+ word|exactly \d+|no more than \d+ sentence|"
    r"\d+ sentences|\d+ paragraphs|all lowercase|capital letters|uppercase|"
    r"without using any comma|no comma|quotation mark|wrap.{0,20}title|postscript|"
    r"highlight|placeholder|bold|section|word count|character limit|table format)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    """Parse the Conifer location, flattening mode, and output settings."""

    parser = argparse.ArgumentParser(
        description="Build a single-turn Conifer SFT pool for IF training or replay."
    )
    parser.add_argument(
        "--conifer_path",
        type=str,
        default="Conifer_data/data/train_sft-00000-of-00001.parquet",
        help="Local parquet file, a directory holding it, or a save_to_disk dir. "
        "Empty falls back to --conifer_repo (needs network).",
    )
    parser.add_argument("--conifer_repo", type=str, default="ConiferLM/Conifer")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="save_to_disk target; pass this as train.py --dataset_path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "sampled"],
        default="full",
        help="full keeps every Easy-to-Hard turn (Stage 1 acquisition pool). "
        "sampled keeps at most one row per conversation (replay pool), which "
        "avoids spending replay budget on paraphrases of the same seed.",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        choices=["hardest", "uniform", "easiest"],
        default="hardest",
        help="Which turn --mode sampled keeps from a multi-turn Easy-to-Hard "
        "conversation. hardest = the last turn, which carries every constraint "
        "the ladder accumulated; uniform = a random turn, which spreads the "
        "replay pool across difficulties instead of concentrating at the top.",
    )
    parser.add_argument(
        "--format_constraints_only",
        type=int,
        default=0,
        help="1 keeps only turns whose instruction carries a constraint of the "
        "kind IFEval and IFBench can verify programmatically (length, casing, "
        "punctuation, list structure, placeholders). That is 15.6% of the "
        "flattened pool, so it trades ~35.6k rows for ~5.5k that are far more "
        "aligned with how IF is scored. Applied before turn selection, so "
        "--mode sampled then keeps the hardest QUALIFYING turn per "
        "conversation rather than dropping conversations whose last turn "
        "happens to carry no format constraint.",
    )
    parser.add_argument(
        "--max_turns_per_conv",
        type=int,
        default=0,
        help="Cap on Easy-to-Hard turns kept per conversation under --mode full; "
        "0 = no cap. Turns are kept from the hardest end, so a cap of 2 keeps "
        "the two most constrained variants.",
    )
    parser.add_argument(
        "--include_process_feedback",
        type=int,
        default=1,
        help="1 (default) adds the Process Feedback subset as one row per "
        "conversation, pairing the first user message with the last assistant "
        "message: (messages[0], messages[3]) for a 4-message conversation, "
        "(messages[0], messages[1]) for a 2-message one. 0 drops the subset, "
        "leaving Easy-to-Hard's single construction rule -- useful for ruling "
        "out the revised-under-criticism answer distribution as a confound.",
    )
    parser.add_argument(
        "--drop_pf_unrevised",
        type=int,
        default=0,
        help="Only read when --include_process_feedback 1. 1 drops the 226 "
        "conversations whose final answer is byte-identical to the first "
        "attempt (the critique concluded the answer already complied).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="Budget the truncation report is headlined against; does not itself truncate.",
    )
    parser.add_argument(
        "--length_sample",
        type=int,
        default=5000,
        help="Rows measured for the length report; 0 = all.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Model dir for the length report. Empty skips the report, which also "
        "skips the only check on --max_len.",
    )
    parser.add_argument(
        "--jsonl_dir",
        type=str,
        default="",
        help="Where the manifest goes. Defaults to out_dir's parent, kept outside "
        "out_dir so save_to_disk owns that directory alone.",
    )
    return parser.parse_args()


def load_conifer(args: argparse.Namespace):
    """Load Conifer from a parquet file, a directory, a save_to_disk dir, or the hub."""

    from datasets import Dataset, load_dataset, load_from_disk

    if not args.conifer_path:
        loaded = load_dataset(args.conifer_repo)
        return loaded["train_sft"] if "train_sft" in loaded else loaded["train"]

    source = Path(args.conifer_path)
    if source.is_file():
        return Dataset.from_parquet(str(source))

    if not source.is_dir():
        raise SystemExit(f"--conifer_path does not exist: {source}")

    if (source / "dataset_info.json").exists() or (source / "dataset_dict.json").exists():
        loaded = load_from_disk(str(source))
        if hasattr(loaded, "keys"):
            for name in ("train_sft", "train"):
                if name in loaded:
                    return loaded[name]
            raise SystemExit(f"No train split among {sorted(loaded)} in {source}")
        return loaded

    matches = sorted(path for path in source.rglob("*.parquet") if path.is_file())
    if not matches:
        raise SystemExit(f"No .parquet files found under {source}")
    return Dataset.from_parquet([str(path) for path in matches])


def turn_pairs(messages: Any) -> list[tuple[str, str]]:
    """Split a message list into (user, assistant) pairs, ignoring a trailing user turn.

    Conifer alternates strictly user/assistant, but a defensive pass costs
    nothing and a role mismatch here would otherwise surface as a training row
    whose "answer" is a question.
    """

    pairs: list[tuple[str, str]] = []
    for index in range(0, len(messages) - 1, 2):
        user_turn, assistant_turn = messages[index], messages[index + 1]
        if user_turn.get("role") != "user" or assistant_turn.get("role") != "assistant":
            break
        pairs.append((user_turn.get("content") or "", assistant_turn.get("content") or ""))
    return pairs


def make_row(
    instruction: str,
    output: str,
    conv_id: int,
    conifer_type: str,
    turn_index: int,
    n_turns: int,
) -> dict[str, Any]:
    """Build one SFT row plus the difficulty metadata a later sampler selects on.

    input is always empty: Conifer's user turn is a self-contained instruction,
    and apply_train_template would otherwise glue an "Input:" header onto it.
    """

    return {
        "instruction": instruction.strip(),
        "input": "",
        "output": output.strip(),
        "conv_id": conv_id,
        "conifer_type": conifer_type,
        "turn_index": turn_index,
        "n_turns": n_turns,
    }


def flatten(dataset, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Flatten conversations into single-turn rows by subset-specific rules."""

    missing = [name for name in REQUIRED_COLUMNS if name not in dataset.column_names]
    if missing:
        raise SystemExit(
            f"Conifer columns {missing} not found; got {dataset.column_names}."
        )

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    stats = {
        "convs": 0,
        "convs_easy_to_hard": 0,
        "convs_process_feedback": 0,
        "convs_pf_excluded": 0,
        "convs_unknown_type": 0,
        "rows_easy_to_hard": 0,
        "rows_process_feedback": 0,
        "pf_four_message": 0,
        "pf_two_message": 0,
        "pf_other_length": 0,
        "pf_revised": 0,
        "pf_unrevised": 0,
        "pf_unrevised_dropped": 0,
        "dropped_empty": 0,
        "dropped_no_pairs": 0,
        "turns_without_format_constraint": 0,
        "convs_no_format_constraint": 0,
    }

    for conv_id, (messages, conifer_type) in enumerate(
        zip(dataset["messages"], dataset["type"])
    ):
        stats["convs"] += 1
        pairs = turn_pairs(messages)
        if not pairs:
            stats["dropped_no_pairs"] += 1
            continue
        n_turns = len(pairs)

        if conifer_type == PROCESS_FEEDBACK:
            stats["convs_process_feedback"] += 1
            if args.include_process_feedback != 1:
                stats["convs_pf_excluded"] += 1
                continue
            # Process Feedback is kept, but only as a clean single-turn pair.
            # A 4-message conversation is [question, attempt, critique, revision];
            # taking messages[0] and messages[3] drops the critique and the
            # rejected first attempt. A 2-message conversation is already a
            # single pair, so messages[0] and messages[1].
            n_messages = len(messages)
            if n_messages >= 4:
                if (
                    messages[0].get("role") != "user"
                    or messages[3].get("role") != "assistant"
                ):
                    stats["dropped_no_pairs"] += 1
                    continue
                first_question = messages[0].get("content") or ""
                final_answer = messages[3].get("content") or ""
                first_attempt = messages[1].get("content") or ""
                stats["pf_four_message"] += 1
            elif n_messages >= 2:
                first_question = messages[0].get("content") or ""
                final_answer = messages[1].get("content") or ""
                first_attempt = final_answer
                stats["pf_two_message"] += 1
            else:
                stats["pf_other_length"] += 1
                stats["dropped_no_pairs"] += 1
                continue
            if args.format_constraints_only == 1 and not FORMAT_CONSTRAINT_PATTERN.search(
                first_question
            ):
                stats["convs_no_format_constraint"] += 1
                continue
            if n_messages >= 4:
                if final_answer.strip() == first_attempt.strip():
                    stats["pf_unrevised"] += 1
                    if args.drop_pf_unrevised == 1:
                        stats["pf_unrevised_dropped"] += 1
                        continue
                else:
                    stats["pf_revised"] += 1
            selected = [(0, first_question, final_answer)]
        else:
            if conifer_type == EASY_TO_HARD:
                stats["convs_easy_to_hard"] += 1
            else:
                stats["convs_unknown_type"] += 1
            # Every turn is a standalone instruction at its own difficulty.
            indexed = [(index, user, answer) for index, (user, answer) in enumerate(pairs)]
            if args.format_constraints_only == 1:
                # Filter before selecting, so "hardest" means the hardest turn
                # that qualifies rather than a coin flip on whether the last
                # turn happens to carry a verifiable constraint.
                qualifying = [
                    item for item in indexed if FORMAT_CONSTRAINT_PATTERN.search(item[1])
                ]
                stats["turns_without_format_constraint"] += len(indexed) - len(qualifying)
                if not qualifying:
                    stats["convs_no_format_constraint"] += 1
                    continue
                indexed = qualifying
            if args.mode == "sampled":
                if args.sample_strategy == "hardest":
                    indexed = [indexed[-1]]
                elif args.sample_strategy == "easiest":
                    indexed = [indexed[0]]
                else:
                    indexed = [rng.choice(indexed)]
            elif args.max_turns_per_conv > 0:
                indexed = indexed[-args.max_turns_per_conv :]
            selected = indexed

        for turn_index, user, answer in selected:
            if not user.strip() or not answer.strip():
                stats["dropped_empty"] += 1
                continue
            rows.append(
                make_row(user, answer, conv_id, conifer_type, turn_index, n_turns)
            )
            if conifer_type == PROCESS_FEEDBACK:
                stats["rows_process_feedback"] += 1
            else:
                stats["rows_easy_to_hard"] += 1

    return rows, stats


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile of an already sorted list."""

    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def length_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Measure real training-time token lengths and the cost of each budget.

    Lengths come from the same apply_chat_template call the trainer makes, so
    the template's role markers and special tokens count against --max_len.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    sample = rows if args.length_sample <= 0 else rows[: args.length_sample]

    full_lengths: list[int] = []
    prompt_lengths: list[int] = []
    output_lengths: list[int] = []
    for row in sample:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        full_lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        prompt_lengths.append(
            len(tokenizer(row["instruction"], add_special_tokens=False)["input_ids"])
        )
        output_lengths.append(
            len(tokenizer(row["output"], add_special_tokens=False)["input_ids"])
        )
    full_lengths.sort()
    prompt_lengths.sort()
    output_lengths.sort()

    truncation = {}
    for budget in sorted({*CANDIDATE_MAX_LENS, args.max_len}):
        over = sum(length > budget for length in full_lengths)
        lost = sum(max(length - budget, 0) for length in full_lengths)
        # A row whose answer alone exceeds the budget loses its entire
        # instruction, not just the first few constraints.
        instruction_gone = sum(length >= budget for length in output_lengths)
        truncation[budget] = {
            "rows_truncated": over,
            "fraction_truncated": over / len(full_lengths),
            "tokens_dropped_mean_over_truncated": lost / over if over else 0.0,
            "rows_losing_whole_instruction": instruction_gone,
        }

    report = {
        "tokenizer": args.tokenizer_path,
        "rows_measured": len(sample),
        "full_length": {f"p{q}": percentile(full_lengths, q) for q in (50, 90, 95, 99)}
        | {"max": full_lengths[-1]},
        "instruction_length": {
            f"p{q}": percentile(prompt_lengths, q) for q in (50, 90, 99)
        }
        | {"max": prompt_lengths[-1]},
        "output_length": {f"p{q}": percentile(output_lengths, q) for q in (50, 90, 99)}
        | {"max": output_lengths[-1]},
        "truncation_by_max_len": truncation,
    }

    print("==== Conifer training-length report ====")
    print(
        f"n={len(sample)}  full: P50={report['full_length']['p50']} "
        f"P90={report['full_length']['p90']} P95={report['full_length']['p95']} "
        f"P99={report['full_length']['p99']} max={report['full_length']['max']}"
    )
    print(
        f"instruction only: P50={report['instruction_length']['p50']} "
        f"P90={report['instruction_length']['p90']} max={report['instruction_length']['max']}"
    )
    print(
        f"answer only: P50={report['output_length']['p50']} "
        f"P90={report['output_length']['p90']} max={report['output_length']['max']}"
    )
    print(
        f"{'max_len':>10}{'truncated':>12}{'share':>9}"
        f"{'mean tokens lost':>20}{'instruction gone':>19}"
    )
    for budget, stats in truncation.items():
        marker = "  <- --max_len" if budget == args.max_len else ""
        print(
            f"{budget:>10}{stats['rows_truncated']:>12}"
            f"{stats['fraction_truncated']:>8.1%}"
            f"{stats['tokens_dropped_mean_over_truncated']:>20.0f}"
            f"{stats['rows_losing_whole_instruction']:>19}{marker}"
        )
    print(
        "Truncation keeps the END of the sequence, so a truncated Conifer row "
        "loses the start of its instruction -- the constraints. The last column "
        "counts rows whose answer alone fills the budget, leaving no instruction "
        "at all. Pick the smallest budget that keeps both near zero."
    )
    return report


def main() -> None:
    """Write the flattened pool as save_to_disk plus a manifest."""

    args = parse_args()
    dataset = load_conifer(args)
    print(f"loaded {len(dataset)} Conifer conversations from {args.conifer_path or args.conifer_repo}")

    rows, stats = flatten(dataset, args)
    if not rows:
        raise SystemExit("Conifer converted to 0 usable rows; check the column contents.")

    print(
        f"mode={args.mode}"
        + (f" strategy={args.sample_strategy}" if args.mode == "sampled" else "")
        + f" -> {len(rows)} single-turn rows"
    )
    print(
        f"  Easy-to-Hard : {stats['convs_easy_to_hard']} convs -> "
        f"{stats['rows_easy_to_hard']} rows"
    )
    if args.include_process_feedback == 1:
        print(
            f"  Process Feedback: {stats['convs_process_feedback']} convs -> "
            f"{stats['rows_process_feedback']} rows "
            f"(4-message={stats['pf_four_message']} as messages[0]+[3], "
            f"2-message={stats['pf_two_message']} as messages[0]+[1]; "
            f"revised={stats['pf_revised']}, unrevised={stats['pf_unrevised']}, "
            f"dropped={stats['pf_unrevised_dropped']})"
        )
    else:
        print(
            f"  Process Feedback: {stats['convs_pf_excluded']} convs excluded "
            "(critique turns do not survive flattening; pass "
            "--include_process_feedback 1 to include them)"
        )
    if stats["convs_unknown_type"]:
        print(
            f"  unknown type : {stats['convs_unknown_type']} convs, "
            "flattened as Easy-to-Hard"
        )
    if args.format_constraints_only == 1:
        print(
            f"  format filter: dropped {stats['turns_without_format_constraint']} turns "
            "carrying no programmatically verifiable constraint, and "
            f"{stats['convs_no_format_constraint']} conversations that had none at any turn"
        )
    if stats["dropped_empty"] or stats["dropped_no_pairs"]:
        print(
            f"  dropped: {stats['dropped_empty']} empty-field turns, "
            f"{stats['dropped_no_pairs']} conversations with no usable pair"
        )

    # The ladder only exists in Easy-to-Hard; every Process Feedback row sits at
    # turn_index 0 by construction, so folding them in would inflate level 0
    # with rows that carry no difficulty information.
    difficulty_histogram: dict[int, int] = {}
    for row in rows:
        if row["conifer_type"] == PROCESS_FEEDBACK:
            continue
        difficulty_histogram[row["turn_index"]] = (
            difficulty_histogram.get(row["turn_index"], 0) + 1
        )
    print(
        "  difficulty (turn_index) histogram, Easy-to-Hard only: "
        + ", ".join(f"{k}={difficulty_histogram[k]}" for k in sorted(difficulty_histogram))
    )

    from datasets import Dataset, DatasetDict

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(out_dir))
    print(f"wrote pool to {out_dir}")

    manifest: dict[str, Any] = {
        "source": args.conifer_path or args.conifer_repo,
        "mode": args.mode,
        "sample_strategy": args.sample_strategy if args.mode == "sampled" else "",
        "max_turns_per_conv": args.max_turns_per_conv,
        "format_constraints_only": args.format_constraints_only,
        "include_process_feedback": args.include_process_feedback,
        "drop_pf_unrevised": args.drop_pf_unrevised,
        "subsets": (
            "Easy-to-Hard + Process Feedback"
            if args.include_process_feedback == 1
            else "Easy-to-Hard only"
        ),
        "seed": args.seed,
        "train": {"path": str(out_dir), "num_rows": len(rows)},
        "counts": stats,
        "difficulty_histogram": {str(k): v for k, v in sorted(difficulty_histogram.items())},
        "note": "train.py splits validation out of this pool with --val_fraction. "
        "turn_index is the Easy-to-Hard difficulty level; Process Feedback rows "
        "are always 0 since they pair the first question with the final answer. "
        "difficulty_histogram counts Easy-to-Hard rows only. build_loader drops "
        "every column except the tokenized three before collating.",
    }
    if args.tokenizer_path:
        manifest["length"] = length_report(rows, args)
    else:
        print("skipping length report: no --tokenizer_path")

    jsonl_dir = Path(args.jsonl_dir) if args.jsonl_dir else out_dir.parent
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = jsonl_dir / f"{out_dir.name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
