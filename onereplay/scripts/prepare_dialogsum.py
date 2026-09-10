"""Convert DialogSum into the {instruction, input, output} SFT schema.

Why a conversion step instead of a loader branch: train.py hands every row of
whatever save_to_disk directory --dataset_path names to build_sft_tokenize_fn,
which reads exactly three columns. DialogSum ships {id, dialogue, summary,
topic}, so all that stands between it and the existing pipeline is a rename.
Renaming here rather than adding a second loader keeps tokenization, label
masking, replay mixing and the OPD loader on the one code path Commonsense170k
already exercises, which is also what makes the two training sets comparable.

Split handling: only the official train split becomes the training pool, and
train.py carves its own validation slice out of it via --val_fraction. That is
deliberate. Commonsense170k is scored the same way (a random slice of train),
and held-out loss only means the same thing across the two training sets if
both validation sets were drawn by the same in-distribution procedure. The
official validation and test splits are written out as JSONL instead, unused
for now, waiting for a generation metric that needs a canonical test set.

One caveat for that future metric: the HuggingFace test split holds 1500 rows
because the official 500 test dialogues each carry three reference summaries,
flattened into one row per reference. ROUGE against a single flattened row is
not the number the DialogSum paper reports -- regroup by `id` and score
multi-reference.

Length is the reason this script prints a report. DialogSum dialogues average
131 words, and tokenizer_to_ids truncates from the left, so an over-long row
loses the opening turns of the conversation, which is the part a summary needs
most. Pick --max_len from the truncation table below, not from the 512 that
Commonsense170k's short prompts got away with.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_INSTRUCTION = "Summarize the following dialogue."

REQUIRED_COLUMNS = ("dialogue", "summary")

# Candidate budgets the truncation table reports on, so --max_len can be chosen
# from measured truncation rates rather than guessed.
CANDIDATE_MAX_LENS = (512, 768, 1024, 1536, 2048)

# Suffix -> datasets loader name, in the order preferred when a directory holds
# several formats. DialogSum is distributed as CSV on the hub and JSONL on
# GitHub, and the hub's auto-conversion serves parquet.
DATA_SUFFIXES = (
    (".parquet", "parquet"),
    (".jsonl", "json"),
    (".json", "json"),
    (".csv", "csv"),
)

# Split name -> filename substrings that identify it. Checked in this order, so
# "validation" wins before the shorter "val" alias can mis-bind. `holdout`
# matches nothing on purpose: it has no summary column and cannot be trained on.
SPLIT_PATTERNS = {
    "train": ("train",),
    "validation": ("validation", "dev", "val"),
    "test": ("test",),
}

METADATA_NAMES = frozenset(
    {"dataset_info.json", "state.json", "dataset_dict.json", "config.json"}
)


def parse_args() -> argparse.Namespace:
    """Parse the DialogSum location, prompt wording, and output settings."""

    parser = argparse.ArgumentParser(description="Build the DialogSum SFT training pool.")
    parser.add_argument(
        "--dialogsum_path",
        type=str,
        default="",
        help="Local save_to_disk dir, split-per-file dir, or single file. "
        "Empty falls back to --dialogsum_repo (needs network).",
    )
    parser.add_argument("--dialogsum_repo", type=str, default="knkarthick/dialogsum")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="save_to_disk target for the train pool; pass this as train.py --dataset_path.",
    )
    parser.add_argument(
        "--jsonl_dir",
        type=str,
        default="",
        help="Where the untouched validation/test splits go. Defaults to out_dir's parent, "
        "kept out of out_dir so save_to_disk owns that directory alone.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Fixed user instruction prepended to every dialogue. One constant string for "
        "the whole set, so the model learns the task rather than instruction variety.",
    )
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
        help="Rows measured for the length report; 0 = all. Tokenizing 12k dialogues is "
        "cheap, but the percentiles are stable well before that.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=(
            str(Path(os.environ["MODEL_DIR"]) / os.environ.get("MODEL_NAME", "Qwen3-8B"))
            if os.environ.get("MODEL_DIR")
            else ""
        ),
        help="Model dir for the length report; defaults to $MODEL_DIR/$MODEL_NAME when set. "
        "Empty skips the report, which also skips the only check on --max_len.",
    )
    return parser.parse_args()


def find_data_files(directory: Path) -> tuple[str, list[str]]:
    """Locate the data files inside a plain directory of downloaded files."""

    for suffix, fmt in DATA_SUFFIXES:
        matches = sorted(
            path
            for path in directory.rglob(f"*{suffix}")
            if path.is_file() and path.name not in METADATA_NAMES
        )
        if matches:
            return fmt, [str(path) for path in matches]
    raise ValueError(
        f"No .parquet/.jsonl/.json/.csv data files found under {directory}. "
        "Point --dialogsum_path at the file itself or a directory containing it."
    )


def load_local(fmt: str, files: list[str]):
    """Read local data files without any Hub round-trip.

    load_dataset("csv", ...) resolves the packaged builder remotely first, which
    hangs on an offline compute node. Dataset.from_csv / from_json /
    from_parquet instantiate that builder directly, so they stay local.
    """

    from datasets import Dataset

    argument = files if len(files) > 1 else files[0]
    if fmt == "parquet":
        return Dataset.from_parquet(argument)
    if fmt == "csv":
        return Dataset.from_csv(argument)
    return Dataset.from_json(argument)


def file_format(path: Path) -> str:
    """Map a data file's suffix onto a datasets loader name."""

    fmt = dict(DATA_SUFFIXES).get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"Unsupported DialogSum file type: {path}")
    return fmt


def load_split_files(directory: Path) -> dict[str, Any]:
    """Bind each split to the file whose name claims it.

    A file matching no split is skipped rather than folded into train: DialogSum
    also ships a `holdout` file with no summary column, and silently training on
    it would fail deep inside tokenization instead of here.
    """

    fmt, files = find_data_files(directory)
    splits: dict[str, Any] = {}
    for name, aliases in SPLIT_PATTERNS.items():
        matched = [
            path
            for path in files
            if any(alias in Path(path).stem.lower() for alias in aliases)
        ]
        if matched:
            splits[name] = load_local(fmt, sorted(matched))
    if not splits:
        raise ValueError(
            f"None of the {len(files)} {fmt} file(s) under {directory} carry a "
            f"train/validation/test name; rename them or pass the train file directly."
        )
    return splits


def load_dialogsum(args: argparse.Namespace) -> dict[str, Any]:
    """Load DialogSum from a save_to_disk dir, a plain dir, a file, or the hub."""

    from datasets import load_dataset, load_from_disk

    if not args.dialogsum_path:
        return dict(load_dataset(args.dialogsum_repo))

    source = Path(args.dialogsum_path)
    if source.is_dir():
        if (source / "dataset_info.json").exists() or (source / "dataset_dict.json").exists():
            loaded = load_from_disk(str(source))
            return dict(loaded) if hasattr(loaded, "keys") else {"train": loaded}
        print(f"loading split-per-file directory {source}")
        return load_split_files(source)

    if not source.is_file():
        raise ValueError(f"--dialogsum_path does not exist: {source}")
    return {"train": load_local(file_format(source), [str(source)])}


def to_sft_rows(dataset, instruction: str) -> tuple[list[dict[str, str]], int]:
    """Rename DialogSum's columns onto the SFT schema, dropping empty rows.

    instruction / input rather than one concatenated prompt: apply_train_template
    joins them as "{instruction}\\n\\nInput:\\n{dialogue}", and the OPD loader
    needs the two halves separately to re-render a prompt at rollout time.
    """

    missing = [name for name in REQUIRED_COLUMNS if name not in dataset.column_names]
    if missing:
        raise SystemExit(
            f"DialogSum columns {missing} not found; got {dataset.column_names}. "
            "The official test split names its three references summary1/2/3 -- "
            "that variant is for a multi-reference metric, not for training."
        )

    rows: list[dict[str, str]] = []
    dropped = 0
    for dialogue, summary in zip(dataset["dialogue"], dataset["summary"]):
        dialogue_text = (dialogue or "").strip()
        summary_text = (summary or "").strip()
        if not dialogue_text or not summary_text:
            dropped += 1
            continue
        rows.append(
            {"instruction": instruction, "input": dialogue_text, "output": summary_text}
        )
    return rows, dropped


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile of an already sorted list."""

    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def length_report(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> dict[str, Any]:
    """Measure real training-time token lengths and the cost of each budget.

    Lengths come from the same apply_chat_template call the trainer makes, not
    from a raw word count, because the template's role markers and the chat
    special tokens are part of what has to fit inside --max_len.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    sample = rows if args.length_sample <= 0 else rows[: args.length_sample]

    full_lengths: list[int] = []
    output_lengths: list[int] = []
    for row in sample:
        user_content = f"{row['instruction']}\n\nInput:\n{row['input']}"
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": row["output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        full_lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        output_lengths.append(
            len(tokenizer(row["output"], add_special_tokens=False)["input_ids"])
        )
    full_lengths.sort()
    output_lengths.sort()

    truncation = {}
    for budget in sorted({*CANDIDATE_MAX_LENS, args.max_len}):
        over = sum(length > budget for length in full_lengths)
        lost = sum(max(length - budget, 0) for length in full_lengths)
        truncation[budget] = {
            "rows_truncated": over,
            "fraction_truncated": over / len(full_lengths),
            "tokens_dropped_mean_over_truncated": lost / over if over else 0.0,
        }

    report = {
        "tokenizer": args.tokenizer_path,
        "rows_measured": len(sample),
        "full_length": {
            f"p{q}": percentile(full_lengths, q) for q in (50, 90, 95, 99)
        }
        | {"max": full_lengths[-1]},
        "output_length": {
            f"p{q}": percentile(output_lengths, q) for q in (50, 90, 99)
        }
        | {"max": output_lengths[-1]},
        "truncation_by_max_len": truncation,
    }

    print("==== DialogSum training-length report ====")
    print(
        f"n={len(sample)}  full: P50={report['full_length']['p50']} "
        f"P90={report['full_length']['p90']} P95={report['full_length']['p95']} "
        f"P99={report['full_length']['p99']} max={report['full_length']['max']}"
    )
    print(
        f"summary only: P50={report['output_length']['p50']} "
        f"P90={report['output_length']['p90']} max={report['output_length']['max']}"
    )
    print(f"{'max_len':>10}{'truncated':>12}{'share':>9}{'mean tokens lost':>20}")
    for budget, stats in truncation.items():
        marker = "  <- --max_len" if budget == args.max_len else ""
        print(
            f"{budget:>10}{stats['rows_truncated']:>12}"
            f"{stats['fraction_truncated']:>8.1%}"
            f"{stats['tokens_dropped_mean_over_truncated']:>20.0f}{marker}"
        )
    print(
        "Truncation is left-side, so a truncated row keeps its summary and loses the "
        "opening turns of the dialogue -- exactly the content the summary covers. "
        "Choose the smallest budget that keeps this under a few percent."
    )
    return report


def main() -> None:
    """Write the train pool as save_to_disk, the other splits as JSONL, plus a manifest."""

    args = parse_args()
    splits = load_dialogsum(args)
    print(f"loaded splits: {', '.join(f'{k}={len(v)}' for k, v in splits.items())}")
    if "train" not in splits:
        raise SystemExit(f"No train split among {sorted(splits)}; nothing to train on.")

    from datasets import Dataset, DatasetDict

    train_rows, dropped = to_sft_rows(splits["train"], args.instruction)
    if not train_rows:
        raise SystemExit("Train split converted to 0 usable rows; check the column contents.")
    print(f"train: {len(train_rows)} rows kept, {dropped} dropped for an empty field")

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(str(out_dir))
    print(f"wrote train pool to {out_dir}")

    jsonl_dir = Path(args.jsonl_dir) if args.jsonl_dir else out_dir.parent
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    held_out: dict[str, Any] = {}
    for name in ("validation", "test"):
        if name not in splits:
            continue
        rows, name_dropped = to_sft_rows(splits[name], args.instruction)
        path = jsonl_dir / f"{out_dir.name}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        held_out[name] = {"path": str(path), "num_rows": len(rows), "dropped": name_dropped}
        print(f"wrote {name}: {len(rows)} rows to {path}")

    manifest: dict[str, Any] = {
        "source": args.dialogsum_path or args.dialogsum_repo,
        "instruction": args.instruction,
        "train": {
            "path": str(out_dir),
            "num_rows": len(train_rows),
            "dropped": dropped,
        },
        "held_out_jsonl": held_out,
        "note": "train.py splits validation out of this pool with --val_fraction; the "
        "JSONL splits are untouched and reserved for a generation metric.",
    }
    if args.tokenizer_path:
        manifest["length"] = length_report(train_rows, args)
    else:
        print("skipping length report: no --tokenizer_path and no $MODEL_DIR")

    manifest_path = jsonl_dir / f"{out_dir.name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
