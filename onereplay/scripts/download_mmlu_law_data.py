"""One-shot download + normalize of the three MMLU law subjects (login node).

Run this ON A NETWORKED (login) NODE; compute nodes have HF_HUB_OFFLINE=1, so
the mmlu_* metrics can only read local files. Writes, into --out_dir, one file
per subject with the exact names onereplay/eval/metrics/mmlu_law.py expects:

  professional_law_test.jsonl   : {question, choices, answer}   ~1534 rows
  international_law_test.jsonl  : {question, choices, answer}    ~121 rows
  jurisprudence_test.jsonl      : {question, choices, answer}    ~108 rows

`choices` is the four option texts in their original order and `answer` is the
0-based index into it. The index is kept rather than pre-rendered into a letter
because the metric owns the A/B/C/D rendering: leaving it here would mean the
letter the grader compares against was decided by whichever version of this
script last ran, instead of by the harness every arm shares.

Each subject downloads independently: if one config is renamed or gated the
others still land, and the failures are reported together at the end. Do NOT set
the HF_*_OFFLINE flags when running this.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Run as a file (`python onereplay/scripts/download_mmlu_law_data.py`) and
# sys.path[0] is this script's directory, not the repo root, so `import
# onereplay` fails. Every other script here carries the same two lines.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.scripts.prepare_math_data import write_jsonl  # noqa: E402

# MMLU ships one config per subject. These three are the law block: Professional
# Law is the bar-exam-style bulk of it, International Law and Jurisprudence are
# the two small ones, and all three are 4-option single-answer.
SUBJECTS = ("professional_law", "international_law", "jurisprudence")

QUESTION_KEYS = ("question", "Question", "input", "query")
CHOICES_KEYS = ("choices", "options", "endings")
ANSWER_KEYS = ("answer", "Answer", "label", "target", "gold")


def parse_args() -> argparse.Namespace:
    """Parse output dir, repo/split, and which subjects to fetch."""

    parser = argparse.ArgumentParser(
        description="Download + normalize the MMLU law subjects (login node)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="datasets/bench/mmlu",
        help="Must match MMLU_DIR in 90/92_*.pbs.",
    )
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--mmlu_repo", type=str, default="cais/mmlu")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--subjects",
        type=str,
        default=",".join(SUBJECTS),
        help="Comma-separated MMLU config names.",
    )
    return parser.parse_args()


def warn_if_offline() -> None:
    """Fail fast if HF offline flags are set; this script needs network."""

    offline = [
        name
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
        if os.environ.get(name) == "1"
    ]
    if offline:
        raise SystemExit(
            f"{', '.join(offline)} set to 1; unset them and run on a networked node."
        )


def load_split(repo: str, config: str, split: str, cache_dir: str):
    """Load one MMLU subject split."""

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"name": config, "split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(repo, **kwargs)


def pick(record: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    """Return the first present, non-None candidate field."""

    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return None


def normalize_answer(value: Any, num_choices: int) -> int | None:
    """Coerce the gold column to a 0-based index into `choices`.

    cais/mmlu gives an int, but mirrors sometimes give the letter instead. Both
    are accepted; anything else is dropped rather than guessed, because a
    silently mis-mapped gold column shifts every answer by a constant and still
    produces a plausible-looking accuracy.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        index = value
    else:
        text = str(value).strip()
        if len(text) == 1 and text.upper().isalpha():
            index = ord(text.upper()) - ord("A")
        elif text.isdigit():
            index = int(text)
        else:
            return None
    return index if 0 <= index < num_choices else None


def build_subject(args, subject: str, out_dir: Path, cache_dir: str) -> None:
    """One MMLU subject -> {question, choices, answer} jsonl."""

    print(f"[{subject}] {args.mmlu_repo}:{subject}[{args.split}]")
    dataset = load_split(args.mmlu_repo, subject, args.split, cache_dir)

    rows: list[dict[str, Any]] = []
    dropped = 0
    for record in dataset:
        question = pick(record, QUESTION_KEYS)
        choices = pick(record, CHOICES_KEYS)
        question = str(question or "").strip()
        choices = [str(choice).strip() for choice in (choices or [])]
        answer = normalize_answer(pick(record, ANSWER_KEYS), len(choices))
        if not question or len(choices) < 2 or answer is None:
            dropped += 1
            continue
        rows.append({"question": question, "choices": choices, "answer": answer})

    if dropped:
        print(f"[{subject}] dropped {dropped} rows with no usable question/choices/answer")
    write_jsonl(rows, out_dir / f"{subject}_test.jsonl")


def main() -> None:
    """Download every requested subject; keep going if one fails."""

    args = parse_args()
    warn_if_offline()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = [name.strip() for name in args.subjects.split(",") if name.strip()]
    failures: list[str] = []
    for subject in subjects:
        try:
            build_subject(args, subject, out_dir, args.cache_dir)
        except Exception as error:  # noqa: BLE001
            print(f"[WARN] {subject} failed: {error}")
            failures.append(f"{subject}: {error}")

    if failures:
        print("\n==== some downloads failed ====")
        for line in failures:
            print("  - " + line)
        print(
            "对失败项可单独重试，例如: "
            "python -m onereplay.scripts.download_mmlu_law_data --subjects jurisprudence"
        )
    else:
        print("\ndone. all MMLU law subjects ready in " + str(out_dir))


if __name__ == "__main__":
    main()
