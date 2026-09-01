"""Estimate mathematical correctness of a completed self-distillation pool.

This is a corpus diagnostic, not a benchmark evaluation: it compares the final
answer extracted from each generated ``targets`` string with the answer
extracted from the same row's ``gold_targets`` string.  Comparing the full
strings would be meaningless because two correct derivations need not match.

Two extraction views are reported:

* marker: accepts an explicit ``\\boxed{...}``, ``####`` or final-answer phrase;
* lenient: falls back to the last number when no explicit marker exists.

The marker score is higher confidence but can under-count correct unformatted
answers.  The lenient score has better coverage but can mistake a number in an
unfinished derivation for the final answer.  Read both rather than selecting
whichever supports a hypothesis.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from onereplay.eval.metrics.gsm8k import normalize_number
from onereplay.eval.metrics.math500 import extract_answer, is_equiv


FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+(?:final\s+)?answer\s+is|"
    r"(?:therefore|thus|hence),?\s+the\s+answer\s+is|"
    r"final\s+answer(?:\s+is)?)\s*[:：]?\s*(.+?)(?:\n|$)",
    flags=re.IGNORECASE,
)


@dataclass
class Counts:
    rows: int = 0
    usable: int = 0
    truncated: int = 0
    empty_target: int = 0
    empty_gold: int = 0
    marker_gold: int = 0
    marker_pred: int = 0
    marker_pairs: int = 0
    marker_correct: int = 0
    lenient_gold: int = 0
    lenient_pred: int = 0
    lenient_pairs: int = 0
    lenient_correct: int = 0

    def add(self, other: "Counts") -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)


def expand_files(patterns: str) -> list[Path]:
    """Expand comma-separated paths/globs deterministically."""

    paths: list[Path] = []
    for pattern in (part.strip() for part in patterns.split(",") if part.strip()):
        matches = sorted(Path(path) for path in glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).is_file():
            paths.append(Path(pattern))
    # A broad glob can overlap a literal path. Do not score a file twice.
    return list(dict.fromkeys(paths))


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def clean_candidate(candidate: str) -> str | None:
    candidate = candidate.strip().strip("`").strip()
    boxed = extract_answer(candidate)
    if boxed is not None:
        return boxed
    candidate = re.sub(r"^[\s$]*(?:\\\(|\\\[)?", "", candidate)
    candidate = re.sub(r"(?:\\\)|\\\])?[\s$]*[。.]?\s*$", "", candidate)
    candidate = candidate.strip()
    return candidate or None


def marker_answer(text: str) -> str | None:
    """Extract only an answer accompanied by an explicit final-answer marker."""

    boxed = extract_answer(text)
    if boxed is not None:
        return boxed

    if "####" in text:
        for chunk in reversed(text.split("####")[1:]):
            answer = normalize_number(chunk)
            if answer is not None:
                return answer

    matches = list(FINAL_ANSWER_RE.finditer(text))
    if matches:
        return clean_candidate(matches[-1].group(1))
    return None


def lenient_answer(text: str) -> str | None:
    return marker_answer(text) or normalize_number(text)


def source_lookup(patterns: str) -> dict[tuple[str, str], str]:
    """Map (prompt, gold) back to MetaMath source when view files are available."""

    lookup: dict[tuple[str, str], str] = {}
    conflicts: set[tuple[str, str]] = set()
    for path in expand_files(patterns):
        for row in load_jsonl(path):
            # generate_replay_targets goes through to_sft_schema, which strips
            # both strings before writing inputs/gold_targets.
            key = (
                str(row.get("inputs", "") or "").strip(),
                str(row.get("targets", "") or "").strip(),
            )
            source = str(row.get("source", "unknown"))
            if key in lookup and lookup[key] != source:
                conflicts.add(key)
            else:
                lookup[key] = source
    for key in conflicts:
        lookup.pop(key, None)
    return lookup


def score_row(row: dict[str, Any]) -> tuple[Counts, dict[str, str] | None]:
    counts = Counts(rows=1)
    pred_text = str(row.get("targets", "") or "").strip()
    gold_text = str(row.get("gold_targets", "") or "").strip()
    if bool(row.get("truncated")):
        counts.truncated = 1
        return counts, None
    if not pred_text:
        counts.empty_target = 1
        return counts, None
    if not gold_text:
        counts.empty_gold = 1
        return counts, None

    counts.usable = 1
    marker_gold = marker_answer(gold_text)
    marker_pred = marker_answer(pred_text)
    lenient_gold = marker_gold or normalize_number(gold_text)
    lenient_pred = marker_pred or normalize_number(pred_text)

    counts.marker_gold = int(marker_gold is not None)
    counts.marker_pred = int(marker_pred is not None)
    counts.lenient_gold = int(lenient_gold is not None)
    counts.lenient_pred = int(lenient_pred is not None)

    if marker_gold is not None and marker_pred is not None:
        counts.marker_pairs = 1
        counts.marker_correct = int(is_equiv(marker_pred, marker_gold))
    if lenient_gold is not None and lenient_pred is not None:
        counts.lenient_pairs = 1
        counts.lenient_correct = int(is_equiv(lenient_pred, lenient_gold))

    wrong = None
    if (
        lenient_gold is not None
        and lenient_pred is not None
        and not is_equiv(lenient_pred, lenient_gold)
    ):
        wrong = {
            "index": str(row.get("index", "")),
            "prompt": str(row.get("inputs", "")),
            "pred_answer": lenient_pred,
            "gold_answer": lenient_gold,
            "targets": pred_text,
            "gold_targets": gold_text,
        }
    return counts, wrong


def pct(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{numerator / denominator:.2%}"


def print_counts(label: str, counts: Counts) -> None:
    print(f"\n[{label}]")
    print(
        f"rows={counts.rows}  usable={counts.usable}  "
        f"truncated={counts.truncated}  empty_target={counts.empty_target}  "
        f"empty_gold={counts.empty_gold}"
    )
    print(
        "marker extraction: "
        f"gold={pct(counts.marker_gold, counts.usable)}  "
        f"selfdistill={pct(counts.marker_pred, counts.usable)}  "
        f"comparable={counts.marker_pairs}"
    )
    print(
        "marker accuracy  : "
        f"{counts.marker_correct}/{counts.marker_pairs} = "
        f"{pct(counts.marker_correct, counts.marker_pairs)}  "
        f"(all-usable lower bound {pct(counts.marker_correct, counts.usable)})"
    )
    print(
        "lenient extraction: "
        f"gold={pct(counts.lenient_gold, counts.usable)}  "
        f"selfdistill={pct(counts.lenient_pred, counts.usable)}  "
        f"comparable={counts.lenient_pairs}"
    )
    print(
        "lenient accuracy : "
        f"{counts.lenient_correct}/{counts.lenient_pairs} = "
        f"{pct(counts.lenient_correct, counts.lenient_pairs)}  "
        f"(all-usable lower bound {pct(counts.lenient_correct, counts.usable)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_files",
        required=True,
        help="Comma-separated self-distillation paths/globs. Quote shell globs.",
    )
    parser.add_argument(
        "--view_files",
        default="",
        help="Optional prepared MetaMath view-file glob; enables GSM/MATH source breakdown.",
    )
    parser.add_argument(
        "--show_wrong",
        type=int,
        default=5,
        help="Print this many leniently scored disagreements.",
    )
    parser.add_argument(
        "--json_out",
        default="",
        help="Optional JSON output containing aggregate counts and shown disagreements.",
    )
    args = parser.parse_args()

    data_files = expand_files(args.data_files)
    if not data_files:
        raise SystemExit(f"no files matched --data_files={args.data_files!r}")
    print("self-distillation files:")
    for path in data_files:
        print(f"  {path}")

    sources = source_lookup(args.view_files) if args.view_files else {}
    if args.view_files:
        print(f"source lookup entries: {len(sources)}")

    total = Counts()
    per_file: dict[str, Counts] = defaultdict(Counts)
    per_source: dict[str, Counts] = defaultdict(Counts)
    wrong_examples: list[dict[str, str]] = []
    matched_source = 0

    for path in data_files:
        for row in load_jsonl(path):
            counts, wrong = score_row(row)
            total.add(counts)
            per_file[path.name].add(counts)

            source = sources.get(
                (
                    str(row.get("inputs", "") or "").strip(),
                    str(row.get("gold_targets", "") or "").strip(),
                ),
                "unknown",
            )
            if sources:
                per_source[source].add(counts)
                matched_source += int(source != "unknown")

            if wrong is not None and len(wrong_examples) < max(args.show_wrong, 0):
                wrong["file"] = path.name
                wrong["source"] = source
                wrong_examples.append(wrong)

    for label, counts in per_file.items():
        print_counts(label, counts)
    print_counts("ALL", total)

    if sources:
        print(f"\nsource matched: {matched_source}/{total.rows} = {pct(matched_source, total.rows)}")
        for label, counts in sorted(per_source.items()):
            print_counts(f"source={label}", counts)

    if wrong_examples:
        print(f"\n[lenient disagreements: first {len(wrong_examples)}]")
        for number, row in enumerate(wrong_examples, 1):
            prompt = " ".join(row["prompt"].split())[:180]
            print(
                f"{number}. {row['file']} index={row['index']} source={row['source']}\n"
                f"   pred={row['pred_answer']!r}  gold={row['gold_answer']!r}\n"
                f"   prompt={prompt}"
            )

    if args.json_out:
        output = {
            "files": [str(path) for path in data_files],
            "total": asdict(total),
            "per_file": {key: asdict(value) for key, value in per_file.items()},
            "per_source": {key: asdict(value) for key, value in per_source.items()},
            "source_matched": matched_source,
            "shown_wrong_examples": wrong_examples,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
