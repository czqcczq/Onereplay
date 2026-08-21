"""Convert already-downloaded MATH assets into the schemas the Q2 pipeline needs.

You download the parquet files yourself (compute nodes are offline); this script
only reads local files and writes normalized jsonl. It never touches the network.

Inputs (any subset; pass the ones you have). Each accepts a file, a glob, a
directory, or a comma-separated list, in parquet / jsonl / json:
  --math_train  MATH-lighteval train split   -> math_train_inputs_targets.jsonl
  --math500     HuggingFaceH4/MATH-500        -> math500_test.jsonl
  --amc         AI-MO/aimo-validation-amc     -> amc_test.jsonl
  --aime        AIME 2024 (+2025, comma-sep)  -> aime_test.jsonl  (merged)
  --gsm8k       GSM8K test                    -> gsm8k_test.jsonl

Outputs (in --out_dir):
  math_train_inputs_targets.jsonl : {inputs=problem, targets=solution}
      `targets` is only a placeholder gold; generate_replay_targets.py overwrites
      it with the base model's own answer during self-distillation.
  math500_test.jsonl : {problem, answer, solution}  (math500 metric)
  amc_test.jsonl     : {problem, answer}            (math500 metric)
  aime_test.jsonl    : {problem, answer}            (aime metric, integer answers)
  gsm8k_test.jsonl   : {question, answer}           (gsm8k metric)

Field names are auto-detected across common casings, so mirror datasets with
slightly different columns still work.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

PROBLEM_KEYS = ("problem", "Problem", "question", "Question", "prompt", "input")
SOLUTION_KEYS = ("solution", "Solution", "solutions")
ANSWER_KEYS = ("answer", "Answer", "final_answer", "solution_answer", "target")


def parse_args() -> argparse.Namespace:
    """Parse output dir and per-dataset source paths."""

    parser = argparse.ArgumentParser(description="Normalize local MATH assets (offline).")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/hpcwork/xsz96350/Chen_logs/onereplay/datasets/math",
    )
    parser.add_argument("--math_train", type=str, default="")
    parser.add_argument("--math500", type=str, default="")
    parser.add_argument("--amc", type=str, default="")
    parser.add_argument("--aime", type=str, default="")
    parser.add_argument("--gsm8k", type=str, default="")
    return parser.parse_args()


def expand_sources(spec: str) -> list[Path]:
    """Turn a comma-separated file/glob/dir spec into a flat list of files."""

    paths: list[Path] = []
    for piece in (item.strip() for item in spec.split(",")):
        if not piece:
            continue
        candidate = Path(piece)
        if candidate.is_dir():
            for pattern in ("*.parquet", "*.jsonl", "*.json"):
                paths.extend(sorted(candidate.glob(pattern)))
        elif any(ch in piece for ch in "*?["):
            paths.extend(sorted(Path(match) for match in glob.glob(piece)))
        else:
            paths.append(candidate)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"source(s) not found: {missing}")
    if not paths:
        raise FileNotFoundError(f"no files matched: {spec!r}")
    return paths


def read_records(spec: str) -> list[dict[str, Any]]:
    """Read all rows from the given sources into a list of dicts."""

    records: list[dict[str, Any]] = []
    for path in expand_sources(spec):
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            import pandas as pd

            frame = pd.read_parquet(path)
            records.extend(frame.to_dict(orient="records"))
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8") as file:
                records.extend(json.loads(line) for line in file if line.strip())
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                records.extend(payload)
            elif isinstance(payload, dict):
                for key in ("test", "train", "validation", "data"):
                    if isinstance(payload.get(key), list):
                        records.extend(payload[key])
                        break
        else:
            raise ValueError(f"unsupported file type: {path}")
    return records


def pick(record: dict[str, Any], candidates: tuple[str, ...]) -> str:
    """Return the first non-empty candidate field as a stripped string."""

    for key in candidates:
        if key in record and record[key] is not None:
            value = str(record[key]).strip()
            if value:
                return value
    return ""


def write_jsonl(rows: list[dict], target: Path) -> None:
    """Write rows to jsonl and report the count."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {target} ({len(rows)} rows)")


def build_math_train(spec: str, out_dir: Path) -> None:
    """MATH train -> {inputs, targets} self-distillation corpus."""

    rows = []
    for record in read_records(spec):
        problem = pick(record, PROBLEM_KEYS)
        solution = pick(record, SOLUTION_KEYS)
        if problem and solution:
            rows.append({"inputs": problem, "targets": solution})
    write_jsonl(rows, out_dir / "math_train_inputs_targets.jsonl")


def build_math500(spec: str, out_dir: Path) -> None:
    """MATH-500 -> {problem, answer, solution}."""

    rows = []
    for record in read_records(spec):
        problem = pick(record, PROBLEM_KEYS)
        answer = pick(record, ANSWER_KEYS)
        solution = pick(record, SOLUTION_KEYS)
        if problem and (answer or solution):
            rows.append({"problem": problem, "answer": answer, "solution": solution})
    write_jsonl(rows, out_dir / "math500_test.jsonl")


def build_amc(spec: str, out_dir: Path) -> None:
    """AMC -> {problem, answer}."""

    rows = []
    for record in read_records(spec):
        problem = pick(record, PROBLEM_KEYS)
        answer = pick(record, ANSWER_KEYS) or pick(record, SOLUTION_KEYS)
        if problem and answer:
            rows.append({"problem": problem, "answer": answer})
    write_jsonl(rows, out_dir / "amc_test.jsonl")


def build_aime(spec: str, out_dir: Path) -> None:
    """AIME (possibly several files) -> merged {problem, answer}."""

    rows = []
    for record in read_records(spec):
        problem = pick(record, PROBLEM_KEYS)
        answer = pick(record, ANSWER_KEYS) or pick(record, SOLUTION_KEYS)
        if problem and answer:
            rows.append({"problem": problem, "answer": answer})
    write_jsonl(rows, out_dir / "aime_test.jsonl")


def build_gsm8k(spec: str, out_dir: Path) -> None:
    """GSM8K test -> {question, answer}."""

    rows = []
    for record in read_records(spec):
        question = pick(record, ("question", "Question", "problem"))
        answer = pick(record, ANSWER_KEYS)
        if question and answer:
            rows.append({"question": question, "answer": answer})
    write_jsonl(rows, out_dir / "gsm8k_test.jsonl")


def main() -> None:
    """Convert every provided source into its normalized jsonl."""

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = [
        (args.math_train, build_math_train),
        (args.math500, build_math500),
        (args.amc, build_amc),
        (args.aime, build_aime),
        (args.gsm8k, build_gsm8k),
    ]
    ran = False
    for spec, builder in builders:
        if spec:
            builder(spec, out_dir)
            ran = True
    if not ran:
        raise SystemExit("nothing to do: pass at least one of --math_train/--math500/--amc/--aime/--gsm8k")
    print("done.")


if __name__ == "__main__":
    main()
