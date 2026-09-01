"""Execute OPC tests against self-distilled and gold code targets.

The self-distillation JSONL deliberately contains only the prompt, generated
target and gold target.  This script joins each row back to the prepared OPC
view by ``(inputs, gold_targets)``, follows ``source_index`` into the original
educational_instruct parquet, and runs its testcase assertions.

Gold is executed too.  The primary score is prediction pass rate restricted to
rows whose gold target passes the same harness; source-data or reconstruction
failures therefore do not get mislabeled as model mistakes.

Run this only on trusted experiment data.  The shared evaluator blocks obvious
filesystem/process/network access and applies a timeout, but it is a pragmatic
benchmark subprocess rather than a hardened security sandbox.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from onereplay.eval.code_exec import (
    assemble_entry_point_program,
    cleanup_body_completion,
    cleanup_program_completion,
    evaluate_assert_program,
)


@dataclass
class Counts:
    rows: int = 0
    usable: int = 0
    truncated: int = 0
    empty_target: int = 0
    join_missing: int = 0
    no_tests: int = 0
    tested: int = 0
    gold_pass: int = 0
    pred_pass: int = 0
    pred_pass_on_gold_valid: int = 0

    def add(self, other: "Counts") -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def row_key(prompt: Any, gold: Any) -> tuple[str, str]:
    # generate_replay_targets writes strings after to_sft_schema strips them.
    return (str(prompt or "").strip(), str(gold or "").strip())


def build_view_lookup(
    path: Path,
) -> dict[tuple[str, str], deque[dict[str, Any]]]:
    lookup: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in load_jsonl(path):
        lookup[row_key(row.get("inputs"), row.get("targets"))].append(
            {
                "source_index": int(row["source_index"]),
                "style": str(row.get("style", "bare")),
                "entry_point": str(row.get("entry_point", "")),
            }
        )
    return lookup


def normalize_tests(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Some parquet writers preserve list columns as a JSON or Python
        # literal string. Executing "['assert ...']" as one statement would
        # merely construct a list and falsely pass every candidate.
        decoded: Any = None
        if stripped.startswith(("[", "(")):
            try:
                decoded = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                decoded = None
        if decoded is None:
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
        if isinstance(decoded, list):
            return [str(test) for test in decoded if str(test).strip()]
        return [str(decoded)]
    try:
        return [str(test) for test in value if str(test).strip()]
    except TypeError:
        return [str(value)]


def make_program(prompt: str, target: str, style: str) -> str:
    if style == "heval":
        completion = cleanup_body_completion(target)
        return assemble_entry_point_program(prompt, completion)
    return cleanup_program_completion(target)


def ratio(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{numerator / denominator:.2%}"


def error_kind(error: str) -> str:
    if not error:
        return "passed"
    if error == "timeout":
        return "timeout"
    if error == "blocked dangerous code pattern":
        return "blocked"
    return error.split("(", 1)[0].strip() or "unknown"


def print_counts(label: str, counts: Counts) -> None:
    print(f"\n[{label}]")
    print(
        f"rows={counts.rows}  usable={counts.usable}  truncated={counts.truncated}  "
        f"empty_target={counts.empty_target}"
    )
    print(
        f"joined/tested={counts.tested}  join_missing={counts.join_missing}  "
        f"no_tests={counts.no_tests}"
    )
    print(
        f"gold harness pass : {counts.gold_pass}/{counts.tested} = "
        f"{ratio(counts.gold_pass, counts.tested)}"
    )
    print(
        f"selfdistill pass  : {counts.pred_pass}/{counts.tested} = "
        f"{ratio(counts.pred_pass, counts.tested)}"
    )
    print(
        "selfdistill accuracy on gold-valid rows: "
        f"{counts.pred_pass_on_gold_valid}/{counts.gold_pass} = "
        f"{ratio(counts.pred_pass_on_gold_valid, counts.gold_pass)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", required=True, help="Completed OPC self-distillation JSONL.")
    parser.add_argument("--view_file", required=True, help="Prepared OPC view JSONL.")
    parser.add_argument("--opc_path", required=True, help="Original educational_instruct parquet.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Seconds allowed per program.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Random usable rows to execute; 0 executes the full pool.",
    )
    parser.add_argument("--sample_seed", type=int, default=1)
    parser.add_argument("--show_failures", type=int, default=10)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    data_path = Path(args.data_file)
    view_path = Path(args.view_file)
    opc_path = Path(args.opc_path)
    for path in (data_path, view_path, opc_path):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    rows = load_jsonl(data_path)
    lookup = build_view_lookup(view_path)
    opc = load_dataset("parquet", data_files=str(opc_path), split="train")
    if "testcase" not in opc.column_names:
        raise SystemExit(f"{opc_path} has no testcase column; columns={opc.column_names}")

    total = Counts(rows=len(rows))
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        target = str(row.get("targets", "") or "").strip()
        if bool(row.get("truncated")):
            total.truncated += 1
            continue
        if not target:
            total.empty_target += 1
            continue
        total.usable += 1
        key = row_key(row.get("inputs"), row.get("gold_targets"))
        if not lookup.get(key):
            total.join_missing += 1
            continue
        candidates.append((row, lookup[key].popleft()))

    random.Random(args.sample_seed).shuffle(candidates)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    scored = Counts()
    per_style: dict[str, Counts] = defaultdict(Counts)
    pred_errors: Counter[str] = Counter()
    gold_errors: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    for number, (row, meta) in enumerate(candidates, 1):
        style = meta["style"]
        counts = Counts(rows=1, usable=1)
        source_index = meta["source_index"]
        source = dict(opc[source_index])
        tests = normalize_tests(source.get("testcase"))
        if not tests:
            counts.no_tests = 1
            scored.add(counts)
            per_style[style].add(counts)
            continue

        prompt = str(row.get("inputs", "") or "")
        pred_target = str(row.get("targets", "") or "")
        gold_target = str(row.get("gold_targets", "") or "")
        pred_program = make_program(prompt, pred_target, style)
        gold_program = make_program(prompt, gold_target, style)

        gold_ok, gold_error = evaluate_assert_program(gold_program, tests, args.timeout)
        pred_ok, pred_error = evaluate_assert_program(pred_program, tests, args.timeout)

        counts.tested = 1
        counts.gold_pass = int(gold_ok)
        counts.pred_pass = int(pred_ok)
        counts.pred_pass_on_gold_valid = int(gold_ok and pred_ok)
        scored.add(counts)
        per_style[style].add(counts)
        gold_errors[error_kind(gold_error)] += int(not gold_ok)
        pred_errors[error_kind(pred_error)] += int(not pred_ok)

        if gold_ok and not pred_ok and len(failures) < max(args.show_failures, 0):
            failures.append(
                {
                    "index": row.get("index"),
                    "source_index": source_index,
                    "style": style,
                    "entry_point": meta["entry_point"],
                    "error": pred_error,
                    "prompt": prompt,
                    "targets": pred_target,
                    "gold_targets": gold_target,
                    "tests": tests,
                }
            )
        if number % 100 == 0:
            print(
                f"tested {number}/{len(candidates)}  "
                f"gold={ratio(scored.gold_pass, scored.tested)}  "
                f"selfdistill|gold-valid="
                f"{ratio(scored.pred_pass_on_gold_valid, scored.gold_pass)}",
                flush=True,
            )

    # Keep full-pool availability counts and sampled execution counts distinct.
    print("\n[POOL]")
    print(
        f"rows={total.rows}  usable={total.usable}  truncated={total.truncated}  "
        f"empty_target={total.empty_target}  join_missing={total.join_missing}"
    )
    print(f"execution sample={len(candidates)} (limit={args.limit}, seed={args.sample_seed})")
    print_counts("EXECUTED", scored)
    for style, counts in sorted(per_style.items()):
        print_counts(f"style={style}", counts)
    print(f"\npred failure types: {dict(pred_errors.most_common())}")
    print(f"gold failure types: {dict(gold_errors.most_common())}")

    if failures:
        print(f"\n[self-distillation failures on gold-valid rows: first {len(failures)}]")
        for number, row in enumerate(failures, 1):
            prompt = " ".join(row["prompt"].split())[:180]
            print(
                f"{number}. index={row['index']} source_index={row['source_index']} "
                f"style={row['style']} error={row['error']}\n"
                f"   prompt={prompt}"
            )

    if args.json_out:
        output = {
            "pool": asdict(total),
            "execution_sample": len(candidates),
            "limit": args.limit,
            "sample_seed": args.sample_seed,
            "executed": asdict(scored),
            "per_style": {key: asdict(value) for key, value in per_style.items()},
            "pred_failure_types": dict(pred_errors),
            "gold_failure_types": dict(gold_errors),
            "shown_failures": failures,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
