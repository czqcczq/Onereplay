"""Re-score finished HumanEval/MBPP runs from their responses.jsonl.

Why this exists: the judge had two bugs that scored correct answers as failures
(bodies de-indented into SyntaxError on HumanEval, programs beheaded at the
first '\\ndef ' on MBPP). Re-generating five runs costs GPU hours, and for
HumanEval it is not even necessary -- the damage there was confined to the first
line's indentation, which is deterministically reversible, so the fixed judge
can be replayed over the text already on disk.

Read the two labels carefully:

* ``exact``  -- the run stored a ``raw`` field, so the original model output was
  recovered and this number equals what a fresh run would produce.
* ``lower bound`` -- only the cleaned completion survived. On HumanEval that is
  still faithful (indentation is recoverable). On MBPP it is *not*: text after
  the first '\\ndef ' was permanently discarded, so this only recovers rows the
  blacklist wrongly blocked, and MBPP must be re-generated for a real number.

Usage
-----
  python -m onereplay.scripts.rejudge_code_eval \
      --metric humaneval \
      --results_root results/humaneval \
      --humaneval_data_file datasets/humaneval/openai_humaneval.parquet

  python -m onereplay.scripts.rejudge_code_eval \
      --metric mbpp \
      --results_root results/mbpp \
      --mbpp_dataset_path datasets/mbpp --dataset_split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from onereplay.eval.code_exec import (
    assemble_entry_point_program,
    cleanup_body_completion,
    cleanup_program_completion,
    evaluate_assert_program,
    evaluate_entry_point_program,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument(
        "--results_root",
        required=True,
        help="directory holding one subdirectory per run, each with responses.jsonl",
    )
    parser.add_argument("--runs", default="", help="comma-separated run names (default: all)")
    parser.add_argument("--humaneval_data_file", default="")
    parser.add_argument("--mbpp_dataset_path", default="")
    parser.add_argument("--dataset_name", default="google-research-datasets/mbpp")
    parser.add_argument("--dataset_config", default="full")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--out",
        default="",
        help="optional json path for the comparison table",
    )
    parser.add_argument(
        "--write_back",
        type=int,
        default=0,
        help="1 = also write responses_rejudged.jsonl into each run directory",
    )
    return parser.parse_args()


def load_reference(args: argparse.Namespace) -> dict[Any, dict[str, Any]]:
    """Index the evaluation set by task_id."""

    if args.metric == "humaneval":
        from onereplay.eval.metrics.humaneval import load_humaneval

        if not args.humaneval_data_file:
            raise SystemExit("--humaneval_data_file is required for --metric humaneval")
        rows = load_humaneval(args.humaneval_data_file, args.cache_dir, args.limit)
    else:
        from onereplay.eval.metrics.mbpp import load_mbpp

        rows = load_mbpp(
            {
                "mbpp_dataset_path": args.mbpp_dataset_path,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
                "cache_dir": args.cache_dir,
                "limit": args.limit,
            }
        )
    return {row.get("task_id"): row for row in rows}


def mbpp_tests(example: dict[str, Any]) -> list[str]:
    tests = example.get("test_list") or example.get("tests") or []
    if isinstance(tests, str):
        tests = [tests]
    return [str(test) for test in tests]


def rejudge_run(
    metric: str,
    response_path: Path,
    reference: dict[Any, dict[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in response_path.open(encoding="utf-8") if line.strip()]
    used_raw = 0
    old_passed = 0
    new_passed = 0
    missing = 0
    recovered: list[Any] = []
    lost: list[Any] = []
    rejudged: list[dict[str, Any]] = []

    for row in rows:
        task_id = row.get("task_id")
        example = reference.get(task_id)
        was_passed = bool(row.get("passed"))
        old_passed += int(was_passed)
        if example is None:
            missing += 1
            new_passed += int(was_passed)
            continue

        raw = row.get("raw")
        if raw is None:
            text = row.get("completion") or ""
        else:
            text = raw
            used_raw += 1

        if metric == "humaneval":
            completion = cleanup_body_completion(text)
            program = assemble_entry_point_program(example["prompt"], completion)
            ok, error = evaluate_entry_point_program(
                program, example["entry_point"], example["test"], timeout
            )
        else:
            completion = cleanup_program_completion(text)
            ok, error = evaluate_assert_program(completion, mbpp_tests(example), timeout)

        new_passed += int(ok)
        if ok and not was_passed:
            recovered.append(task_id)
        elif was_passed and not ok:
            lost.append(task_id)
        rejudged.append(
            {
                "task_id": task_id,
                "passed": ok,
                "error": error,
                "completion": completion,
                "was_passed": was_passed,
            }
        )

    total = max(len(rows), 1)
    return {
        "n": len(rows),
        "old_passed": old_passed,
        "new_passed": new_passed,
        "old_pass_at_1": old_passed / total,
        "new_pass_at_1": new_passed / total,
        "delta": (new_passed - old_passed) / total,
        "recovered": len(recovered),
        "lost": len(lost),
        "recovered_task_ids": recovered[:20],
        "lost_task_ids": lost[:20],
        "rows_with_raw": used_raw,
        "rows_missing_reference": missing,
        "fidelity": "exact" if used_raw == len(rows) else "lower bound (no raw stored)",
        "_rejudged": rejudged,
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_dir():
        raise SystemExit(f"not a directory: {results_root}")

    if args.runs.strip():
        run_names = [name.strip() for name in args.runs.split(",") if name.strip()]
    else:
        run_names = sorted(
            path.name for path in results_root.iterdir() if (path / "responses.jsonl").is_file()
        )
    if not run_names:
        raise SystemExit(f"no run directory with responses.jsonl under {results_root}")

    reference = load_reference(args)
    print(f"reference set: {len(reference)} tasks ({args.metric})")

    table: dict[str, Any] = {}
    for run_name in run_names:
        response_path = results_root / run_name / "responses.jsonl"
        if not response_path.is_file():
            print(f"skip {run_name}: no responses.jsonl")
            continue
        print(f"rejudging {run_name} ...")
        report = rejudge_run(args.metric, response_path, reference, args.timeout)
        rejudged = report.pop("_rejudged")
        if args.write_back:
            out_path = results_root / run_name / "responses_rejudged.jsonl"
            with out_path.open("w", encoding="utf-8") as file:
                for item in rejudged:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
        table[run_name] = report

    header = f"{'run':<52}{'n':>5}{'old':>9}{'new':>9}{'delta':>8}{'+':>5}{'-':>5}  fidelity"
    print()
    print(header)
    print("-" * len(header))
    for run_name, report in table.items():
        print(
            f"{run_name:<52}{report['n']:>5}"
            f"{report['old_pass_at_1'] * 100:>8.1f}%"
            f"{report['new_pass_at_1'] * 100:>8.1f}%"
            f"{report['delta'] * 100:>+7.1f}%"
            f"{report['recovered']:>5}{report['lost']:>5}"
            f"  {report['fidelity']}"
        )
    print()
    print("old = score as recorded by the buggy judge, new = re-scored by the fixed judge.")
    print("'+' rows the fix recovered, '-' rows it newly failed (should be ~0).")
    if any(report["fidelity"] != "exact" for report in table.values()):
        print(
            "note: runs without a stored 'raw' field are replayed from the cleaned text. "
            "For HumanEval that is faithful; for MBPP the beheaded tail is gone for good, "
            "so treat MBPP numbers as a floor and re-generate for the real value."
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"metric": args.metric, "runs": table}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
