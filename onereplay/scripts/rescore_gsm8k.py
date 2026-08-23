"""Re-score finished GSM8K runs from their responses.jsonl (offline, no GPU).

predicted_answer() used to take the text after the *last* "####". Qwen writes
its answer as "#### 18 ####", so that slice is the empty string and correct
answers were recorded as no-answer -- base scored 2.7% instead of 77.1%. The
generations themselves are fine, so only the scoring has to be redone; this
avoids hours of re-generation just to fix a parser bug.

The scoring functions are imported from the metric itself, so this script can
never drift from what a fresh evaluation would produce.

Rewrites, per run:
  <out_dir>/gsm8k/<run>/responses.jsonl   prediction / correct fields
  <out_dir>/gsm8k/<run>/summary.json      num_scored / correct / accuracy
  <out_dir>/gsm8k_summary.csv             the row(s) whose run_name matches

Preview first, then write:
  python -m onereplay.scripts.rescore_gsm8k --out_dir .../results --runs base,cs_vanilla_seed1
  python -m onereplay.scripts.rescore_gsm8k --out_dir .../results --runs base,cs_vanilla_seed1 --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.eval.metrics.gsm8k import predicted_answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score GSM8K runs from cached responses.")
    parser.add_argument("--out_dir", type=str, required=True, help="Evaluation results root.")
    parser.add_argument("--runs", type=str, required=True, help="Comma-separated run names.")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: preview).")
    return parser.parse_args()


def rescore_run(results_root: Path, run: str) -> dict[str, Any] | None:
    """Recompute one run's accuracy; returns the rows and the new totals."""

    response_path = results_root / "gsm8k" / run / "responses.jsonl"
    if not response_path.is_file():
        print(f"[skip] {run}: 缺 {response_path}")
        return None

    rows = [json.loads(line) for line in response_path.open(encoding="utf-8") if line.strip()]
    old_correct = sum(bool(row.get("correct")) for row in rows)
    scored = 0
    correct = 0
    for row in rows:
        gold = row.get("gold")
        prediction = predicted_answer(row.get("response", ""))
        is_correct = gold is not None and prediction == gold
        row["prediction"] = prediction
        row["correct"] = is_correct
        scored += gold is not None
        correct += is_correct

    return {
        "run": run,
        "rows": rows,
        "path": response_path,
        "num_examples": len(rows),
        "num_scored": scored,
        "old_correct": old_correct,
        "correct": correct,
        "accuracy": correct / max(scored, 1),
    }


def write_run(results_root: Path, result: dict[str, Any]) -> None:
    """Persist re-scored responses and summary for one run (keeps a .bak)."""

    run_dir = results_root / "gsm8k" / result["run"]

    response_path = result["path"]
    shutil.copy2(response_path, response_path.with_suffix(".jsonl.bak"))
    with response_path.open("w", encoding="utf-8") as file:
        for row in result["rows"]:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        shutil.copy2(summary_path, summary_path.with_suffix(".json.bak"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "run_name": result["run"],
            "num_examples": result["num_examples"],
            "num_scored": result["num_scored"],
            "correct": result["correct"],
            "accuracy": result["accuracy"],
            "output_dir": str(run_dir),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def update_csv(results_root: Path, results: list[dict[str, Any]]) -> None:
    """Rewrite the summary CSV rows belonging to the re-scored runs."""

    csv_path = results_root / "gsm8k_summary.csv"
    if not csv_path.is_file():
        print(f"[warn] 没有 {csv_path}，跳过 CSV 更新")
        return

    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    by_run = {result["run"]: result for result in results}
    touched = 0
    for row in rows:
        result = by_run.get(row.get("run_name", ""))
        if result is None:
            continue
        row["num_examples"] = str(result["num_examples"])
        row["num_scored"] = str(result["num_scored"])
        row["correct"] = str(result["correct"])
        row["accuracy"] = str(result["accuracy"])
        touched += 1

    shutil.copy2(csv_path, csv_path.with_suffix(".csv.bak"))
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 更新了 {touched} 行: {csv_path}")


def main() -> None:
    args = parse_args()
    results_root = Path(args.out_dir)
    runs = [name.strip() for name in args.runs.split(",") if name.strip()]

    results = [r for r in (rescore_run(results_root, run) for run in runs) if r]
    if not results:
        raise SystemExit("没有可重打分的 run")

    print(f"{'run':<34}{'n':>6}{'scored':>8}{'old':>7}{'new':>7}{'accuracy':>11}")
    for result in results:
        print(
            f"{result['run']:<34}{result['num_examples']:>6}{result['num_scored']:>8}"
            f"{result['old_correct']:>7}{result['correct']:>7}{result['accuracy']:>11.4f}"
        )

    if not args.apply:
        print("\n预览模式，未写入任何文件。确认无误后加 --apply。")
        return

    for result in results:
        write_run(results_root, result)
        print(f"已写入 {results_root / 'gsm8k' / result['run']}（原文件存为 .bak）")
    update_csv(results_root, results)


if __name__ == "__main__":
    main()
