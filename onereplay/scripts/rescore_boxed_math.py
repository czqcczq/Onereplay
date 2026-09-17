"""Re-score finished AMC / Minerva / MATH-500 runs from responses.jsonl (no GPU).

AMCMetric compared answers as strings while the AMC golds arrive from the source
parquet as floats -- "142.0" in the jsonl against the \\boxed{142} every model
writes. _strip_string touches neither, so **every** correct answer was recorded
wrong and the metric reported accuracy 0.000 for every run regardless of how the
model actually did. metrics/math500.py now compares AMC numerically.

The generations are unaffected by that bug, and for the hard sets they are
expensive (355 problems x 4 samples at a 4096-token budget, per arm), so only
the scoring is redone here.

is_correct comes from the metric class itself, so this script cannot drift from
what a fresh evaluation would produce. average@k bookkeeping is recomputed from
the sample_index field the same way the metric computes it.

Rewrites, per run:
  <out_dir>/<metric>/<run>/responses.jsonl   prediction / correct fields
  <out_dir>/<metric>/<run>/summary.json      correct / accuracy / per-sample / pass@k
  <out_dir>/<metric>_summary.csv             the row(s) whose run_name matches

Preview first, then write:
  python -m onereplay.scripts.rescore_boxed_math \\
    --out_dir .../results_stage3_math2if_Qwen3-4B-Base/eval_hard_t1.0_avg4_cap4096 \\
    --metric amc --runs all
  ... --apply
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

from onereplay.eval.metrics.math500 import (  # noqa: E402
    AMCMetric,
    MATH500Metric,
    MinervaMathMetric,
    extract_answer,
    is_equiv,
    to_number,
)

METRICS = {
    "amc": AMCMetric,
    "minervamath": MinervaMathMetric,
    "math500": MATH500Metric,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score boxed-answer math runs offline.")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory holding <metric>/<run>/. For the sampled sets this is the "
        "eval_hard_t*_avg*_cap* subtree, not the results root.",
    )
    parser.add_argument("--metric", type=str, default="amc", choices=sorted(METRICS))
    parser.add_argument(
        "--runs",
        type=str,
        default="all",
        help="Comma-separated run names, or 'all' for every run dir under <out_dir>/<metric>/.",
    )
    # Only read by minervamath; kept here so a rescore can reproduce the value the
    # original job used (the metric's default is 1%).
    parser.add_argument("--minerva_rel_tol", type=float, default=0.01)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: preview).")
    return parser.parse_args()


def rescore_run(metric_dir: Path, run: str, metric, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute one run's accuracy from its cached responses."""

    response_path = metric_dir / run / "responses.jsonl"
    if not response_path.is_file():
        print(f"[skip] {run}: 缺 {response_path}")
        return None

    rows = [json.loads(line) for line in response_path.open(encoding="utf-8") if line.strip()]
    if not rows:
        print(f"[skip] {run}: {response_path} 是空的")
        return None

    old_correct = sum(bool(row.get("correct")) for row in rows)
    # Questions keep their identity across samples, so group on the text rather
    # than on sample_index arithmetic -- a truncated responses.jsonl (killed job)
    # then still yields the right denominators instead of a silent off-by-k.
    questions: dict[str, int] = {}
    per_sample: dict[int, list[int]] = {}
    solved_once: dict[str, bool] = {}
    correct = 0
    strict_correct = 0
    unparsed = 0

    for row in rows:
        question = str(row.get("question", ""))
        questions.setdefault(question, len(questions))
        gold = str(row.get("gold", ""))
        prediction = extract_answer(str(row.get("response", "")))
        is_hit = metric.is_correct(prediction, gold, cfg)
        row["prediction"] = prediction
        row["correct"] = is_hit
        if getattr(metric, "report_strict_string", False):
            strict_hit = is_equiv(prediction, gold)
            row["correct_strict_string"] = strict_hit
            strict_correct += int(strict_hit)
        correct += int(is_hit)
        # The residual failure mode after the .0 fix: the model boxes an
        # expression ("342+103=445") or prose instead of the number, so neither
        # side parses and the row falls back to string equality. A high count
        # here means the *prompt contract* is being broken, not that the model
        # cannot do the problems -- worth knowing before reading the accuracy.
        if to_number(prediction) is None:
            unparsed += 1
        sample_index = int(row.get("sample_index", 0))
        per_sample.setdefault(sample_index, []).append(int(is_hit))
        solved_once[question] = solved_once.get(question, False) or is_hit

    num_examples = len(questions)
    num_samples = len(per_sample)
    return {
        "run": run,
        "rows": rows,
        "path": response_path,
        "num_examples": num_examples,
        "num_scored": len(rows),
        "old_correct": old_correct,
        "correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "num_samples": num_samples,
        "accuracy_per_sample": "|".join(
            f"{sum(per_sample[index]) / max(num_examples, 1):.4f}" for index in sorted(per_sample)
        ),
        "pass_at_k": sum(solved_once.values()) / max(num_examples, 1),
        "accuracy_strict_string": strict_correct / max(len(rows), 1),
        "unparsed": unparsed,
    }


def write_run(metric_dir: Path, result: dict[str, Any]) -> None:
    """Persist re-scored responses and summary for one run (keeps a .bak)."""

    run_dir = metric_dir / result["run"]
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
    # Update in place and add nothing new: <metric>_summary.csv takes its header
    # from the first run that wrote it, so an extra key here would misalign every
    # later row in a file an earlier job created.
    for key in ("num_examples", "num_scored", "correct", "accuracy"):
        summary[key] = result[key]
    for key in ("num_samples", "accuracy_per_sample", "pass_at_k", "accuracy_strict_string"):
        if key in summary:
            summary[key] = result[key]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def update_csv(out_dir: Path, metric_name: str, results: list[dict[str, Any]]) -> None:
    """Rewrite the summary CSV rows belonging to the re-scored runs."""

    csv_path = out_dir / f"{metric_name}_summary.csv"
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
        for key in ("num_examples", "num_scored", "correct", "accuracy"):
            if key in row:
                row[key] = str(result[key])
        for key in ("accuracy_per_sample", "pass_at_k", "accuracy_strict_string"):
            if key in row:
                row[key] = str(result[key])
        touched += 1

    shutil.copy2(csv_path, csv_path.with_suffix(".csv.bak"))
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 更新了 {touched} 行: {csv_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    metric_dir = out_dir / args.metric
    if not metric_dir.is_dir():
        raise SystemExit(f"找不到 {metric_dir}（--out_dir 要指向含 <metric>/ 的那一层）")

    if args.runs.strip() == "all":
        runs = sorted(path.name for path in metric_dir.iterdir() if path.is_dir())
    else:
        runs = [name.strip() for name in args.runs.split(",") if name.strip()]
    if not runs:
        raise SystemExit(f"{metric_dir} 下没有 run 目录")

    metric = METRICS[args.metric]()
    cfg = {"minerva_rel_tol": args.minerva_rel_tol}
    results = [r for r in (rescore_run(metric_dir, run, metric, cfg) for run in runs) if r]
    if not results:
        raise SystemExit("没有可重打分的 run")

    print()
    print(f"{args.metric}  ({metric_dir})")
    header = (
        f"{'run':<40}{'n':>5}{'k':>3}{'scored':>7}"
        f"{'old acc':>9}{'new acc':>9}{'pass@k':>8}{'no-number':>11}"
    )
    print(header)
    for result in results:
        old_accuracy = result["old_correct"] / max(result["num_scored"], 1)
        print(
            f"{result['run']:<40}{result['num_examples']:>5}{result['num_samples']:>3}"
            f"{result['num_scored']:>7}{old_accuracy * 100:>8.1f}%{result['accuracy'] * 100:>8.1f}%"
            f"{result['pass_at_k'] * 100:>7.1f}%"
            f"{result['unparsed'] / max(result['num_scored'], 1) * 100:>10.1f}%"
        )
    print()
    print("no-number = 抽出的答案解析不成数字的比例。这一列高说明模型没在 \\boxed{} 里放")
    print("一个数（放了式子、句子，或者根本没写 boxed），那是**格式**问题，不是能力问题；")
    print("先去 responses.jsonl 抽几条看，再谈 accuracy 高低。")

    if not args.apply:
        print("\n预览模式，未写入任何文件。确认无误后加 --apply。")
        return

    for result in results:
        write_run(metric_dir, result)
        print(f"已写入 {metric_dir / result['run']}（原文件存为 .bak）")
    update_csv(out_dir, args.metric, results)


if __name__ == "__main__":
    main()
