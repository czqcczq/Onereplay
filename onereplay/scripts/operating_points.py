"""Tabulate every run's operating point on the plasticity/retention tradeoff.

Comparing two anti-forgetting methods at whatever lambda each happened to be
run at says nothing: a higher retention score bought with a higher new-task
val_loss is just a more conservative point on the same curve, not a better
method. What settles it is either a matched val_loss or the whole curve. Both
need the same table, which is what this prints.

It also surfaces the two things that silently invalidate a comparison and are
invisible in the results directory: how many epochs a run actually trained (the
_probe<steps>s<epochs>e suffix only appears on some arms, and PBS 71 has no
such suffix at all), and lambda*R / task_loss, whose calibrated band is
0.01-1 -- a lambda outside it is either doing nothing or drowning the task.

Usage
-----
  python -m onereplay.scripts.operating_points \
      --results_root /scratch/.../results/qwen3-8b

  # add the tradeoff plot and a csv for the paper
  python -m onereplay.scripts.operating_points \
      --results_root /scratch/.../results/qwen3-8b \
      --csv operating_points.csv --plot tradeoff.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The one number worth putting in a table for each metric's summary.json.
HEADLINE_FIELD = {
    "ifeval": "strict_prompt_accuracy",
    # IFBench 主报 loose：它的约束比 IFEval 难，strict 在 300 条上贴地板。
    "ifbench": "loose_prompt_accuracy",
    "multiif": "strict_prompt_accuracy",
    "gsm8k": "accuracy",
    "math500": "accuracy",
    "humaneval": "pass_at_1",
    "mbpp": "pass_at_1",
    "commonsense": "val_loss",
}
DEFAULT_METRICS = "commonsense,ifeval,ifbench,gsm8k,math500,humaneval,mbpp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True, help="含 metrics/ 与各 metric 目录的根")
    parser.add_argument("--out_dir", default="", help="评测产物根，默认同 --results_root")
    parser.add_argument("--runs", default="", help="逗号分隔；留空=自动发现")
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--name_width", type=int, default=54)
    parser.add_argument(
        "--evaluated_only",
        action="store_true",
        help="只列跑过至少一个 benchmark 的 run（滤掉短跑探针）",
    )
    parser.add_argument("--csv", default="")
    parser.add_argument("--plot", default="", help="画 val_loss vs 保留指标的权衡图")
    parser.add_argument("--plot_metric", default="ifeval")
    return parser.parse_args()


def read_training_record(metrics_path: Path) -> dict[str, Any]:
    """Pull the last epoch record, which carries the run's final state."""

    if not metrics_path.is_file():
        return {}
    epochs = []
    for line in metrics_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("record_type") == "epoch":
            epochs.append(record)
    if not epochs:
        return {}
    last = max(epochs, key=lambda r: r.get("epoch", 0))
    task_loss = last.get("train_task_loss")
    lambda_reg = last.get("train_lambda_reg")
    ratio = None
    if task_loss:
        ratio = (lambda_reg or 0.0) / task_loss
    return {
        "epochs": last.get("epoch"),
        "lambda": last.get("replay_lambda"),
        "train_task_loss": task_loss,
        "lambda_reg": lambda_reg,
        "reg_ratio": ratio,
        "train_val_loss": last.get("val_loss"),
    }


def read_metric_scores(out_dir: Path, run: str, metrics: list[str]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for metric in metrics:
        summary_path = out_dir / metric / run / "summary.json"
        if not summary_path.is_file():
            scores[metric] = None
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            scores[metric] = None
            continue
        scores[metric] = summary.get(HEADLINE_FIELD.get(metric, "accuracy"))
    return scores


def discover_runs(results_root: Path, out_dir: Path, metrics: list[str]) -> list[str]:
    names: set[str] = set()
    metrics_dir = results_root / "metrics"
    if metrics_dir.is_dir():
        names.update(path.stem for path in metrics_dir.glob("*.jsonl"))
    for metric in metrics:
        metric_dir = out_dir / metric
        if metric_dir.is_dir():
            names.update(p.name for p in metric_dir.iterdir() if (p / "summary.json").is_file())
    return sorted(names)


def shorten(name: str, width: int) -> str:
    """Trim the middle: both ends carry the identifying bits (method, lambda)."""

    if len(name) <= width:
        return name
    keep = width - 3
    head = keep * 2 // 3
    return f"{name[:head]}...{name[-(keep - head):]}"


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_root
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    runs = [r.strip() for r in args.runs.split(",") if r.strip()] or discover_runs(
        results_root, out_dir, metrics
    )
    if not runs:
        raise SystemExit(f"在 {results_root} 下没发现任何 run")

    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {"run": run}
        row.update(read_training_record(results_root / "metrics" / f"{run}.jsonl"))
        row.update(read_metric_scores(out_dir, run, metrics))
        # The commonsense *metric* is a separate eval that often was never run;
        # the trainer's own held-out loss is the same quantity and is always
        # there, so fall back to it rather than leaving the axis blank.
        row["plasticity"] = (
            row.get("commonsense")
            if row.get("commonsense") is not None
            else row.get("train_val_loss")
        )
        if args.evaluated_only and not any(row.get(m) is not None for m in metrics):
            continue
        rows.append(row)

    width = args.name_width
    header = (
        f"{'run':<{width}}{'ep':>4}{'lambda':>10}{'lamR/task':>11}{'val_loss':>10}"
        + "".join(f"{m[:9]:>11}" for m in metrics)
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{shorten(row['run'], width):<{width}}"
            f"{format_number(row.get('epochs'), 2):>4}"
            f"{format_number(row.get('lambda'), 3):>10}"
            f"{format_number(row.get('reg_ratio'), 3):>11}"
            f"{format_number(row.get('plasticity'), 4):>10}"
        )
        for metric in metrics:
            line += f"{format_number(row.get(metric), 4):>11}"
        print(line)

    print()
    print("ep        = metrics jsonl 里实际训到的 epoch 数（run 名不一定反映它）")
    print("lamR/task = 最后一个 epoch 的 lambda*R / task_loss，判据带是 0.01~1")
    print("val_loss  = 新任务 held-out loss，越低=可塑性越好（等 val_loss 才能比保留）")
    print("            优先取 commonsense 评测的值，没有就取训练最后一个 epoch 的")
    print("其余列    = 各 benchmark 的头号指标（ifeval 为 strict prompt acc）")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run", "epochs", "lambda", "train_task_loss", "lambda_reg", "reg_ratio",
                      "train_val_loss", "plasticity", *metrics]
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n写入 {csv_path}")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n没有 matplotlib，跳过绘图（表格和 csv 不受影响）")
            return
        points = [
            (row["plasticity"], row[args.plot_metric], row["run"])
            for row in rows
            if row.get("plasticity") is not None and row.get(args.plot_metric) is not None
        ]
        if not points:
            print(f"\n没有同时具备 val_loss 与 {args.plot_metric} 的 run，跳过绘图")
            return
        figure, axes = plt.subplots(figsize=(9, 6))
        for val_loss, score, run in points:
            axes.scatter(val_loss, score)
            axes.annotate(shorten(run, 28), (val_loss, score), fontsize=7,
                          xytext=(4, 4), textcoords="offset points")
        axes.set_xlabel("new-task held-out val_loss  (lower = more plasticity)")
        axes.set_ylabel(f"{args.plot_metric} ({HEADLINE_FIELD.get(args.plot_metric)})")
        axes.set_title("plasticity / retention operating points")
        axes.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(args.plot, dpi=150)
        print(f"\n写入 {args.plot}")


if __name__ == "__main__":
    main()
