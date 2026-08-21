"""Turn training metrics jsonl files into the efficiency comparison table.

Every training run appends one record per epoch, and each record already
carries the cost fields collected by onereplay.core.profiling: wall time,
throughput, peak memory, the resident size of C, and — when the run used
--profile 1 — the per-phase breakdown.

This script reads those records and renders two markdown tables: one for
throughput and memory, one for the phase breakdown.

    python -m onereplay.scripts.summarize_cost \
      --metrics_dir <RESULTS_ROOT>/metrics/cost \
      --baseline bench_vanilla_seed1

Note on rows the tables cannot contain: the base model never trains, so its
training cost is "—" rather than zero.

Usage: python -m onereplay.scripts.summarize_cost [args]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PHASES = ("prepare_batch", "task_loss", "replay_reg", "backward", "optimizer")

MISSING = "—"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OneReplay training cost.")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="Directory of *.jsonl metrics files written by --metrics_path.",
    )
    parser.add_argument(
        "--extra_metrics_dir",
        type=str,
        default="",
        help=(
            "Second directory to pull runs from, matched by --extra_glob. Used to "
            "fold the replay sweep's own metrics into the same table."
        ),
    )
    parser.add_argument("--extra_glob", type=str, default="cs_replay_*.jsonl")
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help=(
            "Which epoch to report. Epoch 1 is the fair choice because the cost "
            "benchmark only runs one epoch. 0 uses each run's last epoch."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        help="Run name whose cost the overhead columns are relative to.",
    )
    parser.add_argument("--out", type=str, default="", help="Optional markdown output path.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_record(records: list[dict[str, Any]], epoch: int) -> dict[str, Any] | None:
    """Pick the requested epoch, falling back to the last one recorded."""

    if not records:
        return None
    if epoch > 0:
        for record in records:
            if record.get("epoch") == epoch:
                return record
    return records[-1]


def collect_runs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Map run name -> the one record we report for it."""

    paths = sorted(Path(args.metrics_dir).glob("*.jsonl"))
    if args.extra_metrics_dir:
        paths += sorted(Path(args.extra_metrics_dir).glob(args.extra_glob))

    runs: dict[str, dict[str, Any]] = {}
    for path in paths:
        record = select_record(load_records(path), args.epoch)
        if record is None:
            print(f"skipping {path.name}: no records")
            continue
        if record.get("train_sec") is None:
            # Written by the legacy training script, which had no cost fields.
            print(f"skipping {path.name}: no cost fields, re-run with onereplay.scripts.train")
            continue
        runs[path.stem] = record
    return runs


def order_runs(runs: dict[str, dict[str, Any]], baseline: str) -> list[str]:
    names = sorted(runs)
    if baseline in runs:
        names.remove(baseline)
        names.insert(0, baseline)
    return names


def is_profiled(record: dict[str, Any]) -> bool:
    return any(f"time_{phase}_sec" in record for phase in PHASES)


def number(value: Any, spec: str = ".2f", scale: float = 1.0) -> str:
    if value is None:
        return MISSING
    return format(value * scale, spec)


def ratio(value: Any, reference: Any) -> str:
    """Render value/reference as a signed percentage difference."""

    if value is None or not reference:
        return MISSING
    return f"{100.0 * (value / reference - 1.0):+.1f}%"


def render_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_cost_table(
    runs: dict[str, dict[str, Any]],
    names: list[str],
    baseline: str,
) -> str:
    base = runs.get(baseline, {})
    header = [
        "run",
        "λ",
        "replay r",
        "steps",
        "epoch time (s)",
        "vs base",
        "ms/step",
        "vs base",
        "samples/s",
        "tokens/s",
        "peak alloc (GiB)",
        "peak reserved (GiB)",
        "of which C (GiB)",
    ]
    rows = []
    for name in names:
        record = runs[name]
        lam = record.get("replay_lambda") or 0.0
        replay_ratio = record.get("replay_ratio") or 0.0
        rows.append(
            [
                name,
                f"{lam:g}" if lam else "0",
                f"{replay_ratio:g}" if replay_ratio else "0",
                str(record.get("train_steps", MISSING)),
                number(record.get("train_sec"), ".1f"),
                ratio(record.get("train_sec"), base.get("train_sec")),
                number(record.get("sec_per_step"), ".1f", scale=1000.0),
                ratio(record.get("sec_per_step"), base.get("sec_per_step")),
                number(record.get("samples_per_sec"), ".2f"),
                number(record.get("tokens_per_sec"), ".0f"),
                number(record.get("peak_memory_allocated_gb"), ".2f"),
                number(record.get("peak_memory_reserved_gb"), ".2f"),
                number(record.get("covariance_memory_gb"), ".3f"),
            ]
        )
    return render_table(rows, header)


def build_phase_table(runs: dict[str, dict[str, Any]], profiled: list[str]) -> str:
    if not profiled:
        return ""

    header = ["run", "steps"] + [f"{phase} (s / %)" for phase in PHASES]
    rows = []
    for name in profiled:
        record = runs[name]
        train_sec = record.get("train_sec") or 0.0
        cells = [name, str(record.get("train_steps", MISSING))]
        for phase in PHASES:
            seconds = record.get(f"time_{phase}_sec")
            if seconds is None:
                cells.append(MISSING)
            elif train_sec:
                cells.append(f"{seconds:.1f} / {100.0 * seconds / train_sec:.1f}%")
            else:
                cells.append(f"{seconds:.1f}")
        rows.append(cells)
    return render_table(rows, header)


def main() -> None:
    args = parse_args()
    runs = collect_runs(args)
    if not runs:
        raise SystemExit(f"no usable metrics files under {args.metrics_dir}")

    names = order_runs(runs, args.baseline)
    profiled = [name for name in names if is_profiled(runs[name])]
    # Profiled runs are short and synchronize-slowed, so their wall time is not
    # comparable with the full runs and would poison the overhead columns.
    headline = [name for name in names if name not in profiled] or names

    sections = [
        "## 训练开销",
        "",
        build_cost_table(runs, headline, args.baseline),
        "",
        (
            f"epoch={args.epoch}；`vs base` 相对 `{args.baseline}`。"
            if args.baseline in runs
            else "未指定或未找到 baseline，`vs base` 列为空。"
        ),
        (
            "`epoch time` 变长可能来自更多步数（replay 追加了样本），`ms/step` 变长"
            "才是单步变慢。base 模型不训练，训练开销一栏应写 “—”。"
        ),
    ]
    if headline is names and profiled:
        sections.append(
            "警告：没有找到未开 profile 的运行，上表 wall time 含 synchronize 开销，不可对外报。"
        )

    phase_table = build_phase_table(runs, profiled)
    if phase_table:
        sections += [
            "",
            "## 分阶段耗时（仅 --profile 1 的运行）",
            "",
            phase_table,
            "",
            (
                "这些运行每个阶段都做了 cuda synchronize，整体被拖慢，因此只读占比，"
                "对外报的 wall time 取上表 PROFILE=0 的全量运行。"
            ),
        ]

    report = "\n".join(sections)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
