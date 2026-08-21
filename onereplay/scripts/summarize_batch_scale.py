"""Summarize the batch-scaling cost scan from 15_bench_batch_scale.slurm.

Reads metrics/cost/batch_scale/*.jsonl and prints:

  1. Tokens/s and peak memory vs micro-batch (saturation curve)
  2. OneReplay overhead vs same-batch vanilla (ms/step and memory)

Usage:
    python -m onereplay.scripts.summarize_batch_scale \
      --metrics_dir <RESULTS_ROOT>/metrics/cost/batch_scale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MISSING = "—"
BATCH_RE = re.compile(r"_b(\d+)(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize batch-scaling cost scan.")
    parser.add_argument("--metrics_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def load_last_record(path: Path) -> dict[str, Any] | None:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return None
    record = records[-1]
    if record.get("train_sec") is None:
        return None
    return record


def classify(name: str) -> tuple[str, int, bool] | None:
    """Return (kind, batch, is_profile) or None if the name is not a scan run."""

    is_profile = "_prof_" in name
    match = BATCH_RE.search(name)
    if match is None:
        return None
    batch = int(match.group(1))
    if "vanilla" in name:
        return ("vanilla", batch, is_profile)
    if "onereplay" in name:
        return ("onereplay", batch, is_profile)
    return None


def number(value: Any, spec: str = ".2f", scale: float = 1.0) -> str:
    if value is None:
        return MISSING
    return format(value * scale, spec)


def pct_diff(value: Any, reference: Any) -> str:
    if value is None or not reference:
        return MISSING
    return f"{100.0 * (value / reference - 1.0):+.1f}%"


def render_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.is_dir():
        raise SystemExit(f"metrics_dir not found: {metrics_dir}")

    # kind -> batch -> record  (profile runs kept separately)
    clean: dict[str, dict[int, dict[str, Any]]] = {"vanilla": {}, "onereplay": {}}
    profiled: dict[str, dict[int, dict[str, Any]]] = {"vanilla": {}, "onereplay": {}}

    for path in sorted(metrics_dir.glob("*.jsonl")):
        info = classify(path.stem)
        if info is None:
            continue
        kind, batch, is_profile = info
        record = load_last_record(path)
        if record is None:
            print(f"skipping {path.name}: no cost fields")
            continue
        # Prefer the batch_size field when present; fall back to the name.
        batch = int(record.get("batch_size") or batch)
        target = profiled if is_profile else clean
        target[kind][batch] = record

    batches = sorted(set(clean["vanilla"]) | set(clean["onereplay"]))
    if not batches:
        raise SystemExit(f"no usable bscale_*.jsonl under {metrics_dir}")

    # ---- Table 1: saturation curve (vanilla) ----
    sat_header = [
        "batch",
        "accum_steps",
        "ms/step",
        "samples/s",
        "tokens/s",
        "vs b=prev tokens/s",
        "peak alloc (GiB)",
        "peak reserved (GiB)",
    ]
    sat_rows = []
    prev_tps = None
    for batch in batches:
        record = clean["vanilla"].get(batch)
        if record is None:
            continue
        accum = int(record.get("accumulation_size") or 64)
        accum_steps = max(accum // batch, 1)
        tps = record.get("tokens_per_sec")
        sat_rows.append(
            [
                str(batch),
                str(accum_steps),
                number(record.get("sec_per_step"), ".1f", scale=1000.0),
                number(record.get("samples_per_sec"), ".2f"),
                number(tps, ".0f"),
                pct_diff(tps, prev_tps) if prev_tps is not None else "基准",
                number(record.get("peak_memory_allocated_gb"), ".2f"),
                number(record.get("peak_memory_reserved_gb"), ".2f"),
            ]
        )
        prev_tps = tps

    # ---- Table 2: OneReplay overhead at each batch ----
    oh_header = [
        "batch",
        "vanilla ms/step",
        "onereplay ms/step",
        "time overhead",
        "vanilla tokens/s",
        "onereplay tokens/s",
        "vanilla peak (GiB)",
        "onereplay peak (GiB)",
        "Δ peak (GiB)",
        "C (GiB)",
    ]
    oh_rows = []
    for batch in batches:
        van = clean["vanilla"].get(batch)
        one = clean["onereplay"].get(batch)
        if van is None or one is None:
            continue
        van_peak = van.get("peak_memory_allocated_gb")
        one_peak = one.get("peak_memory_allocated_gb")
        delta_peak = (
            f"{one_peak - van_peak:+.2f}"
            if van_peak is not None and one_peak is not None
            else MISSING
        )
        oh_rows.append(
            [
                str(batch),
                number(van.get("sec_per_step"), ".1f", scale=1000.0),
                number(one.get("sec_per_step"), ".1f", scale=1000.0),
                pct_diff(one.get("sec_per_step"), van.get("sec_per_step")),
                number(van.get("tokens_per_sec"), ".0f"),
                number(one.get("tokens_per_sec"), ".0f"),
                number(van_peak, ".2f"),
                number(one_peak, ".2f"),
                delta_peak,
                number(one.get("covariance_memory_gb"), ".3f"),
            ]
        )

    sections = [
        "## Batch scaling：vanilla 吞吐 / 显存",
        "",
        render_table(sat_rows, sat_header),
        "",
        (
            "读法：`tokens/s` 相对上一档的增益开始接近 0（例如 <+10%）时，"
            "GPU 接近饱和；之后再增大 batch 主要换显存，几乎不换速度。"
            "有效 batch（`accumulation_size`）全程固定，只改 micro-batch。"
        ),
        "",
        "## OneReplay 相对同 batch vanilla 的开销",
        "",
        render_table(oh_rows, oh_header),
        "",
        (
            "读法：正则的绝对耗时大致固定（只跟 LoRA / C 有关），"
            "所以 `time overhead` 应随 batch 增大而下降；"
            "`Δ peak` 应接近 C 常驻显存（约 0.875 GiB），与 batch 无关。"
        ),
    ]

    # Optional profile table: replay_reg share vs batch
    prof_batches = sorted(set(profiled["vanilla"]) | set(profiled["onereplay"]))
    if prof_batches:
        ph_header = [
            "batch",
            "vanilla train (s)",
            "onereplay train (s)",
            "replay_reg (s / %)",
            "backward Δ (s)",
            "est. total reg (s / %)",
        ]
        ph_rows = []
        for batch in prof_batches:
            van = profiled["vanilla"].get(batch)
            one = profiled["onereplay"].get(batch)
            if van is None or one is None:
                continue
            van_train = van.get("train_sec") or 0.0
            one_train = one.get("train_sec") or 0.0
            reg_fwd = one.get("time_replay_reg_sec") or 0.0
            van_bwd = van.get("time_backward_sec") or 0.0
            one_bwd = one.get("time_backward_sec") or 0.0
            bwd_delta = one_bwd - van_bwd
            total_reg = reg_fwd + max(bwd_delta, 0.0)
            ph_rows.append(
                [
                    str(batch),
                    number(van_train, ".1f"),
                    number(one_train, ".1f"),
                    f"{reg_fwd:.2f} / {100.0 * reg_fwd / one_train:.1f}%" if one_train else MISSING,
                    f"{bwd_delta:+.2f}",
                    (
                        f"{total_reg:.2f} / {100.0 * total_reg / van_train:.1f}%"
                        if van_train
                        else MISSING
                    ),
                ]
            )
        sections += [
            "",
            "## 分阶段（仅 RUN_PROFILE=1）：正则占比随 batch 下降",
            "",
            render_table(ph_rows, ph_header),
            "",
            (
                "`replay_reg` 只含前向；反向增量在 `backward Δ` 里。"
                "`est. total reg` = 前向 + max(backwardΔ, 0)，相对 vanilla train 的占比。"
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
