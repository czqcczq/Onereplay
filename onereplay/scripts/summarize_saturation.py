"""Summarize the GPU-saturation check from 16_bench_saturation.slurm.

Every micro-batch in that scan consumes the same number of training samples
(b=8 runs 800 steps, b=16 runs 400, ...), so the total wall time answers the
saturation question directly:

  * equal total time  -> time scales with compute -> the small batch already
    saturates the GPU, and doubling the micro-batch buys nothing but memory
  * small batch much slower -> it left compute idle -> a larger micro-batch is
    genuinely cheaper per sample

That ratio also decides the replay layout: 4+4 with doubled accumulation always
costs 2.0x the total time, while 8+8 costs (2 * ratio)x. Comparing ratio
against 1.0 picks the winner.

Repeats are reduced with min(), not mean: interference only ever makes a run
slower, so the fastest observation is the best estimate of the true cost.

Usage:
    python -m onereplay.scripts.summarize_saturation \
      --metrics_dir <RESULTS_ROOT>/metrics/cost/saturation
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
RUN_RE = re.compile(r"^sat_b(\d+)_r(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the GPU saturation check.")
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


def number(value: Any, spec: str = ".2f", scale: float = 1.0) -> str:
    if value is None:
        return MISSING
    return format(value * scale, spec)


def render_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def verdict(ratio: float) -> str:
    if ratio >= 0.95:
        return "**已饱和**：加大 micro-batch 换不到吞吐，只换显存"
    if ratio >= 0.80:
        return "**接近饱和**：加大 micro-batch 只有个位数到两成的红利"
    return "**明显欠载**：加大 micro-batch 能实打实省时间"


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.is_dir():
        raise SystemExit(f"metrics_dir not found: {metrics_dir}")

    observations: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(metrics_dir.glob("*.jsonl")):
        match = RUN_RE.match(path.stem)
        if match is None:
            continue
        record = load_last_record(path)
        if record is None:
            print(f"skipping {path.name}: no cost fields")
            continue
        record["_rep"] = int(match.group(2))
        batch = int(record.get("batch_size") or match.group(1))
        observations.setdefault(batch, []).append(record)

    if not observations:
        raise SystemExit(f"no usable sat_b*_r*.jsonl under {metrics_dir}")

    batches = sorted(observations)
    best = {batch: min(runs, key=lambda r: r["train_sec"]) for batch, runs in observations.items()}

    # ---- Table 1: every observation, so the spread stays visible ----
    obs_header = [
        "batch",
        "rep",
        "steps",
        "samples",
        "tokens",
        "train (s)",
        "ms/step",
        "tokens/s",
        "peak alloc (GiB)",
    ]
    obs_rows = []
    for batch in batches:
        for record in sorted(observations[batch], key=lambda r: r["_rep"]):
            obs_rows.append(
                [
                    str(batch),
                    str(record["_rep"]),
                    str(record.get("train_steps", MISSING)),
                    str(record.get("train_samples", MISSING)),
                    str(record.get("train_tokens", MISSING)),
                    number(record.get("train_sec"), ".2f"),
                    number(record.get("sec_per_step"), ".1f", scale=1000.0),
                    number(record.get("tokens_per_sec"), ".0f"),
                    number(record.get("peak_memory_allocated_gb"), ".2f"),
                ]
            )

    # ---- Consistency check: the comparison is only valid on equal workloads ----
    warnings = []
    sample_counts = {batch: best[batch].get("train_samples") for batch in batches}
    token_counts = {batch: best[batch].get("train_tokens") for batch in batches}
    if len(set(sample_counts.values())) > 1:
        warnings.append(
            f"各档处理的样本数不一致 {sample_counts}，总时间不能直接比；"
            "请检查 SAMPLES 是否被每个 batch 整除。"
        )
    if len(set(token_counts.values())) > 1:
        warnings.append(
            f"各档的 token 总数不一致 {token_counts}，说明 shuffle 出的样本集合随 "
            "batch 变了；请改看下面按 token 归一化的那一列。"
        )

    # ---- Table 2: per-batch best, against the smallest batch ----
    ref_batch = batches[0]
    ref = best[ref_batch]
    ref_sec = ref["train_sec"]
    ref_tokens = ref.get("train_tokens") or 0
    ref_per_token = ref_sec / ref_tokens if ref_tokens else None

    sum_header = [
        "batch",
        "steps",
        "最快 train (s)",
        f"vs b{ref_batch} 总时间",
        "按 token 归一后",
        "单步倍数",
        "ms/step",
        "tokens/s",
        "peak alloc (GiB)",
        "判定",
    ]
    sum_rows = []
    for batch in batches:
        record = best[batch]
        ratio = record["train_sec"] / ref_sec
        tokens = record.get("train_tokens") or 0
        per_token = record["train_sec"] / tokens if tokens else None
        norm = (
            f"{per_token / ref_per_token:.3f}"
            if per_token is not None and ref_per_token
            else MISSING
        )
        sum_rows.append(
            [
                str(batch),
                str(record.get("train_steps", MISSING)),
                number(record["train_sec"], ".2f"),
                "基准" if batch == ref_batch else f"{ratio:.3f}",
                "基准" if batch == ref_batch else norm,
                "基准" if batch == ref_batch else f"{ratio * batch / ref_batch:.2f}x",
                number(record.get("sec_per_step"), ".1f", scale=1000.0),
                number(record.get("tokens_per_sec"), ".0f"),
                number(record.get("peak_memory_allocated_gb"), ".2f"),
                "基准" if batch == ref_batch else verdict(ratio),
            ]
        )

    sections = [
        f"## GPU 饱和检查：每档固定吃 {ref.get('train_samples', '?')} 个样本",
        "",
        "### 逐次观测",
        "",
        render_table(obs_rows, obs_header),
        "",
        (
            "重复之间取**最快**的一次作为该档代表：外部干扰只会让 run 变慢、不会变快，"
            "所以最小值是对真实开销的最好估计。若同一档的多次观测散布很大，"
            "说明节点仍有干扰，结论要打折。"
        ),
        "",
        "### 汇总与判据",
        "",
        render_table(sum_rows, sum_header),
        "",
        (
            f"读法：各档吃掉的样本数、token 数、优化器更新次数都相同，所以 "
            f"`vs b{ref_batch} 总时间` 接近 1.000 就意味着时间正比于计算量，"
            f"即 b{ref_batch} 已经吃满吞吐；明显小于 1 才说明 b{ref_batch} 有闲置算力。"
            "`单步倍数` 是换算到「每步耗时相对基准的倍数」，"
            "等于总时间比 × batch 比。"
        ),
    ]

    # ---- Replay layout implication, only when the 2x batch exists ----
    double = ref_batch * 2
    if double in best:
        ratio = best[double]["train_sec"] / ref_sec
        step_factor = ratio * 2.0
        ref_peak = ref.get("peak_memory_allocated_gb")
        double_peak = best[double].get("peak_memory_allocated_gb")
        if step_factor <= 2.0:
            choice = (
                f"**8+8 更省时间**（×{step_factor:.2f} < ×2.00），"
                f"代价是峰值显存从 {number(ref_peak)} 升到 {number(double_peak)} GiB。"
                "省下的时间是否值这份显存，要看还要不要给更长的序列留余量。"
            )
        else:
            choice = (
                f"**4+4 + accum×2 更优**：它 ×2.00，8+8 要 ×{step_factor:.2f}，"
                f"而且峰值显存维持在 {number(ref_peak)} GiB 而不是 {number(double_peak)} GiB。"
                "两个维度都不差，直接选它。"
            )
        sections += [
            "",
            "### 对 replay 混合方案的结论（replay_ratio=0.5）",
            "",
            (
                f"两种方案都保证每次更新的 new task 样本数不变：\n\n"
                f"- `4+4` + accumulation ×2：单步耗时不变、步数翻倍 → 总时间恒为 **×2.00**，"
                f"峰值显存维持 b{ref_batch} 的水平，与是否饱和无关。\n"
                f"- `8+8`：步数不变、单步变成 b{double} 的耗时 → 总时间 **×{step_factor:.2f}**，"
                f"峰值显存升到 b{double} 的水平。\n\n"
                f"{choice}"
            ),
        ]

    if warnings:
        sections += ["", "### 口径警告", ""] + [f"- {line}" for line in warnings]

    report = "\n".join(sections)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
