"""Solve for the lambda that puts an arm at a target new-task val_loss.

Comparing two anti-forgetting methods only means something at matched
plasticity, so one arm has to be moved onto the other's val_loss. Guessing the
step by decades wastes runs, because val_loss is not linear in lambda: it
approaches the unregularized floor asymptotically, so the same decade of
lambda buys less and less as the floor gets closer.

What is roughly linear is the *excess over the floor* against lambda, in log
space. This fits

    val_loss(lambda) = floor + a * lambda ** b

to the runs already on disk and inverts it at the target. The floor comes from
the matched vanilla run, which is what lambda -> 0 converges to; without it the
fit has no anchor and systematically overshoots.

Usage
-----
  python -m onereplay.scripts.calibrate_lambda \
      --results_root /scratch/.../results/qwen3-8b \
      --pattern 'cs_ewc_8b_ifgold_r16all_ep1_lam*_regonce_probe0s1e' \
      --floor_run cs_vanilla_8b_r16all_ep1_lr1e-4_seed1 \
      --target 0.0297
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--pattern", default="", help="run 名通配，如 'cs_ewc_8b_ifgold_*'")
    parser.add_argument("--runs", default="", help="逗号分隔的 run 名，覆盖 --pattern")
    parser.add_argument("--floor_run", required=True, help="匹配的 vanilla run（lambda=0 的地板）")
    parser.add_argument("--target", type=float, required=True, help="要命中的 val_loss")
    parser.add_argument("--epochs", type=int, default=0, help=">0 时只用训到该 epoch 数的 run")
    return parser.parse_args()


def read_point(metrics_path: Path) -> dict[str, Any] | None:
    """Last epoch record's lambda / val_loss, or None if unusable."""

    if not metrics_path.is_file():
        return None
    records = []
    for line in metrics_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("record_type") == "epoch":
            records.append(record)
    if not records:
        return None
    last = max(records, key=lambda r: r.get("epoch", 0))
    if last.get("val_loss") is None:
        return None
    return {
        "run": metrics_path.stem,
        "lambda": last.get("replay_lambda"),
        "val_loss": float(last["val_loss"]),
        "epochs": last.get("epoch"),
    }


def fit_power_law(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares on log(excess) = log a + b log lambda."""

    xs = [math.log(lam) for lam, _ in points]
    ys = [math.log(excess) for _, excess in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise SystemExit("所有点的 lambda 相同，无法拟合")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    return math.exp(intercept), slope


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.results_root) / "metrics"
    if not metrics_dir.is_dir():
        raise SystemExit(f"缺 {metrics_dir}")

    floor_point = read_point(metrics_dir / f"{args.floor_run}.jsonl")
    if floor_point is None:
        raise SystemExit(f"读不到地板 run 的 val_loss: {args.floor_run}")
    floor = floor_point["val_loss"]
    print(f"地板 (lambda=0): {args.floor_run}  val_loss={floor:.6f}  ep={floor_point['epochs']}")

    if args.runs.strip():
        names = [r.strip() for r in args.runs.split(",") if r.strip()]
    else:
        names = sorted(
            path.stem
            for path in metrics_dir.glob("*.jsonl")
            if fnmatch.fnmatch(path.stem, args.pattern or "*")
        )

    points: list[dict[str, Any]] = []
    for name in names:
        point = read_point(metrics_dir / f"{name}.jsonl")
        if point is None or not point["lambda"]:
            continue
        if args.epochs and point["epochs"] != args.epochs:
            continue
        point["excess"] = point["val_loss"] - floor
        points.append(point)

    if not points:
        raise SystemExit("没有可用的 lambda 点（要求 lambda>0 且有 val_loss）")

    print()
    print(f"{'run':<58}{'ep':>4}{'lambda':>12}{'val_loss':>11}{'excess':>11}")
    print("-" * 96)
    for point in sorted(points, key=lambda p: p["lambda"]):
        print(
            f"{point['run'][:58]:<58}{point['epochs']:>4}{point['lambda']:>12.4g}"
            f"{point['val_loss']:>11.6f}{point['excess']:>11.6f}"
        )

    usable = [(p["lambda"], p["excess"]) for p in points if p["excess"] > 0]
    if len(usable) < 2:
        raise SystemExit(
            "\n至少要两个 excess>0 的点才能拟合。"
            "\nexcess<=0 说明该 run 的 val_loss 已经不高于 vanilla，地板选错了或噪声盖过了差异。"
        )

    target_excess = args.target - floor
    if target_excess <= 0:
        raise SystemExit(
            f"\n目标 val_loss {args.target} 不高于地板 {floor:.6f}。"
            "\n正则化只会让 val_loss 上升，这个目标任何 lambda>0 都到不了。"
            "\n说明另一条臂的可塑性已经等于甚至好于无正则，改为把另一条臂往上调。"
        )

    coefficient, exponent = fit_power_law(usable)
    if exponent <= 0:
        raise SystemExit("\n拟合出的指数非正（val_loss 随 lambda 下降），数据不自洽，先查这几个 run")
    solution = (target_excess / coefficient) ** (1.0 / exponent)

    print()
    print(f"拟合: val_loss = {floor:.6f} + {coefficient:.4g} * lambda^{exponent:.4g}"
          f"   （{len(usable)} 个点）")
    if len(usable) == 2:
        print("      只有两个点 => 恰定拟合，没有残差可以检验，这是外推不是内插")
    print(f"目标 val_loss {args.target}  =>  建议 lambda ≈ {solution:.4g}")

    print()
    print(f"{'lambda':>12}{'预测 val_loss':>16}")
    print("-" * 28)
    observed = sorted(lam for lam, _ in usable)
    candidates = sorted({solution / 3, solution, solution * 3, *observed})
    for lam in candidates:
        predicted = floor + coefficient * lam**exponent
        mark = "  <- 建议" if abs(lam - solution) < 1e-12 else ""
        inside = "" if observed[0] <= lam <= observed[-1] else "  (外推)"
        print(f"{lam:>12.4g}{predicted:>16.6f}{inside}{mark}")

    print()
    print("提醒：这是从 val_loss 反推的，不保证 lambda*R/task_loss 还落在 0.01~1 的判据带里。")
    print("      为了对齐可塑性而把罚项压到带外是正常的，但要在写作里说明。")


if __name__ == "__main__":
    main()
