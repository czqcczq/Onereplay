"""把若干条臂的评测曲线对齐到同一张表，用于跨臂比较。

== 横轴选哪个，决定了你在回答哪个问题 ==

csv 里有两个可以当横轴的量，它们对 replay 臂**不等价**：

  step        走过的优化步数（csv 里存的是 iter_num，即 micro-batch 计数）。
              各臂每步的算力开销相同，所以同 step = **同算力预算**。
  new_tokens  只数新域 token。replay 臂每步有一部分 micro-batch 来自旧域，所以同一步
              它吃到的新域 token 比 vanilla 少。同 new_tokens = **同新域曝光**。

举例：第 200 步时 vanilla 已吃了 419M 新域 token，replay_8b 只吃了约 210M。按 step 对齐
去比遗忘，等于拿「学了两倍新域」的点去比「忘得更少」，replay 会被系统性地高估。

两个都是正当的问题，只是别混着说：
  同 step       →「给定算力预算，谁的综合表现好」
  同 new_tokens →「学到同样多的新东西，谁忘得少」（论文主表该用这个）

C 臂与 vanilla 都是零重放，两个横轴对它们完全重合，怎么对齐都一样。

== 恢复率 ==

    恢复率 = (baseline 的 probe − 本臂的 probe) / (baseline 的 probe − 起点 probe)

分母是 baseline 在该点的**总遗忘量**，所以 0% = 和 baseline 一样忘，100% = 完全没忘。
起点 probe 取 baseline 第一行（step 0 的初始评测）。注意分母随训练增长，所以同一个
绝对改善量在训练后期对应更小的恢复率百分比——跨点比较恢复率时要记得这件事。

用法：

    python -m kres.compare_curves --runs out/vanilla,out/replay_1b,out/replay_4b
    python -m kres.compare_curves --runs out/vanilla,out/c_1b_lam2e-4 --x step
    python -m kres.compare_curves --runs out/vanilla,out/replay_1b --metric loss
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

FIELDS = ("val_loss", "val_ppl", "probe_loss", "probe_ppl")


def load_run(root: Path) -> list[dict]:
    """读一条臂的所有评测行。

    续跑会新开一个 `version_N` 目录，只读 version_0 会缺掉后半段曲线且不报错——那种
    缺失在图上表现为一条提前截断的线，很容易被当成「这条臂没跑完」。所以这里把所有
    version 都读进来，同 step 的行后读的覆盖先读的（续跑段更新）。
    """
    files = sorted(root.glob("logs/csv/version_*/metrics.csv"))
    if not files:
        raise SystemExit(f"{root} 下找不到 logs/csv/version_*/metrics.csv")
    rows: dict[int, dict] = {}
    for f in files:
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                # 训练行没有 val_loss，只有评测行才有；用它筛掉每步都写的那些行
                if not r.get("val_loss"):
                    continue
                step = int(float(r["step"]))
                rec = {k: float(r[k]) for k in FIELDS if r.get(k)}
                rec["step"] = float(step)
                rec["new_tokens"] = float(r["new_tokens"]) if r.get("new_tokens") else float("nan")
                rows[step] = rec
    if not rows:
        raise SystemExit(f"{root} 的 csv 里没有评测行（val_loss 全空），是不是还没跑到第一个评测点")
    return [rows[s] for s in sorted(rows)]


def interp(series: list[dict], x_key: str, x: float, y_key: str) -> float | None:
    """在 series 上按 x_key 线性插值取 y_key。落在数据范围外返回 None 而不是外推。

    外推在这里是有害的：曲线尾部有 LR 退火造成的拐点，线性外推会给出一个看起来合理
    但系统性偏乐观的值，而它和真实测点长得一样，事后无法分辨。
    """
    xs = [r[x_key] for r in series]
    ys = [r.get(y_key) for r in series]
    if x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= x:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if y0 is None or y1 is None:
                return None
            return y1 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", required=True, help="逗号分隔的 out/<臂> 目录，第一个默认当 baseline")
    ap.add_argument("--baseline", default=None, help="用哪条臂算恢复率，默认取 --runs 的第一个")
    ap.add_argument("--x", choices=("new_tokens", "step"), default="new_tokens", help="横轴")
    ap.add_argument("--metric", choices=("ppl", "loss"), default="ppl")
    ap.add_argument("--grid", type=float, default=0.5e9, help="--x new_tokens 时的取点间隔")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    roots = [Path(x.strip()) for x in args.runs.split(",") if x.strip()]
    names = [r.name for r in roots]
    series = {n: load_run(r) for n, r in zip(names, roots)}
    base = args.baseline or names[0]
    if base not in series:
        raise SystemExit(f"--baseline {base} 不在 --runs 里（{names}）")

    val_key, probe_key = f"val_{args.metric}", f"probe_{args.metric}"
    floor = series[base][0].get(probe_key)
    print(f"baseline = {base}，起点 probe {args.metric} = {floor:.4f}"
          f"（取自 {base} 的第一行，step {series[base][0]['step']:.0f}）\n")
    for n in names:
        s = series[n]
        print(f"  {n:<20} {len(s):>3} 个评估点，"
              f"step {s[0]['step']:.0f}~{s[-1]['step']:.0f}，"
              f"新域 {s[0]['new_tokens'] / 1e9:.2f}B~{s[-1]['new_tokens'] / 1e9:.2f}B")

    # 横轴取点：step 用各臂共有的步，new_tokens 用固定间隔的网格（各臂不会落在同样的
    # new_tokens 上，必须插值）
    if args.x == "step":
        common = set.intersection(*(set(r["step"] for r in s) for s in series.values()))
        grid = sorted(common)
    else:
        hi = min(s[-1]["new_tokens"] for s in series.values())
        lo = max(s[0]["new_tokens"] for s in series.values())
        grid, x = [], max(lo, args.grid)
        while x <= hi + 1e-6:
            grid.append(x)
            x += args.grid
        # 终点几乎永远差网格一丁点：预算写的是 8.00B，实际跑出来是 7,999,979,520。照直
        # 算会把终态那一行整个丢掉，而那正是主表要用的一行，且丢得无声无息——表看上去
        # 完好，只是最后一行不见了
        if not grid or hi - grid[-1] > args.grid * 0.02:
            grid.append(hi)
    if not grid:
        raise SystemExit("各臂没有公共的横轴取值，没法对齐")

    xlabel = "新域(B)" if args.x == "new_tokens" else "step"
    head = f"{xlabel:>10} |"
    for n in names:
        head += f" {n[:22]:^26}|"
    print("\n" + head)
    sub = f"{'':>10} |"
    for _ in names:
        sub += f" {'val':>7}{'probe':>9}{'恢复':>9} |"
    print(sub)
    print("-" * len(head))

    for x in grid:
        base_probe = interp(series[base], args.x, x, probe_key)
        cells = []
        for n in names:
            v = interp(series[n], args.x, x, val_key)
            p = interp(series[n], args.x, x, probe_key)
            if v is None or p is None:
                cells.append(f" {'—':>7}{'—':>9}{'—':>9} |")
                continue
            span = (base_probe - floor) if base_probe is not None else None
            rec = f"{(base_probe - p) / span:>8.1%}" if span and abs(span) > 1e-12 else f"{'—':>8}"
            cells.append(f" {v:>7.4f}{p:>9.4f}{rec:>9} |")
        label = f"{x / 1e9:>10.2f}" if args.x == "new_tokens" else f"{x:>10.0f}"
        print(label + " |" + "".join(cells))

    print(f"""
横轴是 {args.x}。{'同 step = 同算力预算；replay 臂在同一步吃到的新域 token 比 vanilla 少，'
       '所以这张表对 replay 的遗忘是偏乐观的读法。' if args.x == 'step' else
       '同 new_tokens = 同新域曝光，跨臂比遗忘该用这个。'}
恢复率的分母是 {base} 在该点的总遗忘量，它随训练增长，所以同样的绝对改善在后期
对应更小的百分比。跨行比较恢复率时记得这件事。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
