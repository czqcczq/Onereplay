"""C 的谱有多尖，惩罚把力气花在哪，以及漂移到底落在不落在 C 看得见的地方。

起因是一组对不上的实测数：c_4b 把 |CΔW| 相对 vanilla 压到 1/13、R 降 58%，遗忘却只
追回 8%；而 α 回滚扫描显示，在**同一批 110 层**上把漂移等比例缩掉 25% 就能追回 34.7%。
也就是说大效应在这个参数空间里是存在的，是 C 没拿到。同时实测 ρ 从第 500 步的 0.0077
到第 2000 步的 0.0069 一直不动——惩罚在它够得着的地方已经到平衡了。

平衡态解释了 ρ 为什么与 λ 无关：某方向上惩罚力 λ·λ_j·Δw 与 LM 梯度分量 g 相抵时
Δw ≈ g/(λ·λ_j)，代回去 |g_reg| = λ·λ_j·Δw ≈ g。所以加大 λ 只是把**已经被约束的那些
方向**压得更贴近 W₀，够不着的地方一分力都不会多。于是「λ 太小」这个假设能不能成立，
就取决于一件可以直接量的事：**漂移的能量有多少落在 C 的高特征值方向上**。

三个产出：

1. **C 到底多尖**（top-k 特征值占迹的比例、有效秩）。重新量，不沿用旧数字——之前那个
   「99.995% 的迹集中在单个方向」出自 `onereplay/scripts/inspect_cov_layers.py`，层名是
   `layers.2.mlp.down_proj`，Llama 式命名，是 SFT/LoRA 那套实验的模型，跟这里的
   open-sci-ref 0.4B 不是一回事。

2. **惩罚用力 vs 漂移能量的错配**。把 ΔW 投到 C 的特征基上，第 j 个方向的漂移能量是
   e_j = ‖(ΔW V)_{:,j}‖²，而惩罚在这个方向上的用力正比于 λ_j·e_j（因为
   R = tr(ΔW C ΔWᵀ) = Σ_j λ_j e_j）。两个分布一对比就知道惩罚是不是在使偏了劲——
   如果它把九成力气花在只占百分之一漂移的方向上，前面所有现象都解释完了。

3. **硬上限**。特征值低于 τ·λ_max 的方向上，惩罚力小到要把 λ 放大 1/τ 倍才有同等效果；
   λ_j=0 的方向则是任何 λ 都碰不到。所以「漂移能量落在 λ_j ≥ τ·λ_max 的方向上的比例」
   这条曲线，就是 λ 能起作用的天花板。

为什么谱尖本身不算 bug：C = (1/N)Σ x xᵀ 是**未中心化**的二阶矩，而惩罚
R = E_x‖ΔW x‖² 要的正是「这层输出在旧数据上变了多少」，中心化会把这个含义弄坏。
未中心化必然带一个 rank-1 的 μμᵀ 项，再叠上 transformer 众所周知的 massive
activation（残差流里少数维度幅度大两三个数量级），谱尖是真实性质。真正可疑的是惩罚
**每个输出行共用同一个 C**，等于假设所有输出方向对 loss 同等重要——正确的 GGN 块是
E[xxᵀ] ⊗ E[δδᵀ]，输出侧那一半被整个丢掉了。这个脚本量不出输出侧，但它能确定「输入侧
挑的方向里到底有没有漂移」，这是先要排除的那一半。

用法：

    python -m kres.inspect_cov_spectrum \\
        --cov        cov/arm_4b.pt \\
        --base       model/<基座>/lit_model.pth \\
        --checkpoint out/vanilla/final/lit_model.pth

纯 CPU，110 层各做一次对称特征分解，几分钟。给 `--compare <C臂的检查点>` 会把 C 臂的
能量分布并排打出来（两个检查点必须是**同一步**的，否则比的是训练进度不是惩罚效果）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from kres.covariance import load_covariance_payload

# 按特征值降序的秩分桶。第一个桶单独留给 top-1：未中心化二阶矩里的 μμᵀ 是 rank-1 的，
# 如果它一个人就吃掉大半条迹，那要改的是惩罚形式而不是 λ
BUCKETS = [(1, 1), (2, 8), (9, 64), (65, 256), (257, 1 << 30)]
TAUS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return state


def bucket_of(rank: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= rank <= hi:
            return i
    return len(BUCKETS) - 1


def analyze_layer(cov: torch.Tensor, deltas: dict[str, torch.Tensor], dtype: torch.dtype):
    """返回该层的 (特征值降序, {臂名: 各方向的漂移能量})。

    特征分解走 `eigh` 而不是 `svd`：C 已经在 `finalize` 里对称化过，`eigh` 只读一个
    三角、也不会在近简并的特征值上给出乱序的奇异向量。默认 float64——d 最大到几千，
    多花的几秒换掉「小特征值被 fp32 舍成负数、τ 曲线尾部整段失真」这个风险。
    """
    evals, evecs = torch.linalg.eigh(cov.to(dtype))
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    energies = {}
    for arm, dw in deltas.items():
        # ΔW 是 (out, in)，C 在输入维上，所以投影是 ΔW @ V，列 j 对应第 j 个特征方向
        projected = dw.to(dtype) @ evecs
        energies[arm] = projected.pow(2).sum(dim=0)
    return evals, energies


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cov", type=Path, required=True, help="C 文件（kres.collect_cov 的产物）")
    ap.add_argument("--base", type=Path, required=True, help="基座 lit_model.pth（W₀）")
    ap.add_argument("--checkpoint", type=Path, required=True, help="要分析的检查点（W），一般是 vanilla")
    ap.add_argument("--compare", type=Path, default=None, help="可选：C 臂**同一步**的检查点")
    ap.add_argument("--per-layer", type=int, default=8, help="额外打印错配最严重的前几层")
    ap.add_argument("--float32", action="store_true", help="用 fp32 做特征分解（快一倍，尾部精度差）")
    args = ap.parse_args(argv)

    # Windows 控制台默认 GBK，打不出 ‖ΔW‖² 里的上标，会在输出到一半时崩掉——本地试跑
    # 时很容易误判成脚本有 bug。集群是 UTF-8，这行是空操作
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dtype = torch.float32 if args.float32 else torch.float64
    for p in (args.cov, args.base, args.checkpoint) + ((args.compare,) if args.compare else ()):
        if not p.is_file():
            raise SystemExit(f"找不到 {p}")

    payload = load_covariance_payload(args.cov)
    covariances = payload["covariances"]
    base = load_state(args.base)
    arms = {"vanilla": load_state(args.checkpoint)}
    if args.compare:
        arms["C 臂"] = load_state(args.compare)

    names = sorted(covariances)
    print(f"C          {args.cov}（{len(names)} 层）")
    print(f"基座       {args.base}")
    for arm, _ in arms.items():
        print(f"{arm:<10} {args.checkpoint if arm == 'vanilla' else args.compare}")

    n_b = len(BUCKETS)
    trace_b = [0.0] * n_b                                  # 各桶占 C 的迹
    energy_b = {a: [0.0] * n_b for a in arms}               # 各桶占漂移能量
    effort_b = {a: [0.0] * n_b for a in arms}               # 各桶占惩罚用力 λ_j·e_j
    tau_energy = {a: [0.0] * len(TAUS) for a in arms}       # 能量落在 λ_j ≥ τ·λmax 的比例
    totals = {"trace": 0.0}
    totals.update({f"energy_{a}": 0.0 for a in arms})
    totals.update({f"effort_{a}": 0.0 for a in arms})
    per_layer = []

    t0 = time.perf_counter()
    for i, name in enumerate(names, 1):
        cov = covariances[name]
        key = f"{name}.weight"
        if key not in base:
            raise SystemExit(f"C 里的 {name} 在权重里找不到 {key}，C 和这个模型不匹配")
        deltas = {a: (sd[key].float() - base[key].float()) for a, sd in arms.items()}
        if deltas["vanilla"].shape[1] != cov.shape[-1]:
            raise SystemExit(
                f"{name}: ΔW 的输入维 {deltas['vanilla'].shape[1]} 与 C 的 {cov.shape[-1]} 不符"
            )
        evals, energies = analyze_layer(cov, deltas, dtype)

        trace = float(evals.clamp(min=0).sum())
        totals["trace"] += trace
        lam_max = float(evals[0])
        ranks = torch.arange(1, evals.numel() + 1)
        bucket_id = torch.tensor([bucket_of(int(r)) for r in ranks])

        for b in range(n_b):
            m = bucket_id == b
            trace_b[b] += float(evals[m].clamp(min=0).sum())
        for arm, e in energies.items():
            effort = evals.clamp(min=0) * e
            totals[f"energy_{arm}"] += float(e.sum())
            totals[f"effort_{arm}"] += float(effort.sum())
            for b in range(n_b):
                m = bucket_id == b
                energy_b[arm][b] += float(e[m].sum())
                effort_b[arm][b] += float(effort[m].sum())
            for k, tau in enumerate(TAUS):
                tau_energy[arm][k] += float(e[evals >= tau * lam_max].sum())

        e_v = energies["vanilla"]
        top1_trace = float(evals[0].clamp(min=0)) / trace if trace > 0 else 0.0
        top1_energy = float(e_v[0]) / float(e_v.sum()) if float(e_v.sum()) > 0 else 0.0
        per_layer.append((top1_trace - top1_energy, name, top1_trace, top1_energy, int(evals.numel())))
        if i % 20 == 0 or i == len(names):
            print(f"  ... {i}/{len(names)} 层，用时 {time.perf_counter() - t0:.0f}s")

    tr = totals["trace"]
    print("\n=== C 的谱（{} 层聚合）===".format(len(names)))
    print(f"{'特征值秩':<14}{'占 C 的迹':>12}")
    for b, (lo, hi) in enumerate(BUCKETS):
        label = f"top-{lo}" if lo == hi else (f"{lo}..{hi}" if hi < (1 << 30) else f"{lo} 及以后")
        print(f"{label:<14}{trace_b[b] / tr:>11.2%}")

    print("\n=== 惩罚用力 vs 漂移能量（vanilla 的 ΔW）===")
    print(f"{'特征值秩':<14}{'占 C 的迹':>12}{'占惩罚用力':>14}{'占漂移能量':>14}")
    for b, (lo, hi) in enumerate(BUCKETS):
        label = f"top-{lo}" if lo == hi else (f"{lo}..{hi}" if hi < (1 << 30) else f"{lo} 及以后")
        print(
            f"{label:<14}{trace_b[b] / tr:>11.2%}"
            f"{effort_b['vanilla'][b] / totals['effort_vanilla']:>13.2%}"
            f"{energy_b['vanilla'][b] / totals['energy_vanilla']:>13.2%}"
        )
    print(
        "\n  「占惩罚用力」= λ_j·e_j 的份额（R = Σ_j λ_j e_j 的分解），"
        "「占漂移能量」= e_j 的份额（‖ΔW‖² 的分解）。\n"
        "  两列错开得越远，惩罚就越是在使偏劲：它按 λ_j 分配力气，而要压的是 e_j。"
    )

    print("\n=== 硬上限：漂移能量落在「λ_j ≥ τ·λmax」的方向上的比例 ===")
    header = f"{'τ':<10}" + "".join(f"{a:>14}" for a in arms)
    print(header)
    for k, tau in enumerate(TAUS):
        row = f"{tau:<10.0e}"
        for a in arms:
            row += f"{tau_energy[a][k] / totals[f'energy_{a}']:>13.2%}"
        print(row)
    print(
        "\n  这一列读作：在特征值低于 τ·λmax 的方向上，惩罚力要靠把 λ 放大 1/τ 倍才补得回来，\n"
        "  λ_j=0 的方向则任何 λ 都碰不到。所以某个 τ 处的比例，就是把 λ 放大 1/τ 倍所能\n"
        "  触及的漂移份额上限。如果 τ=1e-3 这一行还只有个位数百分比，那「λ 太小」就被排除了：\n"
        "  不是力度不够，是够不着。"
    )

    if args.compare:
        print("\n=== 两条臂的能量分布对比（同一组特征基）===")
        print(f"{'特征值秩':<14}" + "".join(f"{a + ' 能量':>14}" for a in arms) + f"{'C 臂/vanilla':>14}")
        for b, (lo, hi) in enumerate(BUCKETS):
            label = f"top-{lo}" if lo == hi else (f"{lo}..{hi}" if hi < (1 << 30) else f"{lo} 及以后")
            v, c = energy_b["vanilla"][b], energy_b["C 臂"][b]
            print(f"{label:<14}{v:>14.4g}{c:>14.4g}{(c / v if v > 0 else float('nan')):>13.3f}")
        print(
            "\n  最后一列 < 1 的桶就是惩罚真正压下去的地方。如果只有最前面几个桶明显小于 1、\n"
            "  而它们在上表里只占很小的漂移份额，那 C 压对了方向但那里本来就没多少东西。"
        )

    per_layer.sort(reverse=True)
    print(f"\n=== 错配最严重的 {args.per_layer} 层（top-1 占迹 − top-1 占漂移能量）===")
    print(f"{'层':<40}{'d':>7}{'top1 占迹':>12}{'top1 占能量':>13}")
    for _, name, t1, e1, d in per_layer[: args.per_layer]:
        print(f"{name:<40}{d:>7}{t1:>11.2%}{e1:>12.2%}")

    print(
        f"\n合计：tr(C) = {tr:.4g}，‖ΔW‖² = {totals['energy_vanilla']:.4g}"
        f"（‖ΔW‖ = {totals['energy_vanilla'] ** 0.5:.3f}），R·N = {totals['effort_vanilla']:.4g}"
    )
    print("  ‖ΔW‖ 应当与 kres.attribute_forgetting 打的「被正则」那个数一致，对不上说明层集合不同。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
