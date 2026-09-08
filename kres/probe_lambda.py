"""从一个检查点反算 λ 的搜索区间：给定目标比例 ρ，输出该用的 λ。

λ 没有自然量纲。C 的量级由语料的激活统计定，ΔW 由学习率乘步数定，`|g_lm|` 由损失地形
定，三者相乘之后 λ=1 可能意味着惩罚是任务梯度的百万分之一（数值上完全等于 baseline），
也可能是一百万倍（模型冻死）。**搜索空间宽到十个数量级，而两端的失效都不显眼**：λ 太小
得到一条与 baseline 难以区分的曲线，太大则新域 loss 降不下去、看起来像别的 bug。盲扫要么
把 run 浪费在纯 no-op 上，要么撞不到有效区间就下结论说方法无效。

救命的是注入梯度对 λ **严格线性**——`regularizer.py` 的 `update = delta_c * 2·scale`、
`scale = λ/层数`。所以只要量出 λ=1 时的 `|g_reg|`，任何目标比例 ρ 都是一次除法：

    λ = ρ · |g_lm| / |g_reg|(λ=1)

十个数量级就压成了「挑几个 ρ」。

**为什么是离线算而不是在训练里探。** `|g_reg|` 只依赖 ΔW = W − W₀ 和 C，两个都在磁盘
上，所以任何一个检查点都能事后算，不需要在训练循环里加探针。这不只是省事：baseline 臂
同时是**开销对比的分母**，让它扛着惩罚计算，"C 臂比 baseline 贵多少"就变成拿一个已经付过
惩罚成本的基线去比，测出来的开销会接近零——要测的量被测没了。

**这只定区间，不定取值。** 最优 λ 是遗忘与新域的权衡，取决于更在乎哪一头，没有公式能算。
反算之后还要在给出的区间里扫几个点。

用法：

    python -m kres.probe_lambda \
        --base       model/<基座>/lit_model.pth \
        --checkpoint out/vanilla/step-00000750/lit_model.pth \
        --cov        cov/seg00.pt \
        --log        logs/vanilla.log

`--log` 会从训练日志里取 `|g_lm|`（也可以 `--g-lm` 直接给数）。取的是检查点附近一段的
**中位数**而不是某一步的瞬时值：`|g_lm|` 逐步抖动能到几十个百分点，拿单点定 λ 等于让一次
随机的 loss spike 决定整条臂的惩罚强度。
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

import torch

from kres.covariance import load_covariance_payload
from kres.regularizer import full_covariance_grad_, load_reference_weights

# 训练日志里那行：`... step 750 | ... |g_lm|: 0.834 ...`
RE_STEP = re.compile(r"\bstep (\d+)\b")
RE_G_LM = re.compile(r"\|g_lm\|: ([\d.]+|nan)")


def step_from_checkpoint(path: Path) -> int | None:
    """从 `.../step-00000750/lit_model.pth` 里抠出步数。抠不出返回 None。"""
    for part in path.parts:
        if part.startswith("step-"):
            try:
                return int(part[len("step-") :])
            except ValueError:
                return None
    return None


def g_lm_from_log(log_path: Path, around_step: int | None, window: float = 0.1) -> tuple[float, int]:
    """从训练日志里取 `|g_lm|` 的中位数，返回 (中位数, 样本数)。

    只取检查点附近 ±`window` 的那些步：`|g_lm|` 随训练缓慢下降，拿全程的中位数会把早期
    偏大的值掺进来。抠不出检查点步数时退回最后 10% 的日志行。
    """
    samples: list[tuple[int, float]] = []
    current_step = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := RE_STEP.search(line):
            current_step = int(m.group(1))
        if m := RE_G_LM.search(line):
            value = m.group(1)
            if value != "nan":
                samples.append((current_step, float(value)))

    if not samples:
        raise SystemExit(
            f"{log_path} 里没有 |g_lm|。训练要带 --kres.log_grad_norms true 和 --train.max_norm，"
            f"两个都给了才会记这个数"
        )

    if around_step is not None:
        low, high = around_step * (1 - window), around_step * (1 + window)
        picked = [v for s, v in samples if low <= s <= high]
        if picked:
            return statistics.median(picked), len(picked)
        print(
            f"  ⚠ 日志里没有第 {around_step} 步附近（±{window:.0%}）的记录，退回用最后 10% 的行。"
            f"  日志与检查点多半不是同一个 run。"
        )
    tail = [v for _, v in samples[-max(1, len(samples) // 10) :]]
    return statistics.median(tail), len(tail)


def compute(
    base: Path, checkpoint: Path, cov: Path, identity: bool = False
) -> tuple[float, float, int, dict]:
    """返回 (归一化的 R, λ=1 时的 |g_reg|, 层数, C 的 metadata)。

    走的是 `full_covariance_grad_` 本身而不是另抄一遍公式：抄一遍就有抄错一次的机会，
    而抄错的表现是一个量级正确、但系统性偏掉的 λ，没有任何东西能对照出来。

    `identity=True` 把 C 换成同尺寸单位阵，和训练时的 `--kres.identity` 走同一个函数。
    单位阵那条臂的 λ 必须在这里单独反算：|g_reg| 的量级由 C 本身定，照抄真 C 那条臂的
    λ 会让两条臂的惩罚强度差出一个未知倍数，比出来的就成了「λ 谁调得更好」而不是
    「协方差结构有没有用」——而后者正是这条消融唯一要回答的问题。
    """
    payload = load_covariance_payload(cov)
    covariances = payload["covariances"]
    if identity:
        from kres.covariance import to_identity_covariances

        covariances = to_identity_covariances(covariances)
    names = sorted(covariances)

    weights = load_reference_weights(checkpoint, names, dtype=torch.float32)
    references = load_reference_weights(base, names, dtype=torch.float32)

    drift = sum(float((weights[n] - references[n]).abs().sum()) for n in names)
    if drift == 0.0:
        raise SystemExit(
            f"ΔW 恒为 0：{checkpoint.name} 与基座逐比特相同，惩罚梯度也就恒为 0，λ 反算不出来。\n"
            f"要么这个检查点是训练第 0 步存的，要么 --base 指错了。"
        )

    layers = [
        (n, torch.nn.Parameter(weights[n]), covariances[n].to(torch.float32), references[n])
        for n in names
    ]
    # scale = λ/层数，λ=1 就是 1/N。allow_tf32 关掉：离线算不缺算力，没必要引入 tf32 的
    # 那点舍入——训练里开着它是为了 H100 上的 7 倍吞吐，这里没有这个约束
    total, g_reg = full_covariance_grad_(
        layers, scale=1.0 / len(layers), allow_tf32=False, compute_dtype=torch.float32
    )
    return total / len(layers), g_reg, len(layers), payload.get("metadata", {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", type=Path, required=True, help="基座 lit_model.pth（W₀）")
    ap.add_argument("--checkpoint", type=Path, required=True, help="要反算的检查点 lit_model.pth（W）")
    ap.add_argument("--cov", type=Path, required=True, help="C 文件（kres.collect_cov 的产物）")
    ap.add_argument(
        "--identity",
        action="store_true",
        help="把 C 换成同尺寸单位阵，给 c_id 那条臂标 λ（与 --kres.identity 对应）",
    )
    ap.add_argument("--log", type=Path, default=None, help="训练日志，用来取 |g_lm| 的中位数")
    ap.add_argument("--g-lm", type=float, default=None, help="直接给 |g_lm|，给了就不读日志")
    ap.add_argument(
        "--rho",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.1, 0.3],
        help="目标比例 |g_reg|/|g_lm|。默认扫四个点，跨两个数量级",
    )
    args = ap.parse_args(argv)

    if args.g_lm is None and args.log is None:
        raise SystemExit("要么给 --log（从日志取 |g_lm| 中位数），要么给 --g-lm")
    for path in (args.base, args.checkpoint, args.cov):
        if not path.is_file():
            raise SystemExit(f"找不到 {path}")

    step = step_from_checkpoint(args.checkpoint)
    print(f"检查点  {args.checkpoint}" + (f"（第 {step} 步）" if step is not None else ""))
    print(f"基座    {args.base}")
    print(f"C       {args.cov}" + ("（换成单位阵，只取层名和维度）" if args.identity else ""))

    R, g_reg, n_layers, meta = compute(args.base, args.checkpoint, args.cov, args.identity)
    print(
        f"\nC 的 metadata：{n_layers} 层，tokens={meta.get('tokens', '?')}，"
        f"include_lm_head={meta.get('include_lm_head', '?')}"
    )
    print(f"R（归一化后）      = {R:.6e}")
    print(f"|g_reg| （λ=1）    = {g_reg:.6e}")

    if args.g_lm is not None:
        g_lm, n_samples = args.g_lm, 0
        print(f"|g_lm|             = {g_lm:.6f}（命令行给的）")
    else:
        g_lm, n_samples = g_lm_from_log(args.log, step)
        print(f"|g_lm|             = {g_lm:.6f}（{args.log.name} 里 {n_samples} 条的中位数）")

    if g_reg <= 0 or g_lm <= 0:
        raise SystemExit("|g_reg| 或 |g_lm| 是 0，λ 反算不出来")

    ratio = g_reg / g_lm
    print(f"\nλ=1 时惩罚/LM 梯度比 = {ratio:.6e}")
    print("\n" + "-" * 58)
    print(f"{'ρ（惩罚占 LM 梯度）':<24}{'λ':>16}")
    print("-" * 58)
    for rho in sorted(args.rho):
        print(f"{rho:<24.4g}{rho / ratio:>16.4g}")
    print("-" * 58)

    print(
        "\n怎么用这张表：\n"
        "  ρ 是惩罚梯度相对 LM 梯度的大小。ρ=0.1 表示惩罚是任务梯度的一成——这个量级\n"
        "  下惩罚能改变优化轨迹但不主导它。往两头走各有一种失效：ρ 太小时 C 臂与\n"
        "  baseline 难以区分，太大则新域 loss 降不下去、看起来像别的 bug。\n"
        "  扫描先取跨一个数量级的三个点，落在中间那档附近再收窄。"
    )
    if step is not None:
        print(
            f"\n  ⚠ 这是第 {step} 步的瞬时值，不是终态。惩罚梯度正比于 ΔW 而 ΔW 随训练增长，\n"
            f"    所以这个比值会继续上升，据此算出的 λ **偏大**。方向是已知的，所以从\n"
            f"    「看起来可接受」的那批里取偏小的一端。"
        )
    print(
        "\n  ⚠ 三条 C 臂必须用**同一个** λ。它们要回答的是「估 C 用的旧数据越多是否越好」，\n"
        "    各自调 λ 就把「C 更好」和「λ 调得更好」混在一起了，那个对比失去意义。\n"
        "    所以扫描只在一条臂上做（选 4B 那条，不偏向两端任何一头），定下来的值三条通用。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
