"""惩罚是摊在它覆盖的那些层上，还是被其中一层独占？

    python -m kres.inspect_cov --cov <采出来的 C.pt>

惩罚项 R = (1/L) Σ_l tr(ΔW_l C_l ΔW_lᵀ) 里没有任何东西会拉平各层。每个 C_l 带着自己
那层输入激活的量纲，某层的 hidden state 大一个数量级，它贡献的惩罚就大两个数量级。
差距大到一定程度，这个和就不再是「对 L 层的惩罚」而是「对一层的惩罚、其余是观众」，
而下游没有任何指标会报告这件事：R 仍然有个合理的数值，λ 仍然能调它，loss 曲线照常动。

这个问题在全参数训练下不是假想。SFT 侧的 LoRA 只覆盖 q_proj/v_proj，它们的输入都是
post-LayerNorm 的残差状态、跨深度尺度相当。全参数把 `mlp.proj` 也纳进来了，而它的输入
是 MLP 中间激活——正是 massive activation 所在的地方，早期层里少数几个坐标能比其余大
几百倍，而 C 是二阶矩，会把这个差距平方。

**指标定义与 `onereplay/scripts/inspect_cov_layers.py` 保持一致**，这样新 C 的数字可以
直接和 SFT 侧那次诊断（层间保护强度差 6.13e4 倍、layers.2.mlp.down_proj 99.995% 的迹
集中在单个方向）对比。三个数回答三个不同的问题：

  share       某层占标量 R 的多少。对 i.i.d. 方差 s² 的 ΔW 有
              E[tr(ΔW C ΔWᵀ)] = s²·d_out·tr(C)，所以在 Adam 实际产生的
              「各层逐元素步长相当」的更新下，某层的期望占比正比于 d_out·tr(C)。
              一层占到 99% 就意味着 λ 是对着那一层调的，别的层都没被约束。
  mean_diag   tr(C_l)/d_in_l，单位更新能量上的惩罚。注入的梯度是 2(λ/L)·ΔW_l·C_l，
              所以这个量决定某层实际被按得多紧。它的层间跨度就是保护强度的跨度。
  top1_share  λ_max/tr(C)，C 的迹有多大比例集中在最强的那一个方向。接近 1 说明这层的
              C 几乎是 rank-1，惩罚只挡住一个方向、其余方向自由漂移。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kres.covariance import load_covariance_payload

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "model" / "open-sci-ref-v0.02-0.4b-fineweb-edu-1.4t-300B-4096"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="诊断 C 的层间分布与谱集中度")
    p.add_argument("--cov", required=True, help="collect_cov 产出的 .pt")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL), help="用来取每层 d_out")
    p.add_argument(
        "--power-iterations",
        type=int,
        default=30,
        help="λ_max 的幂迭代步数。实测谱非常头重（top-1 约占三分之一迹），收敛很快",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--top", type=int, default=20, help="列出最大的几层；--full 列全部")
    p.add_argument("--full", action="store_true")
    p.add_argument(
        "--dominance-fail",
        type=float,
        default=0.5,
        help="单层占 R 的比例达到这个值就以非零码退出",
    )
    p.add_argument(
        "--dominance-warn",
        type=float,
        default=10.0,
        help="单层占比超过均分的这个倍数就告警",
    )
    p.add_argument("--out-json", default="")
    return p.parse_args()


def top_eigenvalue(matrix: torch.Tensor, iterations: int, generator: torch.Generator) -> float:
    """对称 PSD 矩阵的最大特征值，用幂迭代。

    `torch.linalg.eigvalsh` 在 3840×3840 上跑 110 层要好几分钟 CPU，而且返回 3839 个
    没人看的数。这里只要首个特征值，用来把「迹集中在一个方向」和「整体就是大」区分开。
    """
    dimension = matrix.shape[-1]
    vector = torch.randn(dimension, generator=generator, dtype=torch.float32)
    vector = vector / vector.norm()
    value = 0.0
    for _ in range(iterations):
        product = matrix @ vector
        norm = float(product.norm())
        if norm == 0.0:
            return 0.0
        vector = product / norm
        value = float(vector @ (matrix @ vector))
    return value


def load_out_features(model_dir: str) -> dict[str, int]:
    """每层的 d_out，用 meta device 构模型，不占内存也不读权重。

    d_out 只影响 share 的加权（见模块 docstring 的恒等式）。拿不到时退化成只用
    tr(C)，也够用——d_out 在各投影之间只差几倍，而要找的效应跨几个数量级。
    """
    from torch import nn

    from litgpt.config import Config
    from litgpt.model import GPT

    config = Config.from_file(Path(model_dir) / "model_config.yaml")
    with torch.device("meta"):
        model = GPT(config)
    return {n: m.out_features for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def collect_layer_stats(
    covariances: dict[str, torch.Tensor],
    out_features: dict[str, int],
    iterations: int,
    seed: int,
) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    rows: list[dict] = []
    for name in sorted(covariances):
        cov = covariances[name].float()
        dimension = int(cov.shape[-1])
        trace = float(torch.diagonal(cov).sum())
        lambda_max = top_eigenvalue(cov, iterations, generator)
        rows.append(
            {
                "module": name,
                "kind": name.rsplit(".", 1)[-1],
                "d_in": dimension,
                "d_out": int(out_features.get(name, 0)),
                "trace": trace,
                "mean_diag": trace / max(dimension, 1),
                "lambda_max": lambda_max,
                "top1_share": lambda_max / trace if trace > 0 else 0.0,
            }
        )
    return rows


def add_shares(rows: list[dict], use_d_out: bool) -> None:
    for row in rows:
        weight = float(row["d_out"]) if use_d_out and row["d_out"] else 1.0
        row["contribution"] = weight * row["trace"]
    total = sum(row["contribution"] for row in rows) or 1.0
    for row in rows:
        row["share"] = row["contribution"] / total


def print_table(rows: list[dict], limit: int) -> None:
    header = (f"  {'module':<34} {'d_in':>6} {'trace':>12} {'mean_diag':>12} "
              f"{'top1':>7} {'share':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows[:limit]:
        print(f"  {row['module']:<34} {row['d_in']:>6} {row['trace']:>12.4e} "
              f"{row['mean_diag']:>12.4e} {row['top1_share']:>7.1%} {row['share']:>8.2%}")


def main() -> int:
    args = parse_args()
    payload = load_covariance_payload(args.cov)
    covariances = payload["covariances"]
    metadata = payload.get("metadata", {})

    print(f"C 文件 : {args.cov}")
    print(f"层数   : {len(covariances)}")
    if metadata:
        print(f"采集自 : {metadata.get('chunks', '?')}")
        print(f"         {metadata.get('tokens_seen', 0):,} token / "
              f"cov_normalization={metadata.get('cov_normalization', '?')} / "
              f"forward_dtype={metadata.get('forward_dtype', '?')}")

    try:
        out_features = load_out_features(args.model_dir)
        use_d_out = True
    except Exception as exc:  # noqa: BLE001
        print(f"  取不到 d_out（{type(exc).__name__}: {exc}），share 退化成只用 tr(C)")
        out_features, use_d_out = {}, False

    rows = collect_layer_stats(covariances, out_features, args.power_iterations, args.seed)
    add_shares(rows, use_d_out)
    rows.sort(key=lambda r: r["share"], reverse=True)

    print(f"\n按 share 排序（share 的加权{'含' if use_d_out else '不含'} d_out）：")
    print_table(rows, len(rows) if args.full else args.top)
    if not args.full and len(rows) > args.top:
        print(f"  ...（共 {len(rows)} 层，--full 看全部）")

    # 层间跨度：保护强度的真实不均衡程度
    diags = sorted(r["mean_diag"] for r in rows)
    spread = diags[-1] / max(diags[0], 1e-30)
    largest = rows[0]
    even = 1.0 / len(rows)

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    print(f"  最大单层占 R    : {largest['share']:.2%}  ({largest['module']})")
    print(f"  均分应为        : {even:.2%}，即最大层是均分的 {largest['share'] / even:.1f} 倍")
    print(f"  mean_diag 跨度  : {spread:.3e}×（最小层 → 最大层），这就是保护强度的跨度")
    print(f"  top1_share 最大 : {max(r['top1_share'] for r in rows):.2%} "
          f"({max(rows, key=lambda r: r['top1_share'])['module']})")
    print(f"  top1_share 中位 : {sorted(r['top1_share'] for r in rows)[len(rows) // 2]:.2%}")

    by_kind: dict[str, list[float]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row["mean_diag"])
    print("\n  按模块种类看 mean_diag（这能指出不均衡是不是 mlp.proj 一家造成的）：")
    for kind, values in sorted(by_kind.items(), key=lambda kv: -max(kv[1])):
        print(f"    {kind:<10} n={len(values):<4} "
              f"min={min(values):.3e} 中位={sorted(values)[len(values) // 2]:.3e} max={max(values):.3e}")

    failures = 0
    if largest["share"] >= args.dominance_fail:
        print(f"\n  [FAIL] 单层占了 R 的 {largest['share']:.1%} ≥ {args.dominance_fail:.0%}，"
              "λ 实际上只对着这一层在调，其余层没有被约束")
        failures += 1
    elif largest["share"] / even > args.dominance_warn:
        print(f"\n  [WARN] 最大层是均分的 {largest['share'] / even:.1f} 倍 "
              f"(> {args.dominance_warn:g})，惩罚分布明显偏斜")
    else:
        print(f"\n  [PASS] 最大层占 {largest['share']:.2%}，是均分的 "
              f"{largest['share'] / even:.1f} 倍，惩罚摊得够均匀")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "cov_path": str(args.cov),
                    "metadata": metadata,
                    "weighted_by_d_out": use_d_out,
                    "mean_diag_spread": spread,
                    "largest_share": largest["share"],
                    "layers": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\n  明细写入 {args.out_json}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
