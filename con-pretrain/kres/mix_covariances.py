"""把若干段的 C 合并成一个 C。

主用途是 replay 分段：sample-10BT 被切成 seg00/seg01/... 若干段，replay 4B 那条臂读
[seg00, seg01]，它的 C 就该是这两段合起来的 C。分段采集再合并，比每条臂各自在自己的
并集上重采一遍省事——3 条臂共用 3 次采集，而不是 1+2+3=6 次。

**按 token 计数加权时，合并与「直接在并集上采一次」是严格等价的，不是近似。**
collect_cov 存的是 C_k = Σ_k x xᵀ / n_k 和 n_k，于是

    Σ_k (C_k · n_k) / Σ_k n_k  =  Σ_k (Σ_k x xᵀ) / Σ_k n_k  =  并集的 E[x xᵀ]

所以默认就用 counts 加权，`--weights` 只留给「按 domain 配比人为调权」那种用法（预训练
主线用不到，留着是因为 SFT 侧的老脚本有这个口子）。用简单平均只有在各段 token 数完全
相等时才等价，而按主键哈希分段各段会差几个百分点，所以不提供「等权」这个选项。

用法：

    # 直接给文件
    python -m kres.mix_covariances --inputs cov/seg00.pt,cov/seg01.pt --output cov/arm_4b.pt

    # 从 replay_plan.json 取第 2 条臂该合哪几段（推荐，省得手抄目录名）
    python -m kres.mix_covariances \
        --from-plan data/chunks/fineweb_edu/replay_plan.json --arm 2 \
        --cov-dir cov --output cov/arm_4b.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kres.covariance import load_covariance_payload, save_covariance_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", help="逗号分隔的 C 文件，例如 cov/seg00.pt,cov/seg01.pt")
    src.add_argument("--from-plan", type=Path, help="replay_plan.json 的路径，配合 --arm 和 --cov-dir")
    p.add_argument("--arm", type=int, help="--from-plan 时必填：第几条臂（1 起，1=只读 seg00）")
    p.add_argument("--cov-dir", type=Path, help="--from-plan 时必填：存放 seg00.pt / seg01.pt ... 的目录")
    p.add_argument("--output", required=True, help="合并结果的输出路径（.pt）")
    p.add_argument(
        "--weights",
        help="逗号分隔的人工权重，覆盖默认的 token 计数加权。给了这个就不再与"
             "「在并集上采一次」等价，metadata 里会标出来",
    )
    return p.parse_args()


def resolve_from_plan(plan_path: Path, arm: int | None, cov_dir: Path | None) -> tuple[list[Path], dict[str, Any]]:
    """从 replay_plan.json 反推第 arm 条臂要合并哪几个段的 C 文件。"""
    if arm is None or cov_dir is None:
        raise SystemExit("--from-plan 需要同时给 --arm 和 --cov-dir")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    arms = plan["arms"]
    if not 1 <= arm <= len(arms):
        raise SystemExit(f"--arm {arm} 越界，plan 里只有 {len(arms)} 条臂")
    entry = arms[arm - 1]

    # plan 里 replay_dirs 与 cov_dirs 逐字相同是「replay 和 C 同源」的机械保证，
    # 在这里再验一次：C 是从这个字段展开的，它要是错了就直接合出一个不同源的 C。
    if entry["replay_dirs"] != entry["cov_dirs"]:
        raise SystemExit(f"plan 里第 {arm} 条臂的 replay_dirs 与 cov_dirs 不一致，plan 文件已损坏")

    paths = [cov_dir / f"{name}.pt" for name in entry["segments"]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"缺少这些段的 C：{[str(p) for p in missing]}。"
            f"先对每个段跑 collect_cov --chunks <段目录> --output <cov-dir>/<段名>.pt"
        )
    return paths, {"plan": str(plan_path), "arm": arm, "segments": entry["segments"]}


def main() -> int:
    args = parse_args()

    plan_info: dict[str, Any] = {}
    if args.from_plan is not None:
        paths, plan_info = resolve_from_plan(args.from_plan, args.arm, args.cov_dir)
    else:
        paths = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    if len(paths) < 2:
        raise SystemExit(f"要合并的 C 少于两个（{[str(p) for p in paths]}），没必要跑这个脚本")

    payloads = [load_covariance_payload(p) for p in paths]

    # 各段是同一个基座、同一套目标层采的，键集合必须完全相同。老脚本在这里取交集，
    # 那会把「某段少了几层」这种 bug 静默地变成「合出来的 C 少几层」，而少的那几层
    # 到训练时才以 assert_covers 失败的形式暴露，甚至可能压根不报。
    key_sets = [set(pl["covariances"]) for pl in payloads]
    if len(set(map(frozenset, key_sets))) != 1:
        base = key_sets[0]
        diffs = {
            str(p): {"多出": sorted(ks - base)[:5], "缺少": sorted(base - ks)[:5]}
            for p, ks in zip(paths, key_sets) if ks != base
        }
        raise SystemExit(f"各输入的目标层集合不一致（以 {paths[0].name} 为基准）：{diffs}")
    keys = sorted(key_sets[0])

    # 同理，采集设置不一致合出来的 C 没有意义：不同 checkpoint 采的 C 描述的是不同的
    # 激活分布，不同 normalization 的 C 尺度都不一样。
    for field in ("model_name", "cov_normalization", "include_lm_head", "seq_len"):
        values = {str(pl.get("metadata", {}).get(field)) for pl in payloads}
        if len(values) != 1:
            raise SystemExit(f"各输入的 metadata.{field} 不一致：{sorted(values)}，不能合并")

    counts = [pl["counts"] for pl in payloads]
    if args.weights is not None:
        weights = [float(x) for x in args.weights.split(",") if x.strip()]
        if len(weights) != len(paths):
            raise SystemExit(f"--weights 给了 {len(weights)} 个，输入有 {len(paths)} 个")
        total = sum(weights)
        if total <= 0:
            raise SystemExit("--weights 之和必须为正")
        weights = [w / total for w in weights]
        weighting = "manual"
    else:
        # 每层的 token 计数在同一次采集里是相同的（collect_cov 已经硬断言过），
        # 所以取任意一层的即可，不必逐层算权重。
        per_input = [c[keys[0]] for c in counts]
        if any(n <= 0 for n in per_input):
            raise SystemExit(f"有输入的 token 计数为 0：{dict(zip(map(str, paths), per_input))}")
        total = sum(per_input)
        weights = [n / total for n in per_input]
        weighting = "token_count"

    mixed: dict[str, torch.Tensor] = {}
    mixed_counts: dict[str, int] = {}
    for key in keys:
        acc = None
        for w, pl in zip(weights, payloads):
            term = w * pl["covariances"][key].to(torch.float64)
            acc = term if acc is None else acc + term
        mixed[key] = acc.to(torch.float32)
        # counts 累加而不是取 min：合并后的 C 描述的就是这么多 token 的分布，下游要用它
        # 再做一次加权合并（比如先合段、再合 domain）时权重才是对的。
        mixed_counts[key] = sum(int(c[key]) for c in counts)

    metadata = {
        "type": "covariance_mix",
        "weighting": weighting,
        "exact_union": weighting == "token_count",
        "inputs": [str(p) for p in paths],
        "weights": weights,
        "input_tokens": [int(c[keys[0]]) for c in counts],
        "target_layers": len(keys),
        **plan_info,
        # 上游的采集设置原样带下来，几周后要能回答「这个 C 是谁合的」。
        "source_metadata": [pl.get("metadata", {}) for pl in payloads],
    }
    for field in ("model_name", "cov_normalization", "include_lm_head", "seq_len"):
        metadata[field] = payloads[0].get("metadata", {}).get(field)

    path = save_covariance_payload(args.output, mixed, mixed_counts, metadata)
    print(f"合并 {len(paths)} 个 C → {path}（{path.stat().st_size / 1e9:.2f} GB）")
    print(f"  加权方式 {weighting}"
          + ("（与在并集上采一次严格等价）" if weighting == "token_count" else "（人工权重，非等价）"))
    for p, w, n in zip(paths, weights, metadata["input_tokens"]):
        print(f"  {p.name:<16} 权重 {w:.4f}   token {n:,}")
    print(f"  合并后 token {mixed_counts[keys[0]]:,}，目标层 {len(keys)} 个")
    print("\n下一步：python -m kres.inspect_cov --cov " + str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
