"""Mix several OneReplay covariance files into one weighted covariance file.

Each input file is produced by scripts/collect_cov.py and contains one covariance
matrix per LoRA target module. This script combines matching module keys:

    C_mix[layer] = sum_i weight_i * C_i[layer]

The resulting file has the same format as a normal covariance file, so training
can use it directly via --cov_path.
"""

# =============================================================================
# PORT NOTE 文件状态：已落地 → kres/mix_covariances.py，本文件只作原版存档，别再用
#
# 落地版改了三处语义，都是这个原版在预训练场景下会出错的地方：
#   1. 默认按 token 计数加权（Σ C_k·n_k / Σ n_k），与「在并集上采一次」严格等价。
#      原版要求手工传 --weights，随手给的权重合出来的 C 不对应任何真实分布。
#   2. 键集合不一致时报错，不再静默取交集——取交集会把「某段少了几层」悄悄变成
#      「合出来的 C 少几层」。
#   3. counts 累加而不是取 min，这样合并结果还能参与下一级合并。
# 下面是原版内容。
#
# onereplay/mix_covariances.py 的逐字节副本，只加注释、未改代码行。校验：
#     diff onereplay/mix_covariances.py con-pretrain/onereplay_port/mix_covariances.py
#
# 这个脚本只做 C_mix[layer] = sum_i w_i * C_i[layer]，纯张量加权，与训练框架完全无关。
# 唯一要改的是 L26 `from onereplay.core.covariance import ...` 的包路径，以及 L22
# project_root 的 parents[1] 层数（取决于新项目的目录深度）。
#
# 用法上有一点变化值得先想清楚：SFT 时混的是 flan/math/code 这类"任务"，预训练语料里没有
# 这个概念，能混的是不同 domain 的切片（web / code / arxiv ...）。如果第一阶段语料本身
# 就是单一混合分布，那直接采一份 C 就够，这个脚本用不上；只有当你想让 C 的 domain 配比
# 和第一阶段的真实配比对齐、而采集是分 domain 做的时候才需要它。
# 权重应当取第一阶段各 domain 的 token 占比，而不是随手给的数——这个换算要记进 metadata。
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core.covariance import save_covariance_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse weighted covariance inputs and output path."""

    parser = argparse.ArgumentParser(description="Mix OneReplay covariance files.")
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Comma-separated covariance files, e.g. flan.pt,math.pt,code.pt",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Comma-separated weights matching --inputs, e.g. 0.5,0.25,0.25",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--normalize_weights", type=int, default=1)
    return parser.parse_args()


def load_payload(path: str) -> dict[str, Any]:
    """Load one covariance payload from disk."""

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "covariances" not in payload:
        raise ValueError(f"Unsupported covariance payload: {path}")
    return payload


def parse_list(text: str) -> list[str]:
    """Split a comma-separated argument and remove empty pieces."""

    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    """Load all inputs, combine matching module matrices, and save C_mix."""

    args = parse_args()
    input_paths = parse_list(args.inputs)
    weights = [float(item) for item in parse_list(args.weights)]
    if len(input_paths) != len(weights):
        raise ValueError("--inputs and --weights must have the same length")
    if args.normalize_weights == 1:
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        weights = [weight / total_weight for weight in weights]

    payloads = [load_payload(path) for path in input_paths]
    key_sets = [set(payload["covariances"].keys()) for payload in payloads]
    common_keys = sorted(set.intersection(*key_sets))
    if not common_keys:
        raise ValueError("No common covariance keys across inputs")

    missing_by_input = {
        path: sorted(set.union(*key_sets) - set(payload["covariances"].keys()))
        for path, payload in zip(input_paths, payloads)
    }

    mixed: dict[str, torch.Tensor] = {}
    mixed_counts: dict[str, int] = {}
    for key in common_keys:
        value = None
        count_values = []
        for weight, payload in zip(weights, payloads):
            covariance = payload["covariances"][key].float()
            weighted = weight * covariance
            value = weighted if value is None else value + weighted
            count_values.append(int(payload.get("counts", {}).get(key, 0)))
        mixed[key] = value
        # Counts no longer define the normalization exactly after mixing. Store
        # the minimum common count as a conservative diagnostic.
        mixed_counts[key] = min(count_values) if count_values else 0

    metadata = {
        "type": "weighted_covariance_mix",
        "inputs": input_paths,
        "weights": weights,
        "normalize_weights": args.normalize_weights,
        "num_common_keys": len(common_keys),
        "missing_by_input": missing_by_input,
        "source_metadata": [payload.get("metadata", {}) for payload in payloads],
    }
    save_covariance_payload(args.output_path, mixed, mixed_counts, metadata)
    print(
        json.dumps(
            {
                "output_path": args.output_path,
                "num_common_keys": len(common_keys),
                "weights": weights,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
