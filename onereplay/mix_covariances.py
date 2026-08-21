"""Mix several OneReplay covariance files into one weighted covariance file.

Each input file is produced by scripts/collect_cov.py and contains one covariance
matrix per LoRA target module. This script combines matching module keys:

    C_mix[layer] = sum_i weight_i * C_i[layer]

The resulting file has the same format as a normal covariance file, so training
can use it directly via --cov_path.
"""

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
