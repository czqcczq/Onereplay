"""Mix several EWC Fisher files into one weighted Fisher file.

Each input file is produced by scripts/collect_fisher.py and holds one diagonal
Fisher matrix per LoRA target module. This script combines matching module keys:

    F_mix[layer] = sum_i weight_i * F_i[layer]

The result has the same format as a normal Fisher file, so training reads it via
--fisher_path with no special casing.

Weights can be given two ways, and the difference is the whole experiment:

  --weights 0.5,0.5           the raw OneReplay-style mix. F is a sequence-summed
                              score, so if one domain's F is k times larger, it
                              takes k times the share of the penalty. This is the
                              "same as replay 0.5" arm on purpose.

  --equalize mean             ignore --weights and set weight_i = (1/mean_i)
                              normalized to sum to 1, so every domain contributes
                              the same total Fisher mass. This is the "scale-
                              normalized equal-contribution" arm. It reads each
                              file's global mean the same way fisher_summary does,
                              so swapping in a re-collected F recomputes the split
                              instead of trusting a number copied into a script.

The two files this produces are meant to be trained under the same lambda grid
and compared, which is why neither rescales the overall magnitude beyond what the
weights imply: raw 0.5/0.5 lands at mean ~= 0.5*(m_if+m_math) and the equalized
mix at ~= 2/(1/m_if + 1/m_math), same order of magnitude, so one sweep covers
both.

Usage:
    # arm 1: scale-normalized equal contribution
    python -m onereplay.mix_fishers --inputs F_if.pt,F_math.pt \
        --equalize mean --output_path F_mix_equal.pt

    # arm 2: raw 0.5/0.5, i.e. the replay-0.5 analogue
    python -m onereplay.mix_fishers --inputs F_if.pt,F_math.pt \
        --weights 0.5,0.5 --output_path F_mix_half.pt
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

from onereplay.core.fisher import fisher_summary, save_fisher_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse weighted Fisher inputs, the weighting mode, and the output path."""

    parser = argparse.ArgumentParser(description="Mix EWC Fisher files.")
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Comma-separated Fisher files, e.g. F_if.pt,F_math.pt",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help=(
            "Comma-separated weights matching --inputs, e.g. 0.5,0.5. Ignored when "
            "--equalize is set."
        ),
    )
    parser.add_argument(
        "--equalize",
        type=str,
        default="none",
        choices=["none", "mean"],
        help=(
            "none uses --weights as given; mean overrides them with 1/mean_i "
            "(normalized), so every domain contributes equal Fisher mass."
        ),
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--normalize_weights",
        type=int,
        default=1,
        help="1 rescales the explicit weights to sum to 1; --equalize always normalizes.",
    )
    return parser.parse_args()


def load_payload(path: str) -> dict[str, Any]:
    """Load one Fisher payload, refusing a covariance file passed by mistake."""

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "fishers" not in payload:
        if isinstance(payload, dict) and "covariances" in payload:
            raise ValueError(
                f"{path} holds covariances, not Fisher matrices. Use mix_covariances.py."
            )
        raise ValueError(f"Unsupported Fisher payload: {path}")
    return payload


def parse_list(text: str) -> list[str]:
    """Split a comma-separated argument and drop empty pieces."""

    return [item.strip() for item in text.split(",") if item.strip()]


def global_mean(fishers: dict[str, torch.Tensor]) -> float:
    """Global element mean of a Fisher, matching fisher_summary's definition."""

    return float(fisher_summary(fishers)["mean"])


def resolve_weights(
    payloads: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[float], list[float]]:
    """Return the (weights, per-input means) the mix will use."""

    means = [global_mean(payload["fishers"]) for payload in payloads]

    if args.equalize == "mean":
        for path_mean in means:
            if path_mean <= 0:
                raise ValueError("Cannot equalize by mean: an input has a zero mean")
        raw = [1.0 / path_mean for path_mean in means]
        total = sum(raw)
        return [weight / total for weight in raw], means

    weights = [float(item) for item in parse_list(args.weights)]
    if not weights:
        raise ValueError("--weights is required unless --equalize is set")
    if len(weights) != len(payloads):
        raise ValueError("--inputs and --weights must have the same length")
    if args.normalize_weights == 1:
        total = sum(weights)
        if total == 0:
            raise ValueError("Sum of weights cannot be zero")
        weights = [weight / total for weight in weights]
    return weights, means


def main() -> None:
    """Load all inputs, combine matching module matrices, and save F_mix."""

    args = parse_args()
    input_paths = parse_list(args.inputs)
    if len(input_paths) < 2:
        raise ValueError("--inputs needs at least two Fisher files to mix")

    payloads = [load_payload(path) for path in input_paths]
    weights, means = resolve_weights(payloads, args)

    key_sets = [set(payload["fishers"].keys()) for payload in payloads]
    common_keys = sorted(set.intersection(*key_sets))
    if not common_keys:
        raise ValueError("No common Fisher keys across inputs")
    missing_by_input = {
        path: sorted(set.union(*key_sets) - keys)
        for path, keys in zip(input_paths, key_sets)
    }

    mixed: dict[str, torch.Tensor] = {}
    mixed_counts: dict[str, int] = {}
    for key in common_keys:
        value: torch.Tensor | None = None
        count_values: list[int] = []
        reference_shape = payloads[0]["fishers"][key].shape
        for weight, payload in zip(weights, payloads):
            fisher = payload["fishers"][key].float()
            if fisher.shape != reference_shape:
                raise ValueError(
                    f"shape mismatch on {key}: {tuple(fisher.shape)} vs "
                    f"{tuple(reference_shape)}. The inputs came from different models "
                    "or target_modules and cannot be added."
                )
            weighted = weight * fisher
            value = weighted if value is None else value + weighted
            count_values.append(int(payload.get("counts", {}).get(key, 0)))
        mixed[key] = value
        # After mixing, no single example count defines the normalization; keep the
        # minimum as a conservative diagnostic, the way mix_covariances does.
        mixed_counts[key] = min(count_values) if count_values else 0

    # w_i * mean_i is each domain's share of the penalty mass; equalization makes
    # these equal, and the raw mix makes them proportional to the domains' scales.
    contributions = [weight * path_mean for weight, path_mean in zip(weights, means)]
    mixed_scale = fisher_summary(mixed)

    metadata = {
        "type": "weighted_fisher_mix",
        "mode": "equalize:mean" if args.equalize == "mean" else "explicit",
        "inputs": input_paths,
        "weights": weights,
        "input_means": means,
        "input_contributions": contributions,
        "normalize_weights": args.normalize_weights,
        "num_common_keys": len(common_keys),
        "missing_by_input": missing_by_input,
        "fisher_scale": mixed_scale,
        "source_pool_fingerprints": [
            payload.get("metadata", {}).get("pool_fingerprint") for payload in payloads
        ],
        "source_metadata": [payload.get("metadata", {}) for payload in payloads],
    }
    save_fisher_payload(args.output_path, mixed, mixed_counts, metadata)

    print(
        json.dumps(
            {
                "output_path": args.output_path,
                "mode": metadata["mode"],
                "inputs": input_paths,
                "weights": [round(weight, 6) for weight in weights],
                "input_means": [round(path_mean, 8) for path_mean in means],
                "input_contributions": [round(item, 8) for item in contributions],
                "mix_mean": round(mixed_scale["mean"], 8),
                "mix_max": mixed_scale["max"],
                "num_common_keys": len(common_keys),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
