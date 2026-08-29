"""Checks for onereplay/mix_fishers.py.

The mix has to get two things right: the equalize-by-mean weights (so neither
domain silently dominates the penalty) and the raw weighted sum itself (so the
combined F is actually what training will regularize against). Both are verified
against payloads whose per-layer values are chosen so the arithmetic is checkable
by hand.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def fisher_payload(values: dict[str, torch.Tensor], fingerprint: str) -> dict:
    """A minimal but valid Fisher payload with a real per-layer tensor dict."""

    return {
        "fishers": values,
        "counts": {name: 1000 for name in values},
        "metadata": {
            "pool_fingerprint": fingerprint,
            "target_modules": sorted(values.keys()),
        },
    }


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "onereplay.mix_fishers", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def global_mean(values: dict[str, torch.Tensor]) -> float:
    total = sum(float(v.double().sum()) for v in values.values())
    count = sum(v.numel() for v in values.values())
    return total / count


def main() -> None:
    tmp = Path(tempfile.mkdtemp())

    # F_if: mean 0.1 over its entries. F_math: mean 0.025, i.e. 4x smaller, the
    # same 3.78x-ish gap the real files show, rounded for hand arithmetic.
    if_values = {
        "model.layers.0.self_attn.q_proj": torch.full((2, 2), 0.1),
        "model.layers.0.self_attn.v_proj": torch.full((2, 2), 0.1),
    }
    math_values = {
        "model.layers.0.self_attn.q_proj": torch.full((2, 2), 0.025),
        "model.layers.0.self_attn.v_proj": torch.full((2, 2), 0.025),
    }
    m_if = global_mean(if_values)
    m_math = global_mean(math_values)
    assert abs(m_if - 0.1) < 1e-6 and abs(m_math - 0.025) < 1e-6

    f_if = tmp / "F_if.pt"
    f_math = tmp / "F_math.pt"
    torch.save(fisher_payload(if_values, "fp_if"), f_if)
    torch.save(fisher_payload(math_values, "fp_math"), f_math)

    # ---- arm 1: equalize by mean ----
    out_equal = tmp / "F_mix_equal.pt"
    result = run(["--inputs", f"{f_if},{f_math}", "--equalize", "mean",
                  "--output_path", str(out_equal)])
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    # 1/mean weights: (1/0.1, 1/0.025) = (10, 40) -> normalized (0.2, 0.8)
    assert abs(report["weights"][0] - 0.2) < 1e-6, report
    assert abs(report["weights"][1] - 0.8) < 1e-6, report
    # equal contribution means the two w_i*mean_i are equal
    contrib = report["input_contributions"]
    assert abs(contrib[0] - contrib[1]) < 1e-9, report
    # the mixed tensor itself: 0.2*0.1 + 0.8*0.025 = 0.04 everywhere
    payload = torch.load(out_equal, map_location="cpu")
    for tensor in payload["fishers"].values():
        assert torch.allclose(tensor, torch.full_like(tensor, 0.04)), tensor
    assert payload["metadata"]["mode"] == "equalize:mean"
    assert payload["metadata"]["source_pool_fingerprints"] == ["fp_if", "fp_math"]
    print("equalize mean: weights 0.2/0.8, equal contributions, F_mix=0.04 -- ok")

    # ---- arm 2: raw 0.5/0.5 ----
    out_half = tmp / "F_mix_half.pt"
    result = run(["--inputs", f"{f_if},{f_math}", "--weights", "0.5,0.5",
                  "--output_path", str(out_half)])
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["weights"] == [0.5, 0.5], report
    # raw mix: 0.5*0.1 + 0.5*0.025 = 0.0625 -> if takes the larger share, as intended
    payload = torch.load(out_half, map_location="cpu")
    for tensor in payload["fishers"].values():
        assert torch.allclose(tensor, torch.full_like(tensor, 0.0625)), tensor
    # under raw weights the contributions are NOT equal: if is 4x math
    contrib = report["input_contributions"]
    assert abs(contrib[0] / contrib[1] - 4.0) < 1e-6, report
    assert payload["metadata"]["mode"] == "explicit"
    print("raw 0.5/0.5: F_mix=0.0625, if contributes 4x math -- ok")

    # ---- unnormalized weights are rescaled unless told not to ----
    result = run(["--inputs", f"{f_if},{f_math}", "--weights", "1,3",
                  "--output_path", str(tmp / "n.pt")])
    report = json.loads(result.stdout)
    assert abs(report["weights"][0] - 0.25) < 1e-6, report
    assert abs(report["weights"][1] - 0.75) < 1e-6, report
    result = run(["--inputs", f"{f_if},{f_math}", "--weights", "1,3",
                  "--normalize_weights", "0", "--output_path", str(tmp / "n0.pt")])
    report = json.loads(result.stdout)
    assert report["weights"] == [1.0, 3.0], report
    print("weight normalization toggles correctly -- ok")

    # ---- a covariance file must be rejected with a pointer to the right tool ----
    cov = tmp / "C.pt"
    torch.save({"covariances": {"model.layers.0.self_attn.q_proj": torch.eye(2)}}, cov)
    result = run(["--inputs", f"{f_if},{cov}", "--weights", "0.5,0.5",
                  "--output_path", str(tmp / "bad.pt")])
    assert result.returncode != 0, result.stdout
    assert "mix_covariances" in result.stderr, result.stderr
    print("covariance input rejected -- ok")

    # ---- shape mismatch (different target_modules) must fail loudly ----
    wrong = tmp / "F_wrong.pt"
    torch.save(
        fisher_payload({"model.layers.0.self_attn.q_proj": torch.full((3, 3), 0.1),
                        "model.layers.0.self_attn.v_proj": torch.full((3, 3), 0.1)},
                       "fp_wrong"),
        wrong,
    )
    result = run(["--inputs", f"{f_if},{wrong}", "--weights", "0.5,0.5",
                  "--output_path", str(tmp / "bad2.pt")])
    assert result.returncode != 0, result.stdout
    assert "shape mismatch" in result.stderr, result.stderr
    print("shape mismatch rejected -- ok")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
