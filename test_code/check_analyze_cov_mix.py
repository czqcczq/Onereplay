"""Checks for onereplay/scripts/analyze_cov_mix.py.

The audit's whole value is that its numbers can be trusted against a mix nobody
has independently verified, so the checks run it on covariances whose spectrum is
constructed rather than sampled: one shared block both domains excite equally,
one block only the first domain excites, one block only the second does. Every
quantity the script reports then has a closed form.

The construction mirrors the real files' qualitative shape -- the shared block
carries most of the trace, the distinctive blocks carry little -- so the test
also demonstrates the failure mode the audit exists to catch: traces that nearly
agree while the distinctive subspaces are lopsided.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]

DIM = 32
RANK = 4
SHARED = list(range(0, 8))
IF_ONLY = list(range(8, 12))
MATH_ONLY = list(range(12, 20))
TAIL = list(range(20, DIM))


def build_spectra() -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues for the two domains over one shared eigenbasis."""

    if_values = torch.full((DIM,), 1e-4, dtype=torch.float64)
    math_values = torch.full((DIM,), 1e-4, dtype=torch.float64)
    for index in SHARED:
        if_values[index] = 10.0
        math_values[index] = 10.0
    for index in IF_ONLY:
        if_values[index] = 1.0
        math_values[index] = 0.01
    for index in MATH_ONLY:
        if_values[index] = 0.01
        math_values[index] = 2.0
    return if_values, math_values


def covariance_payload(
    basis: torch.Tensor,
    values: torch.Tensor,
    layers: list[str],
    meta: dict,
) -> dict:
    matrix = (basis * values) @ basis.T
    return {
        "covariances": {name: matrix.float().clone() for name in layers},
        "counts": {name: 1_000_000 for name in layers},
        "metadata": meta,
    }


def adapter_dir(root: Path, layers: list[str]) -> Path:
    """A minimal PEFT-shaped adapter directory that load_lora_deltas can read."""

    directory = root / "adapter"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": RANK, "lora_alpha": 2 * RANK}), encoding="utf-8"
    )
    generator = torch.Generator().manual_seed(7)
    state = {}
    for name in layers:
        state[f"base_model.model.{name}.lora_A.weight"] = torch.randn(
            RANK, DIM, generator=generator
        )
        state[f"base_model.model.{name}.lora_B.weight"] = torch.randn(
            DIM, RANK, generator=generator
        )
    torch.save(state, directory / "adapter_model.bin")
    return directory


def expected_penalty(state: dict, name: str, covariance: torch.Tensor) -> float:
    """tr(DeltaW C DeltaW^T) formed the slow, obvious way as the reference."""

    lora_a = state[f"base_model.model.{name}.lora_A.weight"].double()
    lora_b = state[f"base_model.model.{name}.lora_B.weight"].double()
    delta = 2.0 * (lora_b @ lora_a)
    return float(torch.trace(delta @ covariance.double() @ delta.T))


def expected_gradient_cosine(
    state: dict,
    name: str,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[float, float]:
    """Per-layer cosine of the two penalty gradients, and its energy weight."""

    lora_a = state[f"base_model.model.{name}.lora_A.weight"].double()
    lora_b = state[f"base_model.model.{name}.lora_B.weight"].double()
    delta = 2.0 * (lora_b @ lora_a)
    grad_first = delta @ first.double()
    grad_second = delta @ second.double()
    inner = float((grad_first * grad_second).sum())
    denominator = float(grad_first.norm() * grad_second.norm())
    return inner / denominator, denominator


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    layers = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"]

    generator = torch.Generator().manual_seed(0)
    basis, _ = torch.linalg.qr(torch.randn(DIM, DIM, generator=generator, dtype=torch.float64))
    if_values, math_values = build_spectra()

    base_meta = {
        "model_name": "Qwen3-1.7B",
        "target_modules": ["q_proj", "v_proj"],
        "cov_normalization": "none",
        "cov_norm_eps": 1e-6,
        "use_chat_template": 1,
        "include_target_in_chat": 1,
        "require_target": 1,
    }
    if_cov = tmp / "C_if.pt"
    math_cov = tmp / "C_math.pt"
    torch.save(
        covariance_payload(basis, if_values, layers,
                           {**base_meta, "max_len": 512, "truncation_side": "right",
                            "pool_rows": 20000, "pool_fingerprint": "fp_if"}),
        if_cov,
    )
    torch.save(
        covariance_payload(basis, math_values, layers,
                           {**base_meta, "max_len": 2048, "truncation_side": "left",
                            "pool_rows": 25114, "pool_fingerprint": "fp_math"}),
        math_cov,
    )

    directory = adapter_dir(tmp, layers)
    state = torch.load(directory / "adapter_model.bin", map_location="cpu")
    json_out = tmp / "audit.json"

    result = subprocess.run(
        [
            sys.executable, "-m", "onereplay.scripts.analyze_cov_mix",
            "--covs", f"{if_cov},{math_cov}",
            "--labels", "if,math",
            "--weights", "0.5,0.5",
            "--probe_adapter", str(directory),
            "--ref_lambda", "3e-2",
            "--layer_stride", "1",
            "--json_out", str(json_out),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit("analyze_cov_mix exited non-zero")
    print(result.stdout)
    summary = json.loads(json_out.read_text(encoding="utf-8"))

    # ---- scale: the traces are the sums of the constructed eigenvalues ----
    trace_if = float(if_values.sum()) * len(layers)
    trace_math = float(math_values.sum()) * len(layers)
    got_if = summary["scale"]["totals"]["if"]
    got_math = summary["scale"]["totals"]["math"]
    assert abs(got_if - trace_if) < 1e-4 * trace_if, (got_if, trace_if)
    assert abs(got_math - trace_math) < 1e-4 * trace_math, (got_math, trace_math)
    want_weight = (1 / trace_if) / (1 / trace_if + 1 / trace_math)
    assert abs(summary["scale"]["equal_trace_weights"]["if"] - want_weight) < 1e-4
    print(f"scale: trace if={trace_if:.4f} math={trace_math:.4f} "
          f"(ratio {trace_math / trace_if:.4f}) -- ok")

    # ---- influence: rank x rank shortcut must equal the explicit DeltaW form ----
    for label, covariance in (("if", if_values), ("math", math_values)):
        matrix = (basis * covariance) @ basis.T
        want = sum(expected_penalty(state, name, matrix) for name in layers) / len(layers)
        got = summary["influence"]["penalty"][label]
        assert abs(got - want) < 1e-3 * want, (label, got, want)
    print("influence: rank x rank penalty matches the explicit trace -- ok")

    if_matrix = (basis * if_values) @ basis.T
    math_matrix = (basis * math_values) @ basis.T
    per_layer = [expected_gradient_cosine(state, name, if_matrix, math_matrix) for name in layers]
    total_weight = sum(weight for _, weight in per_layer)
    want_cosine = sum(cosine * weight for cosine, weight in per_layer) / total_weight
    got_cosine = summary["influence"]["gradient_cosine"]
    assert abs(got_cosine - want_cosine) < 1e-3, (got_cosine, want_cosine)
    print(f"influence: energy-weighted gradient cosine {got_cosine:.4f} matches the "
          f"explicit form -- ok")

    # ---- the direction split has to sum back to the penalty it decomposes ----
    for label in ("if", "math"):
        recovered = summary["directions"]["total_mass"][label]
        assert abs(recovered - summary["influence"]["penalty"][label]) < 1e-6 * recovered, label
    print("directions: masses sum back to reg(C) exactly -- ok")

    # ---- directions: the census is the construction, read back ----
    census = summary["directions"]["census"]
    assert census["if_unique"] == len(IF_ONLY) * len(layers), census
    assert census["math_unique"] == len(MATH_ONLY) * len(layers), census
    assert census["shared"] == (len(SHARED) + len(TAIL)) * len(layers), census
    print(f"directions: census {census} matches the constructed blocks -- ok")

    # ---- distinctive mass must come back in the original eigenvalue units ----
    # The probe reweights each direction, so the masses are only proportional to
    # the constructed eigenvalues; their ratio is what the construction pins down.
    distinctive = summary["directions"]["distinctive_mass"]
    want_ratio = (
        float(math_values[MATH_ONLY].sum()) / float(if_values[IF_ONLY].sum())
    )
    got_ratio = distinctive["math"] / distinctive["if"]
    assert 0.3 * want_ratio < got_ratio < 3.0 * want_ratio, (got_ratio, want_ratio)
    print(f"directions: distinctive mass {distinctive['if']:.4g} vs {distinctive['math']:.4g} "
          f"({got_ratio:.2f}x, construction says {want_ratio:.2f}x before the probe) -- ok")

    # ---- equal-unique weight must actually equalize, whatever the probe did ----
    got_weights = summary["directions"]["equal_unique_weights"]
    assert got_weights["if"] > 0.6, got_weights
    print(f"directions: equal-unique weight if={got_weights['if']:.4f} "
          f"math={got_weights['math']:.4f} -- ok")

    # The point of the whole exercise: near-equal traces, lopsided protection.
    imbalance = summary["directions"]["current_imbalance"]
    assert imbalance > 1.5, imbalance
    print(f"directions: 0.5/0.5 leaves a {imbalance:.2f}x imbalance despite a "
          f"{trace_math / trace_if:.2f}x trace ratio -- ok")

    # ---- candidates: lambda holds lambda * reg fixed, and equal-unique lands at 1.0 ----
    penalties = summary["influence"]["penalty"]
    reference = 0.5 * penalties["if"] + 0.5 * penalties["math"]
    for record in summary["candidates"]["candidates"]:
        weights = record["weights"]
        mixed = weights["if"] * penalties["if"] + weights["math"] * penalties["math"]
        want_lambda = 3e-2 * reference / mixed
        assert abs(record["equivalent_lambda"] - want_lambda) < 1e-9, record
        if record["name"] == "equal-unique":
            assert abs(record["unique_imbalance"] - 1.0) < 1e-3, record
    print("candidates: equivalent lambda holds lambda * reg fixed, equal-unique "
          "predicts a 1.0x balance -- ok")

    # ---- provenance: a blocking mismatch has to be refused, not reweighted ----
    bad = tmp / "C_bad.pt"
    torch.save(
        covariance_payload(basis, math_values, layers,
                           {**base_meta, "cov_normalization": "base_output_norm",
                            "max_len": 2048, "truncation_side": "left"}),
        bad,
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "onereplay.scripts.analyze_cov_mix",
            "--covs", f"{if_cov},{bad}", "--labels", "if,math",
            "--skip_directions", "1", "--strict", "1",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    assert "cov_normalization" in result.stdout, result.stdout
    print("provenance: mismatched cov_normalization is refused under --strict -- ok")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
