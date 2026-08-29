"""Throwaway check for onereplay/scripts/inspect_fisher.py (no GPU, no cluster).

46_math_fisher.pbs leans on this script to catch the silent ways an F_math
collection can be wrong, so the script itself has to be known-good before a job
spends two GPU hours trusting it. Builds synthetic payloads with the real
metadata shape (F_if's numbers are the measured ones from job 2944210) and runs
the exact invocation the PBS Step 2 uses against a correct file and against ones
with each defect injected.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

FINGERPRINT = "abc123def456"


def fisher_payload(
    mean: float,
    tokens: float,
    max_len: int,
    truncation_side: str,
    zero_rows: int,
    ess_ratio: float,
    examples: int,
    *,
    use_bf16: int = 0,
    modules: list[str] | None = None,
    fingerprint: str = FINGERPRINT,
) -> dict:
    return {
        "fishers": {"model.layers.0.self_attn.q_proj": torch.full((4, 4), mean)},
        "counts": {"model.layers.0.self_attn.q_proj": examples},
        "metadata": {
            "estimator": "assistant_only_sequence_sum_diagonal_empirical_fisher",
            "loss_reduction": "sum",
            "num_examples": examples,
            "pool_rows": examples,
            "pool_fingerprint": fingerprint,
            "use_bf16": use_bf16,
            "target_modules": modules if modules is not None else ["q_proj", "v_proj"],
            "require_target": 1,
            "sample_shuffle": 0,
            "truncation_side": truncation_side,
            "max_len": max_len,
            "prompt_prefix_mismatches": 0,
            "length_weighting": {
                "examples": float(examples),
                "zero_supervision_rows": float(zero_rows),
                "mean_supervised_tokens": tokens,
                "median_supervised_tokens": tokens * 0.8,
                "max_supervised_tokens": tokens * 5,
                "effective_sample_size": examples * ess_ratio,
                "ess_ratio": ess_ratio,
                "length_exponent": 1.4,
            },
            "fisher_scale": {"layers": 56.0, "mean": mean, "max": mean * 100},
        },
    }


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "onereplay.scripts.inspect_fisher", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp())

    if_path = tmp / "fisher_if.pt"
    # The real F_if: 20k rows, max_len 512, right truncation, mean 9.98e-2,
    # 1408 zero-supervision rows, ESS 2547/20000.
    torch.save(fisher_payload(9.98e-2, 24.1, 512, "right", 1408, 0.127, 20000), if_path)

    good = tmp / "fisher_math.pt"
    torch.save(fisher_payload(3.0, 357.1, 2048, "left", 0, 0.42, 25114), good)

    cov = tmp / "cov_math.pt"
    torch.save(
        {
            "covariances": {"model.layers.0.self_attn.q_proj": torch.eye(4)},
            "metadata": {
                "pool_rows": 25114,
                "pool_fingerprint": FINGERPRINT,
                "max_len": 2048,
                "truncation_side": "left",
                "require_target": 1,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
        cov,
    )

    # ---- --print_meta: how Step 0 reads C_math's fingerprint ----
    result = run(["--path", str(cov), "--print_meta", "pool_fingerprint"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == FINGERPRINT, repr(result.stdout)
    print(f"print_meta on a cov file -> {result.stdout.strip()}")

    result = run(["--path", str(cov), "--print_meta", "target_modules"])
    assert result.returncode == 0 and "q_proj" in result.stdout, result.stdout
    print(f"print_meta of a list     -> {result.stdout.strip()}")

    result = run(["--path", str(cov), "--print_meta", "nonexistent_key"])
    assert result.returncode != 0, "missing key should fail loudly"
    print("print_meta of a missing key exits non-zero")

    result = run(["--path", str(tmp / "nope.pt"), "--print_meta", "max_len"])
    assert result.returncode != 0 and "not found" in result.stderr, result.stderr
    print("missing file exits non-zero")

    # ---- the PBS Step 2 invocation, verbatim ----
    step2 = [
        "--expect_max_len", "2048",
        "--expect_truncation_side", "left",
        "--expect_target_modules", "q_proj,v_proj",
        "--expect_require_target", "1",
        "--expect_sample_shuffle", "0",
        "--expect_use_bf16", "0",
        "--expect_zero_supervision_rows", "0",
        "--expect_pool_fingerprint", FINGERPRINT,
        "--reference", str(if_path),
        "--strict", "1",
    ]

    result = run(["--path", str(good), *step2])
    print("\n---- Step 2 on a correct F_math ----")
    print(result.stdout)
    assert result.returncode == 0, f"correct payload rejected:\n{result.stdout}\n{result.stderr}"
    assert "all checks passed" in result.stdout
    # 3.0 / 9.98e-2 = 30.06x, and 357.1 / 24.1 = 14.8x, so the measured ratio must
    # land inside the T^a band [14.8, 219] and trigger the normalize warning.
    assert "30.06x" in result.stdout, "scale ratio not reported"
    assert "14.82x" in result.stdout, "token ratio not reported"
    assert "inverse-scale weights" in result.stdout, "mixing guidance missing"

    # ---- each defect, injected one at a time ----
    defects = {
        "right truncation (the --truncation_side flag never landed)": (
            fisher_payload(2.4, 300.0, 2048, "right", 812, 0.31, 25114),
            ["truncation_side", "zero_supervision_rows"],
        ),
        "max_len 1024 instead of C_math's 2048": (
            fisher_payload(2.0, 300.0, 1024, "left", 0, 0.40, 25114),
            ["max_len"],
        ),
        "bf16 collection": (
            fisher_payload(3.0, 357.1, 2048, "left", 0, 0.42, 25114, use_bf16=1),
            ["use_bf16"],
        ),
        "wrong target_modules": (
            fisher_payload(
                3.0, 357.1, 2048, "left", 0, 0.42, 25114,
                modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            ["target_modules"],
        ),
        "pool fingerprint diverged from C_math": (
            fisher_payload(3.0, 357.1, 2048, "left", 0, 0.42, 25114, fingerprint="deadbeef"),
            ["pool_fingerprint"],
        ),
    }
    print("---- Step 2 on injected defects ----")
    for name, (payload, expected_mentions) in defects.items():
        path = tmp / "defect.pt"
        torch.save(payload, path)
        result = run(["--path", str(path), *step2])
        assert result.returncode == 1, f"{name}: accepted a bad payload"
        assert "FAILED:" in result.stdout, f"{name}: no failure block"
        for mention in expected_mentions:
            assert mention in result.stdout.split("FAILED:")[1], (
                f"{name}: failure block never mentions {mention}\n{result.stdout}"
            )
        print(f"  rejected: {name}")

    # ---- 47_if_fisher.pbs Step 2: the same checks, but shaped for F_if ----
    # F_if is legitimately 512 / right-truncated / 1408 zero-supervision rows, so the
    # re-collection has to be validated against those values rather than F_math's.
    if_step2 = [
        "--expect_max_len", "512",
        "--expect_truncation_side", "right",
        "--expect_target_modules", "q_proj,v_proj",
        "--expect_require_target", "1",
        "--expect_sample_shuffle", "0",
        "--expect_use_bf16", "0",
        "--expect_zero_supervision_rows", "1408",
        "--expect_mean", "9.98e-2",
        "--mean_rtol", "0.02",
        "--strict", "1",
    ]
    result = run(["--path", str(if_path), *if_step2])
    assert result.returncode == 0, f"F_if rejected by its own expectations:\n{result.stdout}"
    print("\nF_if passes the 47-shaped checks (right truncation, 1408 zero rows, mean 9.98e-2)")

    # A mean that drifted past the tolerance must fail: that is the signal that the
    # re-collection produced a different F and the old lambda no longer applies.
    drifted = tmp / "fisher_if_drifted.pt"
    torch.save(fisher_payload(1.20e-1, 24.1, 512, "right", 1408, 0.127, 20000), drifted)
    result = run(["--path", str(drifted), *if_step2])
    assert result.returncode == 1, "a 20% mean drift was accepted"
    assert "fisher_scale.mean" in result.stdout.split("FAILED:")[1], result.stdout
    print("  rejected: mean drifted 20% from the reference")

    # Within tolerance must pass, so float-level differences between GPUs do not
    # block a genuine reproduction.
    close = tmp / "fisher_if_close.pt"
    torch.save(fisher_payload(9.99e-2, 24.1, 512, "right", 1408, 0.127, 20000), close)
    result = run(["--path", str(close), *if_step2])
    assert result.returncode == 0, f"a 0.1% difference was rejected:\n{result.stdout}"
    print("  accepted: mean within 0.1%, i.e. float-level reproduction")

    # ---- --strict 0 reports without failing the job ----
    torch.save(defects["bf16 collection"][0], tmp / "defect.pt")
    result = run(["--path", str(tmp / "defect.pt"), *step2[:-1], "0"])
    assert result.returncode == 0 and "FAILED:" in result.stdout, result.stdout
    print("  --strict 0 reports the same failure but exits 0")

    # ---- a missing reference must not crash the self-check ----
    result = run([
        "--path", str(good), *step2[:-3], str(tmp / "no_such_if.pt"), "--strict", "1",
    ])
    assert result.returncode == 0, result.stdout
    assert "scale comparison skipped" in result.stdout, result.stdout
    print("  absent --reference degrades to a skip, not a crash")

    # ---- per-layer diagnostic (compare_layers) ----
    # Build two Fishers with known per-layer masses so the global ratio and the
    # spread are checkable by hand rather than trusted.
    def multilayer(masses: dict[str, float]) -> dict:
        payload = fisher_payload(sum(masses.values()) / len(masses), 100.0, 2048, "left", 0, 0.5,
                                 1000)
        # One 1x1 matrix per layer whose single entry is that layer's mass, so
        # sum == the number we put in and the ratio is exact.
        payload["fishers"] = {name: torch.tensor([[mass]]) for name, mass in masses.items()}
        return payload

    # target/reference chosen so the global ratio is exactly 4.0 (8+4+4 over 1+1+2),
    # while layer ratios range 2..8 -> a deliberately wide spread.
    tgt_layers = tmp / "tgt_layers.pt"
    ref_layers = tmp / "ref_layers.pt"
    torch.save(multilayer({"l0": 8.0, "l1": 4.0, "l2": 4.0}), tgt_layers)
    torch.save(multilayer({"l0": 1.0, "l1": 1.0, "l2": 2.0}), ref_layers)

    result = run(["--path", str(tgt_layers), "--reference", str(ref_layers), "--strict", "0"])
    assert result.returncode == 0, result.stdout
    print("\n---- per-layer diagnostic output ----")
    print(result.stdout.split("per-layer mass")[1] if "per-layer mass" in result.stdout
          else result.stdout)
    assert "per-layer mass" in result.stdout, "layer diagnostic did not run"
    # global ratio = (8+4+4)/(1+1+2) = 16/4 = 4.0
    assert "global mass ratio        : 4.000x" in result.stdout, result.stdout
    # equal-contribution weight for target = 1/(1+4) = 0.2
    assert "target=0.2000 reference=0.8000" in result.stdout, result.stdout
    # layer ratios are 8, 4, 2 -> spread 8/2 = 4.0x, and all three target-dominant
    assert "target dominates (r_l>1) : 3 / 3 layers" in result.stdout, result.stdout
    assert "spread (max/min)       : 4.0x" in result.stdout, result.stdout
    print("  global ratio, equal-contribution weight, dominance count all correct")

    # A genuinely wide spread must trip the "wide" verdict, since that is the whole
    # point of the per-layer check: one layer 100x, another 1x.
    torch.save(multilayer({"l0": 100.0, "l1": 1.0}), tgt_layers)
    torch.save(multilayer({"l0": 1.0, "l1": 1.0}), ref_layers)
    result = run(["--path", str(tgt_layers), "--reference", str(ref_layers), "--strict", "0"])
    assert "-> wide" in result.stdout, result.stdout
    print("  wide spread trips the wide verdict")
    # restore the balanced pair for later reuse
    torch.save(multilayer({"l0": 8.0, "l1": 4.0, "l2": 4.0}), tgt_layers)
    torch.save(multilayer({"l0": 1.0, "l1": 1.0, "l2": 2.0}), ref_layers)

    # --layer_diagnostic 0 must suppress it
    result = run(["--path", str(tgt_layers), "--reference", str(ref_layers),
                  "--layer_diagnostic", "0", "--strict", "0"])
    assert "per-layer mass" not in result.stdout, "layer diagnostic ran despite the 0 flag"
    print("  --layer_diagnostic 0 suppresses it")

    # It must also work when the reference is a covariance file (trace, not sum).
    cov_layers = tmp / "cov_layers.pt"
    torch.save(
        {"covariances": {"l0": torch.eye(4) * 2.0, "l1": torch.eye(4)},
         "metadata": {"pool_fingerprint": "x"}},
        cov_layers,
    )
    fish_layers = tmp / "fish_layers.pt"
    payload = fisher_payload(1.0, 100.0, 2048, "left", 0, 0.5, 1000)
    payload["fishers"] = {"l0": torch.tensor([[8.0]]), "l1": torch.tensor([[4.0]])}
    torch.save(payload, fish_layers)
    result = run(["--path", str(fish_layers), "--reference", str(cov_layers), "--strict", "0"])
    # target sum masses 8,4 ; reference traces 8,4 -> global ratio 1.0
    assert "(sum target / trace reference)" in result.stdout, result.stdout
    assert "global mass ratio        : 1.000x" in result.stdout, result.stdout
    print("  cov reference uses trace, fisher target uses sum, ratio 1.0 as expected")

    # ---- a payload predating the diagnostics must say so, not KeyError ----
    stale = fisher_payload(3.0, 357.1, 2048, "left", 0, 0.42, 25114)
    del stale["metadata"]["length_weighting"]
    del stale["metadata"]["fisher_scale"]
    torch.save(stale, tmp / "stale.pt")
    result = run(["--path", str(tmp / "stale.pt"), *step2])
    assert result.returncode != 0, "stale payload should be rejected"
    assert "predating" in result.stderr or "predating" in result.stdout, (
        f"stale payload gave an unhelpful error:\n{result.stdout}\n{result.stderr}"
    )
    print("  payload without diagnostics gives a readable error")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
