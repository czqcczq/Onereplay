"""Throwaway check for summarize_cost against synthetic metrics files."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def cost(epoch, train_sec, steps, lam=0.0, ratio=0.0, cov=0.0, phases=None):
    record = {
        "epoch": epoch,
        "train_task_loss": 0.11,
        "val_loss": 0.06,
        "replay_lambda": lam,
        "replay_ratio": ratio,
        "eval_sec": 12.0,
        "train_sec": train_sec,
        "train_steps": steps,
        "train_samples": steps * 8,
        "train_tokens": steps * 8 * 180,
        "sec_per_step": train_sec / steps,
        "samples_per_sec": steps * 8 / train_sec,
        "tokens_per_sec": steps * 8 * 180 / train_sec,
        "covariance_memory_gb": cov,
        "peak_memory_allocated_gb": 12.4 + cov,
        "peak_memory_reserved_gb": 14.1 + cov,
    }
    for phase, seconds in (phases or {}).items():
        record[f"time_{phase}_sec"] = seconds
    return record


def main():
    tmp = Path(tempfile.mkdtemp())
    cost_dir = tmp / "cost"
    metrics_dir = tmp / "metrics"

    write(cost_dir / "bench_vanilla_seed1.jsonl", [cost(1, 2450.0, 21090)])
    write(cost_dir / "bench_onereplay_lam1e-2_seed1.jsonl", [cost(1, 2472.0, 21090, lam=0.01, cov=0.875)])
    write(
        cost_dir / "bench_vanilla_prof_seed1.jsonl",
        [cost(1, 41.0, 300, phases={"prepare_batch": 0.2, "task_loss": 18.0, "replay_reg": 0.0, "backward": 19.0, "optimizer": 0.4})],
    )
    write(
        cost_dir / "bench_onereplay_lam1e-2_prof_seed1.jsonl",
        [cost(1, 43.0, 300, lam=0.01, cov=0.875, phases={"prepare_batch": 0.2, "task_loss": 18.1, "replay_reg": 1.9, "backward": 19.4, "optimizer": 0.4})],
    )
    # 09's replay run: 3 epochs, more steps because rows were appended
    write(
        metrics_dir / "cs_replay_r0.10_seed1.jsonl",
        [cost(e, 2698.0, 23199, ratio=0.10) for e in (1, 2, 3)],
    )
    # a legacy-style record with no cost fields must be skipped, not crash
    write(metrics_dir / "cs_replay_r0.01_seed1.jsonl", [{"epoch": 1, "train_task_loss": 0.1}])

    result = subprocess.run(
        [
            sys.executable, "-m", "onereplay.scripts.summarize_cost",
            "--metrics_dir", str(cost_dir),
            "--extra_metrics_dir", str(metrics_dir),
            "--baseline", "bench_vanilla_seed1",
            "--out", str(tmp / "cost_table.md"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("summarize_cost failed")

    out = result.stdout
    assert "no cost fields" in out, "legacy record should be skipped with a notice"
    assert "cs_replay_r0.10_seed1" in out, "extra_metrics_dir run missing"
    assert "+10.1%" in out, "replay epoch-time overhead not computed"
    assert "+0.9%" in out, "onereplay per-step overhead not computed"
    assert "0.875" in out, "covariance memory missing"
    assert (tmp / "cost_table.md").exists()

    cost_section, phase_section = out.split("## 分阶段耗时")
    assert "_prof_" not in cost_section, "short profiled runs must stay out of the headline table"
    assert "-98" not in cost_section, "short runs leaked a bogus overhead number"
    assert "bench_onereplay_lam1e-2_prof_seed1" in phase_section, "phase table missing its run"
    assert "1.9 / 4.4%" in phase_section, "replay_reg share missing"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
