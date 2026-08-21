"""Compare the A/B runs from 23_check_reg_equivalence.slurm.

Decides one thing: can the existing IFEval / Multi-IF / val-loss numbers be
reused after switching to --reg_once_per_update 1, or does a full run have to be
repeated to confirm them.

The two implementations accumulate the same gradient in exact arithmetic, so any
divergence here is floating-point rounding. To tell rounding from a real change,
the comparison needs a noise floor: two runs of the *same* configuration. The
verdict is based on |new - old| relative to |repeat - old|, never on the absolute
size of |new - old| alone.

Two sources are read per run:
  <prefix>.log    per-step task_loss, printed at 6 decimals, used to locate the
                  first divergence and to see whether it grows with step count
  <prefix>.jsonl  epoch metrics at full float precision, the precise verdict

Usage:
  python -m onereplay.scripts.compare_reg_equivalence \
      --old OUT/A1 --repeat OUT/A2 --new OUT/B
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STEP_PATTERN = re.compile(r"step (\d+)/\d+ task_loss=(\S+) replay_reg=(\S+)")

# A relative difference at or below this is indistinguishable from the rounding
# already present in a bf16 training step, so it cannot change a benchmark score.
EQUIVALENT = 1e-4
# Above this, the trajectories have genuinely separated and the scores of a
# 3-epoch run are no longer guaranteed to carry over.
DIVERGED = 1e-2


def parse_steps(prefix: str) -> tuple[dict[int, float], dict[int, float]]:
    path = Path(f"{prefix}.log")
    if not path.is_file():
        raise SystemExit(f"missing step log: {path}")
    task_loss: dict[int, float] = {}
    replay_reg: dict[int, float] = {}
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            match = STEP_PATTERN.search(line)
            if not match:
                continue
            step = int(match.group(1))
            task_loss[step] = float(match.group(2))
            replay_reg[step] = float(match.group(3))
    if not task_loss:
        raise SystemExit(f"no 'step N/M task_loss=' lines found in {path}")
    return task_loss, replay_reg


def parse_metrics(prefix: str) -> dict[str, float]:
    path = Path(f"{prefix}.jsonl")
    if not path.is_file():
        raise SystemExit(f"missing metrics file: {path}")
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise SystemExit(f"no records in {path}")
    return records[-1]


def relative(new: float, old: float) -> float:
    scale = max(abs(old), 1e-12)
    return abs(new - old) / scale


def compare_series(
    new: dict[int, float], old: dict[int, float]
) -> tuple[int | None, float, float, float, float]:
    """First differing step, max abs, max rel, and mean rel over first/last quarter."""

    shared = sorted(set(new) & set(old))
    first_diff = None
    max_abs = 0.0
    max_rel = 0.0
    for step in shared:
        difference = abs(new[step] - old[step])
        if difference > 0.0 and first_diff is None:
            first_diff = step
        max_abs = max(max_abs, difference)
        max_rel = max(max_rel, relative(new[step], old[step]))

    quarter = max(len(shared) // 4, 1)
    head = shared[:quarter]
    tail = shared[-quarter:]
    head_rel = sum(relative(new[s], old[s]) for s in head) / len(head)
    tail_rel = sum(relative(new[s], old[s]) for s in tail) / len(tail)
    return first_diff, max_abs, max_rel, head_rel, tail_rel


def report_series(label: str, new: dict[int, float], old: dict[int, float]) -> dict[str, float]:
    first_diff, max_abs, max_rel, head_rel, tail_rel = compare_series(new, old)
    location = "never" if first_diff is None else f"step {first_diff}"
    print(
        f"  {label:<22} first diff {location:>10} | "
        f"max abs {max_abs:.3e} | max rel {max_rel:.3e} | "
        f"mean rel {head_rel:.3e} (first quarter) -> {tail_rel:.3e} (last quarter)"
    )
    return {"max_rel": max_rel, "head_rel": head_rel, "tail_rel": tail_rel}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reg_once_per_update A/B runs.")
    parser.add_argument("--old", required=True, help="prefix of the old-behavior run (A1)")
    parser.add_argument("--repeat", required=True, help="prefix of its repeat (A2), the noise floor")
    parser.add_argument("--new", required=True, help="prefix of the new-behavior run (B)")
    args = parser.parse_args()

    old_loss, old_reg = parse_steps(args.old)
    repeat_loss, repeat_reg = parse_steps(args.repeat)
    new_loss, new_reg = parse_steps(args.new)
    print(
        f"steps parsed: old={len(old_loss)} repeat={len(repeat_loss)} new={len(new_loss)}\n"
    )

    print("task_loss per step (log precision: 6 decimals)")
    noise = report_series("A2 vs A1 (noise)", repeat_loss, old_loss)
    signal = report_series("B vs A1 (the fix)", new_loss, old_loss)

    print("\nreplay_reg per step (same weights and same C, so this should track closely)")
    report_series("A2 vs A1 (noise)", repeat_reg, old_reg)
    reg_signal = report_series("B vs A1 (the fix)", new_reg, old_reg)

    print("\nepoch metrics (full precision)")
    old_metrics = parse_metrics(args.old)
    repeat_metrics = parse_metrics(args.repeat)
    new_metrics = parse_metrics(args.new)
    metric_rels: dict[str, float] = {}
    for key in ("train_task_loss", "train_replay_reg", "val_loss"):
        if old_metrics.get(key) is None:
            continue
        old_value = float(old_metrics[key])
        repeat_value = float(repeat_metrics[key])
        new_value = float(new_metrics[key])
        metric_rels[key] = relative(new_value, old_value)
        print(
            f"  {key:<18} A1={old_value!r}\n"
            f"  {'':<18} A2={repeat_value!r}  (rel {relative(repeat_value, old_value):.3e})\n"
            f"  {'':<18} B ={new_value!r}  (rel {metric_rels[key]:.3e})"
        )

    # Sanity check that the runs really did differ in configuration; a mislaunched
    # job where all three used the same flag would otherwise "pass" trivially.
    print("\nreg_once_per_update recorded in metrics:")
    print(
        f"  A1={old_metrics.get('reg_once_per_update')} "
        f"A2={repeat_metrics.get('reg_once_per_update')} "
        f"B={new_metrics.get('reg_once_per_update')}"
    )
    if old_metrics.get("reg_once_per_update") == new_metrics.get("reg_once_per_update"):
        print(
            "FAIL: A1 and B used the same reg_once_per_update, so this comparison "
            "tests nothing. Check the slurm script."
        )
        raise SystemExit(1)
    if float(old_metrics.get("replay_lambda", 0.0)) == 0.0:
        print(
            "FAIL: replay_lambda=0, so the penalty never entered the gradient and the "
            "two implementations are trivially identical. Rerun with the best lambda."
        )
        raise SystemExit(1)

    worst = max([signal["max_rel"], reg_signal["max_rel"], *metric_rels.values()])
    growing = signal["tail_rel"] > max(10 * signal["head_rel"], EQUIVALENT)

    print("\nverdict")
    print(f"  noise floor (A2 vs A1) : {noise['max_rel']:.3e}")
    print(f"  worst B vs A1          : {worst:.3e}")
    if worst <= max(noise["max_rel"], EQUIVALENT) and not growing:
        print(
            "  EQUIVALENT: the fix is within the run-to-run noise floor. Reuse the "
            "existing IFEval / Multi-IF / val loss results and only update the timing "
            "numbers; no training run needs repeating."
        )
    elif worst < DIVERGED and not growing:
        print(
            "  CLOSE: above the noise floor but far below anything that moves a "
            "benchmark score. Reuse the existing results and record this bound in the "
            "results log."
        )
    else:
        reason = "the difference grows with step count" if growing else "the difference is large"
        print(
            f"  DIVERGED: {reason}. Repeat ONE full run (e.g. LoRA lambda=3e-2, 3 epochs "
            "+ IFEval) and compare its scores against the recorded ones. The rest of the "
            "sweep still does not need repeating."
        )


if __name__ == "__main__":
    main()
