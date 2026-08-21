"""Summarize 25_bench_reg_cost.slurm into the numbers that go in the results log.

The two sides of each pair no longer have the same length: the vanilla run only
supplies the ms/step denominator and stops after a few thousand steps, while the
OneReplay run is the full production run that also gets evaluated. That is fine
for the comparison, because `win=` is a windowed rate rather than a whole-run
average, but it does mean the token counts differ by design.

Steady-state definition matches results_log: the 25th percentile of a run's
500-step window speeds (the `win=` field). Node contention only ever makes a
window slower, so a low quantile strips it out while the mean and median absorb
it. Across repeats the fastest run represents the configuration, for the same
reason.

The report also runs one consistency check worth more than the raw numbers.
Before the fix the penalty was paid on every micro-batch, so the measured
per-micro-batch overhead *was* the cost of evaluating it once. After the fix it
is paid once per optimizer step, so:

    (OneReplay - vanilla) * accumulation_steps

should land back on the old per-micro-batch overhead (about 9 ms for LoRA, about
455 ms for full fine-tuning). If it does, the two measurements corroborate each
other and the only thing that changed is how often the penalty is evaluated.

Usage: python -m onereplay.scripts.summarize_reg_cost --out_dir DIR
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WINDOW_PATTERN = re.compile(r"win=([\d.]+)ms/step")

# Measured before the fix, when the penalty ran on every micro-batch. Kept here
# so the report can state the improvement and check its own consistency.
BASELINE = {
    "lora": {"vanilla_ms": 88.1, "onereplay_ms": 97.5, "label": "LoRA q/v, lambda=3e-2"},
    "full": {"vanilla_ms": 96.5, "onereplay_ms": 552.0, "label": "full finetune, lambda=3e-2"},
}
PAIRS = [("lora", "lora_van", "lora_or"), ("full", "full_van", "full_or")]


def steady_ms(path: Path) -> tuple[float, int]:
    """25th percentile of the window speeds in one run log."""

    windows = [float(match.group(1)) for match in WINDOW_PATTERN.finditer(path.read_text(
        encoding="utf-8", errors="replace"
    ))]
    if not windows:
        raise ValueError(f"no 'win=' fields in {path}")
    if len(windows) < 4:
        # quantiles needs enough points to be meaningful; fall back to the min,
        # which is the same intent (discard contention) with less resolution.
        return min(windows), len(windows)
    return statistics.quantiles(windows, n=4, method="inclusive")[0], len(windows)


def load_runs(out_dir: Path) -> dict[str, dict[str, object]]:
    """Collect per-mode steady-state times, keeping the fastest repeat."""

    runs: dict[str, dict[str, object]] = {}
    for log_path in sorted(out_dir.glob("*_rep*.log")):
        mode = log_path.stem.rsplit("_rep", 1)[0]
        try:
            ms, window_count = steady_ms(log_path)
        except ValueError as error:
            print(f"skipping {log_path.name}: {error}")
            continue

        metrics_path = log_path.with_suffix(".jsonl")
        record: dict[str, object] = {}
        if metrics_path.is_file():
            lines = [
                line
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if lines:
                record = json.loads(lines[-1])

        entry = {
            "ms": ms,
            "windows": window_count,
            "source": log_path.name,
            "peak_gb": record.get("peak_memory_allocated_gb"),
            "tokens": record.get("train_tokens"),
            "steps": record.get("train_steps"),
            "reg_once": record.get("reg_once_per_update"),
            "accum": record.get("accumulation_size"),
            "batch": record.get("batch_size"),
        }
        print(
            f"  {log_path.name:<24} {ms:7.1f} ms/step over {window_count} windows"
            + (f", peak {entry['peak_gb']:.2f} GiB" if entry["peak_gb"] is not None else "")
        )
        previous = runs.get(mode)
        if previous is None or ms < float(previous["ms"]):
            runs[mode] = entry
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize regularizer cost benchmark.")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"not a directory: {out_dir}")

    print("per-run steady state (25th percentile of window speeds)")
    runs = load_runs(out_dir)
    if not runs:
        raise SystemExit(f"no *_rep*.log files with win= fields under {out_dir}")

    problems: list[str] = []
    for entry in runs.values():
        if entry["reg_once"] not in (1, None):
            problems.append(
                f"{entry['source']} ran with reg_once_per_update={entry['reg_once']}; "
                "this benchmark is meant to measure the fixed behavior"
            )

    print("\nregularizer cost after the fix")
    header = (
        f"{'config':<28} {'vanilla':>9} {'OneReplay':>10} {'delta':>9} {'relative':>9} "
        f"{'per update':>11}"
    )
    print(header)
    print("-" * len(header))

    for key, vanilla_mode, reg_mode in PAIRS:
        if vanilla_mode not in runs or reg_mode not in runs:
            print(f"{BASELINE[key]['label']:<28} (missing {vanilla_mode} or {reg_mode})")
            continue
        vanilla = runs[vanilla_mode]
        regularized = runs[reg_mode]
        vanilla_ms = float(vanilla["ms"])
        reg_ms = float(regularized["ms"])
        delta = reg_ms - vanilla_ms
        accum = int(regularized["accum"] or 64)
        batch = int(regularized["batch"] or 8)
        accumulation_steps = max(accum // batch, 1)
        per_update = delta * accumulation_steps
        print(
            f"{BASELINE[key]['label']:<28} {vanilla_ms:8.1f}  {reg_ms:9.1f}  "
            f"{delta:+8.1f}  {delta / vanilla_ms * 100:+8.1f}%  {per_update:10.1f}"
        )

        # Only meaningful when the two runs cover the same steps: then a token
        # mismatch means they saw different data and ms/step is not comparable.
        # Different step counts are expected (see the module docstring).
        same_length = bool(vanilla["steps"]) and vanilla["steps"] == regularized["steps"]
        if (
            same_length
            and vanilla["tokens"]
            and regularized["tokens"]
            and vanilla["tokens"] != regularized["tokens"]
        ):
            problems.append(
                f"{key}: train_tokens differ ({vanilla['tokens']} vs {regularized['tokens']}), "
                "so ms/step is not an apples-to-apples comparison"
            )

        old = BASELINE[key]
        old_delta = old["onereplay_ms"] - old["vanilla_ms"]
        old_relative = old_delta / old["vanilla_ms"] * 100
        new_relative = delta / vanilla_ms * 100
        print(
            f"{'':<28} before the fix: {old['vanilla_ms']:.1f} -> "
            f"{old['onereplay_ms']:.1f} ms ({old_relative:+.1f}%), "
            f"now {new_relative:+.1f}%"
            + (
                f", {old_relative / new_relative:.1f}x cheaper"
                if new_relative > 0.01
                else ""
            )
        )
        # The old per-micro-batch overhead was the cost of one evaluation, which
        # is exactly what one update now pays.
        drift = abs(per_update - old_delta) / max(old_delta, 1e-9)
        verdict = "consistent" if drift < 0.35 else f"off by {drift * 100:.0f}%, investigate"
        print(
            f"{'':<28} one evaluation costs {per_update:.1f} ms/update now vs "
            f"{old_delta:.1f} ms/micro-batch before: {verdict}"
        )

        if vanilla["peak_gb"] is not None and regularized["peak_gb"] is not None:
            print(
                f"{'':<28} peak allocated {float(vanilla['peak_gb']):.2f} -> "
                f"{float(regularized['peak_gb']):.2f} GiB "
                f"({float(regularized['peak_gb']) - float(vanilla['peak_gb']):+.2f}; "
                "unchanged by the fix, the peak still comes from the step that "
                "evaluates the penalty)"
            )

    if problems:
        print("\nwarnings")
        for problem in problems:
            print(f"  {problem}")


if __name__ == "__main__":
    main()
