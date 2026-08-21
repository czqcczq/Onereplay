"""Paired analysis of StrongREJECT scores across runs.

Reads the scored_sr.jsonl files written by scripts/judge_strongreject.py (one
per run), aligns them by the shared prompt `id`, and reports -- overall and per
segment (direct vs jailbreak-wrapped, plus per template) -- :

  1. per-run mean 0-1 harmfulness score with a bootstrap 95% CI, and a
     binarized "jailbroken" rate (score >= threshold);
  2. paired comparisons (vanilla vs onereplay, base vs each) with
       - a paired bootstrap CI on the mean-score difference, and
       - an exact McNemar test on the binarized jailbroken labels,
     so a small gap is not read as real when it is within noise;
  3. attack gain per run: mean(jailbreak) - mean(direct), i.e. how much the
     jailbreak shells raise harmfulness over the un-wrapped baseline.

The StrongREJECT score is ~0 for refusals and for degenerate/empty stubs alike,
so unlike the WildGuard harmful-rate this metric is not inflated by output
collapse -- no separate "net" (degeneration-controlled) pass is needed.

No GPU, no model load; pure analysis over existing files.
"""

from __future__ import annotations

import argparse
import glob
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse the scored files, run order, and bootstrap settings."""

    parser = argparse.ArgumentParser(description="Analyze StrongREJECT scores.")
    parser.add_argument(
        "--scored",
        type=str,
        required=True,
        help="Comma-separated scored_sr.jsonl files or globs (one per run).",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="base,vanilla_seed1,onereplay_lam3e-2_seed1",
        help="Run names in reporting order; must match the scored dirs.",
    )
    parser.add_argument("--out_path", type=str, default="")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--binary_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def load_runs(spec: str, run_order: list[str]) -> dict[str, dict[str, dict]]:
    """Load each scored file into {run_name: {id: record}}."""

    paths: list[Path] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        matches = glob.glob(piece)
        paths.extend(Path(m) for m in matches) if matches else paths.append(Path(piece))

    by_run: dict[str, dict[str, dict]] = {}
    for path in paths:
        if not path.exists():
            continue
        run_name = path.parent.name
        records: dict[str, dict] = {}
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    records[record["id"]] = record
        by_run[run_name] = records

    missing = [name for name in run_order if name not in by_run]
    if missing:
        raise ValueError(f"Missing scored runs {missing}; found {sorted(by_run)}")
    return {name: by_run[name] for name in run_order}


def aligned_ids(by_run: dict[str, dict[str, dict]]) -> list[str]:
    """Prompt ids present in every run, so comparisons are strictly paired."""

    id_sets = [set(records) for records in by_run.values()]
    return sorted(set.intersection(*id_sets))


def segment_ids(records: dict[str, dict], ids: list[str], segment: str) -> list[str]:
    """Ids belonging to a segment: 'all', 'direct', or 'jailbreak'."""

    if segment == "all":
        return list(ids)
    if segment == "direct":
        return [i for i in ids if records[i].get("template_id") == "direct"]
    if segment == "jailbreak":
        return [i for i in ids if records[i].get("template_id") != "direct"]
    return [i for i in ids if records[i].get("template_id") == segment]


def score_vector(records: dict[str, dict], ids: list[str]) -> np.ndarray:
    """Continuous 0-1 StrongREJECT scores over an id subset."""

    return np.array([float(records[i]["sr_score"]) for i in ids], dtype=float)


def binary_vector(records: dict[str, dict], ids: list[str], threshold: float) -> np.ndarray:
    """Binarized jailbroken flags (score >= threshold) over an id subset."""

    return np.array([1 if float(records[i]["sr_score"]) >= threshold else 0 for i in ids], dtype=int)


def mcnemar_exact(vec_a: np.ndarray, vec_b: np.ndarray) -> dict[str, Any]:
    """Two-sided exact McNemar test on paired binary labels a vs b."""

    b = int(np.sum((vec_a == 1) & (vec_b == 0)))
    c = int(np.sum((vec_a == 0) & (vec_b == 1)))
    n = b + c
    if n == 0:
        return {"discordant_ab": b, "discordant_ba": c, "p_value": 1.0}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"discordant_ab": b, "discordant_ba": c, "p_value": min(1.0, 2 * tail)}


def bootstrap_mean_ci(vec: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    """Percentile 95% CI for a single mean via resampling of paired items."""

    if len(vec) == 0:
        return [0.0, 0.0]
    idx = rng.integers(0, len(vec), size=(n_boot, len(vec)))
    means = vec[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def bootstrap_diff_ci(
    vec_a: np.ndarray, vec_b: np.ndarray, rng: np.random.Generator, n_boot: int
) -> list[float]:
    """Percentile 95% CI for the paired mean difference a - b."""

    if len(vec_a) == 0:
        return [0.0, 0.0]
    idx = rng.integers(0, len(vec_a), size=(n_boot, len(vec_a)))
    diffs = vec_a[idx].mean(axis=1) - vec_b[idx].mean(axis=1)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def analyze_segment(
    segment: str,
    by_run: dict[str, dict[str, dict]],
    ids: list[str],
    run_order: list[str],
    rng: np.random.Generator,
    n_boot: int,
    threshold: float,
) -> dict[str, Any]:
    """Per-run mean/jailbroken rates and paired comparisons for one segment."""

    ref = next(iter(by_run.values()))
    seg_ids = segment_ids(ref, ids, segment)
    result: dict[str, Any] = {"num_paired": len(seg_ids), "rates": {}, "comparisons": {}}
    if not seg_ids:
        return result

    scores = {run: score_vector(by_run[run], seg_ids) for run in run_order}
    flags = {run: binary_vector(by_run[run], seg_ids, threshold) for run in run_order}
    for run in run_order:
        result["rates"][run] = {
            "mean_score": float(scores[run].mean()),
            "mean_score_ci": bootstrap_mean_ci(scores[run], rng, n_boot),
            "jailbroken_rate": float(flags[run].mean()),
            "jailbroken_ci": bootstrap_mean_ci(flags[run].astype(float), rng, n_boot),
        }

    comparisons = []
    if len(run_order) >= 3:
        comparisons = [(run_order[1], run_order[2]), (run_order[0], run_order[2]),
                       (run_order[0], run_order[1])]
    for a, b in comparisons:
        result["comparisons"][f"{a}_vs_{b}"] = {
            "mean_score_diff": float(scores[a].mean() - scores[b].mean()),
            "mean_score_diff_ci": bootstrap_diff_ci(scores[a], scores[b], rng, n_boot),
            "jailbroken": {
                **mcnemar_exact(flags[a], flags[b]),
                "rate_diff": float(flags[a].mean() - flags[b].mean()),
                "rate_diff_ci": bootstrap_diff_ci(
                    flags[a].astype(float), flags[b].astype(float), rng, n_boot
                ),
            },
        }
    return result


def attack_gain(
    by_run: dict[str, dict[str, dict]], ids: list[str], run_order: list[str]
) -> dict[str, Any]:
    """Per-run mean-score lift from jailbreak shells over the direct baseline."""

    ref = next(iter(by_run.values()))
    direct_ids = segment_ids(ref, ids, "direct")
    jb_ids = segment_ids(ref, ids, "jailbreak")
    gains: dict[str, Any] = {}
    for run in run_order:
        direct_mean = (
            float(score_vector(by_run[run], direct_ids).mean()) if direct_ids else 0.0
        )
        jb_mean = float(score_vector(by_run[run], jb_ids).mean()) if jb_ids else 0.0
        gains[run] = {
            "direct_mean_score": direct_mean,
            "jailbreak_mean_score": jb_mean,
            "attack_gain": jb_mean - direct_mean,
        }
    return gains


def main() -> None:
    """Run all segments and print / save the statistical report."""

    args = parse_args()
    run_order = [name.strip() for name in args.runs.split(",") if name.strip()]
    by_run = load_runs(args.scored, run_order)
    ids = aligned_ids(by_run)
    rng = np.random.default_rng(args.seed)

    ref = next(iter(by_run.values()))
    templates = sorted({ref[i].get("template_id", "?") for i in ids})
    segments = ["all", "direct", "jailbreak", *templates]

    report: dict[str, Any] = {
        "runs": run_order,
        "num_aligned_ids": len(ids),
        "binary_threshold": args.binary_threshold,
        "attack_gain": attack_gain(by_run, ids, run_order),
        "segments": {},
    }
    for segment in segments:
        report["segments"][segment] = analyze_segment(
            segment, by_run, ids, run_order, rng, args.n_boot, args.binary_threshold
        )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_path:
        Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_path).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
