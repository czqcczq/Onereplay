"""Turn judged safety responses into paired significance tests and degeneration
diagnostics -- the statistics needed to move from "descriptive rates" to a
defensible claim.

Reads the judged.jsonl files written by scripts/judge_safety.py (one per run),
aligns them by the shared prompt `id`, and for each track reports:

  1. per-run harmful / refusal / other rates with a paired bootstrap 95% CI;
  2. paired comparisons (vanilla vs onereplay, base vs each) with a McNemar
     exact test on the harmful and refusal labels, so a 6-point gap on 200
     prompts is not read as real when it is within noise;
  3. degeneration diagnostics -- response-length distribution and the share of
     very short / "other" replies -- so a safety change driven by the model
     answering harmful prompts with a stub is not mistaken for retained
     alignment.
  4. net (degeneration-controlled) rates -- harmful/refusal computed only over
     fluent responses, so a model that looks "safer" merely because its outputs
     collapse under jailbreak pressure is exposed. Per-run conditional rates use
     each run's own fluent subset; paired net comparisons restrict to ids where
     both runs answered fluently, keeping the McNemar test strictly paired.

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

DEGENERATE_TOKENS = 8
SHORT_TOKENS = 20


def parse_args() -> argparse.Namespace:
    """Parse the judged files, run order, and bootstrap settings."""

    parser = argparse.ArgumentParser(description="Analyze judged safety responses.")
    parser.add_argument(
        "--judged",
        type=str,
        required=True,
        help="Comma-separated judged.jsonl files or globs (one per run).",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="base,vanilla_seed1,onereplay_lam3e-2_seed1",
        help="Run names in reporting order; must match the judged dirs.",
    )
    parser.add_argument("--out_path", type=str, default="")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument(
        "--net_min_tokens",
        type=int,
        default=SHORT_TOKENS,
        help="Responses with fewer tokens are treated as collapsed/degenerate "
        "and excluded from the net (degeneration-controlled) rates.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def load_runs(spec: str, run_order: list[str]) -> dict[str, dict[str, dict]]:
    """Load each judged file into {run_name: {id: record}}."""

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
        raise ValueError(f"Missing judged runs {missing}; found {sorted(by_run)}")
    return {name: by_run[name] for name in run_order}


def aligned_ids(by_run: dict[str, dict[str, dict]]) -> list[str]:
    """Prompt ids present in every run, so comparisons are strictly paired."""

    id_sets = [set(records) for records in by_run.values()]
    return sorted(set.intersection(*id_sets))


def track_filter(record: dict, track: str) -> bool:
    """Match a record to a track key ('A', 'B', 'C_safe', 'C_unsafe')."""

    if track in ("A", "B", "C"):
        return record.get("track") == track
    if track == "C_safe":
        return record.get("track") == "C" and record.get("label") == "safe"
    if track == "C_unsafe":
        return record.get("track") == "C" and record.get("label") == "unsafe"
    return False


def label_vector(records: dict[str, dict], ids: list[str], outcome: str) -> np.ndarray:
    """Binary vector: 1 where the record's outcome equals `outcome`."""

    return np.array([1 if records[i]["outcome"] == outcome else 0 for i in ids], dtype=int)


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


def bootstrap_rate_ci(vec: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    """Percentile 95% CI for a single rate via resampling of paired items."""

    if len(vec) == 0:
        return [0.0, 0.0]
    idx = rng.integers(0, len(vec), size=(n_boot, len(vec)))
    rates = vec[idx].mean(axis=1)
    return [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]


def bootstrap_diff_ci(
    vec_a: np.ndarray, vec_b: np.ndarray, rng: np.random.Generator, n_boot: int
) -> list[float]:
    """Percentile 95% CI for the paired rate difference a - b."""

    if len(vec_a) == 0:
        return [0.0, 0.0]
    idx = rng.integers(0, len(vec_a), size=(n_boot, len(vec_a)))
    diffs = vec_a[idx].mean(axis=1) - vec_b[idx].mean(axis=1)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def degeneration(records: dict[str, dict], ids: list[str]) -> dict[str, Any]:
    """Length distribution and short/other shares for one run and id set."""

    lengths = np.array([int(records[i].get("response_tokens", 0)) for i in ids])
    outcomes = [records[i]["outcome"] for i in ids]
    n = max(len(ids), 1)
    other_ids = [i for i in ids if records[i]["outcome"] == "other"]
    other_len = np.array([int(records[i].get("response_tokens", 0)) for i in other_ids])
    return {
        "n": len(ids),
        "mean_tokens": float(lengths.mean()) if len(lengths) else 0.0,
        "median_tokens": float(np.median(lengths)) if len(lengths) else 0.0,
        "p10_tokens": float(np.percentile(lengths, 10)) if len(lengths) else 0.0,
        "frac_degenerate": float(np.mean(lengths < DEGENERATE_TOKENS)) if len(lengths) else 0.0,
        "frac_short": float(np.mean(lengths < SHORT_TOKENS)) if len(lengths) else 0.0,
        "other_rate": sum(o == "other" for o in outcomes) / n,
        "other_frac_short": (
            float(np.mean(other_len < SHORT_TOKENS)) if len(other_len) else 0.0
        ),
    }


def response_len(record: dict) -> int:
    """Response length in tokens, defaulting to 0 when unrecorded."""

    return int(record.get("response_tokens", 0))


def outcome_vector(records: dict[str, dict], ids: list[str], outcome: str) -> np.ndarray:
    """Binary vector over a specific id subset (not necessarily all track ids)."""

    return np.array([1 if records[i]["outcome"] == outcome else 0 for i in ids], dtype=int)


def net_section(
    track_ids: list[str],
    by_run: dict[str, dict[str, dict]],
    run_order: list[str],
    rng: np.random.Generator,
    n_boot: int,
    min_tokens: int,
) -> dict[str, Any]:
    """Degeneration-controlled rates and paired comparisons over fluent responses.

    A response is "fluent" when its length is >= min_tokens; collapsed/short
    stubs are dropped. Per-run rates use each run's own fluent subset (so a run
    that collapses a lot has a smaller denominator and its true harmful rate
    surfaces); paired comparisons keep only ids where BOTH runs answered fluently
    so the McNemar test stays strictly paired.
    """

    net: dict[str, Any] = {"min_tokens": min_tokens, "rates": {}, "comparisons": {}}

    for run in run_order:
        records = by_run[run]
        kept = [i for i in track_ids if response_len(records[i]) >= min_tokens]
        harmful = outcome_vector(records, kept, "harmful")
        refusal = outcome_vector(records, kept, "refusal")
        net["rates"][run] = {
            "n_kept": len(kept),
            "n_excluded": len(track_ids) - len(kept),
            "harmful_rate": float(harmful.mean()) if len(harmful) else 0.0,
            "harmful_ci": bootstrap_rate_ci(harmful, rng, n_boot),
            "refusal_rate": float(refusal.mean()) if len(refusal) else 0.0,
            "refusal_ci": bootstrap_rate_ci(refusal, rng, n_boot),
        }

    pairs = []
    if len(run_order) >= 3:
        pairs = [(run_order[1], run_order[2]), (run_order[0], run_order[2]),
                 (run_order[0], run_order[1])]
    for a, b in pairs:
        recs_a, recs_b = by_run[a], by_run[b]
        joint = [
            i
            for i in track_ids
            if response_len(recs_a[i]) >= min_tokens and response_len(recs_b[i]) >= min_tokens
        ]
        ha, hb = outcome_vector(recs_a, joint, "harmful"), outcome_vector(recs_b, joint, "harmful")
        ra, rb = outcome_vector(recs_a, joint, "refusal"), outcome_vector(recs_b, joint, "refusal")
        net["comparisons"][f"{a}_vs_{b}"] = {
            "n_joint": len(joint),
            "harmful": {
                **mcnemar_exact(ha, hb),
                "rate_diff": float(ha.mean() - hb.mean()) if len(ha) else 0.0,
                "rate_diff_ci": bootstrap_diff_ci(ha, hb, rng, n_boot),
            },
            "refusal": {
                **mcnemar_exact(ra, rb),
                "rate_diff": float(ra.mean() - rb.mean()) if len(ra) else 0.0,
                "rate_diff_ci": bootstrap_diff_ci(ra, rb, rng, n_boot),
            },
        }
    return net


def analyze_track(
    track: str,
    by_run: dict[str, dict[str, dict]],
    ids: list[str],
    run_order: list[str],
    rng: np.random.Generator,
    n_boot: int,
    net_min_tokens: int,
) -> dict[str, Any]:
    """Rates+CI, paired McNemar tests, and degeneration for one track."""

    track_ids = [i for i in ids if track_filter(next(iter(by_run.values()))[i], track)]
    result: dict[str, Any] = {"num_paired": len(track_ids), "rates": {}, "degeneration": {}}
    if not track_ids:
        return result

    vectors = {
        run: {
            "harmful": label_vector(by_run[run], track_ids, "harmful"),
            "refusal": label_vector(by_run[run], track_ids, "refusal"),
        }
        for run in run_order
    }
    for run in run_order:
        result["rates"][run] = {
            "harmful_rate": float(vectors[run]["harmful"].mean()),
            "harmful_ci": bootstrap_rate_ci(vectors[run]["harmful"], rng, n_boot),
            "refusal_rate": float(vectors[run]["refusal"].mean()),
            "refusal_ci": bootstrap_rate_ci(vectors[run]["refusal"], rng, n_boot),
        }
        result["degeneration"][run] = degeneration(by_run[run], track_ids)

    comparisons = []
    if len(run_order) >= 3:
        comparisons = [(run_order[1], run_order[2]), (run_order[0], run_order[2]),
                       (run_order[0], run_order[1])]
    result["comparisons"] = {}
    for a, b in comparisons:
        result["comparisons"][f"{a}_vs_{b}"] = {
            "harmful": {
                **mcnemar_exact(vectors[a]["harmful"], vectors[b]["harmful"]),
                "rate_diff": float(vectors[a]["harmful"].mean() - vectors[b]["harmful"].mean()),
                "rate_diff_ci": bootstrap_diff_ci(
                    vectors[a]["harmful"], vectors[b]["harmful"], rng, n_boot
                ),
            },
            "refusal": {
                **mcnemar_exact(vectors[a]["refusal"], vectors[b]["refusal"]),
                "rate_diff": float(vectors[a]["refusal"].mean() - vectors[b]["refusal"].mean()),
                "rate_diff_ci": bootstrap_diff_ci(
                    vectors[a]["refusal"], vectors[b]["refusal"], rng, n_boot
                ),
            },
        }
    result["net"] = net_section(track_ids, by_run, run_order, rng, n_boot, net_min_tokens)
    return result


def main() -> None:
    """Run all tracks and print / save the statistical report."""

    args = parse_args()
    run_order = [name.strip() for name in args.runs.split(",") if name.strip()]
    by_run = load_runs(args.judged, run_order)
    ids = aligned_ids(by_run)
    rng = np.random.default_rng(args.seed)

    report: dict[str, Any] = {"runs": run_order, "num_aligned_ids": len(ids), "tracks": {}}
    for track in ("A", "B", "C_safe", "C_unsafe"):
        report["tracks"][track] = analyze_track(
            track, by_run, ids, run_order, rng, args.n_boot, args.net_min_tokens
        )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_path:
        Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_path).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
