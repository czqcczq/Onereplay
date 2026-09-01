"""Report every run's score twice: on the whole benchmark, and on the subset
the base model already got right.

Overall accuracy answers "how good is this model", which is not the question
these experiments ask. Retention asks "did fine-tuning destroy something the
base model could do", and on the whole benchmark that signal is diluted by the
items base never solved: a run cannot lose what W0 never had, so those items
only contribute noise plus whatever the run happened to learn from
Commonsense170k. On GSM8K base solves ~77% of 1319 items, so ~300 items --
nearly a quarter of the test set -- can only move a retention number by
accident.

Splitting on base's own correctness separates the two effects that overall
accuracy sums together:

    base-correct subset    the retention rate proper. 100% means nothing W0
                           could do was lost. This is the number the anti-
                           forgetting claim is actually about.
    base-wrong subset      items W0 failed. A run scoring above 0 here gained
                           something, either from the new task or from a
                           format change that the scorer now accepts.

A method can look flat overall while having broken 8% of what base knew and
accidentally picked up 8% of what it did not, and only this split shows it.

Pairing is per item, not per count, so the two subsets are fixed by the
baseline run and identical for every method compared against it.

--runs defaults to scanning every run directory that has a verdict file, so the
usual invocation needs no run names at all:

    python -m onereplay.scripts.score_retention_subset \\
        --results_root /path/results --metrics gsm8k,math500

Pass --runs explicitly only to pin the table to a fixed set of arms.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Fields the eval metrics use to identify one item, in preference order. Code
# metrics carry a real id; the math ones only save the question text, which is
# unique within each of these test sets and, more to the point, is what the
# model was actually shown -- so it stays a valid join key even if the eval
# file's row order changes.
KEY_FIELDS = ("task_id", "id", "question", "prompt", "key")

# Per-item verdict fields, in preference order. Every one of these is a bool
# where True means the item was scored as correct.
CORRECT_FIELDS = ("correct", "passed", "follow_all_instructions")

# Which per-item file holds the verdicts. Most metrics score inline while
# generating; IFEval writes responses.jsonl without verdicts and defers judging
# to the vendored Google checker, whose output is a separate file.
RESULT_FILES = {
    "ifeval": "eval_results_strict.jsonl",
    "ifeval_loose": "eval_results_loose.jsonl",
}


def result_filename(metric: str) -> str:
    return RESULT_FILES.get(metric, "responses.jsonl")


def metric_dirname(metric: str) -> str:
    """Directory under results_root. ifeval_loose is a scoring view of ifeval."""

    return "ifeval" if metric == "ifeval_loose" else metric


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def item_verdicts(rows: list[dict[str, Any]], path: Path) -> dict[str, bool]:
    """Map each item to whether this run got it right.

    Duplicate keys are suffixed by occurrence rather than dropped or allowed to
    overwrite each other, so a benchmark that repeats a prompt keeps every copy
    and still lines up across runs (the metrics iterate their eval file in a
    fixed order, so the n-th copy is the same item in every run).
    """

    key_field = next((f for f in KEY_FIELDS if rows and f in rows[0]), "")
    correct_field = next((f for f in CORRECT_FIELDS if rows and f in rows[0]), "")
    if not key_field or not correct_field:
        raise ValueError(
            f"{path} has no usable key/verdict field. Looked for {KEY_FIELDS} and "
            f"{CORRECT_FIELDS}; the file has {sorted(rows[0]) if rows else 'no rows'}"
        )

    verdicts: dict[str, bool] = {}
    seen: dict[str, int] = {}
    for row in rows:
        base_key = str(row.get(key_field))
        count = seen.get(base_key, 0)
        seen[base_key] = count + 1
        key = base_key if count == 0 else f"{base_key}#{count}"
        verdicts[key] = bool(row.get(correct_field))
    return verdicts


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs only."""

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def discover_runs(root: Path, metric: str, baseline: str) -> list[str]:
    """Every run directory under this metric that actually has a verdict file.

    Run names are long and easy to mistype, and which ones exist differs per
    metric because arms get evaluated at different times. Scanning is both
    less error-prone than a hand-written list and honest about what is there:
    an arm you thought you had evaluated simply does not appear.
    """

    metric_dir = root / metric_dirname(metric)
    if not metric_dir.is_dir():
        return []
    filename = result_filename(metric)
    found = sorted(
        entry.name for entry in metric_dir.iterdir() if (entry / filename).is_file()
    )
    # Baseline first: every other row is read as a delta against it.
    head = [baseline] if baseline in found else []
    return head + [run for run in found if run != baseline]


def report(
    metric: str,
    runs: list[str],
    root: Path,
    baseline: str,
) -> list[dict[str, Any]]:
    """Print the split table for one metric and return its rows as records."""

    filename = result_filename(metric)
    loaded: dict[str, dict[str, bool]] = {}
    for run in runs:
        path = root / metric_dirname(metric) / run / filename
        if not path.is_file():
            print(f"[skip] {path} 不存在")
            continue
        try:
            rows = load_rows(path)
            if not rows:
                print(f"[skip] {path} 是空文件（评测未完成或失败）")
                continue
            loaded[run] = item_verdicts(rows, path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            # --runs all intentionally scans directories left by interrupted
            # evaluations too. One bad/incomplete run must not hide every
            # healthy run for this metric.
            print(f"[skip] {path} 无法读取为完整判分结果: {exc}")
            continue

    if baseline not in loaded:
        print(
            f"[skip] metric={metric}: 基线 run {baseline!r} 没有有效判分结果，无法定义子集"
        )
        return []

    base_verdicts = loaded[baseline]
    print(f"\n{'=' * 104}\nmetric = {metric}   (子集由 {baseline} 定义，判分文件 {filename})\n{'=' * 104}")

    records: list[dict[str, Any]] = []
    # ASCII headers: a CJK header is two cells wide but one character long, so
    # str.format's padding would misalign every column under it.
    header = (
        f"{'run':<46} {'n':>5} {'overall':>8} {'n_ok':>6} {'acc|base_ok':>12} "
        f"{'n_bad':>6} {'acc|base_bad':>13}"
    )
    print(header)
    for run, verdicts in loaded.items():
        # Only items both runs scored. A --limit mismatch or a crashed job
        # shows up as a shrunken n instead of a silently skewed rate.
        common = sorted(set(base_verdicts) & set(verdicts))
        if len(common) != len(base_verdicts) or len(common) != len(verdicts):
            print(
                f"   [warn] {run}: 与基线只有 {len(common)} 题重合 "
                f"(基线 {len(base_verdicts)}，该 run {len(verdicts)})，下面只统计重合部分"
            )
        kept = [k for k in common if base_verdicts[k]]
        lost = [k for k in common if not base_verdicts[k]]
        overall = sum(verdicts[k] for k in common)
        on_kept = sum(verdicts[k] for k in kept)
        on_lost = sum(verdicts[k] for k in lost)
        print(
            f"{run:<46} {len(common):>5} {overall / max(len(common), 1):>7.2%} "
            f"{len(kept):>6} {on_kept / max(len(kept), 1):>11.2%} "
            f"{len(lost):>6} {on_lost / max(len(lost), 1):>12.2%}"
        )
        low, high = wilson(on_kept, len(kept))
        records.append(
            {
                "metric": metric,
                "run": run,
                "baseline": baseline,
                "n_common": len(common),
                "overall_accuracy": overall / max(len(common), 1),
                "n_base_correct": len(kept),
                "accuracy_on_base_correct": on_kept / max(len(kept), 1),
                "accuracy_on_base_correct_ci95": [low, high],
                "n_base_wrong": len(lost),
                "accuracy_on_base_wrong": on_lost / max(len(lost), 1),
            }
        )

    print(
        "\n   acc|base_ok  = 保留率：base 会做的题里这个 run 还会做的比例，越接近 100% 遗忘越少。\n"
        "   acc|base_bad = 新增率：base 不会的题里这个 run 做对的比例，来自新任务或判分口径变化。\n"
        "   overall ≈ 保留率×base准确率 + 新增率×(1-base准确率)，所以整体持平也可能是两边互相抵消。"
    )

    # -- what moved, on the retention subset --------------------------------
    print(f"\n-- base 答对子集上的配对变化（vs {baseline}）")
    print(f"{'run':<46} {'d_retention':>12} {'lost':>6} {'p(McNemar)':>12} {'95% CI':>22}")
    for record in records:
        run = record["run"]
        if run == baseline:
            continue
        verdicts = loaded[run]
        kept = [k for k in sorted(set(base_verdicts) & set(verdicts)) if base_verdicts[k]]
        dropped = sum(1 for k in kept if not verdicts[k])
        # b is 0 by construction: every item in this subset was already correct
        # for the baseline, so the run can only lose them, never gain them.
        p = mcnemar_exact(0, dropped)
        low, high = record["accuracy_on_base_correct_ci95"]
        flag = "" if p < 0.05 else "  (不显著)"
        print(
            f"{run:<46} {record['accuracy_on_base_correct'] - 1.0:>+11.2%} "
            f"{dropped:>6} {p:>12.4g}   [{low:.2%},{high:.2%}]{flag}"
        )
    print(
        "\n   这里的 p 是单边退化检验：子集内基线全对，所以任何不一致都是掉分，\n"
        "   b=0 使 McNemar 退化成 0.5^n 的符号检验。"
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument(
        "--metrics",
        default="gsm8k,math500,humaneval,mbpp,ifeval",
        help="Comma-separated. ifeval_loose scores the ifeval run with the loose checker.",
    )
    parser.add_argument(
        "--runs",
        default="all",
        help="Comma-separated run names, or 'all' (default) to score every run "
        "directory that has a verdict file for the metric.",
    )
    parser.add_argument(
        "--baseline",
        default="base",
        help="Run whose correct items define the subset. Keep this at the "
        "un-finetuned model unless you mean to ask a different question.",
    )
    parser.add_argument(
        "--json_out",
        default="",
        help="Write one JSON record per (metric, run) here, for pasting into a results table.",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    explicit = [r.strip() for r in args.runs.split(",") if r.strip()]
    scan = args.runs.strip() == "all"
    records: list[dict[str, Any]] = []
    for metric in (m.strip() for m in args.metrics.split(",") if m.strip()):
        runs = discover_runs(root, metric, args.baseline) if scan else explicit
        if not runs:
            print(f"\n[skip] metric={metric}: {root / metric_dirname(metric)} 下没有判分文件")
            continue
        records.extend(report(metric, runs, root, args.baseline))

    if args.json_out and records:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n写出 {len(records)} 条记录到 {out_path}")


if __name__ == "__main__":
    main()
