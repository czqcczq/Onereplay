"""Paired bootstrap error bars for the probe cross-entropy.

The CE table on its own says C_mix beats pure C_if by 3-5 points of eliminated
forgetting on every corpus, but with no error bars there is no way to tell a
real ordering from three coin flips that happened to land the same way -- and on
MATH-500 the gap is 1% in relative terms, which is exactly where noise lives.

The resampling unit is the row, and the same resampled row indices are applied
to every run. That pairing matters: the runs differ by one training flag and
score identical rows, so most of the variance is "this row is hard", which is
shared and cancels in a difference. An unpaired interval would be several times
wider and would say nothing.

Three quantities, each with a percentile interval:

  CE                     per run
  eliminated forgetting  1 - (method - base) / (vanilla - base), the chapter 3
                         framing, so the numbers are comparable to the FLAN
                         probe's 59.0% / 62.6%
  pairwise difference     CE(A) - CE(B) with a two-sided bootstrap p-value; this
                         is the one that decides whether C_mix > C_if holds up

    python -m onereplay.scripts.bootstrap_probe_ce \\
        --per_row_dir /path/results/probe_ce/per_row \\
        --base base --vanilla cs_vanilla_seed1 \\
        --compare cs_onereplay_balanced_lam3e-2_seed1_regonce,cs_onereplay_ifmath_lam3e-2_seed1_regonce
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_row_dir", required=True, help="Written by score_probe_ce.py")
    parser.add_argument("--base", default="base")
    parser.add_argument(
        "--vanilla",
        default="",
        help="Unregularized run; needed for the eliminated-forgetting column.",
    )
    parser.add_argument(
        "--compare",
        default="",
        help="Comma-separated run pair(s) to difference, e.g. C_if,C_mix. Every "
        "ordered pair among these is reported.",
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05, help="0.05 -> 95% interval")
    parser.add_argument("--out", default="", help="Optional JSON dump of every interval.")
    return parser.parse_args()


def load_per_row(per_row_dir: Path) -> dict[str, dict[str, list[dict[str, float]]]]:
    """corpus -> run -> rows, parsed from '<run>__<corpus>.jsonl' filenames."""

    table: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(dict)
    for path in sorted(per_row_dir.glob("*__*.jsonl")):
        run, corpus = path.stem.split("__", 1)
        with path.open(encoding="utf-8") as file:
            table[corpus][run] = [json.loads(line) for line in file if line.strip()]
    return table


def weighted_ce(rows: list[dict[str, float]], picks: list[int]) -> float:
    loss = 0.0
    tokens = 0.0
    for index in picks:
        row = rows[index]
        loss += row["loss_sum"]
        tokens += row["tokens"]
    return loss / tokens if tokens else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def two_sided_p(values: list[float]) -> float:
    """Fraction of replicates on the wrong side of zero, doubled.

    A percentile p-value rather than a normal approximation: the eliminated
    forgetting ratio has a difference in its denominator and is not remotely
    Gaussian, and the same routine then applies to every quantity here.
    """

    if not values:
        return 1.0
    below = sum(1 for value in values if value <= 0) / len(values)
    above = sum(1 for value in values if value >= 0) / len(values)
    return min(1.0, 2 * min(below, above))


def report_corpus(
    corpus: str,
    runs: dict[str, list[dict[str, float]]],
    args: argparse.Namespace,
    compare: list[str],
) -> dict[str, object]:
    sizes = {run: len(rows) for run, rows in runs.items()}
    if len(set(sizes.values())) != 1:
        print(f"\n[skip] {corpus}: 各 run 行数不一致 {sizes}，无法配对")
        return {}
    n = next(iter(sizes.values()))

    print(f"\n{'=' * 96}")
    print(f"corpus = {corpus}   n={n} 行   {args.replicates} 次配对 bootstrap")
    print("=" * 96)

    rng = random.Random(args.seed)
    point = {run: weighted_ce(rows, list(range(n))) for run, rows in runs.items()}

    draws: dict[str, list[float]] = {run: [] for run in runs}
    for _ in range(args.replicates):
        picks = [rng.randrange(n) for _ in range(n)]
        for run, rows in runs.items():
            draws[run].append(weighted_ce(rows, picks))

    low_q, high_q = args.alpha / 2, 1 - args.alpha / 2
    result: dict[str, object] = {"corpus": corpus, "n": n, "ce": {}}

    print(f"\n-- CE（{100 * (1 - args.alpha):.0f}% 区间）")
    print(f"{'run':<50} {'CE':>10} {'区间':>22}")
    for run in sorted(point, key=lambda r: point[r]):
        low, high = percentile(draws[run], low_q), percentile(draws[run], high_q)
        print(f"{run:<50} {point[run]:>10.5f} {f'[{low:.5f}, {high:.5f}]':>22}")
        result["ce"][run] = {"point": point[run], "low": low, "high": high}

    # -- eliminated forgetting -------------------------------------------
    base, vanilla = args.base, args.vanilla
    if base in runs and vanilla in runs:
        print(f"\n-- 消除的遗忘  1 - (方法 - {base}) / ({vanilla} - {base})")
        print(f"{'run':<50} {'点估计':>10} {'区间':>22}")
        result["eliminated"] = {}
        for run in runs:
            if run in (base, vanilla):
                continue
            def ratio(index: int | None) -> float:
                if index is None:
                    b, v, m = point[base], point[vanilla], point[run]
                else:
                    b, v, m = draws[base][index], draws[vanilla][index], draws[run][index]
                span = v - b
                return 1.0 - (m - b) / span if span else 0.0

            values = [ratio(i) for i in range(args.replicates)]
            low, high = percentile(values, low_q), percentile(values, high_q)
            estimate = ratio(None)
            print(f"{run:<50} {estimate:>9.1%} {f'[{low:.1%}, {high:.1%}]':>22}")
            result["eliminated"][run] = {"point": estimate, "low": low, "high": high}

    # -- pairwise differences --------------------------------------------
    pairs = [(a, b) for i, a in enumerate(compare) for b in compare[i + 1:]]
    pairs = [(a, b) for a, b in pairs if a in runs and b in runs]
    if pairs:
        print("\n-- 两两之差  CE(A) - CE(B)（>0 表示 B 更接近 base，即 B 更好）")
        print(f"{'A - B':<62} {'差':>10} {'区间':>22} {'p':>8}")
        result["differences"] = []
        for a, b in pairs:
            values = [draws[a][i] - draws[b][i] for i in range(args.replicates)]
            estimate = point[a] - point[b]
            low, high = percentile(values, low_q), percentile(values, high_q)
            p = two_sided_p(values)
            label = f"{a.split('_lam')[0]} - {b.split('_lam')[0]}"
            flag = "" if p < args.alpha else "  (不显著)"
            print(
                f"{label:<62} {estimate:>+10.5f} "
                f"{f'[{low:+.5f}, {high:+.5f}]':>22} {p:>8.4f}{flag}"
            )
            result["differences"].append(
                {"a": a, "b": b, "point": estimate, "low": low, "high": high, "p": p}
            )
    return result


def main() -> None:
    args = parse_args()
    table = load_per_row(Path(args.per_row_dir))
    if not table:
        print(f"{args.per_row_dir} 下没有 <run>__<corpus>.jsonl，先用 score_probe_ce 的 --per_row_dir 产出")
        return

    compare = [item.strip() for item in args.compare.split(",") if item.strip()]
    results = []
    for corpus in sorted(table):
        outcome = report_corpus(corpus, table[corpus], args, compare)
        if outcome:
            results.append(outcome)

    print(
        "\n读法：'消除的遗忘' 区间不含 0 说明该方法确实起了作用；两两之差的区间不含 0\n"
        "才能说一份 C 优于另一份。参照第三章 FLAN 池外探针，OneReplay 消除 59.0%。"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n区间已写入 {args.out}")


if __name__ == "__main__":
    main()
