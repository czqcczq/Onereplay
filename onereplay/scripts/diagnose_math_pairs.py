"""Paired diagnostics for math retention runs.

The summary.json accuracy alone cannot tell a real retention effect from an
answer-extraction artifact: MATH-500 scores only the last \\boxed{...} with no
fallback, so a run that stops emitting boxed answers loses points without
losing math ability, while GSM8K falls back to the last number anywhere in the
response, so a truncated chain silently scores a wrong-but-parseable number.

This script joins the per-item responses.jsonl of several runs on the question
text and reports, per metric:

  * extraction health: no-answer rate, missing format marker, hit-cap rate
  * re-scoring under a lenient and a strict extractor, to bound how much of the
    accuracy gap is format compliance rather than math
  * exact McNemar between each run and the baseline, plus how many of the
    discordant items are explained by extraction failure

Scoring functions are imported from the evaluator itself so the re-scores stay
byte-identical to what produced summary.json.

Usage:
    python -m onereplay.scripts.diagnose_math_pairs \\
        --results_root /scratch/.../Onereplay/results \\
        --metrics math500,gsm8k \\
        --runs base,cs_vanilla_seed1,cs_onereplay_ifmath_lam3e-2_seed1_regonce \\
        --baseline base \\
        --cap 4096 \\
        --model_path /path/to/Qwen3-1.7B      # optional, enables token lengths
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

# The metric modules pull in onereplay.eval.generation, which imports torch at
# module level. Diagnostics are meant to run on a login node, so stub it out
# when torch is absent; the scoring helpers below never touch it.
try:  # noqa: SIM105
    import torch  # noqa: F401
except ImportError:
    _stub = types.ModuleType("onereplay.eval.generation")
    _stub.generate_response = None  # type: ignore[attr-defined]
    sys.modules["onereplay.eval.generation"] = _stub

from onereplay.eval.metrics.gsm8k import normalize_number  # noqa: E402
from onereplay.eval.metrics.math500 import extract_answer, is_equiv  # noqa: E402


# --------------------------------------------------------------------------
# alternative extractors: one more permissive, one stricter than production
# --------------------------------------------------------------------------
def math500_lenient(response: str) -> str | None:
    """Production extractor, falling back to the last number when no boxed."""

    boxed = extract_answer(response)
    return boxed if boxed is not None else normalize_number(response)


def gsm8k_production(response: str) -> str | None:
    """Verbatim copy of the evaluator: '####' segments, else last number."""

    for chunk in reversed(response.split("####")):
        value = normalize_number(chunk)
        if value is not None:
            return value
    return None


def gsm8k_strict(response: str) -> str | None:
    """Only accept a number that follows an explicit '####' marker."""

    parts = response.split("####")
    for chunk in reversed(parts[1:]):
        value = normalize_number(chunk)
        if value is not None:
            return value
    return None


METRIC_SPECS: dict[str, dict[str, Any]] = {
    "math500": {
        "marker": "\\boxed",
        "production": extract_answer,
        "lenient": math500_lenient,
        "strict": extract_answer,
        "equal": is_equiv,
    },
    "amc": {
        "marker": "\\boxed",
        "production": extract_answer,
        "lenient": math500_lenient,
        "strict": extract_answer,
        "equal": is_equiv,
    },
    "gsm8k": {
        "marker": "####",
        "production": gsm8k_production,
        "lenient": gsm8k_production,
        "strict": gsm8k_strict,
        "equal": lambda a, b: a is not None and b is not None and a == b,
    },
}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the b/c discordant counts."""

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; better than normal approx near the tails."""

    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def rescore(rows: list[dict[str, Any]], extractor: Callable, equal: Callable) -> list[bool]:
    return [bool(equal(extractor(row.get("response", "")), row.get("gold"))) for row in rows]


def token_lengths(rows: list[dict[str, Any]], tokenizer) -> list[int]:
    return [
        len(tokenizer(row.get("response", ""), add_special_tokens=False)["input_ids"])
        for row in rows
    ]


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(round(q / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def report_metric(metric: str, runs: list[str], root: Path, baseline: str, cap: int, tokenizer) -> None:
    spec = METRIC_SPECS[metric]
    loaded: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        path = root / metric / run / "responses.jsonl"
        if not path.is_file():
            print(f"[skip] {path} 不存在")
            continue
        loaded[run] = load_rows(path)
    if len(loaded) < 1:
        return

    print(f"\n{'=' * 100}")
    print(f"metric = {metric}   (marker = {spec['marker']!r}, cap = {cap})")
    print("=" * 100)

    # 1. question alignment ------------------------------------------------
    keys = {run: [row.get("question", "") for row in rows] for run, rows in loaded.items()}
    ref_run = next(iter(keys))
    aligned = all(keys[run] == keys[ref_run] for run in keys)
    print(f"\n-- 题目对齐: {'逐题同序，可直接配对' if aligned else '不对齐，按 question 文本 join'}")
    for run, rows in loaded.items():
        print(f"   {run:<50} n={len(rows)}")
    if not aligned:
        common = set.intersection(*(set(k) for k in keys.values()))
        print(f"   交集 n={len(common)}")
        for run in loaded:
            index = {row.get("question", ""): row for row in loaded[run]}
            loaded[run] = [index[q] for q in sorted(common)]

    # 2. evaluation batch: same decoding budget? same code version? ---------
    # A run truncated at the cap has its longest responses pinned at the cap,
    # so max length identifies which MATH_MAX_NEW_TOKENS produced the file --
    # summary.json does not record it. mtime separates evaluation batches, which
    # matters because the gsm8k '####' fallback was only added on 2026-08-24.
    print("\n-- 评测批次一致性（charmax 差一倍 => 预算不同 => 不可比）")
    print(f"{'run':<50} {'charmax':>9} {'charP99':>9} {'charP50':>9} {'mtime':>17}")
    for run, rows in loaded.items():
        chars = sorted(len(row.get("response", "")) for row in rows)
        mtime = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime((root / metric / run / "responses.jsonl").stat().st_mtime)
        )
        print(
            f"{run:<50} {chars[-1]:>9} {percentile(chars, 99):>9} "
            f"{percentile(chars, 50):>9} {mtime:>17}"
        )

    # 3. extraction health -------------------------------------------------
    print("\n-- 抽取健康度（no-answer 越高说明分数越受格式影响）")
    print(f"{'run':<50} {'n':>5} {'no-answer':>13} {'no-marker':>13} {'hit-cap':>13} {'tokP50/P90/P99/max':>24}")
    for run, rows in loaded.items():
        n = len(rows)
        no_answer = sum(spec["production"](row.get("response", "")) in (None, "") for row in rows)
        no_marker = sum(spec["marker"] not in row.get("response", "") for row in rows)
        if tokenizer is not None:
            lengths = sorted(token_lengths(rows, tokenizer))
            hit_cap = sum(length >= cap - 8 for length in lengths)
            cap_str = f"{hit_cap} ({hit_cap / n:.1%})"
            len_str = f"{percentile(lengths, 50)}/{percentile(lengths, 90)}/{percentile(lengths, 99)}/{lengths[-1]}"
        else:
            cap_str = "n/a"
            len_str = "n/a (给 --model_path)"
        print(
            f"{run:<50} {n:>5} {no_answer:>5} ({no_answer / n:>5.1%}) "
            f"{no_marker:>5} ({no_marker / n:>5.1%}) {cap_str:>13} {len_str:>24}"
        )

    # 4. re-scoring under three extractors ---------------------------------
    print("\n-- 三种抽取口径下的准确率（若排序随口径改变，差距就是格式而非数学）")
    print(f"{'run':<50} {'production':>22} {'lenient':>12} {'strict':>12}")
    scores: dict[str, list[bool]] = {}
    for run, rows in loaded.items():
        n = len(rows)
        prod = rescore(rows, spec["production"], spec["equal"])
        lenient = rescore(rows, spec["lenient"], spec["equal"])
        strict = rescore(rows, spec["strict"], spec["equal"])
        scores[run] = prod
        low, high = wilson_interval(sum(prod), n)
        print(
            f"{run:<50} {sum(prod) / n:>7.2%} [{low:.2%},{high:.2%}] "
            f"{sum(lenient) / n:>11.2%} {sum(strict) / n:>11.2%}"
        )
        stored = sum(bool(row.get("correct")) for row in rows)
        if stored != sum(prod):
            print(f"{'':<50} !! 重算 {sum(prod)} != 文件里的 {stored}，抽取逻辑或数据已变动")

    # 5. paired McNemar vs baseline ----------------------------------------
    if baseline not in scores:
        print(f"\n-- 跳过配对检验：baseline {baseline!r} 无数据")
        return
    print(f"\n-- 配对 McNemar（vs {baseline}）")
    print(
        f"{'run':<50} {'Δacc':>8} {'b(赢)':>7} {'c(输)':>7} {'discord':>9} "
        f"{'p':>9} {'b中对方无答案':>14}"
    )
    base_scores = scores[baseline]
    base_rows = loaded[baseline]
    for run, run_scores in scores.items():
        if run == baseline:
            continue
        rows = loaded[run]
        b = c = 0
        b_from_extraction = 0
        for i, (base_ok, run_ok) in enumerate(zip(base_scores, run_scores)):
            if run_ok and not base_ok:
                b += 1
                if spec["production"](base_rows[i].get("response", "")) in (None, ""):
                    b_from_extraction += 1
            elif base_ok and not run_ok:
                c += 1
        n = len(run_scores)
        delta = (sum(run_scores) - sum(base_scores)) / n
        p = mcnemar_exact(b, c)
        flag = "" if p < 0.05 else "  (不显著)"
        print(
            f"{run:<50} {delta:>+7.2%} {b:>7} {c:>7} {b + c:>9} "
            f"{p:>9.4f} {b_from_extraction:>10} ({b_from_extraction / max(b, 1):.0%}){flag}"
        )
    print(
        "\n   b = 该 run 对而 baseline 错的题数，c = 反之。p 是双侧精确检验。\n"
        "   最后一列是 b 里 baseline 压根没抽出答案的比例：越高，说明这个\n"
        "   '提升' 越是格式服从度而不是解题能力。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True, help="含 math500/ gsm8k/ 的目录")
    parser.add_argument("--metrics", default="math500,gsm8k")
    parser.add_argument("--runs", required=True, help="逗号分隔的 run_name，第一个通常是 base")
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--cap", type=int, default=4096, help="评测时的 MATH_MAX_NEW_TOKENS")
    parser.add_argument("--model_path", default="", help="给了才算 token 长度与 hit-cap")
    args = parser.parse_args()

    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    root = Path(args.results_root)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    for metric in (m.strip() for m in args.metrics.split(",") if m.strip()):
        if metric not in METRIC_SPECS:
            print(f"[skip] 不支持的 metric: {metric}")
            continue
        report_metric(metric, runs, root, args.baseline, args.cap, tokenizer)


if __name__ == "__main__":
    main()
