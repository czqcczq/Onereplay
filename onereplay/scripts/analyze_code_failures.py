"""Bucket HumanEval/MBPP failures by cause, and pair runs on task_id.

pass@1 alone cannot say whether a model stopped being able to write code or
merely stopped emitting code cleanly. On GSM8K that distinction turned out to
be the whole story: vanilla lost the '#### ' marker on 22.4% of items while its
math was intact, and the scorer's fallback compressed a 30-point format
collapse into a 1.67pp accuracy drop.

The code metrics happen to record exactly what is needed to separate the two,
in the `error` field their harness already writes:

    SyntaxError / IndentationError  the completion is not valid Python at all
                                    -> the model emitted prose, or the cleanup
                                       cut a chatty preamble off mid-function
    AssertionError                  the code ran and returned wrong answers
                                    -> a genuine capability failure
    NameError / TypeError / ...     the code ran but is broken in some way
    timeout                         non-terminating program
    blocked                         the harness' own safety blocklist fired,
                                    which is a scorer artifact, not the model

Two caveats worth carrying into any writeup, both from onereplay/eval/code_exec.py:

  * cleanup_completion truncates at the first '\\ndef ', so a model that writes
    "Here is the solution:\\n\\ndef f(...)" keeps only the prose and scores 0
    with perfectly good code. This penalizes chatty models specifically.
  * has_dangerous_code rejects any completion containing 'import os', 'open(',
    'pathlib' and friends, including legitimate uses. Absolute pass@1 will sit
    below published numbers for the same model. It applies equally to every
    run, so comparisons stay valid; the absolute value does not.

    python -m onereplay.scripts.analyze_code_failures \\
        --results_root /path/results --metrics humaneval,mbpp \\
        --runs base,cs_vanilla_seed1 --baseline base
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

# Ordered: the first pattern that matches wins, so the specific ones come first.
BUCKETS = (
    ("blocked（判分器拦截）", lambda e: "blocked" in e),
    ("timeout（不终止）", lambda e: "timeout" in e),
    ("语法错误（不是合法 Python）", lambda e: bool(re.search(r"SyntaxError|IndentationError", e))),
    ("断言失败（能跑，答案错）", lambda e: "AssertionError" in e),
    ("名字/属性错误（残缺代码）", lambda e: bool(re.search(r"NameError|AttributeError", e))),
    ("类型/取值错误", lambda e: bool(re.search(r"TypeError|ValueError|IndexError|KeyError", e))),
)


def bucket(error: str) -> str:
    if not error:
        return "其他"
    for name, matches in BUCKETS:
        if matches(error):
            return name
    return "其他"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def report(metric: str, runs: list[str], root: Path, baseline: str) -> None:
    loaded: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        path = root / metric / run / "responses.jsonl"
        if path.is_file():
            loaded[run] = load_rows(path)
        else:
            print(f"[skip] {path} 不存在")
    if not loaded:
        return

    print(f"\n{'=' * 96}\nmetric = {metric}\n{'=' * 96}")

    print(f"\n-- pass@1")
    print(f"{'run':<46} {'n':>5} {'passed':>7} {'pass@1':>9} {'95% CI':>20}")
    scores: dict[str, dict[str, bool]] = {}
    for run, rows in loaded.items():
        keyed = {str(row.get("task_id")): bool(row.get("passed")) for row in rows}
        scores[run] = keyed
        passed = sum(keyed.values())
        low, high = wilson(passed, len(rows))
        print(
            f"{run:<46} {len(rows):>5} {passed:>7} {passed / max(len(rows), 1):>8.2%} "
            f"  [{low:.2%},{high:.2%}]"
        )

    print(f"\n-- 失败原因分布（占全部题目的比例）")
    names = [name for name, _ in BUCKETS] + ["其他"]
    print(f"{'run':<46}" + "".join(f"{n.split('（')[0]:>14}" for n in names))
    for run, rows in loaded.items():
        counts = {name: 0 for name in names}
        for row in rows:
            if row.get("passed"):
                continue
            counts[bucket(str(row.get("error", "")))] += 1
        line = f"{run:<46}"
        for name in names:
            line += f"{counts[name] / max(len(rows), 1):>13.1%} "
        print(line)
    print(
        "\n   语法错误占比暴涨而断言失败没怎么变 => 模型是不肯好好输出代码了，不是不会写。\n"
        "   断言失败占比上升 => 代码写得出来但是错的，那才是能力退化。"
    )

    if baseline not in scores:
        return
    print(f"\n-- 配对 McNemar（vs {baseline}）")
    print(f"{'run':<46} {'Δpass@1':>9} {'b(赢)':>7} {'c(输)':>7} {'discord':>9} {'p':>9}")
    base_scores = scores[baseline]
    for run, run_scores in scores.items():
        if run == baseline:
            continue
        common = sorted(set(base_scores) & set(run_scores))
        b = sum(1 for k in common if run_scores[k] and not base_scores[k])
        c = sum(1 for k in common if base_scores[k] and not run_scores[k])
        delta = (
            sum(run_scores[k] for k in common) - sum(base_scores[k] for k in common)
        ) / max(len(common), 1)
        p = mcnemar_exact(b, c)
        flag = "" if p < 0.05 else "  (不显著)"
        print(f"{run:<46} {delta:>+8.2%} {b:>7} {c:>7} {b + c:>9} {p:>9.4f}{flag}")

    discord_any = any(True for _ in loaded)
    if discord_any:
        print(
            "\n   配对可分辨下限约 1.96*sqrt(discord率/n)。HumanEval 只有 164 题，"
            "\n   所以它只能读出十几个点以上的差异；MBPP test 有 500 题，好一些。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--metrics", default="humaneval,mbpp")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--baseline", default="base")
    args = parser.parse_args()

    root = Path(args.results_root)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    for metric in (m.strip() for m in args.metrics.split(",") if m.strip()):
        report(metric, runs, root, args.baseline)


if __name__ == "__main__":
    main()
