"""Validate the APPS judge against gold solutions. No GPU, no model.

    python -m onereplay.scripts.check_apps_judge \\
      --apps_data_file datasets/code/apps_test.parquet \\
      --apps_difficulties introductory --limit 100

Why this has to run before any model is evaluated
-------------------------------------------------
If the judge is broken, every run scores near zero and the log gives no way to
tell that from "the model cannot code". 39_code_eval_base_vanilla.pbs already
carries a self-check of the MBPP/HumanEval sandbox for the same reason; this is
the APPS equivalent, except that it also produces a number worth reporting:
strict pass on gold solutions is the CEILING of this harness. Model scores have
to be read against it, and it belongs in the paper next to them.

APPS gold solutions are human submissions scraped from contest sites, so a few
percent genuinely fail (wrong language assumptions, missing imports, reliance on
the site's exact I/O quirks). A ceiling in the low-to-mid 90s is expected. Much
lower than that and the judge, not the data, is the problem -- look at the error
buckets to see which way it is failing.

--num_solutions > 1 separates the two: if solution 0 fails but one of the first
three passes, that is a bad submission, not a broken runner.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.eval.apps_exec import evaluate_stdin_program  # noqa: E402
from onereplay.eval.metrics.apps import load_apps  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps_data_file", type=str, required=True)
    parser.add_argument(
        "--apps_split",
        type=str,
        default="test",
        help="Only used when --apps_data_file is a save_to_disk directory.",
    )
    parser.add_argument("--apps_difficulties", type=str, default="")
    parser.add_argument("--apps_stdin_only", type=int, default=1)
    parser.add_argument("--apps_max_tests", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument(
        "--num_solutions",
        type=int,
        default=1,
        help="Try this many gold solutions per problem; a problem counts as "
        "passed if any of them passes.",
    )
    parser.add_argument("--out_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = vars(args)
    rows, dropped = load_apps(cfg)
    print(
        f"scoring {len(rows)} problems out of {dropped['scanned']} scanned; dropped "
        f"{dropped['call_based']} call-based, {dropped['no_tests']} without tests, "
        f"{dropped['bad_json']} unparseable"
    )
    if not rows:
        raise SystemExit("no problems selected -- check --apps_difficulties and --limit")

    tmp_root = os.environ.get("TMPDIR", "")
    passed = 0
    ratios: list[float] = []
    durations: list[float] = []
    errors: Counter[str] = Counter()
    no_solution = 0

    for index, example in enumerate(rows, start=1):
        try:
            solutions = json.loads(example["solutions"])
        except (json.JSONDecodeError, TypeError):
            solutions = []
        if not solutions:
            no_solution += 1
            ratios.append(0.0)
            errors["no gold solution"] += 1
            continue

        started = time.time()
        best: dict | None = None
        for solution in solutions[: max(args.num_solutions, 1)]:
            result = evaluate_stdin_program(
                str(solution),
                example["test_inputs"],
                example["test_outputs"],
                args.timeout,
                tmp_root=tmp_root,
                max_tests=args.apps_max_tests,
            )
            # Strictly-greater would keep the placeholder whenever every attempt
            # scores 0.0, hiding the real failure reason behind the initial value.
            if best is None or result["pass_ratio"] > best["pass_ratio"]:
                best = result
            if result["passed"]:
                break
        durations.append(time.time() - started)
        if best is None:
            best = {"passed": False, "pass_ratio": 0.0, "error": "no gold solution"}

        passed += int(best["passed"])
        ratios.append(best["pass_ratio"])
        if not best["passed"]:
            label = best["error"] or "unknown"
            errors[label.split(":", 1)[0].strip() if ":" in label else label] += 1

        if index % 10 == 0:
            print(
                f"  {index}/{len(rows)}  strict={passed}/{index} "
                f"({passed / index:.1%})  mean_ratio={statistics.fmean(ratios):.3f}"
            )

    count = max(len(rows), 1)
    summary = {
        "num_problems": len(rows),
        "gold_strict_pass": passed,
        "gold_pass_at_1": passed / count,
        "gold_mean_pass_ratio": statistics.fmean(ratios) if ratios else 0.0,
        "problems_without_gold_solution": no_solution,
        "num_solutions_tried": args.num_solutions,
        "max_tests": args.apps_max_tests,
        "timeout": args.timeout,
        "seconds_per_problem_mean": statistics.fmean(durations) if durations else 0.0,
        "seconds_per_problem_p90": (
            sorted(durations)[int(0.9 * (len(durations) - 1))] if durations else 0.0
        ),
        "error_buckets": dict(errors),
    }
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    ceiling = summary["gold_pass_at_1"]
    print(
        f"\n判分器天花板 = {ceiling:.1%}。所有模型分数都要对着这个数读，论文里必须一起报。"
    )
    if ceiling < 0.80:
        print(
            "低于 80%：先别评模型。看 error_buckets ——\n"
            "  timeout 占多数        -> 调大 --timeout（竞赛题参考解本来就慢）\n"
            "  wrong answer 占多数   -> 判等太严或期望输出的格式没对上\n"
            "  EOFError 占多数       -> stdin 喂法有问题（多组输入的题读法不同）\n"
            "  SyntaxError 占多数    -> gold 里有 Python 2 代码，属数据噪声，可接受"
        )
    per_problem = summary["seconds_per_problem_mean"]
    print(
        f"判分耗时 {per_problem:.2f} s/题 -> 500 题约 {per_problem * 500 / 60:.0f} 分钟/模型"
    )

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
