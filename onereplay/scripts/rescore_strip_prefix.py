"""Re-score finished runs after deleting the "the correct answer is" preamble.

Commonsense170k trains the model to answer with a fixed "the correct answer is
X" template, and after vanilla SFT that template leaks into every benchmark.
On HumanEval it turns a correct function body into a SyntaxError, on IFEval it
adds text the constraint checkers forbid -- so the recorded score conflates two
very different failures: a wrong answer, and a right answer wearing the wrong
costume.

This script separates them. It replays each metric's own judge over the stored
responses with the preamble removed, so the delta is exactly the part of the
gap that formatting alone explains. Whatever gap survives is a real capability
loss.

Read ``leak_rate`` alongside the delta: a metric where the preamble never
appears (MATH500, MBPP) is untouched by construction, and that contrast is the
evidence for *why* only some benchmarks collapsed.

Nothing is overwritten. Results land next to the originals as
``responses_stripped.jsonl`` / ``summary_stripped.json`` when ``--write_back 1``.

Usage
-----
  # math + instruction following need no dataset files (gold is in the jsonl)
  python -m onereplay.scripts.rescore_strip_prefix \
      --out_dir /scratch/.../results --metrics gsm8k,math500,ifeval

  # code needs the eval sets to re-run the tests
  python -m onereplay.scripts.rescore_strip_prefix \
      --out_dir /scratch/.../results --metrics humaneval,mbpp \
      --humaneval_data_file datasets/humaneval/openai_humaneval.parquet \
      --mbpp_dataset_path datasets/mbpp --dataset_split test \
      --out /scratch/.../results/strip_prefix_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ALL_METRICS = ("gsm8k", "math500", "amc", "aime", "humaneval", "mbpp", "ifeval")
MATH_METRICS = ("gsm8k", "math500", "amc", "aime")
CODE_METRICS = ("humaneval", "mbpp")

# The Commonsense170k answer template, plus the two abbreviations the model
# drifts into. Kept as plain phrases so --phrases can extend them without
# anyone having to write a regex.
DEFAULT_PHRASES = (
    "the correct answer is",
    "correct answer is",
    "the answer is",
)


def build_patterns(phrases: tuple[str, ...]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Compile the anchored and floating forms of the preamble matcher.

    Only horizontal whitespace is allowed *inside* a phrase, and at most one
    newline is eaten *after* it. That matters for HumanEval: the body's leading
    indentation is load-bearing, so "the correct answer is\\n    balance = 0"
    must become "    balance = 0", not "balance = 0".
    """

    alternatives = "|".join(
        r"[ \t]+".join(re.escape(word) for word in phrase.split())
        for phrase in sorted(phrases, key=len, reverse=True)
    )
    tail = r"\b[ \t]*[:：]?[ \t]*(?:\r?\n)?"
    anchored = re.compile(rf"^[\s]*(?:{alternatives}){tail}", re.IGNORECASE)
    floating = re.compile(rf"(?:{alternatives}){tail}", re.IGNORECASE)
    return anchored, floating


def strip_preamble(text: str, anchored: re.Pattern[str], floating: re.Pattern[str], everywhere: bool) -> str:
    """Delete the preamble: every occurrence, or repeatedly at the head only.

    The head case loops because the collapsed template sometimes stutters
    ("the correct answer is the correct answer is 240").
    """

    if everywhere:
        return floating.sub("", text)
    previous = None
    while previous != text:
        previous = text
        text = anchored.sub("", text, count=1)
    return text


# --------------------------------------------------------------------------
# per-metric judges: each returns (verdict, extra fields) for one row
# --------------------------------------------------------------------------


def make_math_judge(metric: str) -> Callable[[dict[str, Any], str], tuple[bool, dict[str, Any]]]:
    if metric == "gsm8k":
        from onereplay.eval.metrics.gsm8k import predicted_answer

        def judge(row: dict[str, Any], text: str) -> tuple[bool, dict[str, Any]]:
            gold = row.get("gold")
            pred = predicted_answer(text)
            return bool(gold is not None and pred == gold), {"prediction": pred}

        return judge

    from onereplay.eval.metrics.math500 import extract_answer, is_equiv

    def judge(row: dict[str, Any], text: str) -> tuple[bool, dict[str, Any]]:
        pred = extract_answer(text)
        return bool(is_equiv(pred, row.get("gold"))), {"prediction": pred}

    return judge


def make_code_judge(
    metric: str, reference: dict[Any, dict[str, Any]], timeout: float
) -> Callable[[dict[str, Any], str], tuple[bool, dict[str, Any]]]:
    from onereplay.eval.code_exec import (
        assemble_entry_point_program,
        cleanup_body_completion,
        cleanup_program_completion,
        evaluate_assert_program,
        evaluate_entry_point_program,
    )

    def judge(row: dict[str, Any], text: str) -> tuple[bool, dict[str, Any]]:
        example = reference.get(row.get("task_id"))
        if example is None:
            return bool(row.get("passed")), {"error": "no reference", "completion": ""}
        if metric == "humaneval":
            completion = cleanup_body_completion(text)
            program = assemble_entry_point_program(example["prompt"], completion)
            ok, error = evaluate_entry_point_program(
                program, example["entry_point"], example["test"], timeout
            )
        else:
            completion = cleanup_program_completion(text)
            tests = example.get("test_list") or example.get("tests") or []
            if isinstance(tests, str):
                tests = [tests]
            ok, error = evaluate_assert_program(completion, [str(t) for t in tests], timeout)
        return ok, {"error": error, "completion": completion}

    return judge


def load_code_reference(metric: str, args: argparse.Namespace) -> dict[Any, dict[str, Any]]:
    if metric == "humaneval":
        from onereplay.eval.metrics.humaneval import load_humaneval

        if not args.humaneval_data_file:
            raise SystemExit("humaneval 需要 --humaneval_data_file")
        rows = load_humaneval(args.humaneval_data_file, args.cache_dir, args.limit)
    else:
        from onereplay.eval.metrics.mbpp import load_mbpp

        rows = load_mbpp(
            {
                "mbpp_dataset_path": args.mbpp_dataset_path,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
                "cache_dir": args.cache_dir,
                "limit": args.limit,
            }
        )
    return {row.get("task_id"): row for row in rows}


# --------------------------------------------------------------------------
# shape statistics: what the response looks like, independent of the score
# --------------------------------------------------------------------------


def shape_stats(metric: str, texts: list[str]) -> dict[str, Any]:
    """Summarize response shape so a parsing failure can be told from a real one.

    A run whose responses collapsed to a bare templated answer has short text
    and no reasoning markers; one that still reasons and merely wears the wrong
    prefix does not.
    """

    if not texts:
        return {}
    lengths = sorted(len(t) for t in texts)
    stats: dict[str, Any] = {
        "median_chars": lengths[len(lengths) // 2],
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "frac_under_120_chars": round(sum(n < 120 for n in lengths) / len(lengths), 4),
    }
    if metric == "gsm8k":
        stats["frac_with_hash_marker"] = round(sum("####" in t for t in texts) / len(texts), 4)
    if metric in ("math500", "amc", "aime"):
        stats["frac_with_boxed"] = round(sum("\\boxed" in t for t in texts) / len(texts), 4)
    if metric in CODE_METRICS:
        stats["frac_with_def"] = round(sum("def " in t for t in texts) / len(texts), 4)
    return stats


# --------------------------------------------------------------------------
# per-run drivers
# --------------------------------------------------------------------------


def rescore_scored_metric(
    metric: str,
    rows: list[dict[str, Any]],
    judge: Callable[[dict[str, Any], str], tuple[bool, dict[str, Any]]],
    text_field: str,
    fallback_field: str,
    verdict_field: str,
    anchored: re.Pattern[str],
    floating: re.Pattern[str],
    everywhere: bool,
) -> dict[str, Any]:
    """Judge every row twice -- as stored and with the preamble gone."""

    total = len(rows)
    old_correct = 0
    new_correct = 0
    baseline_correct = 0
    leaked = 0
    recovered: list[Any] = []
    lost: list[Any] = []
    stripped_rows: list[dict[str, Any]] = []
    raw_texts: list[str] = []
    used_fallback = 0

    for row in rows:
        text = row.get(text_field)
        if text is None:
            text = row.get(fallback_field) or ""
            used_fallback += 1
        raw_texts.append(text)
        old_correct += int(bool(row.get(verdict_field)))

        # Re-judge the untouched text too: the stored verdict may predate a
        # judge fix, and comparing against it would attribute that fix to the
        # preamble removal.
        base_ok, _ = judge(row, text)
        baseline_correct += int(base_ok)

        cleaned = strip_preamble(text, anchored, floating, everywhere)
        has_leak = cleaned != text
        leaked += int(has_leak)

        new_ok, extra = judge(row, cleaned) if has_leak else (base_ok, {})
        new_correct += int(new_ok)
        if new_ok and not base_ok:
            recovered.append(row.get("task_id", row.get("question", "")))
        elif base_ok and not new_ok:
            lost.append(row.get("task_id", row.get("question", "")))

        stripped_rows.append(
            {
                **{k: v for k, v in row.items() if k not in (text_field,)},
                **extra,
                verdict_field: new_ok,
                "verdict_as_stored": bool(row.get(verdict_field)),
                "verdict_rejudged_unstripped": base_ok,
                "preamble_stripped": has_leak,
                text_field: cleaned,
            }
        )

    denominator = max(total, 1)
    return {
        "metric": metric,
        "n": total,
        "leak_rate": round(leaked / denominator, 4),
        "n_leaked": leaked,
        "score_as_stored": round(old_correct / denominator, 4),
        "score_rejudged": round(baseline_correct / denominator, 4),
        "score_stripped": round(new_correct / denominator, 4),
        "delta_from_stripping": round((new_correct - baseline_correct) / denominator, 4),
        "recovered": len(recovered),
        "lost": len(lost),
        "recovered_examples": recovered[:10],
        "lost_examples": lost[:10],
        "rows_using_cleaned_text": used_fallback,
        "fidelity": "exact" if used_fallback == 0 else "lower bound (raw text not stored)",
        "shape": shape_stats(metric, raw_texts),
        "_rows": stripped_rows,
    }


def rescore_ifeval(
    rows: list[dict[str, Any]],
    input_data: str,
    anchored: re.Pattern[str],
    floating: re.Pattern[str],
    everywhere: bool,
) -> dict[str, Any]:
    third_party = Path(__file__).resolve().parents[1] / "third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from instruction_following_eval import evaluation_lib

    inputs = evaluation_lib.read_prompt_list(input_data)
    stored = {row["prompt"]: row.get("response", "") for row in rows}
    inputs = [inp for inp in inputs if inp.prompt in stored]

    cleaned = {
        prompt: strip_preamble(text, anchored, floating, everywhere)
        for prompt, text in stored.items()
    }
    leaked = sum(cleaned[p] != stored[p] for p in stored)

    def accuracies(mapping: dict[str, str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for label, tester in (
            ("strict", evaluation_lib.test_instruction_following_strict),
            ("loose", evaluation_lib.test_instruction_following_loose),
        ):
            outputs = [tester(inp, mapping) for inp in inputs]
            prompt_total = max(len(outputs), 1)
            instruction_total = max(sum(len(o.follow_instruction_list) for o in outputs), 1)
            out[f"{label}_prompt_accuracy"] = round(
                sum(o.follow_all_instructions for o in outputs) / prompt_total, 4
            )
            out[f"{label}_instruction_accuracy"] = round(
                sum(sum(o.follow_instruction_list) for o in outputs) / instruction_total, 4
            )
        return out

    before = accuracies(stored)
    after = accuracies(cleaned)
    return {
        "metric": "ifeval",
        "n": len(inputs),
        "leak_rate": round(leaked / max(len(stored), 1), 4),
        "n_leaked": leaked,
        "score_as_stored": before["strict_prompt_accuracy"],
        "score_rejudged": before["strict_prompt_accuracy"],
        "score_stripped": after["strict_prompt_accuracy"],
        "delta_from_stripping": round(
            after["strict_prompt_accuracy"] - before["strict_prompt_accuracy"], 4
        ),
        "detail_before": before,
        "detail_after": after,
        "fidelity": "exact",
        "shape": shape_stats("ifeval", list(stored.values())),
        "_rows": [
            {"prompt": p, "response": cleaned[p], "preamble_stripped": cleaned[p] != stored[p]}
            for p in stored
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, help="评测结果根目录（含 gsm8k/ ifeval/ ...）")
    parser.add_argument("--metrics", default=",".join(ALL_METRICS))
    parser.add_argument("--runs", default="", help="逗号分隔 run 名；留空=该 metric 下全部")
    parser.add_argument(
        "--phrases",
        default="",
        help="分号分隔的待删前缀短语，覆盖默认值",
    )
    parser.add_argument(
        "--scope",
        choices=["prefix", "anywhere"],
        default="prefix",
        help="prefix=只删开头（默认，保守）；anywhere=删全部出现位置",
    )
    parser.add_argument("--ifeval_input", default="")
    parser.add_argument("--humaneval_data_file", default="")
    parser.add_argument("--mbpp_dataset_path", default="")
    parser.add_argument("--dataset_name", default="google-research-datasets/mbpp")
    parser.add_argument("--dataset_config", default="full")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--write_back", type=int, default=0)
    parser.add_argument("--out", default="", help="对比表输出 json 路径")
    return parser.parse_args()


def discover_runs(metric_dir: Path, requested: list[str]) -> list[str]:
    if requested:
        return requested
    return sorted(p.name for p in metric_dir.iterdir() if (p / "responses.jsonl").is_file())


def main() -> None:
    args = parse_args()
    results_root = Path(args.out_dir)
    if not results_root.is_dir():
        raise SystemExit(f"不是目录: {results_root}")

    phrases = tuple(
        p.strip() for p in args.phrases.split(";") if p.strip()
    ) or DEFAULT_PHRASES
    anchored, floating = build_patterns(phrases)
    everywhere = args.scope == "anywhere"
    requested_runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    report: dict[str, Any] = {
        "phrases": list(phrases),
        "scope": args.scope,
        "runs": {},
    }
    failures: list[str] = []

    for metric in metrics:
        metric_dir = results_root / metric
        if not metric_dir.is_dir():
            print(f"[skip] 没有 {metric_dir}")
            continue

        reference: dict[Any, dict[str, Any]] = {}
        if metric in CODE_METRICS:
            try:
                reference = load_code_reference(metric, args)
            except SystemExit as exc:
                print(f"[skip] {metric}: {exc}")
                continue
            print(f"{metric} 参考集: {len(reference)} 题")

        for run in discover_runs(metric_dir, requested_runs):
            response_path = metric_dir / run / "responses.jsonl"
            if not response_path.is_file():
                continue
            rows = [
                json.loads(line)
                for line in response_path.open(encoding="utf-8")
                if line.strip()
            ]
            print(f"处理 {metric}/{run} ({len(rows)} 行) ...", flush=True)

            # One unreadable run (or a missing judge dependency) must not throw
            # away the runs that already succeeded -- these jobs take minutes.
            try:
                if metric == "ifeval":
                    result = rescore_ifeval(
                        rows,
                        args.ifeval_input
                        or str(
                            Path(__file__).resolve().parents[1]
                            / "third_party"
                            / "instruction_following_eval"
                            / "data"
                            / "input_data.jsonl"
                        ),
                        anchored,
                        floating,
                        everywhere,
                    )
                elif metric in CODE_METRICS:
                    result = rescore_scored_metric(
                        metric,
                        rows,
                        make_code_judge(metric, reference, args.timeout),
                        text_field="raw",
                        fallback_field="completion",
                        verdict_field="passed",
                        anchored=anchored,
                        floating=floating,
                        everywhere=everywhere,
                    )
                else:
                    result = rescore_scored_metric(
                        metric,
                        rows,
                        make_math_judge(metric),
                        text_field="response",
                        fallback_field="response",
                        verdict_field="correct",
                        anchored=anchored,
                        floating=floating,
                        everywhere=everywhere,
                    )
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                print(f"[fail] {metric}/{run}: {exc!r}")
                failures.append(f"{metric}/{run}: {exc!r}")
                continue

            stripped_rows = result.pop("_rows")
            if args.write_back:
                out_path = metric_dir / run / "responses_stripped.jsonl"
                with out_path.open("w", encoding="utf-8") as file:
                    for item in stripped_rows:
                        file.write(json.dumps(item, ensure_ascii=False) + "\n")
                (metric_dir / run / "summary_stripped.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            report["runs"].setdefault(metric, {})[run] = result

    header = (
        f"{'metric/run':<58}{'n':>5}{'leak':>7}{'stored':>8}{'rejudge':>9}"
        f"{'strip':>8}{'delta':>8}{'+':>4}{'-':>4}"
    )
    print()
    print(header)
    print("-" * len(header))
    for metric, runs in report["runs"].items():
        for run, r in runs.items():
            print(
                f"{metric + '/' + run:<58}{r['n']:>5}"
                f"{r['leak_rate'] * 100:>6.1f}%"
                f"{r['score_as_stored'] * 100:>7.1f}%"
                f"{r['score_rejudged'] * 100:>8.1f}%"
                f"{r['score_stripped'] * 100:>7.1f}%"
                f"{r['delta_from_stripping'] * 100:>+7.1f}%"
                f"{r.get('recovered', 0):>4}{r.get('lost', 0):>4}"
            )
    print()
    print("leak    = 该 run 里带 'the correct answer is' 前缀的样本比例")
    print("stored  = responses.jsonl 里记录的分数；rejudge = 用当前判分器重跑原文")
    print("strip   = 删掉前缀后的分数；delta = strip - rejudge，即格式问题独占的部分")
    print("delta 约等于 0 且 leak 很高 => 前缀不是元凶，是模型真的不会做了")

    if failures:
        print("\n以下 run 处理失败：")
        for line in failures:
            print(f"  {line}")

    if args.out:
        report["failures"] = failures
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {out_path}")


if __name__ == "__main__":
    main()
