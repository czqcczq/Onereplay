"""Run every metric's real loading path against the real downloaded files.

    python test_code/check_bench_loaders.py

check_bench_metrics.py already proves the scoring logic on hand-built fixtures.
This checks the other half, the half fixtures cannot: that the loaders agree
with the bytes actually on disk.

That half is where the silent failures live. A loader reading "answer" from a
file that spells it "answers" does not crash -- it yields an empty gold, skips
the row, and reports 0.0, which is indistinguishable from a model that got
everything wrong. So the check here is not "did it parse" but "did it keep the
rows it should have kept, with distinct golds", and a bad count is a bug
report rather than a statistic.

The three families are loaded differently because they are built differently:
the medical and finance metrics expose load_examples, while the math ones
inline their parsing inside run(), so the math path below mirrors that code.
"""

from __future__ import annotations

import json
import time
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# math lives under datasets/math/<name>_test.jsonl (where this project has always
# kept it), the rest under datasets/bench/<name>/. Same split as the PBS script.
BENCH = REPO_ROOT / "datasets"


def load_gsm8k(path: Path) -> list[dict]:
    from onereplay.eval.metrics.gsm8k import build_prompt, gold_answer

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            gold = gold_answer(record.get("answer", ""))
            rows.append(
                {
                    "prompt": build_prompt(record["question"]),
                    "gold": "" if gold is None else str(gold),
                }
            )
    return rows


def load_math(path: Path) -> list[dict]:
    """Mirror MATH500Metric.run's parsing, which drives Minerva too."""

    from onereplay.eval.metrics.math500 import (
        ANSWER_KEYS,
        QUESTION_KEYS,
        extract_answer,
        first_existing_key,
        load_json_records,
    )

    rows = []
    for record in load_json_records(str(path)):
        if not isinstance(record, dict):
            continue
        question_key = first_existing_key(record, "", QUESTION_KEYS)
        answer_key = first_existing_key(record, "", ANSWER_KEYS)
        if not question_key or not answer_key:
            continue
        question = str(record[question_key]).strip()
        raw = str(record[answer_key]).strip()
        gold = extract_answer(raw) or raw
        if question and gold:
            rows.append({"prompt": question, "gold": gold})
    return rows


def load_metric(spec: str, cfg_key: str, path: Path) -> list[dict]:
    module_name, _, class_name = spec.partition(":")
    module = __import__(module_name, fromlist=[class_name])
    metric = getattr(module, class_name)()
    examples = metric.load_examples({cfg_key: str(path)})
    for example in examples:
        if "gold" in example:
            continue
        annotation = example.get("annotation")
        if isinstance(annotation, dict):
            # TAT-QA keeps the whole question dict, because its official scoring
            # needs answer_type and scale, not just the answer string.
            scale = str(annotation.get("scale") or "").strip()
            example["gold"] = f"{annotation.get('answer')}" + (f" [{scale}]" if scale else "")
        else:
            # Finance rows carry the gold as text plus its executed value.
            value = example.get("gold_value")
            example["gold"] = example.get("gold_text") or ("" if value is None else str(value))
    return examples


def count_tatqa(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sum(len(doc.get("questions", [])) for doc in payload)


# name, file, loader, expected row count (int or callable), expected distinct
# golds. An int pins a closed label set -- MedQA has exactly 5 options, so 4
# would mean a whole letter went missing. None means open-ended, where the test
# is only that golds are not collapsing onto a handful of shared values.
CASES = [
    ("gsm8k", "math/gsm8k_test.jsonl", lambda p: load_gsm8k(p), 1319, None),
    ("math500", "math/math500_test.jsonl", lambda p: load_math(p), 500, None),
    ("minervamath", "math/minervamath_test.jsonl", lambda p: load_math(p), 272, None),
    (
        "medqa",
        "bench/medqa/test.jsonl",
        lambda p: load_metric("onereplay.eval.metrics.medical:MedQAMetric", "medqa_data_path", p),
        1273,
        5,
    ),
    (
        "pubmedqa",
        "bench/pubmedqa/test.jsonl",
        lambda p: load_metric("onereplay.eval.metrics.medical:PubMedQAMetric", "pubmedqa_data_path", p),
        1000,
        3,
    ),
    (
        "medxpertqa",
        "bench/medxpertqa/test.jsonl",
        lambda p: load_metric("onereplay.eval.metrics.medical:MedXpertQAMetric", "medxpertqa_data_path", p),
        2450,
        10,
    ),
    (
        "finqa",
        "bench/finqa/test.jsonl",
        lambda p: load_metric("onereplay.eval.metrics.finance:FinQAMetric", "finqa_data_path", p),
        1147,
        None,
    ),
    (
        "convfinqa",
        "bench/convfinqa/dev_turn.json",
        lambda p: load_metric("onereplay.eval.metrics.finance:ConvFinQAMetric", "convfinqa_data_path", p),
        1490,
        None,
    ),
    (
        "tatqa",
        "bench/tatqa/tatqa_dataset_dev.json",
        lambda p: load_metric("onereplay.eval.metrics.tatqa:TATQAMetric", "tatqa_data_path", p),
        count_tatqa,
        None,
    ),
]


def check_code() -> list[str]:
    """HumanEval and MBPP, which do not fit CASES and need a different check.

    Two reasons they are separate. Their metrics expose module-level loaders
    instead of ``load_examples``, so ``load_metric`` cannot reach them. And they
    have no gold string to count distinct values of -- correctness is decided by
    running the code, which makes the judging chain itself the thing worth
    testing. So this runs each task's own canonical solution: anything that
    fails is the grader rejecting a known-good answer, and every such rejection
    comes straight off the reported pass@1 for every model in the table.
    """

    problems: list[str] = []

    print("=" * 78)
    print("humaneval   code/humaneval_test.parquet")
    path = BENCH / "code/humaneval_test.parquet"
    if not path.exists():
        print("  SKIP 文件不存在\n")
    else:
        from onereplay.eval.code_exec import (
            assemble_entry_point_program,
            evaluate_entry_point_program,
        )
        from onereplay.eval.metrics.humaneval import load_humaneval

        rows = load_humaneval(str(path), "", 0)
        count = len(rows)
        print(f"  {'OK  ' if count == 164 else 'FAIL'} 载入 {count} / 源 164 条")
        if count != 164:
            problems.append(f"humaneval: 载入 {count}，应为 164")

        missing = [f for f in ("prompt", "entry_point", "test", "canonical_solution")
                   if rows and f not in rows[0]]
        if missing:
            problems.append(f"humaneval: 缺字段 {missing}")
            print(f"  FAIL 缺字段 {missing}")
        else:
            print("  OK   prompt / entry_point / test / canonical_solution 齐全")

        passed, failures_here = 0, []
        for row in rows:
            program = assemble_entry_point_program(row["prompt"], row["canonical_solution"])
            good, error = evaluate_entry_point_program(
                program, row["entry_point"], row["test"], 5.0
            )
            if good:
                passed += 1
            else:
                failures_here.append((row["task_id"], error))
        rate = passed / max(count, 1)
        print(f"  {'OK  ' if rate == 1.0 else 'FAIL'} 标准答案自测 {passed}/{count} ({rate:.1%})")
        for task_id, error in failures_here[:5]:
            print(f"       {task_id}: {error[:90]}")
        if rate < 1.0:
            # Most likely has_dangerous_code: its blacklist matches "import os"
            # and friends, which some canonical solutions legitimately use.
            problems.append(
                f"humaneval: 判分器判错了 {count - passed} 道标准答案，这部分会从每个模型的 pass@1 里扣掉"
            )
        print()

    print("=" * 78)
    print("mbpp   code/mbpp_full")
    path = BENCH / "code/mbpp_full"
    if not path.exists():
        print("  SKIP 目录不存在\n")
        return problems

    from onereplay.eval.code_exec import evaluate_assert_program
    from onereplay.eval.metrics.mbpp import load_mbpp

    rows = load_mbpp({"mbpp_dataset_path": str(path), "dataset_split": "test"})
    count = len(rows)
    print(f"  {'OK  ' if count == 500 else 'FAIL'} test split 载入 {count} / 源 500 条")
    if count != 500:
        # 90 means the validation split came back instead -- evaluate.py's
        # default, and far too few tasks to read a pass@1 difference off.
        hint = "（90 = 拿成 validation 了，要显式传 --dataset_split test）" if count == 90 else ""
        problems.append(f"mbpp: test split 载入 {count}，应为 500{hint}")

    missing = [f for f in ("text", "test_list", "code") if rows and f not in rows[0]]
    if missing:
        problems.append(f"mbpp: 缺字段 {missing}")
        print(f"  FAIL 缺字段 {missing}")
    else:
        print("  OK   text / test_list / code 齐全")

    passed, failures_here = 0, []
    for row in rows:
        tests = list(row.get("test_list") or [])
        setup = list(row.get("test_setup_code") and [row["test_setup_code"]] or [])
        good, error = evaluate_assert_program(row["code"], setup + tests, 5.0)
        if good:
            passed += 1
        else:
            failures_here.append((row.get("task_id"), error))
    rate = passed / max(count, 1)
    print(f"  {'OK  ' if rate >= 0.98 else 'FAIL'} 标准答案自测 {passed}/{count} ({rate:.1%})")
    for task_id, error in failures_here[:5]:
        print(f"       task {task_id}: {error[:90]}")
    if rate < 0.98:
        problems.append(
            f"mbpp: 判分器判错了 {count - passed} 道标准答案（{1 - rate:.1%}），这部分是所有模型的共同扣分"
        )
    problems.extend(check_evalplus())
    return problems


def check_evalplus() -> list[str]:
    """HumanEval+ and MBPP+, where the tests are the whole point.

    Both run each task's reference solution. For the plus sets that check is
    doing more than confirming the files parse: they ship ~80x the assertions,
    and a timeout counts as a failure, so this is also how you find out whether
    30 seconds is enough before a whole eval run reports depressed pass@1 for
    every arm at once.
    """

    problems: list[str] = []

    print("=" * 78)
    print("humanevalplus   code/humanevalplus_test.parquet")
    path = BENCH / "code/humanevalplus_test.parquet"
    if not path.exists():
        print("  SKIP 文件不存在\n")
    else:
        from onereplay.eval.code_exec import (
            assemble_entry_point_program,
            evaluate_entry_point_program,
        )
        from onereplay.eval.metrics.evalplus import HumanEvalPlusMetric
        from onereplay.eval.metrics.humaneval import load_humaneval

        rows = load_humaneval(str(path), "", 0)
        count = len(rows)
        print(f"  {'OK  ' if count == 164 else 'FAIL'} 载入 {count} / 源 164 条")
        if count != 164:
            problems.append(f"humanevalplus: 载入 {count}，应为 164")

        # The reason to use the plus set at all: if `test` is the same size as
        # the base set's, the wrong file got downloaded.
        sizes = [len(str(row.get("test", ""))) for row in rows]
        mean = sum(sizes) / max(count, 1)
        print(f"  test 字段平均 {mean:,.0f} 字符（基础版约 700）")
        if mean < 5000:
            problems.append(f"humanevalplus: test 平均才 {mean:.0f} 字符，可能下成基础版了")

        timeout = HumanEvalPlusMetric.default_timeout
        passed, slow, failures_here = 0, 0, []
        for row in rows:
            program = assemble_entry_point_program(row["prompt"], row["canonical_solution"])
            start = time.time()
            good, error = evaluate_entry_point_program(
                program, row["entry_point"], row["test"], timeout
            )
            elapsed = time.time() - start
            if elapsed > timeout * 0.5:
                slow += 1
            if good:
                passed += 1
            else:
                failures_here.append((row["task_id"], f"{error[:80]} ({elapsed:.1f}s)"))
        rate = passed / max(count, 1)
        print(f"  {'OK  ' if rate == 1.0 else 'FAIL'} 标准答案自测 {passed}/{count} "
              f"({rate:.1%}, timeout={timeout}s)")
        for task_id, error in failures_here[:5]:
            print(f"       {task_id}: {error}")
        if slow:
            print(f"  !! {slow} 道用掉了超过一半的 timeout，余量偏小，考虑调大 CODE_TIMEOUT")
        if rate < 1.0:
            problems.append(
                f"humanevalplus: 判分器判错 {count - passed} 道标准答案"
                f"（超时的话调大 CODE_TIMEOUT，当前 {timeout}s）"
            )
        print()

    print("=" * 78)
    print("mbppplus   code/mbppplus_test.parquet")
    path = BENCH / "code/mbppplus_test.parquet"
    if not path.exists():
        print("  SKIP 文件不存在\n")
        return problems

    from onereplay.eval.code_exec import evaluate_assert_program
    from onereplay.eval.metrics.evalplus import MBPPPlusMetric

    metric = MBPPPlusMetric()
    rows = metric.load_rows({"mbppplus_data_file": str(path)})
    count = len(rows)
    # 378, not 500: EvalPlus dropped the ambiguous tasks. So MBPP and MBPP+ are
    # different task sets and their scores are not a before/after pair.
    print(f"  {'OK  ' if count == 378 else 'FAIL'} 载入 {count} / 源 378 条")
    if count != 378:
        problems.append(f"mbppplus: 载入 {count}，应为 378")

    # The override that matters: grading on `test`, not the original `test_list`.
    sample = rows[0]
    chosen = metric.select_tests(sample)
    base_tests = sample.get("test_list") or []
    if chosen and str(chosen[-1]) not in [str(t) for t in base_tests]:
        ok_len = sum(len(str(item)) for item in chosen)
        print(f"  OK   判分读的是 test 而不是 test_list（{ok_len:,} 字符 vs "
              f"test_list {sum(len(str(t)) for t in base_tests):,}）")
    else:
        problems.append("mbppplus: select_tests 返回的是 test_list，那等于用基础版的测试冒充加号版")
        print("  FAIL select_tests 返回了 test_list")

    timeout = MBPPPlusMetric.default_timeout
    passed, slow, failures_here = 0, 0, []
    for row in rows:
        start = time.time()
        good, error = evaluate_assert_program(row["code"], metric.select_tests(row), timeout)
        elapsed = time.time() - start
        if elapsed > timeout * 0.5:
            slow += 1
        if good:
            passed += 1
        else:
            failures_here.append((row.get("task_id"), f"{error[:80]} ({elapsed:.1f}s)"))
    rate = passed / max(count, 1)
    print(f"  {'OK  ' if rate >= 0.98 else 'FAIL'} 标准答案自测 {passed}/{count} "
          f"({rate:.1%}, timeout={timeout}s)")
    for task_id, error in failures_here[:5]:
        print(f"       task {task_id}: {error}")
    if slow:
        print(f"  !! {slow} 道用掉了超过一半的 timeout")
    if rate < 0.98:
        problems.append(
            f"mbppplus: 判分器判错 {count - passed} 道标准答案（{1 - rate:.1%}），"
            "要么 test 的执行方式不对，要么 timeout 太短"
        )
    print()
    return problems


def main() -> None:
    failures: list[str] = []
    for name, relative, loader, expected, expect_distinct in CASES:
        path = BENCH / relative
        print("=" * 78)
        print(f"{name}   {relative}")
        if not path.exists():
            print("  SKIP 文件不存在\n")
            continue

        want = expected(path) if callable(expected) else expected
        try:
            examples = loader(path)
        except Exception as error:  # noqa: BLE001
            print(f"  FAIL {type(error).__name__}: {error}\n")
            failures.append(f"{name}: {type(error).__name__}: {error}")
            continue

        kept = len(examples)
        rate = kept / want if want else 0.0
        ok = 0.98 <= rate <= 1.0
        print(f"  {'OK  ' if ok else 'FAIL'} 载入 {kept} / 源 {want} 条 ({rate:.1%})")
        if not ok:
            reason = "重复展开" if rate > 1.0 else "字段大概率没对上"
            failures.append(f"{name}: 载入 {kept}/{want}，{reason}")

        if not examples:
            print()
            continue

        golds = [str(example.get("gold", "")) for example in examples]
        empty = sum(1 for gold in golds if not gold.strip())
        distinct = Counter(golds)
        print(f"  gold 空值 {empty}，不同取值 {len(distinct)}")
        if len(distinct) <= 12:
            print(f"  gold 分布 {dict(distinct.most_common())}")
        else:
            print(f"  gold 样例 {golds[:6]}")
        if empty:
            failures.append(f"{name}: {empty} 条 gold 为空")
        if expect_distinct is not None:
            if len(distinct) != expect_distinct:
                failures.append(
                    f"{name}: 选项应有 {expect_distinct} 种 gold，实际 {len(distinct)} 种"
                )
        # One gold shared by hundreds of rows means the loader latched onto a
        # conversation-level or document-level field instead of the row's own --
        # exactly what ConvFinQA did before the turn fields were read correctly.
        elif kept > 50 and len(distinct) < kept * 0.1:
            failures.append(f"{name}: 只有 {len(distinct)} 个不同 gold，疑似读错层级")

        sample = examples[0]
        prompt = str(sample.get("prompt", ""))
        print(f"  首条 gold = {sample.get('gold')!r}")
        print(f"  首条 prompt ({len(prompt)} chars): {prompt[:160]!r}...")
        print()

    failures.extend(check_code())

    print("=" * 78)
    if failures:
        print(f"发现 {len(failures)} 个问题：")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("全部 loader 与真实数据对齐。")


if __name__ == "__main__":
    main()
