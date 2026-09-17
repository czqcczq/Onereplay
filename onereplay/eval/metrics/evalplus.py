"""HumanEval+ and MBPP+ (EvalPlus), as thin variants of the base two metrics.

EvalPlus keeps the tasks and replaces the tests. The base HumanEval ships around
9 test cases per task and MBPP ships 3, which is few enough that a solution can
pass while being wrong -- the paper's finding is that this inflates pass@1 by
double digits. The plus sets run roughly 80x more cases, so a pass means more.

The two arrive in different shapes, and only one of them is a drop-in:

  * **HumanEval+ is.** evalplus/humanevalplus carries the same five columns as
    openai/openai_humaneval -- task_id, prompt, canonical_solution, entry_point,
    test -- with `test` grown from a few hundred characters to ~77K. The base
    metric already execs whatever is in `test`, so pointing it at the other file
    is the entire change.

  * **MBPP+ is not.** evalplus/mbppplus has 378 rows, not 500: EvalPlus dropped
    the tasks whose original statement was ambiguous or whose reference solution
    was wrong. More importantly it keeps the ORIGINAL three assertions in
    `test_list` and puts its own expanded set in `test`. Reading `test_list`
    there would run the base set's tests over a subset of its tasks and report
    it as MBPP+, which is a strictly worse measurement wearing a better name.
    So select_tests is overridden to read `test`, with `test_imports` executed
    first because the expanded assertions rely on them.

Because the task sets differ, MBPP and MBPP+ scores are not comparable as
before/after -- 500 tasks vs 378 different ones. Report them side by side.

The timeouts are longer than the base metrics'. 80x the assertions is 80x the
work, and a timeout is scored as a failure, so a too-short limit quietly
converts correct answers into a lower pass@1 for every arm.
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from onereplay.eval.metrics.humaneval import HumanEvalMetric
from onereplay.eval.metrics.mbpp import MBPPMetric


class HumanEvalPlusMetric(HumanEvalMetric):
    """HumanEval+ -- the base metric reading the extended test file."""

    name = "humanevalplus"
    data_key = "humanevalplus_data_file"
    default_timeout = 30.0


class MBPPPlusMetric(MBPPMetric):
    """MBPP+ -- 378 tasks, graded on `test` rather than `test_list`."""

    name = "mbppplus"
    default_timeout = 30.0

    def load_rows(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        data_file = cfg.get("mbppplus_data_file", "")
        if not data_file:
            raise ValueError(
                "mbppplus 需要 --mbppplus_data_file（evalplus/mbppplus 的 parquet）。"
                "先跑 python test_code/download_bench.py --only mbppplus"
            )
        dataset = load_dataset(
            "parquet",
            data_files=data_file,
            split="train",
            cache_dir=cfg.get("cache_dir", "") or None,
        )
        rows = [dict(dataset[i]) for i in range(len(dataset))]
        limit = int(cfg.get("limit", 0))
        return rows[:limit] if limit > 0 else rows

    def select_tests(self, example: dict[str, Any]) -> list[str]:
        # test_imports first: the expanded assertions reference what it pulls in,
        # and each element is exec'd into one shared namespace in order.
        imports = example.get("test_imports") or []
        if isinstance(imports, str):
            imports = [imports]
        expanded = example.get("test") or ""
        if not expanded:
            # Never silently fall back to test_list -- that would report the base
            # set's tests under the plus set's name.
            raise ValueError(f"mbppplus 行 {example.get('task_id')} 没有 test 字段")
        return [str(item) for item in imports] + [str(expanded)]
