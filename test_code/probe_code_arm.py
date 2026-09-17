"""Measure the Code arm before building it: EvalPlus fields, then Magicoder.

Run on the login node (needs network):

    python test_code/probe_code_arm.py --tokenizer_path models/Qwen3-4B-Base

Two stages, in the order that can change the plan:

1. EvalPlus field check. HumanEval+ and MBPP+ are only cheap if their rows carry
   the same fields the existing HumanEvalMetric / MBPPMetric already read -- then
   they are a data path away. If the test cases arrive in a different shape, the
   judging code has to be written, and that is worth knowing before promising
   four benchmarks instead of two.

2. Magicoder sizing. How many rows survive decontamination against
   HumanEval/MBPP, how many survive 4096, and what the Python / non-Python mix
   looks like on the way out.

   The whole corpus is measured, not the Python part. The benchmarks are Python,
   which makes filtering to Python look obvious, and the paper's ablation
   (Table 5) says it costs accuracy: HumanEval+ pass@1 is 47.6 for Python-only
   (43K) against 55.5 for the full 75K, and non-Python data *alone* gets 44.5
   from a 34.1 base. So the language readings are printed for reference and the
   pipeline runs on everything.

   Where Python does need identifying -- the mix report -- it is the ```python
   fence, not the `lang` column. `lang` records the language of the SEED
   snippet, and the paper explicitly declines to classify by it "because LLMs
   performing OSS-Instruct may produce code in a different programming language
   than the seed".

Nothing is written except the raw downloads; this only measures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The same filters prepare_magicoder_ccode.py applies, imported rather than
# recopied: if the pool builder's notion of "contaminated" drifts from the
# probe's, the probe stops predicting what the builder will produce.
from onereplay.scripts.prepare_magicoder_ccode import (  # noqa: E402
    NGRAM,
    humaneval_docstrings,
    mbpp_texts,
    ngram_hashes,
    normalize,
)

# Parallel to DEFAULT_SYSTEM_PROMPT, differing only in where the final answer
# goes. Keeping the "reason step by step" half identical holds the per-domain
# difference down to the one clause that has to differ: a code answer belongs in
# a fenced block, and the code metrics extract fences, not \boxed{}.
CODE_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final code in a Python code block."
)

EVALPLUS = {
    "humanevalplus": "evalplus/humanevalplus",
    "mbppplus": "evalplus/mbppplus",
}
# What the existing metrics read. A plus-set carrying these is a data swap.
NEEDED = {
    "humanevalplus": ("task_id", "prompt", "entry_point", "test"),
    "mbppplus": ("task_id", "prompt", "test_list"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Size up the Code arm before building it.")
    parser.add_argument("--dest", type=str, default="datasets/raw")
    parser.add_argument(
        "--magicoder_path",
        type=str,
        default="datasets/code_replay/magicoder_oss_instruct_75k.parquet",
        help="Local parquet from download_magicoder_data.py. Missing falls back to the Hub.",
    )
    parser.add_argument("--magicoder_repo", type=str, default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    parser.add_argument("--humaneval_data_file", type=str, default="datasets/code/humaneval_test.parquet")
    parser.add_argument("--mbpp_dataset_path", type=str, default="datasets/code/mbpp_full")
    parser.add_argument("--mbpp_split", type=str, default="test")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Qwen3-4B-Base. Without it the 4096 question cannot be answered.",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--system_prompt", type=str, default=CODE_SYSTEM_PROMPT)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--skip_evalplus", action="store_true")
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarize(label: str, values: list[int]) -> None:
    if not values:
        print(f"  {label}: (空)")
        return
    print(
        f"  {label}: n={len(values)} mean={sum(values) / len(values):.0f} "
        f"median={percentile(values, 0.5)} p90={percentile(values, 0.9)} "
        f"p95={percentile(values, 0.95)} max={max(values)}"
    )


def stage_evalplus(args: argparse.Namespace) -> None:
    print("=" * 78)
    print("阶段 1：EvalPlus 字段检查（决定 +  版本是换数据还是要写判分）")
    print("=" * 78)
    if args.skip_evalplus:
        print("  --skip_evalplus，跳过")
        return

    from datasets import load_dataset

    for name, repo in EVALPLUS.items():
        print(f"\n[{name}] {repo}")
        try:
            dataset = load_dataset(repo, split="test", cache_dir=args.cache_dir or None)
        except Exception as error:  # noqa: BLE001
            print(f"  !! 拉取失败: {error}")
            print("     如果是 split 名不对，先 python -c \"from datasets import get_dataset_split_names;"
                  f" print(get_dataset_split_names('{repo}'))\"")
            continue

        columns = list(dataset.column_names)
        print(f"  {len(dataset)} 行   字段: {columns}")
        missing = [field for field in NEEDED[name] if field not in columns]
        if missing:
            print(f"  !! 缺少现有 metric 依赖的字段: {missing}")
            print("     -> 不能只换数据路径，判分逻辑要改写")
        else:
            print("  现有 metric 需要的字段齐全 -> 很可能只要换数据路径")

        row = dict(dataset[0])
        for field in NEEDED[name]:
            if field not in row:
                continue
            value = row[field]
            shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
            print(f"    {field}: {str(shown)[:150]!r}")
        # The plus sets' whole point is more test cases; if the count per task
        # matches the base set, the download was the wrong one.
        tests = row.get("test") or row.get("test_list") or ""
        length = len(tests) if isinstance(tests, (list, tuple)) else len(str(tests))
        print(f"    第一题的测试规模: {length}"
              f"{' 条断言' if isinstance(tests, (list, tuple)) else ' 字符'}")


def load_magicoder_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    local = Path(args.magicoder_path)
    if local.is_file():
        print(f"  读本地 {local}")
        dataset = load_dataset(
            "parquet", data_files=str(local), split="train", cache_dir=args.cache_dir or None
        )
    else:
        print(f"  本地没有 {local}，从 Hub 拉 {args.magicoder_repo}")
        dataset = load_dataset(args.magicoder_repo, split="train", cache_dir=args.cache_dir or None)
    return [dict(dataset[i]) for i in range(len(dataset))]


def stage_magicoder(args: argparse.Namespace) -> None:
    print()
    print("=" * 78)
    print("阶段 2：Magicoder 量化")
    print("=" * 78)
    rows = load_magicoder_rows(args)
    print(f"  {len(rows)} 行   字段: {list(rows[0].keys()) if rows else '(空)'}")

    problem_key = "problem" if rows and "problem" in rows[0] else "instruction"
    solution_key = "solution" if rows and "solution" in rows[0] else "response"
    print(f"  取列: problem={problem_key!r} solution={solution_key!r}")

    print()
    print("---- 语言口径（lang 记的是种子片段的语言，不是生成内容的）")
    by_lang, by_fence, by_both, nonempty = [], [], [], []
    for index, row in enumerate(rows):
        problem = str(row.get(problem_key, "") or "").strip()
        solution = str(row.get(solution_key, "") or "").strip()
        if not problem or not solution:
            continue
        nonempty.append(index)
        lang_ok = str(row.get("lang", "")).strip().lower() == "python"
        fence_ok = "```python" in solution.lower()
        if lang_ok:
            by_lang.append(index)
        if fence_ok:
            by_fence.append(index)
        if lang_ok and fence_ok:
            by_both.append(index)
    total = max(len(rows), 1)
    print(f"  lang == python           {len(by_lang):>6}  ({len(by_lang)/total:5.1%})")
    print(f"  含 ```python（论文口径） {len(by_fence):>6}  ({len(by_fence)/total:5.1%})")
    print(f"  两者交集                 {len(by_both):>6}  ({len(by_both)/total:5.1%})")
    print(f"  只满足一边               {len(set(by_lang) ^ set(by_fence)):>6}")
    print(f"  全部非空行（现用口径）   {len(nonempty):>6}  ({len(nonempty)/total:5.1%})")
    print()
    print("  用全量，不是只用 Python —— 论文 Table 5（CodeLlama-Python-7B，HumanEval+ pass@1）:")
    print("    不微调 34.1 / 只用 Python(43K) 47.6 / 只用非 Python(32K) 44.5 / 全量(75K) 55.5")
    print("  非 Python 那一半对 Python 是净贡献，不是稀释。")

    kept = nonempty
    is_python = set(by_fence)

    print()
    print("---- 去重（题面归一后完全相同）")
    seen: set[str] = set()
    deduped = []
    for index in kept:
        key = normalize(rows[index][problem_key])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(index)
    print(f"  去掉 {len(kept) - len(deduped)} -> 剩 {len(deduped)}")

    print()
    print(f"---- 去污染（vs HumanEval / MBPP，{NGRAM}-gram）")
    grams: set[int] = set()
    if Path(args.humaneval_data_file).is_file():
        docstrings = humaneval_docstrings(args.humaneval_data_file, args.cache_dir)
        for text in docstrings:
            grams |= ngram_hashes(text)
        print(f"  HumanEval 签名 {len(docstrings)} 条")
    else:
        print(f"  !! 找不到 {args.humaneval_data_file}，跳过 HumanEval 一侧")
        print("     先跑 python -m onereplay.scripts.download_code_data")
    if Path(args.mbpp_dataset_path).exists():
        texts = mbpp_texts(args.mbpp_dataset_path, args.mbpp_split)
        for text in texts:
            grams |= ngram_hashes(text)
        print(f"  MBPP 签名 {len(texts)} 条")
    else:
        print(f"  !! 找不到 {args.mbpp_dataset_path}，跳过 MBPP 一侧")

    clean, hits = [], []
    if grams:
        for index in deduped:
            combined = f"{rows[index][problem_key]}\n{rows[index][solution_key]}"
            (hits if ngram_hashes(combined) & grams else clean).append(index)
        print(f"  命中 {len(hits)} 条 -> 剩 {len(clean)}")
        for index in hits[: args.samples]:
            print(f"    [命中样例] {str(rows[index][problem_key])[:110]}")
    else:
        clean = deduped
        print("  两侧都缺，本轮没做去污染 —— 这个数字不能当最终结论")

    print()
    print(f"---- {args.max_tokens} token 过滤")
    if not args.tokenizer_path:
        print("  没给 --tokenizer_path，只能报字符数（约 4 字符/token，仅供估算）")
        summarize("问题字符", [len(str(rows[i][problem_key])) for i in clean])
        summarize("回答字符", [len(str(rows[i][solution_key])) for i in clean])
        return

    from onereplay.scripts.domain_sft.common import load_tokenizer, serialize_many

    print(f"  system prompt: {args.system_prompt!r}")
    tokenizer = load_tokenizer(args.tokenizer_path, args.system_prompt)
    pairs = [
        (str(rows[i][problem_key]).strip(), str(rows[i][solution_key]).strip())
        for i in clean
    ]
    measured = serialize_many(tokenizer, pairs, progress_every=5000)
    survivors = [item for item in measured if item.total_tokens <= args.max_tokens]

    summarize("全部 total_tokens", [item.total_tokens for item in measured])
    print(
        f"  <= {args.max_tokens}: {len(survivors)} / {len(measured)} "
        f"({len(survivors) / max(len(measured), 1):.1%})"
    )
    if survivors:
        summarize("存活行 assistant", [item.response_tokens for item in survivors])
        print(f"  存活行 assistant token 合计: {sum(i.response_tokens for i in survivors):,}")
        # A prompt that is not a token prefix of the full render means the label
        # mask lands on the wrong positions for that row.
        violations = sum(1 for item in survivors if not item.prompt_is_prefix)
        if violations:
            print(f"  !! prompt 不是 full 的前缀: {violations} 行，label mask 会错位")

        # 存活行里 Python / 非 Python 各占多少 —— 论文是 ~57% Python，差太多说明
        # 某一侧被前面的过滤打得更狠。
        py_tokens = non_py_tokens = py_rows = 0
        for row_index, item in zip(clean, measured):
            if item.total_tokens > args.max_tokens:
                continue
            if row_index in is_python:
                py_rows += 1
                py_tokens += item.response_tokens
            else:
                non_py_tokens += item.response_tokens
        print(
            f"  其中 Python {py_rows} 行 / {py_tokens:,} assistant token"
            f"（{py_rows / max(len(survivors), 1):.1%} 的行，"
            f"{py_tokens / max(py_tokens + non_py_tokens, 1):.1%} 的 token）"
        )
        print(f"  非 Python {len(survivors) - py_rows} 行 / {non_py_tokens:,} assistant token")
        print("  论文 OSS-Instruct 75K 约 57% 是 Python；偏离太多说明某一侧被过滤打得更狠")

    print()
    print("---- 对照：另外两臂")
    for domain in ("math", "medical"):
        path = Path("data/stats") / f"{domain}_stats.json"
        if not path.exists():
            print(f"  {domain:8} 没有 {path}")
            continue
        stats = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"  {domain:8} rows={stats.get('num_examples'):>7}  "
            f"assistant_tokens={stats.get('total_assistant_tokens'):>12,}"
        )

    print()
    print("---- 抽样看内容")
    for index in clean[: args.samples]:
        print("=" * 70)
        print(f"  problem : {str(rows[index][problem_key])[:300]}")
        print(f"  solution: {str(rows[index][solution_key])[:300]}")


def main() -> None:
    args = parse_args()
    stage_evalplus(args)
    stage_magicoder(args)
    print()
    print("=" * 78)
    print("看完这两段就能定：+ 版本怎么接、Python 用哪个口径、最终有多少行")
    print("=" * 78)


if __name__ == "__main__":
    main()
