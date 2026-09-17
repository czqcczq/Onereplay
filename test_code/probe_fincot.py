"""Fetch FinCoT / Fino1 and answer the three questions that decide the swap.

Run on the login node (needs network), after download_bench.py has put the
benchmarks in place:

    python test_code/probe_fincot.py --tokenizer_path models/Qwen3-4B-Base

The three questions, in the order they can kill the plan:

1. Is it contaminated? The Fino1 paper never states whether FinCoT was built
   from the train splits. Their own results require it -- they report on FinQA
   test -- but "required for their paper to hold" is not a check. Every question
   is matched against the local FinQA test / ConvFinQA dev / TAT-QA dev. A
   non-trivial hit rate means the Finance arm cannot use this corpus as is.
2. How much survives 4096? FinCoT's Question field runs to 450k characters,
   which is DocFinQA shipping entire filings. Those rows are gone, and the
   question is what is left after they go.
3. How does it compare to the other two arms? Row counts across the three
   domains are not comparable (Medical's 37257 rows are ~9500 questions with
   ~4 traces each), so this reports assistant tokens, which is what the SFT
   gradient is actually made of.

Nothing is written except the raw download; this only measures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPECS = {
    "fincot": {
        "repo_id": "TheFinAI/FinCoT",
        "question": "Question",
        "reasoning": "Reasoning_process",
        "answer_text": "Final_response",
        "gold": "",
        "note": "FinQA + ConvFinQA + TAT-QA + DocFinQA + DocMath + Econ-Logic + BizBench",
    },
    "fino1": {
        "repo_id": "TheFinAI/Fino1_Reasoning_Path_FinQA_v2",
        "question": "Open-ended Verifiable Question",
        "reasoning": "Complex_CoT",
        "answer_text": "Response",
        "gold": "Ground-True Answer",
        "note": "FinQA only, but carries the gold value -- enables correctness filtering",
    },
}

# FinCoT wraps every row as "...Context: {ctx} Question: {q} Answer:". The last
# occurrence wins because a filing's own text can contain the word "Question".
QUESTION_SPAN = re.compile(r"Question:\s*(.+?)\s*Answer:\s*$", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure FinCoT/Fino1 before switching the Finance corpus.")
    parser.add_argument("--dest", type=str, default="datasets/raw")
    parser.add_argument("--bench_root", type=str, default="datasets/bench")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Qwen3-4B-Base. Without it only character lengths are reported, "
        "which cannot decide the 4096 question.",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Defaults to common.DEFAULT_SYSTEM_PROMPT. Must match training: it "
        "contributes tokens that count against --max_tokens.",
    )
    parser.add_argument("--skip_download", action="store_true")
    return parser.parse_args()


def download(name: str, repo_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    target = dest / name
    print(f"[{name}] {repo_id}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(target))
    return target


def load_rows(directory: Path) -> list[dict[str, Any]]:
    import pandas as pd

    files = sorted(directory.rglob("*.parquet")) or sorted(directory.rglob("*.json*"))
    if not files:
        raise SystemExit(f"{directory} 里没有 parquet/json，下载可能没成功")
    rows: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() == ".parquet":
            rows.extend(pd.read_parquet(path).to_dict("records"))
        elif path.suffix.lower() in (".jsonl", ".ndjson"):
            with path.open(encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows.extend(record for record in payload if isinstance(record, dict))
    return rows


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def bare_question(wrapped: str) -> str:
    """The question alone, with the context prompt stripped off.

    Comparing the wrapped form against a benchmark question would never match:
    the wrapped form is thousands of characters of filing text.
    """

    match = QUESTION_SPAN.search(str(wrapped or ""))
    return match.group(1) if match else str(wrapped or "")[-300:]


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def eval_questions(bench_root: Path) -> dict[str, set[str]]:
    """Normalized questions from the three Finance benchmarks."""

    out: dict[str, set[str]] = {}

    finqa = read_json(bench_root / "finqa" / "test.jsonl")
    if finqa:
        found = set()
        for record in finqa:
            blocks = [record[key] for key in ("qa", "qa_0", "qa_1") if isinstance(record.get(key), dict)]
            if not blocks and isinstance(record.get("question"), str):
                blocks = [record]
            for block in blocks:
                text = normalize(block.get("question", ""))
                if text:
                    found.add(text)
        out["finqa_test"] = found

    conv = read_json(bench_root / "convfinqa" / "dev_turn.json")
    if conv:
        records = conv if isinstance(conv, list) else conv.get("dev", [])
        found = set()
        for record in records:
            annotation = record.get("annotation") if isinstance(record, dict) else None
            annotation = annotation if isinstance(annotation, dict) else {}
            dialogue = record.get("cur_dial") or annotation.get("cur_dial")
            if isinstance(dialogue, (list, tuple)) and dialogue:
                text = normalize(dialogue[-1])
                if text:
                    found.add(text)
        out["convfinqa_dev"] = found

    tatqa = read_json(bench_root / "tatqa" / "tatqa_dataset_dev.json")
    if tatqa:
        found = set()
        for record in tatqa if isinstance(tatqa, list) else []:
            for question in record.get("questions", []) or []:
                text = normalize(question.get("question", ""))
                if text:
                    found.add(text)
        out["tatqa_dev"] = found

    return out


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


def main() -> None:
    args = parse_args()
    dest = Path(args.dest)
    bench_root = Path(args.bench_root)

    print("=" * 78)
    print("阶段 0：下载")
    print("=" * 78)
    directories: dict[str, Path] = {}
    for name, spec in SPECS.items():
        target = dest / name
        if args.skip_download or (target.exists() and any(target.rglob("*.parquet"))):
            print(f"[{name}] 已存在，跳过")
            directories[name] = target
        else:
            directories[name] = download(name, spec["repo_id"], dest)
    print()

    datasets: dict[str, list[dict[str, Any]]] = {}
    for name, directory in directories.items():
        rows = load_rows(directory)
        datasets[name] = rows
        spec = SPECS[name]
        print(f"[{name}] {len(rows)} 行   {spec['note']}")
        if rows:
            print(f"         字段: {list(rows[0].keys())}")
    print()

    print("=" * 78)
    print("阶段 1：污染检查（最先做，命中多就不用往下看了）")
    print("=" * 78)
    benches = eval_questions(bench_root)
    if not benches:
        print(f"!! {bench_root} 下没找到 benchmark 数据，先跑 download_bench.py")
    for bench, questions in benches.items():
        print(f"  {bench}: {len(questions)} 道题")
    print()

    for name, rows in datasets.items():
        field = SPECS[name]["question"]
        asked = [normalize(bare_question(row.get(field, ""))) for row in rows]
        asked_set = {text for text in asked if text}
        print(f"[{name}] {len(asked_set)} 道去重后的训练题")
        for bench, questions in benches.items():
            overlap = asked_set & questions
            share = len(overlap) / max(len(questions), 1)
            flag = "  <-- 污染" if len(overlap) > 5 else ""
            print(f"    vs {bench:15} 命中 {len(overlap):5}  (占该 bench {share:.1%}){flag}")
            for sample in list(overlap)[:3]:
                print(f"        {sample[:110]}")
    print()

    print("=" * 78)
    print("阶段 2：两个数据集的重叠（决定能不能合并）")
    print("=" * 78)
    keys = list(datasets)
    if len(keys) == 2:
        left = {normalize(bare_question(row.get(SPECS[keys[0]]["question"], ""))) for row in datasets[keys[0]]}
        right = {normalize(bare_question(row.get(SPECS[keys[1]]["question"], ""))) for row in datasets[keys[1]]}
        both = left & right
        print(f"  {keys[0]} 独有 {len(left - right)}")
        print(f"  {keys[1]} 独有 {len(right - left)}")
        print(f"  两边都有 {len(both)}")
        print(f"  合并去重后 {len(left | right)}")
    print()

    print("=" * 78)
    print(f"阶段 3：{args.max_tokens} token 过滤后还剩多少")
    print("=" * 78)
    if not args.tokenizer_path:
        print("  没给 --tokenizer_path，只能报字符数（1 token 大约 4 字符，仅供估算）")
        for name, rows in datasets.items():
            spec = SPECS[name]
            summarize(
                f"{name} 问题字符",
                [len(str(row.get(spec["question"], ""))) for row in rows],
            )
            summarize(
                f"{name} 回答字符",
                [
                    len(str(row.get(spec["reasoning"], ""))) + len(str(row.get(spec["answer_text"], "")))
                    for row in rows
                ],
            )
        return

    from onereplay.scripts.domain_sft.common import (
        DEFAULT_SYSTEM_PROMPT,
        load_tokenizer,
        serialize_many,
    )

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    print(f"  system prompt: {system_prompt!r}")
    tokenizer = load_tokenizer(args.tokenizer_path, system_prompt)

    for name, rows in datasets.items():
        spec = SPECS[name]
        pairs = []
        for row in rows:
            question = str(row.get(spec["question"], "") or "").strip()
            reasoning = str(row.get(spec["reasoning"], "") or "").strip()
            final = str(row.get(spec["answer_text"], "") or "").strip()
            if not question or not reasoning:
                continue
            pairs.append((question, f"{reasoning}\n\n{final}".strip()))

        print(f"\n[{name}] 量 {len(pairs)} 行")
        measured = serialize_many(tokenizer, pairs, progress_every=2000)
        totals = [item.total_tokens for item in measured]
        kept = [item for item in measured if item.total_tokens <= args.max_tokens]
        summarize("全部 total_tokens", totals)
        print(
            f"  <= {args.max_tokens}: {len(kept)} / {len(measured)} "
            f"({len(kept) / max(len(measured), 1):.1%})"
        )
        if kept:
            assistant = sum(item.response_tokens for item in kept)
            print(f"  存活行的 assistant token 合计: {assistant:,}")
            summarize("存活行 assistant", [item.response_tokens for item in kept])
            violations = sum(1 for item in kept if not item.prompt_is_prefix)
            if violations:
                print(f"  !! prompt 不是 full 的前缀: {violations} 行，label mask 会错位")

    print()
    print("=" * 78)
    print("对照：另外两臂（来自各自的 stats）")
    print("=" * 78)
    for domain in ("math", "medical"):
        path = Path("data/stats") / f"{domain}_stats.json"
        if not path.exists():
            print(f"  {domain}: 没有 {path}")
            continue
        stats = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"  {domain:8} rows={stats.get('num_examples'):>7}  "
            f"assistant_tokens={stats.get('total_assistant_tokens'):>12,}"
        )


if __name__ == "__main__":
    main()
