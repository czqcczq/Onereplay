"""Turn existing eval outputs into probe corpora, without generating anything.

A probe needs {prompt, base model's own answer} pairs. For math that data has
already been produced twice over: `results/<metric>/base/responses.jsonl` holds
the bare Qwen3 answer to every MATH-500 and GSM8K problem, written by the
benchmark run itself. Re-generating it would cost the same hours the benchmark
cost, so this script reuses it.

Building the probe out of the benchmark's own items buys something a fresh
corpus would not: cross-entropy and accuracy are then measured on *identical*
problems, which makes "CE moved 40% while accuracy moved 0.8pp" a statement
about one set of items rather than two.

A third corpus is cut from the MetaMath self-distillation pool that C_math was
collected from. Comparing it against the benchmark corpora is the in-pool /
out-of-pool contrast from chapter 3, adapted: nothing here trains on MetaMath
text, but C_math is built from those activations, so a regularizer anchored to
them could look better on rows that fed it.

    python -m onereplay.scripts.build_math_probe \\
        --responses math500=/path/results/math500/base/responses.jsonl \\
        --responses gsm8k=/path/results/gsm8k/base/responses.jsonl \\
        --cpool /path/results/replay/metamath_cmath_30k_canonical_selfdistill_seed1.jsonl \\
        --out_dir /path/results/probe_ce/corpora \\
        --model_path /path/models/Qwen3-1.7B --cap 4096
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from onereplay.eval.metrics.aime import build_prompt as aime_prompt
from onereplay.eval.metrics.gsm8k import build_prompt as gsm8k_prompt
from onereplay.eval.metrics.math500 import build_prompt as math500_prompt

# Imported from the metrics rather than copied: the probe is only measuring the
# right thing if it renders the byte-identical prompt the benchmark rendered,
# and a local copy would drift the first time either side is edited.
PROMPT_BUILDERS = {
    "math500": math500_prompt,
    "amc": math500_prompt,
    "gsm8k": gsm8k_prompt,
    "aime": aime_prompt,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable. NAME picks the prompt template (math500/gsm8k/amc/aime).",
    )
    parser.add_argument("--cpool", type=str, default="", help="MetaMath self-distill jsonl.")
    parser.add_argument("--cpool_size", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Tokenizer for length stats.")
    parser.add_argument(
        "--cap",
        type=int,
        default=4096,
        help="MATH_MAX_NEW_TOKENS the responses were generated under; rows within "
        "8 tokens of it were cut off mid-answer and are dropped.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Rows per corpus; 0 = all.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))]


def build_from_responses(
    name: str, rows: list[dict[str, Any]], tokenizer, cap: int, limit: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert benchmark responses into self-distill schema rows.

    A response pinned at the generation cap never emitted a stop token, so it is
    a fragment rather than an answer. Chapter 1's replay corpus dropped those for
    the same reason -- training or scoring on them measures the model's ability
    to not finish.
    """

    if name not in PROMPT_BUILDERS:
        raise ValueError(f"unknown probe corpus {name!r}; known: {sorted(PROMPT_BUILDERS)}")
    build = PROMPT_BUILDERS[name]

    out: list[dict[str, Any]] = []
    dropped_truncated = 0
    dropped_empty = 0
    target_lengths: list[int] = []

    for index, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        response = str(row.get("response", "")).strip()
        if not question or not response:
            dropped_empty += 1
            continue
        target_tokens = len(tokenizer(response, add_special_tokens=False)["input_ids"])
        if target_tokens >= cap - 8:
            dropped_truncated += 1
            continue
        prompt = build(question)
        out.append(
            {
                "index": index,
                "inputs": prompt,
                "targets": response,
                "question": question,
                "truncated": False,
                "prompt_tokens": len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                "target_tokens": target_tokens,
            }
        )
        target_lengths.append(target_tokens)
        if limit > 0 and len(out) >= limit:
            break

    stats = {
        "corpus": name,
        "rows_in": len(rows),
        "rows_out": len(out),
        "dropped_truncated": dropped_truncated,
        "dropped_empty": dropped_empty,
        "target_tokens_p50": percentile(target_lengths, 50),
        "target_tokens_p90": percentile(target_lengths, 90),
        "target_tokens_p99": percentile(target_lengths, 99),
        "target_tokens_max": max(target_lengths) if target_lengths else 0,
    }
    return out, stats


def subsample(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Even stride, matching data/probe.py's _subsample: the pool was already
    shuffled once, so position carries no information and a stride needs no seed
    of its own to be reproducible."""

    if size <= 0 or size >= len(rows):
        return rows
    stride = len(rows) // size
    return [rows[i] for i in range(0, stride * size, stride)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    out_dir = Path(args.out_dir)
    manifest: dict[str, Any] = {"cap": args.cap, "corpora": []}
    bench_questions: dict[str, set[str]] = {}

    for spec in args.responses:
        if "=" not in spec:
            raise ValueError(f"--responses must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name, path = name.strip(), Path(path.strip())
        if not path.is_file():
            print(f"[skip] {name}: 找不到 {path}")
            continue
        rows, stats = build_from_responses(name, load_jsonl(path), tokenizer, args.cap, args.limit)
        out_path = out_dir / f"probe_{name}_base.jsonl"
        write_jsonl(out_path, rows)
        stats["path"] = str(out_path)
        stats["source"] = str(path)
        manifest["corpora"].append(stats)
        bench_questions[name] = {row["question"] for row in rows}
        print(
            f"{name:<10} {stats['rows_in']:>5} -> {stats['rows_out']:>5} 行 "
            f"(丢弃 截断={stats['dropped_truncated']} 空={stats['dropped_empty']}) "
            f"target token P50/P90/P99/max="
            f"{stats['target_tokens_p50']}/{stats['target_tokens_p90']}/"
            f"{stats['target_tokens_p99']}/{stats['target_tokens_max']}"
        )

    if args.cpool:
        cpool_path = Path(args.cpool)
        if not cpool_path.is_file():
            print(f"[skip] cpool: 找不到 {cpool_path}")
        else:
            pool = load_jsonl(cpool_path)
            usable = [
                row
                for row in pool
                if not row.get("truncated") and str(row.get("targets", "")).strip()
            ]
            cut = subsample(usable, args.cpool_size)
            out_path = out_dir / "probe_metamath_cpool.jsonl"
            write_jsonl(out_path, cut)
            manifest["corpora"].append(
                {
                    "corpus": "metamath_cpool",
                    "rows_in": len(pool),
                    "rows_usable": len(usable),
                    "rows_out": len(cut),
                    "path": str(out_path),
                    "source": str(cpool_path),
                }
            )
            print(f"{'cpool':<10} {len(pool):>5} -> {len(usable)} 可用 -> {len(cut):>5} 行")

            # The benchmark items must not be in the pool C_math was built from.
            # Chapter 3 found 98 of 1500 held-out FLAN rows leaking back into the
            # training pool because an offset silently did nothing, so this check
            # is cheap insurance rather than a formality.
            pool_prompts = {str(row.get("inputs", "")).strip() for row in pool}
            for name, questions in bench_questions.items():
                overlap = len(questions & pool_prompts)
                manifest[f"{name}_cpool_overlap"] = overlap
                verdict = "OK，无重叠" if overlap == 0 else f"!! {overlap} 题与 C 池重叠"
                print(f"重叠自检 {name} vs C 池: {verdict}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nmanifest -> {manifest_path}")
    print(
        "提示：score_probe_ce 的 --max_len 要覆盖上面的 target token 分位数，"
        "否则长回答会被截断，CE 只覆盖前缀。"
    )


if __name__ == "__main__":
    main()
