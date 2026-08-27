"""Turn finished code-eval outputs into CE probe corpora, without generating anything.

This is the code-side twin of build_math_probe.py. The math probe took the
accuracy story -- a 1.67pp drop on GSM8K that the scorer's fallback nearly hid --
and re-read it as cross-entropy, where the same run showed a 7.5x drift. The
code side has the same problem in reverse (MBPP's 3.8pp is significant but is
carried entirely by a 21x jump in syntax errors, not by wrong algorithms), so it
gets the same instrument.

A probe needs {prompt, base model's own answer} pairs. Two things make the code
case awkward, both noted in the handover:

  * The code responses.jsonl only stores `completion` (already run through
    cleanup_completion) plus `task_id`, `passed`, `error`. There is no prompt
    and no raw generation. So the prompt has to be rebuilt from the source
    dataset and joined back on task_id.
  * Using the cleaned completion as the target is actually the cleaner choice --
    the target is then "the valid, solving-ish code base emitted", with the
    chatty preamble and trailing junk already stripped -- but it must be stated
    that the target is post-cleanup, not the raw sample.

The prompt is rebuilt with the metric's own build_prompt/build_mbpp_prompt so
the probe feeds the model the byte-identical string the benchmark fed it; a
local copy would drift the first time either side is edited. CE and pass@1 are
then measured on identical items, which is what lets "syntax errors 21x while
assertion failures held" and "CE moved Nx" be statements about one set of tasks.

    python -m onereplay.scripts.build_code_probe \\
        --responses humaneval=/path/results/humaneval/base/responses.jsonl \\
        --responses mbpp=/path/results/mbpp/base/responses.jsonl \\
        --humaneval_data_file /path/datasets/code/humaneval_test.parquet \\
        --mbpp_dataset_path /path/datasets/code/mbpp_full --mbpp_split test \\
        --out_dir /path/results/probe_ce_code/corpora \\
        --model_path /path/models/Qwen3-1.7B --cap 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from onereplay.eval.metrics.humaneval import build_prompt as humaneval_prompt
from onereplay.eval.metrics.mbpp import build_mbpp_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable. NAME is humaneval or mbpp; PATH is that metric's "
        "base/responses.jsonl.",
    )
    parser.add_argument(
        "--humaneval_data_file",
        type=str,
        default="",
        help="The parquet download_code_data wrote; used to rebuild prompts.",
    )
    parser.add_argument(
        "--mbpp_dataset_path",
        type=str,
        default="",
        help="The save_to_disk directory download_code_data wrote (mbpp_full).",
    )
    parser.add_argument("--mbpp_split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Tokenizer for length stats.")
    parser.add_argument(
        "--cap",
        type=int,
        default=512,
        help="code_max_new_tokens the completions were generated under; targets "
        "within 8 tokens of it were probably cut off mid-code and are dropped. "
        "Note cleanup_completion may have shortened a capped sample below this, "
        "so this catches only the completions that stayed long -- the same "
        "heuristic the math probe uses.",
    )
    parser.add_argument("--cache_dir", type=str, default="", help="datasets cache for the parquet load.")
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


def load_humaneval_prompts(data_file: str, cache_dir: str) -> dict[str, str]:
    """task_id -> user prompt, rebuilt the way HumanEvalMetric.run does it.

    The metric wraps example["prompt"] (signature + docstring) with build_prompt
    and lets the model continue the body, so the probe's target is that body and
    the prompt is the wrapped signature -- identical to generation time.
    """

    from datasets import load_dataset

    dataset = load_dataset(
        "parquet", data_files=data_file, split="train", cache_dir=cache_dir or None
    )
    prompts: dict[str, str] = {}
    for i in range(len(dataset)):
        row = dict(dataset[i])
        prompts[str(row.get("task_id"))] = humaneval_prompt(row["prompt"])
    return prompts


def load_mbpp_prompts(dataset_path: str, split: str) -> dict[str, str]:
    """task_id -> user prompt, rebuilt the way MBPPMetric.run does it."""

    from datasets import load_from_disk

    dataset_dict = load_from_disk(dataset_path)
    dataset = dataset_dict[split] if split in dataset_dict else dataset_dict
    prompts: dict[str, str] = {}
    for i in range(len(dataset)):
        row = dict(dataset[i])
        prompts[str(row.get("task_id"))] = build_mbpp_prompt(row)
    return prompts


PROMPT_LOADERS: dict[str, Callable[[argparse.Namespace], dict[str, str]]] = {
    "humaneval": lambda a: load_humaneval_prompts(a.humaneval_data_file, a.cache_dir),
    "mbpp": lambda a: load_mbpp_prompts(a.mbpp_dataset_path, a.mbpp_split),
}


def build_from_responses(
    name: str,
    rows: list[dict[str, Any]],
    prompts: dict[str, str],
    tokenizer,
    cap: int,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join base completions onto rebuilt prompts and emit self-distill rows.

    A completion pinned at the generation cap never emitted a stop token, so it
    is a fragment. Empty completions (the model produced only prose that
    cleanup_completion discarded) have no supervised tokens. Both are dropped,
    matching what the math probe and the training replay pool do.
    """

    out: list[dict[str, Any]] = []
    dropped_truncated = 0
    dropped_empty = 0
    dropped_no_prompt = 0
    target_lengths: list[int] = []

    for index, row in enumerate(rows):
        task_id = str(row.get("task_id"))
        completion = str(row.get("completion", "")).strip()
        prompt = prompts.get(task_id)
        if prompt is None:
            dropped_no_prompt += 1
            continue
        if not completion:
            dropped_empty += 1
            continue
        target_tokens = len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        if target_tokens >= cap - 8:
            dropped_truncated += 1
            continue
        out.append(
            {
                "index": index,
                "inputs": prompt,
                "targets": completion,
                "task_id": task_id,
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
        "dropped_no_prompt": dropped_no_prompt,
        "target_tokens_p50": percentile(target_lengths, 50),
        "target_tokens_p90": percentile(target_lengths, 90),
        "target_tokens_p99": percentile(target_lengths, 99),
        "target_tokens_max": max(target_lengths) if target_lengths else 0,
    }
    return out, stats


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

    for spec in args.responses:
        if "=" not in spec:
            raise ValueError(f"--responses must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name, path = name.strip(), Path(path.strip())
        if name not in PROMPT_LOADERS:
            raise ValueError(f"unknown code probe corpus {name!r}; known: {sorted(PROMPT_LOADERS)}")
        if not path.is_file():
            print(f"[skip] {name}: 找不到 {path}")
            continue

        prompts = PROMPT_LOADERS[name](args)
        if not prompts:
            print(f"[skip] {name}: 源数据集为空或路径没传，无法重建 prompt")
            continue

        rows, stats = build_from_responses(
            name, load_jsonl(path), prompts, tokenizer, args.cap, args.limit
        )
        out_path = out_dir / f"probe_{name}_base.jsonl"
        write_jsonl(out_path, rows)
        stats["path"] = str(out_path)
        stats["source"] = str(path)
        manifest["corpora"].append(stats)
        print(
            f"{name:<10} {stats['rows_in']:>5} -> {stats['rows_out']:>5} 行 "
            f"(丢弃 截断={stats['dropped_truncated']} 空={stats['dropped_empty']} "
            f"无prompt={stats['dropped_no_prompt']}) "
            f"target token P50/P90/P99/max="
            f"{stats['target_tokens_p50']}/{stats['target_tokens_p90']}/"
            f"{stats['target_tokens_p99']}/{stats['target_tokens_max']}"
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nmanifest -> {manifest_path}")
    print(
        "提示：target 是清洗后的 completion（已过 cleanup_completion），写论文时标明；"
        "score_probe_ce 的 --max_len 要覆盖上面的 target token 分位数。代码回答比数学短，"
        "2048 一般绰绰有余。"
    )


if __name__ == "__main__":
    main()
