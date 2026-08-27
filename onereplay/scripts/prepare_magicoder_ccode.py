"""Build the Magicoder OSS-Instruct prompt pool for C_code.

The code-side counterpart of prepare_metamath_cmath.py, and deliberately much
simpler -- because the thing that script exists to fix does not exist here.

MetaMathQA's 395K rows are four augmented views (AnsAug / Rephrased / SV /
FOBAR) of ~40K unique original questions, so a random draw would hit the same
underlying problem several times with an arbitrary multiplicity. C is a
token-mean, so those duplicates would silently up-weight whichever problems got
drawn twice; hence the dedup-by-original_question and one-view-per-question
quota logic over there.

Magicoder rows carry no such structure: each is generated independently from a
distinct seed snippet, and the authors already "exclud[ed] samples that are
identical or share the same seed code snippet" (OSS-Instruct paper, 2.2). A
uniform random sample is therefore the correct default and needs no per-view
bookkeeping.

What does still need handling, in the order applied:

  1. `lang` is the language of the SEED snippet, not of the generated problem.
     The paper is explicit that OSS-Instruct may emit a different language than
     the seed, and classifies Python by whether ```python appears in the
     generated text (~43K) rather than by `lang` (~38K). This script takes the
     conjunction, so nothing non-Python leaks into C_code.
  2. Exact-duplicate check anyway. Cheap, and chapter 3 already lost time to a
     pool-overlap bug that a two-line check would have caught.
  3. Contamination against HumanEval / MBPP. The authors decontaminated (only 9
     extra samples were filtered, since starcoderdata was already clean), but
     their filter is string matching, and if any eval item survived into the
     pool then the code CE probe stops being an out-of-pool measurement and
     cannot serve as independent evidence. Verified here with word-level
     8-gram overlap, which catches reformatted copies that substring matching
     would miss.
  4. Prompt length. C is a token-mean, so long prompts dominate it -- the same
     reason 31_math_data_cov.pbs rejected thinking mode. Long rows are also
     dropped downstream by generate_replay_targets (empty target +
     truncated=1), so leaving them in means asking for 20k and silently
     getting fewer.

    python -m onereplay.scripts.prepare_magicoder_ccode \\
        --magicoder_path /path/datasets/code_replay/magicoder_oss_instruct_75k.parquet \\
        --humaneval_data_file /path/datasets/code/humaneval_test.parquet \\
        --mbpp_dataset_path /path/datasets/code/mbpp_full --mbpp_split test \\
        --out_dir /path/datasets/Magicoder_views --prefix magicoder_ccode_20k \\
        --num_samples 20000 --seed 1 \\
        --tokenizer_path /path/models/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

# Generic local-file loaders, shared with the math pool builder rather than
# copied: both need the "read parquet/jsonl without a Hub round-trip" behaviour,
# and a second copy would drift.
from onereplay.scripts.prepare_metamath_cmath import DATA_SUFFIXES, find_data_files, load_local

PROBLEM_KEYS = ("problem", "instruction")
SOLUTION_KEYS = ("solution", "response")

NGRAM = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Magicoder C_code prompt pool.")
    parser.add_argument(
        "--magicoder_path",
        type=str,
        default="",
        help="Local parquet/jsonl/save_to_disk path. Empty falls back to --magicoder_repo.",
    )
    parser.add_argument("--magicoder_repo", type=str, default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="magicoder_ccode_20k")
    parser.add_argument(
        "--lang",
        type=str,
        default="python",
        help="Value required in the `lang` column. Empty disables the seed-language filter.",
    )
    parser.add_argument(
        "--require_python_fence",
        type=int,
        default=1,
        help="1 also requires ```python in the solution, which is the criterion the "
        "OSS-Instruct paper itself uses to call a row Python. Kept separate from "
        "--lang because the two disagree (~38K vs ~43K).",
    )
    parser.add_argument("--num_samples", type=int, default=20000, help="0 keeps the whole pool.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=1024,
        help="Drop problems longer than this before sampling. Two reasons: C is a "
        "token-mean so long prompts dominate it, and generate_replay_targets "
        "discards rows that leave no room for an answer inside --max_len. 1024 "
        "leaves ~1000 tokens for the self-distilled solution at max_len=2048. "
        "0 disables the filter (then expect fewer usable rows than requested).",
    )
    parser.add_argument(
        "--humaneval_data_file",
        type=str,
        default="",
        help="Contamination check against HumanEval; skipped when empty.",
    )
    parser.add_argument("--mbpp_dataset_path", type=str, default="")
    parser.add_argument("--mbpp_split", type=str, default="test")
    parser.add_argument(
        "--drop_contaminated",
        type=int,
        default=1,
        help="1 removes rows sharing an 8-gram with a HumanEval/MBPP item. Leave on: "
        "a single leaked eval item turns the code CE probe from out-of-pool "
        "evidence into in-pool self-scoring.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=(
            str(Path(os.environ["MODEL_DIR"]) / os.environ.get("MODEL_NAME", "Qwen3-1.7B"))
            if os.environ.get("MODEL_DIR")
            else ""
        ),
        help="Model dir for token stats; defaults to $MODEL_DIR/$MODEL_NAME when set.",
    )
    parser.add_argument("--cache_dir", type=str, default="")
    return parser.parse_args()


def load_magicoder(args: argparse.Namespace):
    """Load Magicoder from a save_to_disk dir, a plain dir, a file, or the hub."""

    from datasets import load_dataset, load_from_disk

    if not args.magicoder_path:
        return load_dataset(
            args.magicoder_repo, split=args.split, cache_dir=args.cache_dir or None
        )

    source = Path(args.magicoder_path)
    if source.is_dir():
        if (source / "dataset_info.json").exists() or (source / "dataset_dict.json").exists():
            dataset = load_from_disk(str(source))
            if hasattr(dataset, "keys"):
                dataset = dataset[args.split if args.split in dataset else next(iter(dataset))]
            return dataset
        fmt, files = find_data_files(source)
        print(f"loading {len(files)} {fmt} file(s) from {source}")
        return load_local(fmt, files)

    fmt = dict(DATA_SUFFIXES).get(source.suffix.lower())
    if fmt is None:
        raise ValueError(f"Unsupported Magicoder file type: {source}")
    return load_local(fmt, [str(source)])


def pick_column(columns: list[str], candidates: tuple[str, ...], role: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise SystemExit(
        f"Magicoder {role} column not found; tried {candidates}, got {columns}. "
        "Both the 75K release (problem/solution) and the Instruction-Response "
        "sibling (instruction/response) are supported."
    )


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, so reformatting does not hide a copy."""

    return re.sub(r"\s+", " ", str(text).lower()).strip()


def ngram_hashes(text: str, n: int = NGRAM) -> set[int]:
    """Hashes of every n-word window of normalized text.

    Word-level n-grams rather than pairwise substring search: the eval sets
    contribute ~660 signatures and the pool ~38K rows, so the naive nested loop
    is tens of millions of substring scans. Hashing both sides into sets makes
    it one pass each. n=8 is long enough that a shared window means copied text
    rather than a coincidence of phrasing -- matching on a function name like
    `is_prime` would flag legitimate rows constantly.
    """

    words = normalize(text).split()
    if len(words) < n:
        return {hash(" ".join(words))} if words else set()
    return {hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


def humaneval_docstrings(data_file: str, cache_dir: str) -> list[str]:
    """The docstring of every HumanEval task.

    The docstring is the contamination signature the OSS-Instruct authors
    themselves used, and it is far more distinctive than the signature line.
    """

    from datasets import load_dataset

    dataset = load_dataset(
        "parquet", data_files=data_file, split="train", cache_dir=cache_dir or None
    )
    out: list[str] = []
    for i in range(len(dataset)):
        prompt = str(dict(dataset[i]).get("prompt", ""))
        found = re.findall(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', prompt, flags=re.DOTALL)
        for triple in found:
            text = next((part for part in triple if part), "")
            if len(normalize(text)) >= 40:
                out.append(text)
    return out


def mbpp_texts(dataset_path: str, split: str) -> list[str]:
    """Task description plus asserts for every MBPP task."""

    from datasets import load_from_disk

    dataset_dict = load_from_disk(dataset_path)
    dataset = dataset_dict[split] if split in dataset_dict else dataset_dict
    out: list[str] = []
    for i in range(len(dataset)):
        row = dict(dataset[i])
        text = str(row.get("text", ""))
        if normalize(text):
            out.append(text)
        tests = row.get("test_list") or []
        if isinstance(tests, str):
            tests = [tests]
        out.extend(str(test) for test in tests if normalize(str(test)))
    return out


def build_eval_ngrams(args: argparse.Namespace) -> tuple[set[int], dict[str, int]]:
    """Union of eval-item n-gram hashes, plus how many items each side gave."""

    grams: set[int] = set()
    counts = {"humaneval_signatures": 0, "mbpp_signatures": 0}
    if args.humaneval_data_file and Path(args.humaneval_data_file).is_file():
        docstrings = humaneval_docstrings(args.humaneval_data_file, args.cache_dir)
        counts["humaneval_signatures"] = len(docstrings)
        for text in docstrings:
            grams |= ngram_hashes(text)
    else:
        print("跳过 HumanEval 污染自检：未给 --humaneval_data_file 或文件不存在")
    if args.mbpp_dataset_path and Path(args.mbpp_dataset_path).exists():
        texts = mbpp_texts(args.mbpp_dataset_path, args.mbpp_split)
        counts["mbpp_signatures"] = len(texts)
        for text in texts:
            grams |= ngram_hashes(text)
    else:
        print("跳过 MBPP 污染自检：未给 --mbpp_dataset_path 或路径不存在")
    return grams, counts


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))]


def count_tokens(texts: list[str], tokenizer_path: str, chunk: int = 1000) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    lengths: list[int] = []
    for start in range(0, len(texts), chunk):
        batch = texts[start : start + chunk]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def main() -> None:
    args = parse_args()
    dataset = load_magicoder(args)
    columns = list(dataset.column_names)
    print(f"loaded {len(dataset)} rows, columns={columns}")

    problem_key = pick_column(columns, PROBLEM_KEYS, "problem")
    solution_key = pick_column(columns, SOLUTION_KEYS, "solution")
    print(f"using columns: problem={problem_key!r} solution={solution_key!r}")

    problems = dataset[problem_key]
    solutions = dataset[solution_key]
    langs = dataset["lang"] if "lang" in columns else [""] * len(dataset)

    stats: dict[str, Any] = {"rows_in": len(dataset)}

    # -- step 1: language ---------------------------------------------------
    want_lang = args.lang.strip().lower()
    fence = args.require_python_fence == 1
    kept: list[int] = []
    dropped_lang = 0
    dropped_fence = 0
    for i, (problem, solution, lang) in enumerate(zip(problems, solutions, langs)):
        if want_lang and "lang" in columns and str(lang).strip().lower() != want_lang:
            dropped_lang += 1
            continue
        if fence and "```python" not in str(solution).lower():
            dropped_fence += 1
            continue
        if not str(problem).strip() or not str(solution).strip():
            continue
        kept.append(i)
    stats["dropped_lang"] = dropped_lang
    stats["dropped_no_python_fence"] = dropped_fence
    stats["after_language"] = len(kept)
    print(
        f"语言过滤: lang=={want_lang!r} 去掉 {dropped_lang}，"
        f"缺 ```python 去掉 {dropped_fence} -> 剩 {len(kept)}"
    )

    # -- step 2: exact duplicates ------------------------------------------
    seen: set[str] = set()
    deduped: list[int] = []
    for i in kept:
        key = normalize(problems[i])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(i)
    stats["dropped_duplicate"] = len(kept) - len(deduped)
    stats["after_dedup"] = len(deduped)
    print(f"去重: 去掉 {stats['dropped_duplicate']} 条重复题面 -> 剩 {len(deduped)}")

    # -- step 3: contamination against the eval sets ------------------------
    eval_grams, sig_counts = build_eval_ngrams(args)
    stats.update(sig_counts)
    contaminated: list[int] = []
    if eval_grams:
        for i in deduped:
            combined = f"{problems[i]}\n{solutions[i]}"
            if ngram_hashes(combined) & eval_grams:
                contaminated.append(i)
        stats["contaminated_hits"] = len(contaminated)
        verdict = "OK，无重叠" if not contaminated else f"!! {len(contaminated)} 条与评测集共享 8-gram"
        print(f"污染自检 (HumanEval+MBPP, {NGRAM}-gram): {verdict}")
        if contaminated and args.drop_contaminated == 1:
            flagged = set(contaminated)
            deduped = [i for i in deduped if i not in flagged]
            print(f"  已剔除，剩 {len(deduped)} 条")
    stats["after_decontamination"] = len(deduped)

    # -- step 4: prompt length ---------------------------------------------
    prompt_tokens: dict[int, int] = {}
    if args.tokenizer_path:
        lengths = count_tokens([str(problems[i]) for i in deduped], args.tokenizer_path)
        prompt_tokens = dict(zip(deduped, lengths))
        stats["prompt_tokens_p50"] = percentile(lengths, 50)
        stats["prompt_tokens_p90"] = percentile(lengths, 90)
        stats["prompt_tokens_p99"] = percentile(lengths, 99)
        stats["prompt_tokens_max"] = max(lengths) if lengths else 0
        print(
            f"题面 token: P50={stats['prompt_tokens_p50']} P90={stats['prompt_tokens_p90']} "
            f"P99={stats['prompt_tokens_p99']} max={stats['prompt_tokens_max']}"
        )
        if args.max_prompt_tokens > 0:
            before = len(deduped)
            deduped = [i for i in deduped if prompt_tokens[i] <= args.max_prompt_tokens]
            stats["dropped_too_long"] = before - len(deduped)
            print(
                f"长度过滤: > {args.max_prompt_tokens} token 去掉 "
                f"{stats['dropped_too_long']} -> 剩 {len(deduped)}"
            )
    else:
        print("未给 --tokenizer_path，跳过 token 统计与长度过滤")
    stats["after_length"] = len(deduped)

    # -- step 5: uniform random sample --------------------------------------
    rng = random.Random(args.seed)
    if args.num_samples > 0 and args.num_samples < len(deduped):
        selected = rng.sample(deduped, args.num_samples)
    else:
        if args.num_samples > len(deduped):
            print(
                f"!! 只剩 {len(deduped)} 条可用，少于请求的 {args.num_samples}。"
                "放宽 --max_prompt_tokens 或改小 --num_samples；"
                "别默默用一个更小的池，λ 会对不上。"
            )
        selected = list(deduped)
    selected.sort()
    stats["rows_out"] = len(selected)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.prefix}_python.jsonl"
    with out_path.open("w", encoding="utf-8") as file:
        for i in selected:
            file.write(
                json.dumps(
                    {
                        # `inputs` is the bare problem statement, matching what
                        # prepare_metamath_cmath writes: C is collected on the
                        # corpus prompt rendered through the chat template, not
                        # on the eval-time instruction wrapper that
                        # humaneval.build_prompt adds.
                        "inputs": str(problems[i]),
                        # Overwritten by generate_replay_targets with the base
                        # model's own answer and kept as gold_targets; the
                        # original solution never enters C.
                        "targets": str(solutions[i]),
                        "lang": str(langs[i]),
                        "source_index": i,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\n{stats['rows_out']} 行 -> {out_path}")

    if prompt_tokens:
        chosen = [prompt_tokens[i] for i in selected]
        stats["selected_prompt_tokens_p50"] = percentile(chosen, 50)
        stats["selected_prompt_tokens_p99"] = percentile(chosen, 99)
        stats["selected_prompt_tokens_total"] = sum(chosen)

    manifest = {
        "prefix": args.prefix,
        "seed": args.seed,
        "num_samples_requested": args.num_samples,
        "lang": args.lang,
        "require_python_fence": args.require_python_fence,
        "max_prompt_tokens": args.max_prompt_tokens,
        "drop_contaminated": args.drop_contaminated,
        "problem_column": problem_key,
        "solution_column": solution_key,
        "path": str(out_path),
        "stats": stats,
        "note": (
            "targets 是数据集原始解答，仅作参照；generate_replay_targets 会用 base "
            "自蒸馏覆盖它，真正进入 C 的是 prompt + base 自己生成的解答。"
        ),
    }
    manifest_path = out_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
