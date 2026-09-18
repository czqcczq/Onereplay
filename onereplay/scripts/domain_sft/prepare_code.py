"""Build the Code specialist pool from Magicoder-OSS-Instruct-75K (Python only).

The third arm. It replaced Finance, which was abandoned after ODA-Fin-SFT-318k
turned out to carry programming exercises, grammar drills and multilingual
entity extraction under a "finance" label -- the training signal did not match
FinQA/ConvFinQA/TAT-QA, and no filter rescues a corpus whose domain boundary is
that porous. Code does not have that problem: HumanEval and MBPP ask for exactly
what the corpus teaches.

Four things here differ from the other two domains, and each is forced:

  * **The system turn is the code one.** A code answer cannot go inside
    \\boxed{}; the graders in code_exec.py extract a markdown fence. Both
    strings share their first clause verbatim, so the per-domain difference is
    one subordinate clause -- see 90_specialist_sft.pbs.
  * **All languages by default, not Python only.** The benchmarks are Python,
    which makes filtering to Python look obvious, and the paper's own ablation
    (Table 5, CodeLlama-Python-7B) says otherwise:

        finetuning data      HumanEval+   MultiPL-E
        none (base)               34.1        29.6
        Python only (43K)         47.6        32.7
        non-Python only (32K)     44.5        38.3
        both (75K)                55.5        37.8

    Training on non-Python data alone lifts *Python* pass@1 by more than ten
    points, and the full mix beats Python-only on Python by another eight. The
    non-Python half is not dilution; dropping it would cost accuracy on the only
    benchmarks this arm reports. It also doubles the pool, which narrows the
    token gap against the other two domains.

    --lang_filter can still restrict to Python. If it does, note that the paper
    classifies by whether ```python appears, explicitly *not* by the `lang`
    column, "because LLMs performing OSS-Instruct may produce code in a
    different programming language than the seed" -- `lang == python` admits
    rows whose solution is Go.
  * **Decontamination is not optional here, but it has to be tuned.**
    HumanEval/MBPP are this arm's only benchmarks, so a surviving copy turns the
    headline number into a memorization measurement. The first attempt reused
    prepare_magicoder_ccode's 8-word window and a single-hit rule, and flagged
    3099 of 37757 rows -- against the 9 the OSS-Instruct authors found with
    string matching. The diagnostic said why: two phrases accounted for 86% of
    it, "your task is to implement a function that" (1431 rows) and "your task
    is to write a function that" (1245 rows), both lifted from HumanEval
    docstrings. Those are how GPT-3.5 opens a programming problem, and both
    corpora are full of them; the split by source confirms it, with HumanEval
    docstrings flagging 7.97% and MBPP's terser statements only 0.41%.

    So: a 13-word window, and at least two distinct windows must match. A real
    copy shares whole sentences and trips many; a shared turn of phrase trips
    one. prepare_magicoder_ccode keeps n=8 -- changing it there would silently
    alter a pool earlier experiments were built on.
  * **There is no correctness filter, because there is nothing to check
    against.** Medical keeps only traces whose answer matches the gold letter.
    Magicoder rows carry no tests, so the strongest available substitute is
    applied instead: every kept row must contain a code block, and where that
    block is Python it must actually parse. Non-Python rows only get the weaker
    check -- there is no ast module for Rust here -- so the guarantee is
    deliberately uneven, and the stats say so rather than papering over it. Both
    checks catch truncated and malformed generations, neither catches wrong
    ones. If the arm needs a real correctness filter later, OpenCoder's
    educational_instruct ships `testcase` and can be executed.

    python -m onereplay.scripts.domain_sft.prepare_code \\
        --magicoder_path datasets/code_replay/magicoder_oss_instruct_75k.parquet \\
        --humaneval_data_file datasets/code/humaneval_test.parquet \\
        --mbpp_dataset_path datasets/code/mbpp_full --mbpp_split test \\
        --tokenizer_path models/Qwen3-4B-Base \\
        --system_prompt '...' --max_tokens 4096 --seed 42
"""

from __future__ import annotations

import argparse
import ast
import random
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from onereplay.scripts.domain_sft.common import (  # noqa: E402
    FilterLog,
    budget_scan,
    build_metadata,
    build_record,
    load_tokenizer,
    serialize_many,
    summarize,
    write_dataset_dir,
    write_jsonl,
    write_stats,
)
from onereplay.scripts.prepare_magicoder_ccode import (  # noqa: E402
    NGRAM,
    humaneval_docstrings,
    mbpp_texts,
    ngram_hashes,
    normalize,
)

DOMAIN = "code"
DATASET = "ise-uiuc/Magicoder-OSS-Instruct-75K"

# Not the 8 that prepare_magicoder_ccode uses; see the docstring. 13 words is
# long enough that the stock problem-statement openers stop matching, and two
# distinct windows are required so one shared phrase is not a verdict.
CODE_NGRAM = 13
CODE_MIN_HITS = 2

# The languages code_exec._extract_code_block treats as Python, so a row that
# trains here is one whose shape the grader can read back at eval time. The
# empty tag is in the set there and so is in it here; ast.parse sorts out the
# untagged blocks that hold shell commands.
PYTHON_FENCE_LANGS = frozenset({"", "python", "py", "python3"})

CODE_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final code in a Python code block."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Code SFT pool from Magicoder-OSS-Instruct-75K under the shared rules."
    )
    parser.add_argument("--magicoder_path", type=str, default="")
    parser.add_argument("--magicoder_repo", type=str, default=DATASET)
    parser.add_argument("--dataset_revision", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, default="data/processed/code_train.jsonl")
    parser.add_argument("--out_stats", type=str, default="data/stats/code_stats.json")
    parser.add_argument("--out_dir", type=str, default="data/processed/code_train")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--lang_filter",
        type=str,
        default="all",
        choices=("all", "python", "python_strict"),
        help="all = every language (default; the paper's best Python score, see "
        "module docstring). python = rows containing a ```python fence, which is "
        "the paper's own criterion. python_strict = that AND lang==python.",
    )
    parser.add_argument(
        "--require_parsable",
        type=int,
        default=1,
        help="Drop rows with no usable code block. Python blocks must ast.parse; "
        "other languages only have to be non-empty. The stand-in for the "
        "correctness filter Medical gets and this corpus cannot support.",
    )
    parser.add_argument("--humaneval_data_file", type=str, default="")
    parser.add_argument("--mbpp_dataset_path", type=str, default="")
    parser.add_argument("--mbpp_split", type=str, default="test")
    parser.add_argument(
        "--decontam_ngram",
        type=int,
        default=CODE_NGRAM,
        help=f"Word-window length for the overlap check (default {CODE_NGRAM}).",
    )
    parser.add_argument(
        "--decontam_min_hits",
        type=int,
        default=CODE_MIN_HITS,
        help="How many DISTINCT windows must match before a row is called "
        f"contaminated (default {CODE_MIN_HITS}). 1 reproduces the old rule, "
        "which flagged 8% of the corpus on shared phrasing alone.",
    )
    parser.add_argument(
        "--require_decontamination",
        type=int,
        default=1,
        help="Refuse to build the pool if neither eval set was given. The default "
        "is on because a silently un-decontaminated Code pool invalidates the "
        "only two benchmarks this arm reports.",
    )
    parser.add_argument(
        "--target_rows",
        type=int,
        default=0,
        help="0 keeps every surviving row, which is the intent: take what the "
        "filters leave rather than padding back to a round number.",
    )
    parser.add_argument("--system_prompt", type=str, default=CODE_SYSTEM_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--max_rows", type=int, default=0, help="Debug aid; 0 reads all.")
    return parser.parse_args()


def load_magicoder(args: argparse.Namespace):
    """Load Magicoder from a local parquet/save_to_disk copy, or the Hub."""

    from datasets import load_dataset, load_from_disk

    if args.magicoder_path:
        path = Path(args.magicoder_path)
        if path.is_file():
            return load_dataset(
                "parquet",
                data_files=str(path),
                split="train",
                cache_dir=args.cache_dir or None,
            )
        if (path / "dataset_dict.json").exists():
            return load_from_disk(str(path))[args.split]
        if (path / "dataset_info.json").exists():
            return load_from_disk(str(path))
        shards = sorted(str(item) for item in path.glob("**/*.parquet"))
        if not shards:
            raise SystemExit(f"no parquet under {path}")
        return load_dataset(
            "parquet", data_files=shards, split="train", cache_dir=args.cache_dir or None
        )
    return load_dataset(args.magicoder_repo, split=args.split, cache_dir=args.cache_dir or None)


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    """(language tag, body) for every fenced block, by scanning for fence pairs.

    Deliberately not a regex. Fences come in pairs, and a ```` ```(.*?)``` ````
    pattern pairs the CLOSING fence of one block with the OPENING fence of the
    next, so it returns the gap between two blocks instead of either block.
    Magicoder solutions routinely put a shell command or a usage example beside
    the implementation, which makes that mispairing the common case rather than
    an edge one -- a regex here silently drops real code and the rows then get
    thrown out as having none.
    """

    blocks: list[tuple[str, str]] = []
    current: list[str] | None = None
    lang = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if current is None:
                lang = stripped[3:].strip().lower()
                current = []
            else:
                blocks.append((lang, "\n".join(current)))
                current = None
                lang = ""
            continue
        if current is not None:
            current.append(line)
    # An unclosed trailing block means a truncated generation. Keep it and let
    # the language-specific check rule on whether enough of it survived.
    if current is not None:
        blocks.append((lang, "\n".join(current)))
    return [(lang, body) for lang, body in blocks if body.strip()]


def python_blocks(text: str) -> list[str]:
    """Just the Python-tagged block bodies."""

    return [body for lang, body in fenced_blocks(text) if lang in PYTHON_FENCE_LANGS]


def has_parsable_python(text: str) -> bool:
    """True when at least one fenced block is syntactically valid Python.

    Any block, not all of them: solutions routinely show a usage example or a
    shell command alongside the implementation, and those are not required to
    parse. One parsable block means there is real code to learn from.
    """

    for block in python_blocks(text):
        try:
            ast.parse(block)
        except (SyntaxError, ValueError):
            continue
        return True
    return False


def has_usable_code(text: str) -> bool:
    """The quality gate, applied at whatever strength the language allows.

    Python rows must parse -- that is the whole value of the check, and Python
    is over half the corpus. Rows in other languages get only "has a non-empty
    code block", because there is no ast module for Rust or Swift sitting in the
    standard library and pulling in seven parsers to reject a handful of rows is
    not worth it.

    The asymmetry is real and is recorded in the stats. It means the non-Python
    half is filtered more loosely than the Python half, not that it is unfiltered.
    """

    blocks = fenced_blocks(text)
    if not blocks:
        return False
    if any(lang in PYTHON_FENCE_LANGS for lang, _ in blocks):
        return has_parsable_python(text)
    return True


def build_eval_ngrams(args: argparse.Namespace) -> tuple[set[int], dict[str, int]]:
    """Union of HumanEval/MBPP n-gram signatures, plus how many each side gave."""

    grams: set[int] = set()
    counts = {"humaneval_signatures": 0, "mbpp_signatures": 0}

    if args.humaneval_data_file and Path(args.humaneval_data_file).is_file():
        docstrings = humaneval_docstrings(args.humaneval_data_file, args.cache_dir)
        counts["humaneval_signatures"] = len(docstrings)
        for text in docstrings:
            grams |= ngram_hashes(text, args.decontam_ngram)
    else:
        print(f"!! 没查 HumanEval 污染：--humaneval_data_file={args.humaneval_data_file!r}")

    if args.mbpp_dataset_path and Path(args.mbpp_dataset_path).exists():
        texts = mbpp_texts(args.mbpp_dataset_path, args.mbpp_split)
        counts["mbpp_signatures"] = len(texts)
        for text in texts:
            grams |= ngram_hashes(text, args.decontam_ngram)
    else:
        print(f"!! 没查 MBPP 污染：--mbpp_dataset_path={args.mbpp_dataset_path!r}")

    if not grams and args.require_decontamination:
        raise SystemExit(
            "两个评测集都没给，去污染没法做。\n"
            "  先跑 python -m onereplay.scripts.download_code_data，然后传\n"
            "  --humaneval_data_file datasets/code/humaneval_test.parquet\n"
            "  --mbpp_dataset_path datasets/code/mbpp_full\n"
            "  真要跳过（不建议，HumanEval/MBPP 是这一臂仅有的两个 benchmark）：\n"
            "  --require_decontamination 0"
        )
    return grams, counts


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path, args.system_prompt)
    dataset = load_magicoder(args)
    total_rows = len(dataset) if not args.max_rows else min(args.max_rows, len(dataset))
    print(f"loaded {len(dataset)} rows from {args.magicoder_path or args.magicoder_repo}")
    print(f"columns: {list(dataset.column_names)}")

    problem_key = "problem" if "problem" in dataset.column_names else "instruction"
    solution_key = "solution" if "solution" in dataset.column_names else "response"
    has_lang = "lang" in dataset.column_names
    if args.lang_filter == "python_strict" and not has_lang:
        raise SystemExit(f"--lang_filter=python_strict 需要 lang 列，但只有 {dataset.column_names}")

    log = FilterLog()
    usable: list[dict[str, str]] = []
    seen: set[str] = set()
    # Recorded, not enforced: the Python share of the final pool is the number
    # Table 5 turns on, so it belongs in the stats even when nothing filters on it.
    python_rows = 0

    # Steps 1-3 in one pass: language, then emptiness/usable code, then the
    # duplicate check. Decontamination needs the eval signatures and runs after.
    for index in range(total_rows):
        row = dataset[index]
        problem = str(row.get(problem_key) or "").strip()
        solution = str(row.get(solution_key) or "").strip()
        seed_lang = str(row.get("lang") or "").strip().lower() if has_lang else ""

        fence_ok = "```python" in solution.lower()
        if args.lang_filter == "python_strict":
            keep_lang = fence_ok and seed_lang == "python"
        elif args.lang_filter == "python":
            keep_lang = fence_ok
        else:
            keep_lang = True
        if not keep_lang:
            log.bump("dropped_by_lang_filter")
            continue

        if not problem:
            log.bump("dropped_empty_problem")
            continue
        if not solution:
            log.bump("dropped_empty_solution")
            continue
        if args.require_parsable and not has_usable_code(solution):
            log.bump("dropped_no_usable_code")
            continue

        key = normalize(problem)
        if key in seen:
            log.bump("dropped_duplicate")
            continue
        seen.add(key)

        python_rows += int(fence_ok)
        usable.append(
            {
                "source": DATASET,
                "question": problem,
                "response": solution,
                "seed_lang": seed_lang,
                "is_python": "1" if fence_ok else "0",
                "original_id": f"magicoder-{args.split}-{index}",
            }
        )

    print(f"{len(usable)} rows usable before decontamination")

    eval_grams, sig_counts = build_eval_ngrams(args)
    near_misses = 0
    if eval_grams:
        clean: list[dict[str, str]] = []
        hits: list[tuple[int, str]] = []
        for item in usable:
            combined = f"{item['question']}\n{item['response']}"
            overlap = ngram_hashes(combined, args.decontam_ngram) & eval_grams
            if len(overlap) >= args.decontam_min_hits:
                hits.append((len(overlap), item["question"][:100]))
                continue
            # Matched, but on too few windows to call it a copy. Counted rather
            # than ignored: if this number is large the threshold is hiding
            # something and belongs in the write-up.
            near_misses += int(bool(overlap))
            clean.append(item)
        log.bump("dropped_contaminated", len(hits))
        print(
            f"污染自检 (HumanEval+MBPP, {args.decontam_ngram}-gram, "
            f">={args.decontam_min_hits} 个窗口): "
            + ("无重叠" if not hits else f"命中 {len(hits)} 条，已剔除")
        )
        print(f"    另有 {near_misses} 条只撞上 <{args.decontam_min_hits} 个窗口，判为措辞雷同，保留")
        for count, sample in sorted(hits, reverse=True)[:5]:
            print(f"    [命中 {count} 个窗口] {sample}")
        usable = clean
    contaminated_hits = log.as_dict().get("dropped_contaminated", 0)

    print(f"{len(usable)} rows survive decontamination; measuring lengths...")
    measured = serialize_many(
        tokenizer,
        [(item["question"], item["response"]) for item in usable],
        batch_size=args.batch_size,
        progress_every=20000,
    )
    scan_totals = [item.total_tokens for item in measured]
    scan_responses = [item.response_tokens for item in measured]
    prefix_violations = sum(1 for item in measured if not item.prompt_is_prefix)

    fitting: list[dict[str, Any]] = []
    for item, serialized in zip(usable, measured):
        if serialized.total_tokens > args.max_tokens:
            log.bump("dropped_over_max_tokens")
            continue
        item["serialized"] = serialized
        fitting.append(item)
    print(f"{len(fitting)} rows fit within {args.max_tokens} tokens")

    # Magicoder rows carry no source column -- every row comes from the one
    # corpus -- so there is no stratum to balance and a uniform draw is correct.
    if args.target_rows and args.target_rows < len(fitting):
        rng = random.Random(args.seed)
        chosen = sorted(rng.sample(range(len(fitting)), args.target_rows))
        log.bump("dropped_sampling", len(fitting) - args.target_rows)
    else:
        chosen = list(range(len(fitting)))

    records = [
        build_record(
            domain=DOMAIN,
            index=position,
            source=fitting[index]["source"],
            question=fitting[index]["question"],
            response=fitting[index]["response"],
            serialized=fitting[index]["serialized"],
            original_id=fitting[index]["original_id"],
        )
        for position, index in enumerate(chosen)
    ]

    print("\nfilter tally:")
    print(log.report())
    if prefix_violations:
        print(
            f"\nWARNING: {prefix_violations} rows failed the prompt-prefix check; "
            "label masking would be misaligned for those rows."
        )

    write_jsonl(args.out_jsonl, records)
    print(f"\nwrote {len(records)} rows to {args.out_jsonl}")
    if args.out_dir:
        write_dataset_dir(args.out_dir, records)
        print(f"wrote trainable dataset to {args.out_dir}")

    stats = summarize(records)
    stats["budget_scan"] = budget_scan(scan_totals, scan_responses)
    stats["counts"] = log.as_dict()
    stats["decontamination"] = {
        "ngram": args.decontam_ngram,
        "min_distinct_hits": args.decontam_min_hits,
        "eval_sets": ["openai_humaneval", f"mbpp/{args.mbpp_split}"],
        "hits_removed": contaminated_hits,
        "kept_single_window_matches": near_misses,
        "note": (
            "n=13 / >=2 windows, not prepare_magicoder_ccode's n=8 / >=1: that "
            "rule flagged 8% of the corpus, 86% of it on two stock problem-"
            "statement openers taken from HumanEval docstrings"
        ),
        **sig_counts,
    }
    stats["sampling"] = {
        "method": "uniform (single-source corpus, no stratum to balance)",
        "seed": args.seed,
        "target_rows": args.target_rows,
    }
    # The mix Table 5 is about. Roughly 57% Python is the paper's own ratio and
    # the one that scored best on HumanEval+; a large drift from it means the
    # filters hit one language group much harder than the other.
    final_python = sum(1 for index in chosen if fitting[index]["is_python"] == "1")
    stats["language_mix"] = {
        "lang_filter": args.lang_filter,
        "python_rows": final_python,
        "non_python_rows": len(records) - final_python,
        "python_share": round(final_python / max(len(records), 1), 4),
        "paper_reference": "OSS-Instruct 75K is ~43K Python / ~32K non-Python (~57% Python)",
        "note": (
            "non-Python rows are kept on purpose: paper Table 5 has both (75K) at "
            "55.5 HumanEval+ vs Python-only (43K) at 47.6"
        ),
    }
    stats["metadata"] = build_metadata(
        domain=DOMAIN,
        dataset_name=DATASET,
        dataset_revision=args.dataset_revision,
        tokenizer_path=args.tokenizer_path,
        max_tokens=args.max_tokens,
        seed=args.seed,
        raw_count=total_rows,
        filtered_count=len(fitting),
        final_count=len(records),
        filter_rules={
            "lang_filter": args.lang_filter,
            "require_parsable": bool(args.require_parsable),
            # Stated positively so a later reader does not assume this pool was
            # verified the way the Medical one was.
            "correctness_filter": (
                "none -- Magicoder ships no tests; code-block presence is the "
                "stand-in, strengthened to ast.parse for Python rows only"
            ),
            "dedup": "exact match on the whitespace-normalized problem",
            "decontamination": (
                f"word-level {args.decontam_ngram}-gram vs HumanEval + MBPP, "
                f">={args.decontam_min_hits} distinct windows"
            ),
            "target_rows": args.target_rows,
        },
        prompt_prefix_violations=prefix_violations,
        system_prompt=args.system_prompt,
    )
    write_stats(args.out_stats, stats)
    print(f"wrote stats to {args.out_stats}")
    print(
        f"\nnum_examples={stats['num_examples']} "
        f"total_tokens={stats['total_tokens']} "
        f"assistant_tokens={stats['total_assistant_tokens']}"
    )


if __name__ == "__main__":
    main()
