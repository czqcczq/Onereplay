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
  * **"Python" is the conjunction of two readings, not `lang`.** `lang` records
    the language of the SEED snippet, and OSS-Instruct is documented to emit a
    different language than its seed (paper 2.2), so `lang == python` admits
    rows whose solution is Go. Requiring a ```python fence too costs some real
    Python rows and keeps the pool honest. --python_rule can relax this.
  * **Decontamination is not optional here.** The authors decontaminated with
    string matching and caught 9 rows. HumanEval/MBPP are this arm's only
    benchmarks, so a survivor turns the headline number into a memorization
    measurement. Word-level 8-grams catch reformatted copies that substring
    matching misses; the signatures and hashing come from
    prepare_magicoder_ccode so the two pools mean the same thing by "clean".
  * **There is no correctness filter, because there is nothing to check
    against.** Medical keeps only traces whose answer matches the gold letter.
    Magicoder rows carry no tests, so the strongest available substitute is
    applied instead: every kept row must contain a Python block that actually
    parses. That catches truncated and malformed generations, not wrong ones,
    and the gap is recorded in the stats rather than papered over. If the arm
    needs a real correctness filter later, OpenCoder's educational_instruct
    ships `testcase` and can be executed.

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
        "--python_rule",
        type=str,
        default="both",
        choices=("both", "lang", "fence"),
        help="both = lang==python AND a ```python fence (default; see module docstring).",
    )
    parser.add_argument(
        "--require_parsable",
        type=int,
        default=1,
        help="Drop rows whose Python block does not ast.parse. The stand-in for "
        "the correctness filter Medical gets and this corpus cannot support.",
    )
    parser.add_argument("--humaneval_data_file", type=str, default="")
    parser.add_argument("--mbpp_dataset_path", type=str, default="")
    parser.add_argument("--mbpp_split", type=str, default="test")
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


def python_blocks(text: str) -> list[str]:
    """Every Python-tagged fenced block, found by scanning lines for fence pairs.

    Deliberately not a regex. Fences come in pairs, and a ```` ```(.*?)``` ````
    pattern pairs the CLOSING fence of one block with the OPENING fence of the
    next, so it returns the gap between two blocks instead of either block.
    Magicoder solutions routinely put a shell command or a usage example beside
    the implementation, which makes that mispairing the common case rather than
    an edge one -- a regex here silently drops real Python and the rows get
    thrown out as unparsable.
    """

    blocks: list[str] = []
    current: list[str] | None = None
    lang = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if current is None:
                lang = stripped[3:].strip().lower()
                current = []
            else:
                if lang in PYTHON_FENCE_LANGS:
                    blocks.append("\n".join(current))
                current = None
                lang = ""
            continue
        if current is not None:
            current.append(line)
    # An unclosed trailing block means a truncated generation. Keep it if it is
    # tagged Python and let ast.parse rule on whether enough of it survived.
    if current is not None and lang in PYTHON_FENCE_LANGS:
        blocks.append("\n".join(current))
    return [block for block in blocks if block.strip()]


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


def build_eval_ngrams(args: argparse.Namespace) -> tuple[set[int], dict[str, int]]:
    """Union of HumanEval/MBPP n-gram signatures, plus how many each side gave."""

    grams: set[int] = set()
    counts = {"humaneval_signatures": 0, "mbpp_signatures": 0}

    if args.humaneval_data_file and Path(args.humaneval_data_file).is_file():
        docstrings = humaneval_docstrings(args.humaneval_data_file, args.cache_dir)
        counts["humaneval_signatures"] = len(docstrings)
        for text in docstrings:
            grams |= ngram_hashes(text)
    else:
        print(f"!! 没查 HumanEval 污染：--humaneval_data_file={args.humaneval_data_file!r}")

    if args.mbpp_dataset_path and Path(args.mbpp_dataset_path).exists():
        texts = mbpp_texts(args.mbpp_dataset_path, args.mbpp_split)
        counts["mbpp_signatures"] = len(texts)
        for text in texts:
            grams |= ngram_hashes(text)
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
    if args.python_rule in ("both", "lang") and not has_lang:
        raise SystemExit(f"--python_rule={args.python_rule} 需要 lang 列，但只有 {dataset.column_names}")

    log = FilterLog()
    usable: list[dict[str, str]] = []
    seen: set[str] = set()

    # Steps 1-3 in one pass: language, then emptiness/parsability, then the
    # duplicate check. Decontamination needs the eval signatures and runs after.
    for index in range(total_rows):
        row = dataset[index]
        problem = str(row.get(problem_key) or "").strip()
        solution = str(row.get(solution_key) or "").strip()
        lang = str(row.get("lang") or "").strip().lower() if has_lang else ""

        lang_ok = lang == "python"
        fence_ok = "```python" in solution.lower()
        if args.python_rule == "both":
            python_ok = lang_ok and fence_ok
        elif args.python_rule == "lang":
            python_ok = lang_ok
        else:
            python_ok = fence_ok
        if not python_ok:
            log.bump("dropped_not_python")
            continue

        if not problem:
            log.bump("dropped_empty_problem")
            continue
        if not solution:
            log.bump("dropped_empty_solution")
            continue
        if args.require_parsable and not has_parsable_python(solution):
            log.bump("dropped_unparsable_python")
            continue

        key = normalize(problem)
        if key in seen:
            log.bump("dropped_duplicate")
            continue
        seen.add(key)

        usable.append(
            {
                "source": DATASET,
                "question": problem,
                "response": solution,
                "original_id": f"magicoder-{args.split}-{index}",
            }
        )

    print(f"{len(usable)} rows usable before decontamination")

    eval_grams, sig_counts = build_eval_ngrams(args)
    if eval_grams:
        clean: list[dict[str, str]] = []
        hits: list[str] = []
        for item in usable:
            combined = f"{item['question']}\n{item['response']}"
            if ngram_hashes(combined) & eval_grams:
                hits.append(item["question"][:100])
                continue
            clean.append(item)
        log.bump("dropped_contaminated", len(hits))
        print(
            f"污染自检 (HumanEval+MBPP, {NGRAM}-gram): "
            + ("无重叠" if not hits else f"命中 {len(hits)} 条，已剔除")
        )
        for sample in hits[:5]:
            print(f"    [命中] {sample}")
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
        "ngram": NGRAM,
        "eval_sets": ["openai_humaneval", f"mbpp/{args.mbpp_split}"],
        "hits_removed": contaminated_hits,
        **sig_counts,
    }
    stats["sampling"] = {
        "method": "uniform (single-source corpus, no stratum to balance)",
        "seed": args.seed,
        "target_rows": args.target_rows,
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
            "python_rule": args.python_rule,
            "require_parsable": bool(args.require_parsable),
            # Stated positively so a later reader does not assume this pool was
            # verified the way the Medical one was.
            "correctness_filter": (
                "none -- Magicoder ships no tests; ast-parsability is the stand-in"
            ),
            "dedup": "exact match on the whitespace-normalized problem",
            "decontamination": f"word-level {NGRAM}-gram vs HumanEval + MBPP",
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
