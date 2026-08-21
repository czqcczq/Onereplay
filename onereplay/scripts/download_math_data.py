"""One-shot download + normalize of all MATH assets on a networked (login) node.

Run this ON THE RWTH LOGIN NODE (compute nodes are offline). It downloads every
dataset the Q2 (math retention) pipeline needs and writes them, with the exact
file names / columns the slurm scripts expect, into --out_dir:

  math_train_inputs_targets.jsonl : MATH-lighteval train -> {inputs, targets}
        self-distillation corpus for C_math. `targets` is a placeholder gold that
        generate_replay_targets.py overwrites with the base model's own answer.
  math500_test.jsonl : HuggingFaceH4/MATH-500 -> {problem, answer, solution}   (math500 metric)
  amc_test.jsonl     : AI-MO/aimo-validation-amc -> {problem, answer}          (math500 metric)
  aime_test.jsonl    : AIME 2024 + 2025 merged -> {problem, answer}            (aime metric)
  gsm8k_test.jsonl   : GSM8K test -> {question, answer}                        (gsm8k metric, optional)

Each download is isolated: if one repo fails (renamed / gated / offline), the
others still complete and the failure is reported at the end. Do NOT set the
HF_*_OFFLINE flags when running this.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from onereplay.scripts.prepare_math_data import (
    ANSWER_KEYS,
    PROBLEM_KEYS,
    SOLUTION_KEYS,
    pick,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse output dir, repo ids/splits, and which assets to fetch."""

    parser = argparse.ArgumentParser(description="Download + normalize MATH assets (login node).")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/hpcwork/xsz96350/Chen_logs/onereplay/datasets/math",
    )
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--math_train_repo", type=str, default="DigitalLearningGmbH/MATH-lighteval")
    parser.add_argument("--math_train_split", type=str, default="train")
    parser.add_argument("--math500_repo", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--math500_split", type=str, default="test")
    parser.add_argument("--amc_repo", type=str, default="AI-MO/aimo-validation-amc")
    parser.add_argument("--amc_split", type=str, default="train")
    parser.add_argument(
        "--aime_repos",
        type=str,
        default="Maxwell-Jia/AIME_2024,yentinglin/aime_2025",
        help="Comma-separated AIME repos (merged into aime_test.jsonl).",
    )
    parser.add_argument("--aime_split", type=str, default="train")
    parser.add_argument("--gsm8k_repo", type=str, default="openai/gsm8k")
    parser.add_argument("--gsm8k_config", type=str, default="main")
    parser.add_argument("--gsm8k_split", type=str, default="test")
    parser.add_argument("--with_math_train", type=int, default=1)
    parser.add_argument("--with_math500", type=int, default=1)
    parser.add_argument("--with_amc", type=int, default=1)
    parser.add_argument("--with_aime", type=int, default=1)
    parser.add_argument("--with_gsm8k", type=int, default=1)
    return parser.parse_args()


def warn_if_offline() -> None:
    """Fail fast if HF offline flags are set; this script needs network."""

    offline = [
        name
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
        if os.environ.get(name) == "1"
    ]
    if offline:
        raise SystemExit(
            f"{', '.join(offline)} set to 1; unset them and run on a networked node."
        )


def load_split(repo: str, split: str, config: str | None, cache_dir: str):
    """Load one dataset split, tolerating datasets versions w/o trust_remote_code."""

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if config:
        kwargs["name"] = config
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        return load_dataset(repo, trust_remote_code=True, **kwargs)
    except TypeError:
        return load_dataset(repo, **kwargs)


def build_math_train(args, out_dir, cache_dir) -> None:
    """MATH-lighteval train -> {inputs, targets} self-distillation corpus."""

    print(f"[math-train] {args.math_train_repo}[{args.math_train_split}]")
    dataset = load_split(args.math_train_repo, args.math_train_split, None, cache_dir)
    rows = []
    for record in dataset:
        problem = pick(record, PROBLEM_KEYS)
        solution = pick(record, SOLUTION_KEYS)
        if problem and solution:
            rows.append({"inputs": problem, "targets": solution})
    write_jsonl(rows, out_dir / "math_train_inputs_targets.jsonl")


def build_math500(args, out_dir, cache_dir) -> None:
    """MATH-500 -> {problem, answer, solution}."""

    print(f"[math-500] {args.math500_repo}[{args.math500_split}]")
    dataset = load_split(args.math500_repo, args.math500_split, None, cache_dir)
    rows = []
    for record in dataset:
        problem = pick(record, PROBLEM_KEYS)
        answer = pick(record, ANSWER_KEYS)
        solution = pick(record, SOLUTION_KEYS)
        if problem and (answer or solution):
            rows.append({"problem": problem, "answer": answer, "solution": solution})
    write_jsonl(rows, out_dir / "math500_test.jsonl")


def build_amc(args, out_dir, cache_dir) -> None:
    """AMC -> {problem, answer}."""

    print(f"[amc] {args.amc_repo}[{args.amc_split}]")
    dataset = load_split(args.amc_repo, args.amc_split, None, cache_dir)
    rows = []
    for record in dataset:
        problem = pick(record, PROBLEM_KEYS)
        answer = pick(record, ANSWER_KEYS) or pick(record, SOLUTION_KEYS)
        if problem and answer:
            rows.append({"problem": problem, "answer": answer})
    write_jsonl(rows, out_dir / "amc_test.jsonl")


def build_aime(args, out_dir, cache_dir) -> None:
    """AIME (several repos) -> merged {problem, answer}."""

    repos = [r.strip() for r in args.aime_repos.split(",") if r.strip()]
    rows = []
    for repo in repos:
        print(f"[aime] {repo}[{args.aime_split}]")
        dataset = load_split(repo, args.aime_split, None, cache_dir)
        for record in dataset:
            problem = pick(record, PROBLEM_KEYS)
            answer = pick(record, ANSWER_KEYS) or pick(record, SOLUTION_KEYS)
            if problem and answer:
                rows.append({"problem": problem, "answer": answer})
    write_jsonl(rows, out_dir / "aime_test.jsonl")


def build_gsm8k(args, out_dir, cache_dir) -> None:
    """GSM8K test -> {question, answer}."""

    print(f"[gsm8k] {args.gsm8k_repo}:{args.gsm8k_config}[{args.gsm8k_split}]")
    dataset = load_split(args.gsm8k_repo, args.gsm8k_split, args.gsm8k_config, cache_dir)
    rows = []
    for record in dataset:
        question = pick(record, ("question", "Question", "problem"))
        answer = pick(record, ANSWER_KEYS)
        if question and answer:
            rows.append({"question": question, "answer": answer})
    write_jsonl(rows, out_dir / "gsm8k_test.jsonl")


def main() -> None:
    """Download every requested asset; keep going if one repo fails."""

    args = parse_args()
    warn_if_offline()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir

    jobs = [
        ("math_train", args.with_math_train, build_math_train),
        ("math500", args.with_math500, build_math500),
        ("amc", args.with_amc, build_amc),
        ("aime", args.with_aime, build_aime),
        ("gsm8k", args.with_gsm8k, build_gsm8k),
    ]
    failures: list[str] = []
    for label, enabled, builder in jobs:
        if not enabled:
            continue
        try:
            builder(args, out_dir, cache_dir)
        except Exception as error:  # noqa: BLE001
            print(f"[WARN] {label} failed: {error}")
            failures.append(f"{label}: {error}")

    if failures:
        print("\n==== some downloads failed ====")
        for line in failures:
            print("  - " + line)
        print(
            "对失败项可单独重试并覆盖 repo，例如: "
            "python -m onereplay.scripts.download_math_data --with_math_train 0 "
            "--with_math500 0 --with_amc 0 --with_gsm8k 0 --aime_repos <repo1,repo2>"
        )
    else:
        print("\ndone. all assets ready in " + str(out_dir))


if __name__ == "__main__":
    main()
