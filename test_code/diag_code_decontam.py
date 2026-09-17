"""Show WHICH 8-gram caused each decontamination hit, so the hits can be judged.

The probe flagged 3099 of 34658 Magicoder rows (8.9%) as overlapping HumanEval
or MBPP. The OSS-Instruct authors' own string-matching filter caught 9. A gap
that large is either a real finding or a broken filter, and the sampled hits
lean toward broken: "robotic arm kinematics" and "Harris corner detection" are
not HumanEval or MBPP tasks.

The filter reports set intersection, which throws away the evidence -- you get
"these hashes collided" and no way to see the text. This rebuilds the map from
hash back to phrase, so each hit can be read and judged.

The suspicion worth testing: mbpp_texts contributes the task description of
every MBPP row, and those descriptions are formulaic ("Write a function to find
the maximum..."). An 8-word window over that is a phrase any Python tutorial
would contain, so it matches on style rather than on copying.

    python test_code/diag_code_decontam.py \\
        --magicoder_path datasets/code_replay/magicoder_oss_instruct_75k.parquet \\
        --humaneval_data_file datasets/code/humaneval_test.parquet \\
        --mbpp_dataset_path datasets/code/mbpp_full
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from onereplay.scripts.prepare_magicoder_ccode import (  # noqa: E402
    NGRAM,
    humaneval_docstrings,
    mbpp_texts,
    normalize,
)


def ngram_texts(text: str, n: int = NGRAM) -> list[str]:
    """The phrases behind ngram_hashes, same windows, kept as text."""

    words = normalize(text).split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain the code decontamination hits.")
    parser.add_argument(
        "--magicoder_path",
        type=str,
        default="datasets/code_replay/magicoder_oss_instruct_75k.parquet",
    )
    parser.add_argument(
        "--humaneval_data_file", type=str, default="datasets/code/humaneval_test.parquet"
    )
    parser.add_argument("--mbpp_dataset_path", type=str, default="datasets/code/mbpp_full")
    parser.add_argument("--mbpp_split", type=str, default="test")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--show", type=int, default=12, help="How many top phrases to print.")
    parser.add_argument("--examples", type=int, default=4, help="Hit rows to print per phrase.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    print("读 Magicoder ...")
    dataset = load_dataset(
        "parquet",
        data_files=args.magicoder_path,
        split="train",
        cache_dir=args.cache_dir or None,
    )
    rows = [dict(dataset[i]) for i in range(len(dataset))]

    kept = [
        row
        for row in rows
        if str(row.get("lang", "")).strip().lower() == "python"
        and "```python" in str(row.get("solution", "")).lower()
        and str(row.get("problem", "")).strip()
        and str(row.get("solution", "")).strip()
    ]
    print(f"  Python 口径下 {len(kept)} 行")

    # phrase -> which eval item it came from, so a hit can be traced back.
    phrase_origin: dict[str, str] = {}
    he_count = mb_count = 0
    if Path(args.humaneval_data_file).is_file():
        for text in humaneval_docstrings(args.humaneval_data_file, args.cache_dir):
            he_count += 1
            for phrase in ngram_texts(text):
                phrase_origin.setdefault(phrase, f"HumanEval: {text.strip()[:60]}")
    if Path(args.mbpp_dataset_path).exists():
        for text in mbpp_texts(args.mbpp_dataset_path, args.mbpp_split):
            mb_count += 1
            for phrase in ngram_texts(text):
                phrase_origin.setdefault(phrase, f"MBPP: {text.strip()[:60]}")
    print(f"  签名 HumanEval {he_count} 条 / MBPP {mb_count} 条 -> {len(phrase_origin)} 个 8-gram")

    print("\n扫描命中 ...")
    phrase_hits: Counter[str] = Counter()
    phrase_rows: dict[str, list[str]] = {}
    hit_rows = 0
    for index, row in enumerate(kept):
        if index and index % 10000 == 0:
            print(f"  ...{index}")
        combined = f"{row['problem']}\n{row['solution']}"
        matched = {p for p in ngram_texts(combined) if p in phrase_origin}
        if not matched:
            continue
        hit_rows += 1
        for phrase in matched:
            phrase_hits[phrase] += 1
            phrase_rows.setdefault(phrase, []).append(str(row["problem"])[:150])

    print(f"\n命中 {hit_rows} / {len(kept)} 行 ({hit_rows / max(len(kept), 1):.1%})")
    print(f"被触发的不同 8-gram: {len(phrase_hits)}")

    print()
    print("=" * 78)
    print(" 触发最多的 8-gram —— 判断误杀就看这里")
    print("=" * 78)
    print("一条 8-gram 如果匹配了成百上千行，它描述的就是通用写法而不是某道题的独有措辞。")
    print()
    covered: set[int] = set()
    for phrase, count in phrase_hits.most_common(args.show):
        share = count / max(hit_rows, 1)
        print(f"[{count} 行, 占命中的 {share:.1%}] {phrase!r}")
        print(f"    来自 {phrase_origin[phrase]}")
        for sample in phrase_rows[phrase][: args.examples]:
            print(f"    · {sample}")
        print()

    # How much of the damage a few generic phrases account for: if dropping the
    # top handful recovers most of the rows, the filter is matching style.
    for top in (1, 3, 5, 10, 20):
        phrases = {phrase for phrase, _ in phrase_hits.most_common(top)}
        still: set[int] = set()
        for index, row in enumerate(kept):
            combined = f"{row['problem']}\n{row['solution']}"
            matched = {p for p in ngram_texts(combined) if p in phrase_origin}
            if matched and not matched <= phrases:
                still.add(index)
        recovered = hit_rows - len(still)
        print(
            f"  忽略最高频的 {top:>2} 条 8-gram -> 命中降到 {len(still)}，"
            f"救回 {recovered} 行 ({recovered / max(hit_rows, 1):.1%})"
        )
        covered = still

    print()
    print("=" * 78)
    print(" 只用 HumanEval docstring 的话（不含 MBPP 题面）")
    print("=" * 78)
    # The likeliest culprit, isolated: HumanEval docstrings are distinctive
    # prose, MBPP descriptions are a template.
    he_only = {p for p, origin in phrase_origin.items() if origin.startswith("HumanEval")}
    he_hits = sum(
        1
        for row in kept
        if {p for p in ngram_texts(f"{row['problem']}\n{row['solution']}")} & he_only
    )
    print(f"  命中 {he_hits} 行 ({he_hits / max(len(kept), 1):.2%})")
    mb_only = {p for p, origin in phrase_origin.items() if origin.startswith("MBPP")}
    mb_hits = sum(
        1
        for row in kept
        if {p for p in ngram_texts(f"{row['problem']}\n{row['solution']}")} & mb_only
    )
    print(f"  只用 MBPP 的话命中 {mb_hits} 行 ({mb_hits / max(len(kept), 1):.2%})")
    print()
    print("如果 MBPP 那边命中远多于 HumanEval，且高频 8-gram 读起来像通用说法，")
    print("那就是 MBPP 的模板化题面在误杀，应该只拿 HumanEval 或改用更长的窗口。")


if __name__ == "__main__":
    main()
