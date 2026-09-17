"""Dry-run the proposed ODA-Fin rules and print what each one would remove.

    python test_code/probe_fin_rules.py --fin_json datasets/raw/ODA-Fin-SFT-318k/train.json

Measures only; writes nothing. The point is to decide whether the correctness
rule is sound *before* it is wired into prepare_finance.py, and the way to
decide that is not the deletion count -- it is whether the rows it deletes
deserve to go. So the report ends with samples from every verdict bucket,
including the ones that were kept.

The rule set, matching the Medical pipeline's shape:

    1. instruction / output non-empty      (answer is provenance, not required)
    2. <think>/<answer> structure intact
    3. wrong answers are not kept          <-- the one being validated here
    4. English only
    5. dedup on normalized instruction
    6. total tokens <= 4096                (not applied here; needs a tokenizer,
                                            and the earlier run already put it
                                            at 2691 rows)
    7. no source above 15%                 (not applied here; it is a sampling
                                            step, not a filter)

Rule 3 splits three ways, because "compare the strings" deletes correct answers
on this corpus:

    numeric     both sides parse as numbers -> 1% relative tolerance, with
                scale retries for percent/thousand/million mismatches
    short text  gold is a label or entity   -> LaTeX-normalized equality
    long text   gold is a sentence          -> undecidable, kept and counted

Only the first two can produce a deletion. A row is never deleted for being
hard to judge.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onereplay.scripts.domain_sft.prepare_finance import (
    CJK_PATTERN,
    extract_boxed,
    iter_json_records,
    strip_tags,
)

# Scale retries: golds are quoted in whatever unit the filing used, while the
# trace usually boxes the raw figure. "$275 million" vs 275000000 and "11.1%"
# vs 0.111 are the same answer, and deleting them would gut the pool.
SCALES = (1.0, 100.0, 0.01, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)
SHORT_TEXT_WORDS = 6
NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
MAGNITUDE = (
    ("trillion", 1e12),
    ("billion", 1e9),
    ("million", 1e6),
    ("thousand", 1e3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run the ODA-Fin filter rules.")
    parser.add_argument("--fin_json", type=str, default="datasets/raw/ODA-Fin-SFT-318k/train.json")
    parser.add_argument("--rel_tol", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=12, help="samples printed per bucket")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=0, help="0 = whole file")
    return parser.parse_args()


def latex_clean(text: str) -> str:
    """Strip the LaTeX a boxed answer carries so it can be compared to a plain gold."""

    out = str(text or "")
    for _ in range(3):  # \text{\mathrm{x}} nests in practice
        out = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname|textbf)\s*\{([^{}]*)\}", r"\1", out)
    out = out.replace("\\%", "%").replace("\\$", "$")
    out = re.sub(r"\\[,;:!]", " ", out)
    out = out.replace("\\ ", " ").replace("\\left", "").replace("\\right", "")
    out = re.sub(r"\\+", "", out)
    return re.sub(r"\s+", " ", out).strip()


def norm_text(text: str) -> str:
    cleaned = latex_clean(text).lower()
    cleaned = re.sub(r"[_/\-]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9%. ]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" .")


# Function words only. "not" and "no" are deliberately absent: dropping them
# would make "positive" and "not positive" the same answer.
STOPWORDS = frozenset("a an the of or and in to for on at is was be as by with".split())


def token_set(text: str) -> frozenset[str]:
    return frozenset(word for word in norm_text(text).split() if word not in STOPWORDS)


# Everything allowed to surround the figure and still call it "a number":
# units, currency, hedges. Anything else means the digits are incidental to a
# sentence ("revenue grew 15% in 2019 because...") and the row belongs in the
# long-text bucket, where it is kept rather than judged.
UNIT_NOISE = re.compile(
    r"approximately|approx|about|around|roughly|nearly|usd|dollars?|cents?|"
    r"percentage\s*points?|percent|pct|per\s*share|shares?|years?|days?|times|"
    r"trillion|billion|million|thousand|[\s$%()\[\]{}~≈+:,'\"]|"
    r"^\W+|\W+$"
)


def as_number(text: str) -> float | None:
    """The figure this string *is*, or None if it merely contains one."""

    cleaned = latex_clean(text).lower().replace(",", "")
    match = NUMBER.search(cleaned)
    if not match:
        return None
    leftover = UNIT_NOISE.sub(" ", cleaned.replace(match.group(0), " ", 1))
    if leftover.strip():
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    for word, factor in MAGNITUDE:
        if word in cleaned:
            value *= factor
            break
    return value


def numeric_match(left: float, right: float, rel_tol: float) -> bool:
    for scale in SCALES:
        scaled = left * scale
        if math.isclose(scaled, right, rel_tol=rel_tol, abs_tol=1e-9):
            return True
    return False


def judge(gold_raw: str, pred_raw: str, rel_tol: float) -> tuple[str, bool]:
    """(bucket, matched). Only numeric/short buckets can report False."""

    gold, pred = str(gold_raw or "").strip(), str(pred_raw or "").strip()
    if not gold:
        return "no_gold", True
    if not pred:
        return "no_pred", True

    gold_value, pred_value = as_number(gold), as_number(pred)
    if gold_value is not None and pred_value is not None:
        return "numeric", numeric_match(pred_value, gold_value, rel_tol)

    gold_norm, pred_norm = norm_text(gold), norm_text(pred)
    if not gold_norm or not pred_norm:
        return "long_text", True
    if len(gold_norm.split()) <= SHORT_TEXT_WORDS:
        hit = gold_norm == pred_norm or gold_norm in pred_norm or pred_norm in gold_norm
        if not hit:
            # Same label, different spelling: FinRED writes
            # "product_or_material_produced" where the trace writes
            # "product/material produced". Ignoring function words makes those
            # equal without making "positive" equal to "not positive".
            gold_tokens, pred_tokens = token_set(gold), token_set(pred)
            hit = bool(gold_tokens) and gold_tokens == pred_tokens
        return "short_text", hit
    return "long_text", True


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    log: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    reservoir: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    seen: set[str] = set()

    def offer(bucket: str, source: str, gold: str, pred: str, instruction: str) -> None:
        pool = reservoir[bucket]
        item = (source, gold, pred, instruction)
        if len(pool) < args.samples:
            pool.append(item)
        else:
            index = rng.randrange(buckets[bucket] + 1)
            if index < args.samples:
                pool[index] = item

    raw = 0
    for row in iter_json_records(args.fin_json):
        raw += 1
        if args.max_rows and raw > args.max_rows:
            raw -= 1
            break
        if raw % 50000 == 0:
            print(f"  ...{raw}")

        source = str(row.get("source") or "(unknown)")
        per_source[source]["total"] += 1

        instruction = str(row.get("instruction") or "").strip()
        output = str(row.get("output") or "").strip()
        if not instruction or not output:
            log["1_empty_field"] += 1
            continue

        body = strip_tags(output)
        if not body:
            log["2_broken_structure"] += 1
            continue

        if CJK_PATTERN.search(instruction) or CJK_PATTERN.search(output):
            log["4_non_english"] += 1
            continue

        key = re.sub(r"\s+", " ", instruction.lower()).strip()
        if key in seen:
            log["5_duplicate"] += 1
            continue
        seen.add(key)

        gold = row.get("answer")
        gold = "" if gold is None else str(gold).strip()
        pred = extract_boxed(body)

        bucket, matched = judge(gold, pred, args.rel_tol)
        label = bucket if matched else f"{bucket}_WRONG"
        offer(label, source, gold, pred, instruction)
        buckets[label] += 1
        per_source[source]["judged"] += 1
        if not matched:
            per_source[source]["wrong"] += 1
            log["3_wrong_answer"] += 1
        else:
            log["kept"] += 1

    print()
    print("=" * 78)
    print(f"规则逐条：读入 {raw} 行")
    print("=" * 78)
    order = ["1_empty_field", "2_broken_structure", "4_non_english", "5_duplicate", "3_wrong_answer", "kept"]
    for name in order:
        if name in log:
            print(f"  {name:22} {log[name]:>7}  ({log[name] / max(raw, 1):5.1%})")
    print(f"\n  长度和源平衡还没算；上一轮里 4096 刷掉 2691 行、源平衡刷掉 61562 行")

    print()
    print("=" * 78)
    print("规则 3 的判定分布")
    print("=" * 78)
    judged = sum(buckets.values())
    for name, count in buckets.most_common():
        print(f"  {name:20} {count:>7}  ({count / max(judged, 1):5.1%})")
    wrong = sum(count for name, count in buckets.items() if name.endswith("_WRONG"))
    decidable = sum(
        count for name, count in buckets.items() if name.split("_WRONG")[0] in ("numeric", "short_text")
    )
    print(f"\n  可判定 {decidable}，其中判错 {wrong} ({wrong / max(decidable, 1):.1%})")
    print(f"  判错的会被删，占全部读入行的 {wrong / max(raw, 1):.1%}")

    print()
    print("=" * 78)
    print("判错的样本 —— 确认这些确实该删（规则有没有误杀，看这里）")
    print("=" * 78)
    for label in ("numeric_WRONG", "short_text_WRONG"):
        print(f"\n---- {label} ----")
        for source, gold, pred, instruction in reservoir.get(label, []):
            print(f"  [{source}]")
            print(f"    Q     : {instruction[:110]}")
            print(f"    gold  : {gold[:90]!r}")
            print(f"    boxed : {pred[:90]!r}")

    print()
    print("=" * 78)
    print("判对/跳过的样本 —— 确认没有漏杀，以及跳过的确实判不了")
    print("=" * 78)
    for label in ("numeric", "short_text", "long_text", "no_gold", "no_pred"):
        pool = reservoir.get(label, [])
        if not pool:
            continue
        print(f"\n---- {label} ({buckets[label]}) ----")
        for source, gold, pred, instruction in pool[: max(4, args.samples // 3)]:
            print(f"  [{source}]")
            print(f"    gold  : {gold[:90]!r}")
            print(f"    boxed : {pred[:90]!r}")

    print()
    print("=" * 78)
    print("按源看判错率（错误集中在个别源的话，说明是源的问题不是规则的问题）")
    print("=" * 78)
    print(f"  {'source':46} {'总数':>7} {'判定':>7} {'判错':>7}  错误率")
    rows = sorted(per_source.items(), key=lambda item: -item[1]["total"])
    for source, counter in rows:
        judged_n, wrong_n = counter["judged"], counter["wrong"]
        rate = wrong_n / judged_n if judged_n else 0.0
        print(f"  {source:46} {counter['total']:>7} {judged_n:>7} {wrong_n:>7}  {rate:6.1%}")


if __name__ == "__main__":
    main()
