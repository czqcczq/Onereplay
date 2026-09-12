"""Slice the sampled Flan 2021 rows into a Part 1 IF training pool plus replay/probe pools.

Reads what download_flan_v2.py left on disk (one jsonl per template style, rows
already in random order) and writes three things:

  --out_dir              save_to_disk {instruction, input, output} for train.py
                         --dataset_path. This is the Part 1 pool: Base --FLANv2-->
                         Model-IF, scored on IFEval + IFBench, the second point of
                         the Conifer comparison.
  --replay_out           jsonl {inputs, targets, task} for a later Part 2, read by
                         replay.py --replay_data_files and by collect_cov
                         --data_files. Those two must see the SAME rows or the
                         OneReplay-vs-EWC comparison is confounded by which data
                         each method got, which is why one file serves both.
  --heldout_out          jsonl in the same shape, for probe.py's flan_heldout
                         curve. Always disjoint from the other two.

Remix weighting
---------------
--weights is the row split across the four template styles, default 40/40/10/10
(zs_opt, zs_noopt, fs_opt, fs_noopt). The official flan2021_submix weights all
four equally (run_example.py: 1/1/1/1), and by rows that is the defensible
setting, but it is not what this line wants:

  - few-shot rows are 2.5-3.3x longer than zero-shot ones, so an equal-row draw
    spends ~72% of its token budget on few-shot prompts;
  - IFEval and IFBench both score single-turn zero-shot prompts, so few-shot
    exemplars train a format the benchmarks never present.

40/40/10/10 keeps both few-shot styles present -- they are part of what Flan
2021 is -- without letting them own the token budget. The manifest records the
resulting token shares so the choice stays visible rather than implicit.

What this dump cannot do
------------------------
SirNeural/flan_v2's `task` column is the constant string "flan" on every row, so
there is no task granularity: FLAN's own examples-proportional mixing with a rate
maximum (which old_knowledge.allocate_task_quota applies to the v1 dump, and
which Longpre et al. measured at ~5 points of MMLU and ~11 of BBH) cannot be
applied here, and no per-task table can be reported. Measured against a dump that
does carry task names, the raw Flan 2021 mixture is roughly 15% machine
translation, which is close enough to the capped-proportional target (5 of 69
tasks are WMT) that leaving it alone is defensible. The family histogram in the
manifest is a lexical fingerprint of that mixture, not a task distribution.

Replay overlap
--------------
--replay_mode nested (default) draws the replay/C pool from inside the training
pool. That is Experience Replay as normally defined -- rehearsing rows from the
previous task -- and it keeps the Part 2 comparison honest: disjoint replay rows
would let the replay baseline keep *learning* IF during Stage 2 from examples it
had never seen, while OneReplay only gets a penalty matrix and no new gradient
signal, so disjointness quietly hands one arm an advantage the other cannot have.

What must be disjoint is the measurement, not the rehearsal. probe.py's two FLAN
curves are exactly this distinction: flan_inpool is "rows replay actually trains
on", flan_heldout is "rows the training pool never contained", and the gap
between them separates retained ability from memorized rows. So --heldout_out is
always cut away first and never overlaps anything.

--replay_mode disjoint is available for the ablation, and because the download
defaults pull enough rows for it.

Token budget against the Conifer line
-------------------------------------
90_part1_if_conifer.pbs trains 38906 rows at MAXLEN=1536 for EPOCHS=3, measured
at 11.41M tok/epoch, so 34.2M tokens total. The length report below prints this
pool's tok/epoch and the epoch count that matches it, because "1 epoch" and
"same token budget" are not the same statement at this pool size. It also prints
supervised (answer-only) tokens separately: Flan 2021 targets are classification
labels and short spans, so a pool matched on total tokens is not matched on the
tokens that carry loss.

Truncation here is benign, unlike on Conifer. tokenizer_to_ids keeps the END of
the sequence, and a FLAN few-shot prompt puts its exemplars first and the actual
question last, so an over-long row degrades into a lower-shot version of itself
with the question intact. The Conifer failure mode -- losing the constraints at
the head and training "constrained-looking answer to an unconstrained question"
-- has no analogue.

Usage
-----
    python -m onereplay.scripts.prepare_flan_v2 \\
        --sample_dir /scratch/.../datasets/flan_v2 \\
        --out_dir /scratch/.../datasets/flan_v2_if_50k \\
        --tokenizer_path /scratch/.../models/Qwen3-1.7B-Base
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

REMIX_ORDER = ("zs_opt", "zs_noopt", "fs_opt", "fs_noopt")

# Budgets the truncation table reports on, same set as prepare_conifer so the
# two IF pools can be read side by side.
CANDIDATE_MAX_LENS = (512, 768, 1024, 1536, 2048, 3072)

# The Conifer line this pool is the counterpart of, for the token arithmetic.
CONIFER_TOKENS_PER_EPOCH = 11.41e6
CONIFER_EPOCHS = 3

FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "translation",
        re.compile(
            r"translat|Übersetz|traduire|"
            r"\b(French|German|Russian|Czech|Finnish|Romanian|Turkish)\b"
            r".{0,40}\b(say|sentence|text)\b",
            re.I,
        ),
    ),
    (
        "summarization",
        re.compile(r"summar|\bshort summary\b|\bheadline\b|\btl;?dr\b|in one sentence", re.I),
    ),
    ("nli", re.compile(r"\bentail|hypothesis|premise|can we (conclude|infer)", re.I)),
    (
        "sentiment",
        re.compile(r"sentiment|positive or negative|\breview\b.{0,60}\b(rating|star)", re.I),
    ),
    (
        "qa_reading",
        re.compile(
            r"\bquestion:|answer the question|based on (the|this) (passage|article|paragraph)",
            re.I,
        ),
    ),
    ("multiple_choice", re.compile(r"^\s*Options:|\bOPT:|\n- |choose|which of", re.I | re.M)),
    ("math", re.compile(r"\bsolve\b|\bwhat is the value\b|=\s*-?\d|\bequation\b", re.I)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Part 1 FLAN-v2 IF pool plus its replay and probe slices."
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="Directory download_flan_v2.py wrote flan_v2_{remix}.jsonl into.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="save_to_disk target for the training pool; pass as train.py --dataset_path.",
    )
    parser.add_argument(
        "--target_rows",
        type=int,
        default=50000,
        help="Rows in the training pool, split across remixes by --weights.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="40,40,10,10",
        help="Row split over zs_opt,zs_noopt,fs_opt,fs_noopt. '25,25,25,25' is "
        "the official flan2021_submix weighting; see the module docstring for "
        "why the default tilts away from few-shot.",
    )
    parser.add_argument(
        "--replay_rows",
        type=int,
        default=20000,
        help="Rows written to --replay_out for a later Part 2. 0 skips the file.",
    )
    parser.add_argument(
        "--replay_mode",
        type=str,
        choices=["nested", "disjoint"],
        default="nested",
        help="nested draws the replay pool from inside the training pool "
        "(Experience Replay as normally defined); disjoint takes rows the "
        "training pool never contained, which lets the replay arm acquire IF "
        "during Stage 2 rather than only retain it.",
    )
    parser.add_argument(
        "--heldout_rows",
        type=int,
        default=2000,
        help="Rows written to --heldout_out for probe.py's flan_heldout curve. "
        "Always cut away before anything else, so it never overlaps the "
        "training or replay pools. 0 skips the file.",
    )
    parser.add_argument(
        "--replay_out",
        type=str,
        default="",
        help="jsonl path for the replay/C pool. Empty defaults to "
        "<out_dir>_replay.jsonl next to out_dir.",
    )
    parser.add_argument(
        "--heldout_out",
        type=str,
        default="",
        help="jsonl path for the held-out probe slice. Empty defaults to "
        "<out_dir>_heldout.jsonl next to out_dir.",
    )
    parser.add_argument(
        "--dedup",
        type=int,
        default=1,
        help="1 drops rows whose (inputs, targets) pair was already taken within "
        "the same remix. Flan 2021 re-templates each example ~10 ways, and the "
        "small tasks have few enough examples that a draw this size can hit the "
        "same pair twice.",
    )
    parser.add_argument(
        "--dedup_across_remixes",
        type=int,
        default=0,
        help="0 (default) keeps a pair that appears in two remixes, which is "
        "what flan2021_submix does: opt and noopt render identically for any "
        "task without answer options, so equal-weight mixing upweights those "
        "tasks by construction. 1 drops the repeats, which also strips zs_noopt "
        "down to the option-bearing tasks -- see carve_all. Either way the "
        "held-out and disjoint-replay slices never collide with the training pool.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_len",
        type=int,
        default=1536,
        help="Budget the truncation report is headlined against; does not itself "
        "truncate. 1536 matches the Conifer line so the two are comparable.",
    )
    parser.add_argument(
        "--length_sample",
        type=int,
        default=5000,
        help="Rows measured per remix for the length report; 0 = all.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Model dir for the length report. Empty skips it, which also skips "
        "the only check on --max_len and on the token budget arithmetic.",
    )
    parser.add_argument(
        "--jsonl_dir",
        type=str,
        default="",
        help="Where the manifest goes. Defaults to out_dir's parent, kept outside "
        "out_dir so save_to_disk owns that directory alone.",
    )
    return parser.parse_args()


def classify(text: str) -> str:
    for name, pattern in FAMILY_PATTERNS:
        if pattern.search(text):
            return name
    return "other"


def parse_weights(spec: str) -> dict[str, float]:
    pieces = [piece.strip() for piece in spec.split(",") if piece.strip()]
    if len(pieces) != len(REMIX_ORDER):
        raise SystemExit(
            f"--weights needs {len(REMIX_ORDER)} numbers for {REMIX_ORDER}, got {spec!r}"
        )
    values = [float(piece) for piece in pieces]
    total = sum(values)
    if total <= 0:
        raise SystemExit("--weights must sum to something positive")
    return {label: value / total for label, value in zip(REMIX_ORDER, values)}


def split_budget(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder split so the per-remix quotas sum to exactly `total`."""

    exact = {label: total * weights[label] for label in REMIX_ORDER}
    quota = {label: int(exact[label]) for label in REMIX_ORDER}
    order = sorted(
        REMIX_ORDER, key=lambda label: (-(exact[label] - int(exact[label])), label)
    )
    index = 0
    while sum(quota.values()) < total:
        quota[order[index % len(order)]] += 1
        index += 1
    return quota


def load_remix(sample_dir: Path, label: str) -> list[dict[str, str]]:
    path = sample_dir / f"flan_v2_{label}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"没有 {path}。先在登录节点跑 download_flan_v2.py，或用 --sample_dir 指对目录。"
        )
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({"inputs": str(row["inputs"]), "targets": str(row["targets"])})
    return rows


def pair_hash(row: dict[str, str]) -> str:
    return hashlib.sha1(
        (row["inputs"].strip() + "\x00" + row["targets"].strip()).encode("utf-8")
    ).hexdigest()


def shuffled_pool(
    rows: list[dict[str, str]],
    label: str,
    args: argparse.Namespace,
    stats: collections.Counter,
) -> list[dict[str, str]]:
    """Shuffle one remix and drop repeats of the same pair inside it."""

    pool = list(rows)
    random.Random(f"{args.seed}:{label}").shuffle(pool)
    if args.dedup != 1:
        return pool

    seen: set[str] = set()
    deduped = []
    for row in pool:
        digest = pair_hash(row)
        if digest in seen:
            stats[f"{label}_dropped_duplicate"] += 1
            continue
        seen.add(digest)
        deduped.append(row)
    return deduped


def carve_all(
    pools: dict[str, list[dict[str, str]]],
    budgets: dict[str, dict[str, int]],
    args: argparse.Namespace,
    stats: collections.Counter,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Cut train / replay / heldout out of the four remixes, globally deconflicted.

    Deduplication has to be global, not per remix, because opt and noopt differ
    only in whether a multiple-choice task's answer options appear in the prompt:
    for every task that has no answer options the two templates render the SAME
    text, so one example shows up once in zs_opt and again, byte-identical, in
    zs_noopt. A per-remix dedup cannot see that, which is how a held-out row ends
    up inside the training pool.

    The training pool keeps those cross-remix repeats on purpose. They are part
    of what flan2021_submix is -- mixing all four styles at equal weight
    upweights option-free tasks exactly this way -- and dropping them would be
    worse than cosmetic: every collision is resolved against zs_noopt, so
    de-duplicating would leave that remix holding only the option-bearing tasks
    and quietly turn it into a multiple-choice pool. The count is reported
    instead. --dedup_across_remixes 1 overrides this.

    What cannot tolerate the overlap is the measurement. probe.py's flan_heldout
    curve means "rows the training pool never contained", so heldout is filtered
    against the training pool's hashes and topped up from the unused tail. Under
    --replay_mode disjoint the replay pool gets the same treatment, since its
    whole point is being rows Stage 1 never trained on.
    """

    slices: dict[str, dict[str, list[dict[str, str]]]] = {
        label: {"train": [], "replay": [], "heldout": []} for label in pools
    }
    cursors: dict[str, int] = {label: 0 for label in pools}

    def take(label: str, count: int, forbidden: set[str] | None) -> list[dict[str, str]]:
        """Next `count` rows of a remix, skipping any pair in `forbidden`."""

        pool = pools[label]
        picked: list[dict[str, str]] = []
        index = cursors[label]
        while index < len(pool) and len(picked) < count:
            row = pool[index]
            index += 1
            if forbidden is not None and pair_hash(row) in forbidden:
                stats[f"{label}_skipped_collision"] += 1
                continue
            picked.append(row)
        cursors[label] = index
        return picked

    # 1. Training pool first: it is the thing the other two are defined relative
    #    to, so it gets the unconstrained draw.
    for label in pools:
        slices[label]["train"] = take(label, budgets["train"].get(label, 0), None)

    train_hashes = {
        pair_hash(row) for label in pools for row in slices[label]["train"]
    }
    train_total = sum(len(slices[label]["train"]) for label in pools)
    stats["train_rows_before_cross_remix_dedup"] = train_total
    stats["train_cross_remix_duplicate_pairs"] = train_total - len(train_hashes)

    if args.dedup_across_remixes == 1:
        kept_hashes: set[str] = set()
        for label in pools:
            unique = []
            for row in slices[label]["train"]:
                digest = pair_hash(row)
                if digest in kept_hashes:
                    stats[f"{label}_dropped_cross_remix_duplicate"] += 1
                    continue
                kept_hashes.add(digest)
                unique.append(row)
            slices[label]["train"] = unique
        train_hashes = kept_hashes

    # 2. Held-out probe slice, from the unused tail and never colliding with train.
    for label in pools:
        slices[label]["heldout"] = take(
            label, budgets["heldout"].get(label, 0), train_hashes
        )
    heldout_hashes = {
        pair_hash(row) for label in pools for row in slices[label]["heldout"]
    }

    # 3. Replay pool.
    for label in pools:
        need = budgets["replay"].get(label, 0)
        if need <= 0:
            continue
        if args.replay_mode == "nested":
            # A prefix of the already-shuffled training slice, so the pool is
            # reproducible, nested inside larger replay budgets at the same
            # seed, and identical for replay.py and collect_cov.
            slices[label]["replay"] = slices[label]["train"][:need]
        else:
            slices[label]["replay"] = take(
                label, need, train_hashes | heldout_hashes
            )

    for label in pools:
        for name in ("train", "replay", "heldout"):
            want = budgets[name].get(label, 0)
            got = len(slices[label][name])
            if got < want:
                stats[f"{label}_short_{name}"] += want - got
        stats[f"{label}_available"] += len(pools[label])
        stats[f"{label}_unused"] += max(len(pools[label]) - cursors[label], 0)
    return slices


def to_sft_rows(
    rows: list[dict[str, str]], label: str, start_index: int
) -> list[dict[str, Any]]:
    """Map {inputs, targets} onto the {instruction, input, output} schema.

    `input` stays empty: apply_train_template only appends an "Input:" block when
    it is non-empty, and a FLAN prompt is already one self-contained user turn.
    remix / template_type / family survive into save_to_disk but never reach the
    model -- build_loader drops every column except the tokenized three -- so a
    later analysis can slice results by template style without re-deriving it.
    """

    template_type = label.replace("_", "-")
    out = []
    for offset, row in enumerate(rows):
        out.append(
            {
                "instruction": row["inputs"].strip(),
                "input": "",
                "output": row["targets"].strip(),
                "remix": label,
                "template_type": template_type,
                "family": classify(row["inputs"]),
                "source_index": start_index + offset,
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict[str, str]], label_of: list[str]) -> None:
    """Write the replay/probe shape: {inputs, targets, task}.

    replay.py, collect_cov and probe.py all read inputs/targets by column name,
    and `task` is carried so a later run can still tell the template styles apart
    even though the upstream dump's own task column was a constant.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row, label in zip(rows, label_of):
            handle.write(
                json.dumps(
                    {
                        "inputs": row["inputs"].strip(),
                        "targets": row["targets"].strip(),
                        "task": f"flan2021_{label}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile of an already sorted list."""

    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def length_report(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """Measure real training-time token lengths, per remix and overall.

    Lengths come from the same apply_chat_template call the trainer makes, so the
    template's role markers and special tokens count against --max_len.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    by_remix: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_remix[row["remix"]].append(row)

    per_remix: dict[str, Any] = {}
    full_all: list[int] = []
    answer_all: list[int] = []
    pool_tokens = 0.0
    pool_answer_tokens = 0.0

    for label in REMIX_ORDER:
        subset = by_remix.get(label, [])
        if not subset:
            continue
        sample = subset if args.length_sample <= 0 else subset[: args.length_sample]
        full: list[int] = []
        prompt: list[int] = []
        answer: list[int] = []
        for row in sample:
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["instruction"]},
                    {"role": "assistant", "content": row["output"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            full.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
            prompt.append(
                len(tokenizer(row["instruction"], add_special_tokens=False)["input_ids"])
            )
            answer.append(
                len(tokenizer(row["output"], add_special_tokens=False)["input_ids"])
            )
        full.sort()
        prompt.sort()
        answer.sort()
        full_all.extend(full)
        answer_all.extend(answer)

        mean_full = sum(full) / len(full)
        mean_answer = sum(answer) / len(answer)
        # Truncation caps what a row contributes, so the pool estimate uses the
        # clipped mean rather than the raw one.
        mean_kept = sum(min(value, args.max_len) for value in full) / len(full)
        pool_tokens += mean_kept * len(subset)
        pool_answer_tokens += mean_answer * len(subset)

        per_remix[label] = {
            "rows": len(subset),
            "rows_in_pool": len(subset),
            "full_mean": round(mean_full, 1),
            "full_mean_at_max_len": round(mean_kept, 1),
            "full": {f"p{q}": percentile(full, q) for q in (50, 90, 95, 99)}
            | {"max": full[-1]},
            "prompt_mean": round(sum(prompt) / len(prompt), 1),
            "answer_mean": round(mean_answer, 1),
            "answer": {f"p{q}": percentile(answer, q) for q in (50, 90, 99)}
            | {"max": answer[-1]},
            "fraction_truncated_at_max_len": round(
                sum(value > args.max_len for value in full) / len(full), 4
            ),
        }

    full_all.sort()
    answer_all.sort()
    truncation = {}
    for budget in sorted({*CANDIDATE_MAX_LENS, args.max_len}):
        over = sum(value > budget for value in full_all)
        lost = sum(max(value - budget, 0) for value in full_all)
        # A row whose answer alone fills the budget keeps no prompt at all. On
        # FLAN this is rare and, unlike on Conifer, not the dangerous case: the
        # question sits at the tail of the prompt, so ordinary truncation only
        # eats few-shot exemplars.
        prompt_gone = sum(value >= budget for value in answer_all)
        truncation[budget] = {
            "rows_truncated": over,
            "fraction_truncated": round(over / len(full_all), 4),
            "tokens_dropped_mean_over_truncated": round(lost / over, 1) if over else 0.0,
            "rows_losing_whole_prompt": prompt_gone,
        }

    epochs_to_match = (
        CONIFER_TOKENS_PER_EPOCH * CONIFER_EPOCHS / pool_tokens if pool_tokens else 0.0
    )
    report = {
        "tokenizer": args.tokenizer_path,
        "max_len": args.max_len,
        "rows_measured": len(full_all),
        "per_remix": per_remix,
        "full_length": {f"p{q}": percentile(full_all, q) for q in (50, 90, 95, 99)}
        | {"max": full_all[-1]},
        "answer_length": {f"p{q}": percentile(answer_all, q) for q in (50, 90, 99)}
        | {"max": answer_all[-1]},
        "truncation_by_max_len": truncation,
        "pool_tokens_per_epoch": round(pool_tokens),
        "pool_answer_tokens_per_epoch": round(pool_answer_tokens),
        "conifer_tokens_per_epoch": CONIFER_TOKENS_PER_EPOCH,
        "conifer_epochs": CONIFER_EPOCHS,
        "epochs_to_match_conifer_total_tokens": round(epochs_to_match, 2),
    }

    print("==== FLAN-v2 training-length report ====")
    print(
        f"{'remix':<10}{'rows':>8}{'full mean':>11}{'P50':>7}{'P90':>7}{'P99':>7}"
        f"{'answer mean':>13}{'trunc@%d' % args.max_len:>12}"
    )
    for label, entry in per_remix.items():
        print(
            f"{label:<10}{entry['rows_in_pool']:>8}{entry['full_mean']:>11.0f}"
            f"{entry['full']['p50']:>7}{entry['full']['p90']:>7}{entry['full']['p99']:>7}"
            f"{entry['answer_mean']:>13.0f}"
            f"{entry['fraction_truncated_at_max_len']:>11.1%}"
        )
    print(
        f"{'ALL':<10}{len(full_all):>8}"
        f"{sum(full_all) / len(full_all):>11.0f}"
        f"{report['full_length']['p50']:>7}{report['full_length']['p90']:>7}"
        f"{report['full_length']['p99']:>7}"
        f"{sum(answer_all) / len(answer_all):>13.0f}"
    )
    print()
    print(f"{'max_len':>10}{'truncated':>12}{'share':>9}{'mean lost':>12}{'prompt gone':>14}")
    for budget, entry in truncation.items():
        marker = "  <- --max_len" if budget == args.max_len else ""
        print(
            f"{budget:>10}{entry['rows_truncated']:>12}{entry['fraction_truncated']:>8.1%}"
            f"{entry['tokens_dropped_mean_over_truncated']:>12.0f}"
            f"{entry['rows_losing_whole_prompt']:>14}{marker}"
        )
    print(
        "  截断保尾丢头，而 FLAN few-shot 把示例放在前、真正的问题放在最后，"
        "所以被截的行退化成更少 shot 的同一题，题干不丢。这和 Conifer 丢约束的失效模式不同。"
    )
    print()
    print("==== token 预算 ====")
    print(f"  本池 1 epoch (max_len={args.max_len})   : {pool_tokens / 1e6:6.2f}M tok")
    print(f"  其中带 loss 的答案 token             : {pool_answer_tokens / 1e6:6.2f}M tok")
    print(
        f"  Conifer 线 (90 号, {CONIFER_EPOCHS} epoch) : "
        f"{CONIFER_TOKENS_PER_EPOCH * CONIFER_EPOCHS / 1e6:6.2f}M tok "
        f"({CONIFER_TOKENS_PER_EPOCH / 1e6:.2f}M/epoch)"
    )
    print(f"  要对齐总 token 需要                  : {epochs_to_match:.2f} epoch")
    print(
        "  注意答案 token 那一行：FLAN 的 targets 是分类标签和短答案，"
        "所以按总 token 对齐并不等于按监督 token 对齐。"
    )
    return report


def main() -> None:
    args = parse_args()
    weights = parse_weights(args.weights)

    out_dir = Path(args.out_dir)
    replay_out = Path(args.replay_out) if args.replay_out else out_dir.parent / f"{out_dir.name}_replay.jsonl"
    heldout_out = (
        Path(args.heldout_out)
        if args.heldout_out
        else out_dir.parent / f"{out_dir.name}_heldout.jsonl"
    )

    budgets = {
        "train": split_budget(args.target_rows, weights),
        "replay": split_budget(args.replay_rows, weights) if args.replay_rows > 0 else {},
        "heldout": split_budget(args.heldout_rows, weights) if args.heldout_rows > 0 else {},
    }

    print("==== FLAN-v2 -> Part 1 IF 池 ====")
    print(f"  取样目录   : {args.sample_dir}")
    print(f"  训练池     : {out_dir}   {args.target_rows} 行")
    print(
        f"  权重       : "
        + ", ".join(f"{label}={weights[label]:.0%}" for label in REMIX_ORDER)
    )
    if args.replay_rows > 0:
        print(f"  replay 池  : {replay_out}   {args.replay_rows} 行  ({args.replay_mode})")
    if args.heldout_rows > 0:
        print(f"  heldout 池 : {heldout_out}   {args.heldout_rows} 行  (总是互斥)")
    print()

    stats: collections.Counter = collections.Counter()
    pools: dict[str, list[dict[str, str]]] = {}
    for label in REMIX_ORDER:
        wanted = sum(budgets[name].get(label, 0) for name in ("train", "replay", "heldout"))
        if wanted == 0:
            continue
        raw = load_remix(Path(args.sample_dir), label)
        pools[label] = shuffled_pool(raw, label, args, stats)

    slices = carve_all(pools, budgets, args, stats)

    train_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, str]] = []
    replay_labels: list[str] = []
    heldout_rows: list[dict[str, str]] = []
    heldout_labels: list[str] = []

    for label in REMIX_ORDER:
        if label not in slices:
            continue
        cut = slices[label]
        notes = []
        if stats[f"{label}_dropped_duplicate"]:
            notes.append(f"同 remix 去重丢 {stats[f'{label}_dropped_duplicate']}")
        if stats[f"{label}_skipped_collision"]:
            notes.append(f"跨 remix 撞车跳过 {stats[f'{label}_skipped_collision']}")
        print(
            f"  {label:<10} 可用 {len(pools[label]):>7,} 行 -> "
            f"train {len(cut['train']):>6,}  "
            f"replay {len(cut['replay']):>6,}  "
            f"heldout {len(cut['heldout']):>5,}"
            + (f"  ({'; '.join(notes)})" if notes else "")
        )
        train_rows.extend(to_sft_rows(cut["train"], label, len(train_rows)))
        replay_rows.extend(cut["replay"])
        replay_labels.extend([label] * len(cut["replay"]))
        heldout_rows.extend(cut["heldout"])
        heldout_labels.extend([label] * len(cut["heldout"]))

    if not train_rows:
        raise SystemExit("训练池 0 行，检查 --sample_dir 和 --weights")

    dup_pairs = stats["train_cross_remix_duplicate_pairs"]
    if dup_pairs:
        print(
            f"\n  训练池里有 {dup_pairs} 对跨 remix 重复的 (inputs, targets)"
            f"（{dup_pairs / len(train_rows):.2%}）。这是 flan2021_submix 的固有行为："
            "opt/noopt 对无选项任务渲染完全相同，等权混合等于给这些任务加权。"
            "--dedup_across_remixes 1 可以去掉，代价见 carve_all 的注释。"
        )

    short = {key: value for key, value in stats.items() if "_short_" in key}
    if short:
        print(
            "\n!! 配额没满足："
            + ", ".join(f"{key}={value}" for key, value in sorted(short.items()))
            + "\n   调大 download_flan_v2.py 的 --rows / --read_mb 后重拉那一路。"
        )

    # Interleave remixes so a truncated run, a --max_train_samples cut or any
    # debug printing sees all four template styles rather than 20k zs_opt rows
    # followed by 20k zs_noopt ones. train.py shuffles anyway; this makes the
    # pool itself readable.
    random.Random(args.seed).shuffle(train_rows)

    from datasets import Dataset, DatasetDict

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(str(out_dir))
    print(f"\n写出训练池 {out_dir}  ({len(train_rows)} 行)")

    if replay_rows:
        write_jsonl(replay_out, replay_rows, replay_labels)
        print(f"写出 replay 池 {replay_out}  ({len(replay_rows)} 行, {args.replay_mode})")
    if heldout_rows:
        write_jsonl(heldout_out, heldout_rows, heldout_labels)
        print(f"写出 heldout 池 {heldout_out}  ({len(heldout_rows)} 行)")

    remix_counts = collections.Counter(row["remix"] for row in train_rows)
    family_counts = collections.Counter(row["family"] for row in train_rows)
    print("\n训练池构成:")
    for label in REMIX_ORDER:
        if remix_counts[label]:
            print(
                f"  {label:<10} {remix_counts[label]:>7,} 行  "
                f"{remix_counts[label] / len(train_rows):>6.1%}"
            )
    print("  模板家族（词法近似，仅作混合比例指纹，不是任务分布）:")
    for name, count in family_counts.most_common():
        print(f"    {name:<18} {count:>7,}  {count / len(train_rows):>6.1%}")

    manifest: dict[str, Any] = {
        "source": str(Path(args.sample_dir) / "flan_v2_{remix}.jsonl"),
        "upstream": "SirNeural/flan_v2, flan_* files only = Flan 2021 sub-mixture",
        "weights": {label: weights[label] for label in REMIX_ORDER},
        "weights_note": "official flan2021_submix is 25/25/25/25 by row; this pool "
        "tilts away from few-shot because fs rows are 2.5-3.3x longer and IFEval / "
        "IFBench score zero-shot single-turn prompts",
        "seed": args.seed,
        "dedup": args.dedup,
        "dedup_across_remixes": args.dedup_across_remixes,
        "replay_mode": args.replay_mode,
        "train": {
            "path": str(out_dir),
            "num_rows": len(train_rows),
            "by_remix": dict(remix_counts),
            "by_family_lexical": dict(family_counts),
        },
        "replay": {
            "path": str(replay_out) if replay_rows else "",
            "num_rows": len(replay_rows),
            "mode": args.replay_mode,
            "note": "read by replay.py --replay_data_files and collect_cov "
            "--data_files; one file for both so C and the replay rows are the "
            "same pool. nested means these rows are a prefix of the training "
            "slice, which is Experience Replay as normally defined.",
        },
        "heldout": {
            "path": str(heldout_out) if heldout_rows else "",
            "num_rows": len(heldout_rows),
            "note": "disjoint from train and replay; probe.py's flan_heldout curve.",
        },
        "counts": dict(stats),
        "schema_note": "train.py splits validation out of the training pool with "
        "--val_fraction. remix / template_type / family are written into "
        "save_to_disk but never reach the model: build_loader drops every column "
        "except input_ids / labels / attention_mask before collating.",
        "task_column_note": "the upstream dump's `task` column is the constant "
        "string 'flan', so FLAN's capped-proportional task balancing cannot be "
        "applied and no per-task table can be reported.",
    }
    if args.tokenizer_path:
        manifest["length"] = length_report(train_rows, args)
    else:
        print("\n跳过长度报告：没给 --tokenizer_path（也就没有对 --max_len 和 epoch 数的校验）")

    jsonl_dir = Path(args.jsonl_dir) if args.jsonl_dir else out_dir.parent
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = jsonl_dir / f"{out_dir.name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n写出 manifest {manifest_path}")


if __name__ == "__main__":
    main()
