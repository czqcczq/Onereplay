"""Build the NuminaMath-CoT SFT pool for the Math stage, as a drop-in for the
OpenR1 one.

Why a second math pool exists
-----------------------------
The OpenR1 pool's targets are DeepSeek R1 traces: a median of 3767 answer
tokens, dense with "wait, but let me check that again". A 1.7B model trained on
them copied the style without the capacity to converge, so on MATH500 more than
half of its responses looped until the token budget ran out and scored 0 for
never reaching a \\boxed span. Raising the budget from 4096 to 8192 changed
nothing, which is what non-termination looks like as opposed to a budget that is
merely too small.

NuminaMath-CoT is the same problems -- OpenR1-Math-220k was built by having R1
solve NuminaMath 1.5 -- with short human/GPT-4o-written solutions instead of
long traces. Measured over all 859494 rows with the Qwen3 tokenizer: solutions
run 269-800 tokens at the median depending on source, the longest single
solution is 2832 tokens, and the self-verification vocabulary that drove the
looping comes in at a median of 0 occurrences per solution against 88 in the
capped OpenR1 responses. So this swaps solution style while holding the problem
distribution fixed, and it removes the pathology at the source rather than
patching the decoder.

Sampling
--------
--sample_rows draws a proportional stratified sample: every source keeps its
share of the usable pool, so a 50k draw has the same source mix as the full
859k. That matters because the sources are wildly unequal -- cn_k12,
synthetic_math and orca_math are 70% of the corpus and are grade-school to
mid-difficulty, while olympiads, aops_forum, amc_aime and math carry the
competition-level problems MATH500 actually tests. Fixed per-source quotas would
be an undeclared difficulty decision; proportional keeps it the corpus's.

Quotas use the largest-remainder method so they sum to exactly --sample_rows,
and each source is drawn by reservoir sampling under its own seed, so adding a
source or changing --sample_rows does not reshuffle the others.

--require_boxed defaults to 1 here, unlike the OpenR1 script. The graders read
the \\boxed span and nothing else, and NuminaMath's pipeline normalizes answers to
either \\boxed{} or a ■ marker: coverage is 100% for most sources but 61% for
aops_forum and 93% for olympiads, so the filter is doing real work.

There is no length filter and no need for one: nothing in this corpus reaches
4096 tokens, so --max_len 4096 truncates zero rows. Read the length report
before setting the training budget anyway, and note the eval budget can come
down with it -- targets this short give the model no reason to generate 8192.

The official 100-row test split is written next to the pool as JSONL, not into
--out_dir. train.py only reads the train split and carves --val_fraction from
it, so that 200-row holdout stays the in-run val loss. The JSONL is a
seed-independent CE probe for when the 50k draw itself changes.

Example:

    python onereplay/scripts/prepare_numina_math.py \\
        --numina_path data_check/AI-MO/NuminaMath-CoT/data \\
        --out_dir data/numina_math_50k \\
        --sample_rows 50000 --require_boxed 1 \\
        --tokenizer_path models/Qwen3-1.7B-Base
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reused rather than reimplemented: the instruction wording has to stay
# byte-identical to the OpenR1 pool's or the two Math stages are not comparable,
# and the length report is what every --max_len decision is read off.
from onereplay.scripts.prepare_openr1_math import (  # noqa: E402
    EVAL_PROMPT_PREFIX,
    REPLAY_COLUMNS,
    build_instruction,
    carve_replay_pool,
    has_boxed,
    length_report,
)

NUMINA_COLUMNS = ("source", "problem", "solution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the NuminaMath-CoT SFT training pool for the Math stage."
    )
    parser.add_argument(
        "--numina_path",
        type=str,
        default="",
        help="Directory of NuminaMath-CoT parquet shards (train-*.parquet), or a "
        "save_to_disk directory. Takes precedence over --numina_repo.",
    )
    parser.add_argument("--numina_repo", type=str, default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument(
        "--sample_rows",
        type=int,
        default=50000,
        help="Rows to draw, proportionally across sources. 0 keeps the whole "
        "usable pool.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated source allowlist, e.g. "
        "'olympiads,aops_forum,amc_aime,math'. Empty keeps all nine. Narrowing "
        "this changes the difficulty mix, so it belongs in the manifest, which "
        "is why it is a flag rather than a hand edit.",
    )
    parser.add_argument(
        "--instruction_style",
        type=str,
        choices=["eval_aligned", "bare"],
        default="eval_aligned",
        help="eval_aligned reuses math500.py's prompt wording so training and "
        "evaluation match. There is no openr1_system option: NuminaMath carries "
        "no R1 directive to prepend.",
    )
    parser.add_argument(
        "--require_boxed",
        type=int,
        default=1,
        help="1 drops rows whose solution has no \\boxed{...} span. Defaults on "
        "because the graders score exactly that span and this corpus ends some "
        "solutions with a ■ marker instead.",
    )
    parser.add_argument(
        "--max_solution_tokens",
        type=int,
        default=0,
        help="Drop rows whose solution exceeds this many tokens. 0 = off, which "
        "is the right setting: the longest solution in the corpus is 2832 "
        "tokens. Requires --tokenizer_path when non-zero.",
    )
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Model dir for the length report. Empty skips it, which also skips "
        "the only check on the training budget.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=4096,
        help="Training budget the truncation table is headlined against; does "
        "not itself truncate. 4096 is provably lossless here -- the longest "
        "solution in the corpus is 2832 tokens -- so the report should show 0 "
        "rows truncated at this value.",
    )
    parser.add_argument("--length_sample", type=int, default=3000)
    parser.add_argument("--jsonl_dir", type=str, default="")
    parser.add_argument("--replay_rows", type=int, default=0)
    parser.add_argument("--replay_sample_seed", type=int, default=1)
    parser.add_argument("--replay_out", type=str, default="")
    parser.add_argument(
        "--save_test",
        type=int,
        default=1,
        help="1 writes the official NuminaMath test split (100 rows) as JSONL. "
        "Kept out of --out_dir: train.py only reads the train split and carves "
        "val_fraction from it. This file is a seed-independent CE probe, not "
        "the training val set. 0 skips it.",
    )
    parser.add_argument(
        "--test_out",
        type=str,
        default="",
        help="JSONL path for the official test split. Empty writes "
        "{jsonl_dir}/{out_dir.name}_test.jsonl.",
    )
    return parser.parse_args()


def iter_shards(args: argparse.Namespace):
    """Yield (source, problem, solution) column triples one shard at a time.

    Streamed per shard rather than concatenated because the corpus is 1.2 GB of
    parquet and only the drawn sample needs to be resident.
    """

    path = Path(args.numina_path) if args.numina_path else None
    if path and path.is_dir() and list(path.glob("*.parquet")):
        import pyarrow.parquet as pq

        # The HF snapshot puts test-00000-of-00001.parquet in the same data/
        # directory as the five train shards, so the fallback has to exclude it
        # by name. Sweeping the whole directory would fold the held-out split
        # into the training pool, and nothing downstream would report it.
        shards = sorted(path.glob("train-*.parquet")) or [
            shard for shard in sorted(path.glob("*.parquet")) if not shard.name.startswith("test-")
        ]
        for shard in shards:
            table = pq.read_table(shard, columns=list(NUMINA_COLUMNS))
            yield (
                shard.name,
                table.column("source").to_pylist(),
                table.column("problem").to_pylist(),
                table.column("solution").to_pylist(),
            )
        return

    from datasets import load_dataset, load_from_disk

    if path and path.is_dir():
        dataset = load_from_disk(str(path))
    else:
        dataset = load_dataset(args.numina_repo)
    if hasattr(dataset, "keys"):
        dataset = dataset["train"]
    missing = [c for c in NUMINA_COLUMNS if c not in dataset.column_names]
    if missing:
        raise SystemExit(f"NuminaMath columns {missing} not found; got {dataset.column_names}")
    yield (
        "train",
        dataset["source"],
        dataset["problem"],
        dataset["solution"],
    )


def make_sft_row(
    problem: str,
    solution: str,
    source: str,
    source_index: int,
    style: str,
) -> dict[str, Any]:
    """One training-schema row. gold_answer is filled later by fill_gold_answers."""

    return {
        "instruction": build_instruction(problem, "", style),
        "input": "",
        "output": solution.strip(),
        "gold_answer": "",
        "data_source": source or "?",
        "source_index": int(source_index),
    }


def fill_gold_answers(rows: list[dict[str, Any]]) -> None:
    """Set gold_answer to the same \\boxed span the MATH500 grader reads."""

    from onereplay.eval.metrics.math500 import extract_answer

    for row in rows:
        row["gold_answer"] = extract_answer(row["output"]) or ""


def load_official_test(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Return (source, problem, solution) triples from the official test split.

    The HF snapshot keeps test-*.parquet next to the train shards. This loader
    never feeds those rows into the training draw; it only exists so --save_test
    can write a fixed probe that does not move when --seed or --sample_rows
    change.
    """

    path = Path(args.numina_path) if args.numina_path else None
    if path and path.is_dir():
        shards = sorted(path.glob("test-*.parquet"))
        if not shards and (path / "data").is_dir():
            shards = sorted((path / "data").glob("test-*.parquet"))
        if shards:
            import pyarrow.parquet as pq

            triples: list[tuple[str, str, str]] = []
            for shard in shards:
                table = pq.read_table(shard, columns=list(NUMINA_COLUMNS))
                triples.extend(
                    zip(
                        table.column("source").to_pylist(),
                        table.column("problem").to_pylist(),
                        table.column("solution").to_pylist(),
                    )
                )
            return triples

    from datasets import load_dataset, load_from_disk

    if path and path.is_dir():
        if not (path / "dataset_dict.json").exists():
            return []
        dataset = load_from_disk(str(path))
    else:
        dataset = load_dataset(args.numina_repo)
    if not hasattr(dataset, "keys") or "test" not in dataset:
        return []
    split = dataset["test"]
    missing = [c for c in NUMINA_COLUMNS if c not in split.column_names]
    if missing:
        raise SystemExit(f"NuminaMath test columns {missing} not found; got {split.column_names}")
    return list(zip(split["source"], split["problem"], split["solution"]))


def convert_official_test(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Map the official test split onto the training schema, unfiltered.

    require_boxed / --sources / --sample_rows stay off: filtering would make
    this a different set every time those flags change, which is the opposite
    of a fixed probe. Empty problem/solution rows are dropped because a CE
    probe with no supervised tokens is a NaN waiting to happen.
    """

    triples = load_official_test(args)
    rows: list[dict[str, Any]] = []
    dropped = 0
    for index, (source, problem, solution) in enumerate(triples):
        if not (problem or "").strip() or not (solution or "").strip():
            dropped += 1
            continue
        rows.append(
            make_sft_row(problem, solution, source or "?", index, args.instruction_style)
        )
    fill_gold_answers(rows)
    if dropped:
        print(f"official test: dropped {dropped} empty rows of {len(triples)}", flush=True)
    return rows


def usable(source: str, problem: str, solution: str, allow: set[str], args) -> bool:
    if allow and source not in allow:
        return False
    if not (problem or "").strip() or not (solution or "").strip():
        return False
    if args.require_boxed == 1 and not has_boxed(solution):
        return False
    return True


def proportional_quotas(available: dict[str, int], target: int) -> dict[str, int]:
    """Split target across sources in proportion to available, summing exactly.

    Largest-remainder rather than plain rounding so the quotas add up to target
    instead of drifting a few rows either way, and so the smallest sources
    (amc_aime is 0.5% of the corpus) are not rounded out of existence.
    """

    total = sum(available.values())
    if target <= 0 or target >= total:
        return dict(available)

    exact = {name: available[name] * target / total for name in available}
    quota = {name: min(int(exact[name]), available[name]) for name in exact}
    order = sorted(exact, key=lambda name: (-(exact[name] - quota[name]), name))
    while sum(quota.values()) < target:
        progressed = False
        for name in order:
            if quota[name] < available[name]:
                quota[name] += 1
                progressed = True
                if sum(quota.values()) >= target:
                    break
        if not progressed:
            break
    return quota


def draw_sample(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Two passes: count the usable pool per source, then reservoir-draw quotas.

    Two passes rather than one because the quotas cannot be computed until the
    usable counts are known, and holding 859k problem/solution pairs in memory
    to avoid a second 20-second read is the wrong trade.
    """

    allow = {s.strip() for s in args.sources.split(",") if s.strip()}
    max_sol = args.max_solution_tokens
    tokenizer = None
    if max_sol > 0:
        if not args.tokenizer_path:
            raise SystemExit("--max_solution_tokens needs --tokenizer_path")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    raw = collections.Counter()
    avail = collections.Counter()
    dropped = collections.Counter()

    print("pass 1/2: counting the usable pool per source", flush=True)
    for name, sources, problems, solutions in iter_shards(args):
        for source, problem, solution in zip(sources, problems, solutions):
            source = source or "?"
            raw[source] += 1
            if not usable(source, problem, solution, allow, args):
                dropped[source] += 1
                continue
            if tokenizer is not None:
                if len(tokenizer(solution, add_special_tokens=False)["input_ids"]) > max_sol:
                    dropped[source] += 1
                    continue
            avail[source] += 1
        print(f"  {name}", flush=True)

    if not avail:
        raise SystemExit("NuminaMath yielded 0 usable rows; check --sources / --require_boxed.")

    quota = proportional_quotas(dict(avail), args.sample_rows)
    total_avail = sum(avail.values())

    print(f"\npass 2/2: drawing {sum(quota.values())} of {total_avail} usable rows", flush=True)
    reservoir: dict[str, list[dict[str, Any]]] = {name: [] for name in quota}
    rng = {name: random.Random(f"{args.seed}:{name}") for name in quota}
    seen = collections.Counter()

    # source_index has to identify a row across the whole corpus, not within a
    # shard, or the five shards hand out five copies of every index and the
    # replay pool loses its only link back to the original row.
    offset = 0
    for name, sources, problems, solutions in iter_shards(args):
        for local, (source, problem, solution) in enumerate(zip(sources, problems, solutions)):
            index = offset + local
            source = source or "?"
            if source not in quota or not usable(source, problem, solution, allow, args):
                continue
            if tokenizer is not None:
                if len(tokenizer(solution, add_special_tokens=False)["input_ids"]) > max_sol:
                    continue
            seen[source] += 1
            row = make_sft_row(problem, solution, source, index, args.instruction_style)
            pool = reservoir[source]
            if len(pool) < quota[source]:
                pool.append(row)
            else:
                j = rng[source].randrange(seen[source])
                if j < quota[source]:
                    pool[j] = row
        offset += len(sources)
        print(f"  {name}", flush=True)

    rows: list[dict[str, Any]] = []
    for source in sorted(reservoir):
        rows.extend(reservoir[source])
    fill_gold_answers(rows)

    # The pool is grouped by source up to here, and train.py carves validation
    # off by row position, so an unshuffled pool would hand it one or two
    # sources as the whole validation set.
    random.Random(args.seed).shuffle(rows)

    stats = {
        "raw_rows": sum(raw.values()),
        "usable_rows": total_avail,
        "sampled_rows": len(rows),
        "per_source": {
            source: {
                "raw": raw[source],
                "usable": avail[source],
                "dropped": dropped[source],
                "quota": quota.get(source, 0),
                "share_of_usable": avail[source] / total_avail,
                "share_of_sample": quota.get(source, 0) / max(len(rows), 1),
            }
            for source in sorted(raw, key=lambda s: -raw[s])
        },
    }
    return rows, stats


def check_out_dir(out_dir: Path) -> None:
    """Refuse an --out_dir that is not empty and not a pool we wrote before.

    save_to_disk lays dataset_dict.json and a split directory straight into the
    path it is given, so pointing --out_dir at a directory that holds other
    things -- a datasets/ root next to math500_test.jsonl, say -- mixes a pool
    into it and the eval data quietly goes missing. Checked before the two-pass
    draw rather than after, so the failure costs seconds instead of minutes.
    """

    if not out_dir.exists():
        return
    entries = sorted(path.name for path in out_dir.iterdir())
    if not entries or "dataset_dict.json" in entries:
        return
    raise SystemExit(
        f"--out_dir {out_dir} already holds {len(entries)} entries and none of "
        f"them is dataset_dict.json, so this is not a pool directory:\n"
        f"  {', '.join(entries[:8])}{' ...' if len(entries) > 8 else ''}\n"
        f"save_to_disk would write into it. Point --out_dir at a new "
        f"subdirectory, e.g. {out_dir / 'numina_math_50k'}."
    )


def main() -> None:
    args = parse_args()
    check_out_dir(Path(args.out_dir))
    rows, stats = draw_sample(args)

    print(f"\n==== 采样结果（seed={args.seed}, require_boxed={args.require_boxed}）====")
    header = f"{'source':<18}{'raw':>9}{'usable':>9}{'quota':>8}{'池占比':>10}{'样本占比':>11}"
    print(header)
    print("-" * (len(header) + 8))
    for source, info in stats["per_source"].items():
        print(
            f"{source[:18]:<18}{info['raw']:>9}{info['usable']:>9}{info['quota']:>8}"
            f"{info['share_of_usable']:>10.2%}{info['share_of_sample']:>11.2%}"
        )
    print(
        f"{'合计':<18}{stats['raw_rows']:>9}{stats['usable_rows']:>9}"
        f"{stats['sampled_rows']:>8}"
    )
    print("  最后两列应当逐行接近：那就是「按原始分布采样」的意思。")

    boxed = sum(1 for row in rows if has_boxed(row["output"]))
    print(f"\n  答案带 \\boxed 的行：{boxed} ({boxed / len(rows):.1%})")

    from datasets import Dataset, DatasetDict

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    jsonl_dir = Path(args.jsonl_dir) if args.jsonl_dir else out_dir.parent
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(out_dir))
    print(f"wrote pool to {out_dir}")

    replay_out = (
        Path(args.replay_out)
        if args.replay_out
        else out_dir.parent / f"{out_dir.name}_replay.jsonl"
    )
    replay_rows: list[dict[str, Any]] = []
    if args.replay_rows > 0:
        replay_rows = carve_replay_pool(rows, args)
        replay_out.parent.mkdir(parents=True, exist_ok=True)
        with replay_out.open("w", encoding="utf-8") as file:
            for row in replay_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        mix = collections.Counter(row["data_source"] for row in replay_rows)
        print(
            f"wrote replay/C pool to {replay_out}  {len(replay_rows)} rows "
            f"(nested subset, replay_sample_seed={args.replay_sample_seed})"
        )
        print(
            "  data_source mix: "
            + ", ".join(f"{name or '<empty>'}={count}" for name, count in mix.most_common())
        )

    test_rows: list[dict[str, Any]] = []
    test_out = (
        Path(args.test_out) if args.test_out else jsonl_dir / f"{out_dir.name}_test.jsonl"
    )
    if args.save_test == 1:
        test_rows = convert_official_test(args)
        if not test_rows:
            raise SystemExit(
                "official NuminaMath test split not found next to the train shards. "
                "Pass --save_test 0 to skip, or point --numina_path at the directory "
                "that holds test-*.parquet (or a DatasetDict with a test split)."
            )
        test_out.parent.mkdir(parents=True, exist_ok=True)
        with test_out.open("w", encoding="utf-8") as file:
            for row in test_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        mix = collections.Counter(row["data_source"] for row in test_rows)
        boxed_test = sum(1 for row in test_rows if has_boxed(row["output"]))
        print(
            f"wrote official test split to {test_out}  {len(test_rows)} rows "
            f"(not used as val; train.py still carves val_fraction from the pool)"
        )
        print(
            "  data_source mix: "
            + ", ".join(f"{name or '<empty>'}={count}" for name, count in mix.most_common())
        )
        print(f"  答案带 \\boxed 的行：{boxed_test}/{len(test_rows)}")

    manifest: dict[str, Any] = {
        "source": args.numina_path or args.numina_repo,
        "dataset": "NuminaMath-CoT",
        "instruction_style": args.instruction_style,
        "eval_prompt_prefix": EVAL_PROMPT_PREFIX
        if args.instruction_style == "eval_aligned"
        else "",
        "require_boxed": args.require_boxed,
        "max_solution_tokens": args.max_solution_tokens,
        "sources_allowlist": args.sources,
        "sample_rows": args.sample_rows,
        "sampling": "proportional stratified, largest-remainder quotas, "
        "per-source reservoir draw under seed:{source}",
        "seed": args.seed,
        "train": {"path": str(out_dir), "num_rows": len(rows)},
        "official_test": {
            "path": str(test_out) if test_rows else "",
            "num_rows": len(test_rows),
            "source_index": "row index in the official test split, not the train corpus",
            "note": "Not the training val set. train.py carves --val_fraction from "
            "the train pool; this JSONL is a seed-independent CE probe.",
        },
        "counts": stats,
        "replay": {
            "path": str(replay_out) if replay_rows else "",
            "num_rows": len(replay_rows),
            "mode": "nested",
            "sample_seed": args.replay_sample_seed,
            "columns": list(REPLAY_COLUMNS),
        },
        "boxed_answer_rows": boxed,
        "note": "Drop-in for the OpenR1 pool: same columns, same instruction "
        "wording, same manifest shape. train.py splits validation out of the "
        "train pool with --val_fraction. The official 100-row test split is "
        "written beside the pool and is not read at train time. gold_answer / "
        "data_source / source_index are metadata.",
    }
    if args.tokenizer_path:
        # The length report is a convenience; the manifest is the pool's only
        # record of which rows came from where. Losing provenance because a
        # percentile table raised is the wrong failure, so the report is allowed
        # to fail loudly and the manifest still gets written.
        try:
            manifest["length"] = length_report(rows, args)
        except Exception as error:  # noqa: BLE001
            print(f"length report failed: {type(error).__name__}: {error}", flush=True)
            manifest["length"] = {"error": f"{type(error).__name__}: {error}"}
    else:
        print("skipping length report: no --tokenizer_path")

    manifest_path = jsonl_dir / f"{out_dir.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
