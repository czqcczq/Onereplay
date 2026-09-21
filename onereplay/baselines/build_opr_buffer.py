"""Build the OPR replay buffer: score the on-policy answers, keep the top-k.

OPR (On-Policy Replay, arXiv:2605.29495) differs from plain replay in two
places, and both live in this script. The rehearsal rows carry the model's *own*
answers rather than the corpus's gold ones, and only the highest-scoring
fraction of them is kept. Everything else -- how the buffer is mixed in, how big
it is relative to the new task -- is ordinary data-level replay and is handled
by train.py.

The selection logic is transcribed from the authors' repository, kept as an
unmodified clone under baseline/OnPolicyReplay:

    upstream  https://github.com/lqzxt/On-Policy-Replay  (see the clone's remote)
    files     tools/generate_opr_ru.py, tools/generate_opr_sc.py

Their two scripts generate and select in one pass; we cannot reuse them
directly because the scoring is dispatched by TRACE task index and the training
harness is a subprocess launcher around ms-swift. What is reproduced here, line
for line, is the part that constitutes the method:

    combined_list.sort(key=lambda x: x["score"], reverse=True)
    replay_list.extend(combined_list[:int(args.buffer_size / (args.task_id + 1))])

that is, sort each source's candidates by score descending and take an equal
share of the buffer from each. ``task_id + 1`` is the number of sources seen so
far, which here is simply the number of protected pools.

Three departures, all forced and all recorded in the run log:

* OPR-SC's score is the mean log-probability of the generated tokens. Their
  tools/generate_opr_sc.py computes it from vLLM's per-token logprobs; ours is
  computed at generation time by scripts/generate_replay_targets.py
  --record_logprob and read here from the avg_logprob field. Same quantity, same
  averaging span, different decoder.
* OPR-RU's scores come from five TRACE-specific functions dispatched by task
  index, none of which applies to a math or code pool. Their principle is to
  score with the benchmark's own metric -- edit similarity for Py150 line
  completion, last-number match for NumGLUE -- so the translation is final
  answer equivalence for MetaMath and testcase execution for OPC. Transplanting
  their fuzz.ratio onto instruction-style code would be the unfaithful choice:
  two correct solutions need not resemble each other.
* Rows whose target is empty are dropped before scoring rather than scored as 0.
  Their scorers return 0 for an empty response (``if not response: return 0``),
  so such a row could only enter the buffer if the buffer were larger than the
  number of non-empty candidates. That is asserted rather than assumed.

Usage:
    python -m onereplay.baselines.build_opr_buffer \
        --reward sc --buffer_size 766 \
        --pool math=<generated_math.jsonl> --pool code=<generated_code.jsonl> \
        --output_path <opr_buffer_sc.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score and select the OPR replay buffer.")
    parser.add_argument(
        "--reward",
        type=str,
        required=True,
        choices=["sc", "ru"],
        help=(
            "sc ranks by the model's own confidence (mean log-probability of the "
            "answer it generated) and needs no gold answer. ru ranks by a "
            "rule-based correctness score against the gold answer. The authors "
            "provide both and report ru as the default."
        ),
    )
    parser.add_argument(
        "--pool",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "A generated candidate file, one per protected domain, e.g. "
            "math=/path/opr_math.jsonl. Repeat the flag. The buffer is split "
            "equally between them, which is what OPR does across the tasks it "
            "has seen."
        ),
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        required=True,
        help=(
            "Total rows to keep, i.e. OPR's int(dataset_size * rho). Each pool "
            "contributes int(buffer_size / n_pools), so the written file holds "
            "slightly fewer rows when the division is not exact -- their "
            "flooring, kept as is."
        ),
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--scorer",
        action="append",
        default=[],
        metavar="LABEL=KIND",
        help=(
            "Which rule-based scorer a pool uses under --reward ru. KIND is math "
            "or code. Defaults to the pool's own label when that label is itself "
            "math or code."
        ),
    )
    # --- inputs the code scorer needs to find the testcases ---
    parser.add_argument(
        "--code_view_file",
        type=str,
        default="",
        help="Prepared OPC view JSONL, joined by (inputs, gold_targets) to recover source_index.",
    )
    parser.add_argument(
        "--code_opc_path",
        type=str,
        default="",
        help="Original educational_instruct parquet, read for its testcase column.",
    )
    parser.add_argument("--code_timeout", type=float, default=3.0)
    parser.add_argument(
        "--scores_out",
        type=str,
        default="",
        help=(
            "Optional sidecar for the per-row scores. Executing the code pool is "
            "slow, so writing them lets a requeued job re-select without "
            "re-running the harness."
        ),
    )
    parser.add_argument(
        "--scores_in",
        type=str,
        default="",
        help="Read scores from a sidecar written by --scores_out instead of recomputing.",
    )
    return parser.parse_args()


def parse_mapping(entries: list[str], flag: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"{flag} expects LABEL=VALUE, got {entry!r}")
        label, value = entry.split("=", 1)
        label, value = label.strip(), value.strip()
        if not label or not value:
            raise SystemExit(f"{flag} expects LABEL=VALUE, got {entry!r}")
        if label in mapping:
            raise SystemExit(f"{flag} names {label!r} twice")
        mapping[label] = value
    return mapping


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: invalid JSON: {error}") from error
    return rows


def usable_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """Drop rows with nothing to train on, and say how many went."""

    kept = [row for row in rows if str(row.get("targets", "") or "").strip()]
    dropped = len(rows) - len(kept)
    truncated = sum(1 for row in kept if row.get("truncated"))
    print(
        f"  {label}: {len(rows)} candidates, {dropped} with an empty target dropped, "
        f"{len(kept)} scoreable ({truncated} of them truncated and kept, per OPR)"
    )
    return kept


def score_self_confidence(rows: list[dict[str, Any]], label: str) -> list[float]:
    """OPR-SC: the mean log-probability of the answer the model produced.

    Read rather than recomputed. generate_replay_targets.py --record_logprob
    averages exactly the tokens vLLM would have reported to their
    calculate_avg_logprob, including the stop token.
    """

    missing = [row for row in rows if "avg_logprob" not in row]
    if missing:
        raise SystemExit(
            f"{label}: {len(missing)} rows have no avg_logprob field. Regenerate the "
            "pool with scripts/generate_replay_targets.py --record_logprob 1; "
            "OPR-SC has no other score to rank by."
        )
    return [float(row["avg_logprob"]) for row in rows]


def score_math(rows: list[dict[str, Any]], label: str) -> list[float]:
    """OPR-RU on a math pool: does the final answer match the gold one.

    100 / 0 rather than a similarity, matching the scale their score_acc and
    score_math return. The extraction is the one already used to audit the
    self-distilled math corpus, which accepts \\boxed{}, #### and an explicit
    final-answer phrase before falling back to the last number.
    """

    from onereplay.eval.metrics.math500 import is_equiv
    from onereplay.scripts.score_math_selfdistill_quality import lenient_answer

    scores = []
    matched = 0
    unscoreable = 0
    for row in rows:
        predicted = lenient_answer(str(row.get("targets", "") or ""))
        gold = lenient_answer(str(row.get("gold_targets", "") or ""))
        if predicted is None or gold is None:
            # No answer could be extracted from one side, so correctness is
            # undecidable. Their scorers return 0 in the equivalent situation.
            unscoreable += 1
            scores.append(0.0)
            continue
        correct = bool(is_equiv(predicted, gold))
        matched += int(correct)
        scores.append(100.0 if correct else 0.0)
    print(
        f"  {label}: rule=math, {matched}/{len(rows)} answers match gold "
        f"({unscoreable} undecidable, scored 0)"
    )
    return scores


def score_code(rows: list[dict[str, Any]], label: str, args: argparse.Namespace) -> list[float]:
    """OPR-RU on a code pool: does the generated program pass the testcases.

    Their Py150 scorer is fuzz.ratio because edit similarity is that
    benchmark's own metric. OPC's own metric is execution, so that is what is
    used here. A row that cannot be joined back to its testcases, or has none,
    scores 0 rather than being dropped, so the buffer size arithmetic is
    unaffected by how many rows happen to be checkable.
    """

    from datasets import load_dataset

    from onereplay.eval.code_exec import evaluate_assert_program
    from onereplay.scripts.score_code_selfdistill_quality import (
        build_view_lookup,
        make_program,
        normalize_tests,
        row_key,
    )

    if not args.code_view_file or not args.code_opc_path:
        raise SystemExit(
            f"{label}: --reward ru on a code pool needs --code_view_file and "
            "--code_opc_path to recover the testcases."
        )
    lookup = build_view_lookup(Path(args.code_view_file))
    source = load_dataset("parquet", data_files=args.code_opc_path, split="train")
    if "testcase" not in source.column_names:
        raise SystemExit(f"{args.code_opc_path} has no testcase column")

    scores = []
    passed = 0
    join_missing = 0
    no_tests = 0
    for number, row in enumerate(rows, 1):
        key = row_key(row.get("inputs"), row.get("gold_targets"))
        if not lookup.get(key):
            join_missing += 1
            scores.append(0.0)
            continue
        meta = lookup[key].popleft()
        tests = normalize_tests(dict(source[meta["source_index"]]).get("testcase"))
        if not tests:
            no_tests += 1
            scores.append(0.0)
            continue
        program = make_program(
            str(row.get("inputs", "") or ""),
            str(row.get("targets", "") or ""),
            meta["style"],
        )
        ok, _ = evaluate_assert_program(program, tests, args.code_timeout)
        passed += int(ok)
        scores.append(100.0 if ok else 0.0)
        if number % 500 == 0:
            print(f"    executed {number}/{len(rows)}, {passed} passing", flush=True)
    print(
        f"  {label}: rule=code, {passed}/{len(rows)} programs pass their testcases "
        f"({join_missing} unjoinable, {no_tests} without tests, all scored 0)"
    )
    return scores


def build_scorer(
    label: str, args: argparse.Namespace, scorers: dict[str, str]
) -> Callable[[list[dict[str, Any]]], list[float]]:
    if args.reward == "sc":
        return lambda rows: score_self_confidence(rows, label)
    kind = scorers.get(label, label)
    if kind == "math":
        return lambda rows: score_math(rows, label)
    if kind == "code":
        return lambda rows: score_code(rows, label, args)
    raise SystemExit(
        f"pool {label!r} has no rule-based scorer. Pass --scorer {label}=math or "
        f"--scorer {label}=code; got kind {kind!r}."
    )


def select_top_k(
    scored: dict[str, list[tuple[float, dict[str, Any]]]], buffer_size: int
) -> list[tuple[str, float, dict[str, Any]]]:
    """Sort each pool descending and take an equal share of the buffer.

    This is the whole of OPR's selection:

        combined_list.sort(key=lambda x: x["score"], reverse=True)
        replay_list.extend(combined_list[:int(args.buffer_size / (args.task_id + 1))])

    with ``task_id + 1`` standing for the number of pools. The floor is theirs:
    two pools and a buffer of 767 give 383 rows each and 766 in total.
    """

    quota = int(buffer_size / len(scored))
    if quota <= 0:
        raise SystemExit(
            f"buffer_size {buffer_size} split across {len(scored)} pools leaves "
            "nothing per pool."
        )

    taken: dict[str, list[tuple[str, float, dict[str, Any]]]] = {}
    for label, entries in scored.items():
        if quota > len(entries):
            raise SystemExit(
                f"pool {label!r} holds {len(entries)} scoreable rows but the buffer "
                f"wants {quota} from it. Generate a larger pool, or lower "
                "--buffer_size."
            )
        # Python's sort is stable, so equal scores keep generation order; with a
        # 0/100 rule-based score that is most of the pool, and stability is what
        # makes the selection reproducible across reruns.
        ordered = sorted(entries, key=lambda item: item[0], reverse=True)[:quota]
        taken[label] = [(label, score, row) for score, row in ordered]
        kept_scores = [score for score, _ in ordered]
        print(
            f"  {label}: kept {len(ordered)}/{len(entries)} "
            f"(score {kept_scores[-1]:.4f} .. {kept_scores[0]:.4f})"
        )

    # Round-robin rather than one pool after the other. The row *set* is
    # identical either way and train.py shuffles before training, so this
    # changes nothing about what is learned. It matters only if the loader ever
    # cuts the buffer short: a prefix of a round-robin ordering keeps the pools
    # balanced and keeps the best rows of each, where a prefix of a concatenated
    # ordering would be one domain only.
    merged: list[tuple[str, float, dict[str, Any]]] = []
    for position in range(quota):
        for label in scored:
            merged.append(taken[label][position])
    return merged


def main() -> int:
    args = parse_args()
    pools = parse_mapping(args.pool, "--pool")
    scorers = parse_mapping(args.scorer, "--scorer")
    if not pools:
        raise SystemExit("at least one --pool LABEL=PATH is required")
    unknown = set(scorers) - set(pools)
    if unknown:
        raise SystemExit(f"--scorer names pools that were not given: {sorted(unknown)}")

    print(f"OPR buffer: reward={args.reward}, buffer_size={args.buffer_size}")
    print(f"pools: {', '.join(f'{k}={v}' for k, v in pools.items())}")

    cached: dict[str, dict[str, float]] = {}
    if args.scores_in:
        cached = json.loads(Path(args.scores_in).read_text(encoding="utf-8"))
        print(f"reusing scores from {args.scores_in}")

    scored: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    fresh: dict[str, dict[str, float]] = {}
    for label, path in pools.items():
        rows = usable_rows(load_jsonl(Path(path)), label)
        if label in cached:
            by_index = cached[label]
            missing = [row for row in rows if str(row["index"]) not in by_index]
            if missing:
                raise SystemExit(
                    f"{label}: the sidecar is missing {len(missing)} of the rows in "
                    f"{path}; regenerate it without --scores_in."
                )
            scores = [by_index[str(row["index"])] for row in rows]
            print(f"  {label}: {len(scores)} scores read from the sidecar")
        else:
            scores = build_scorer(label, args, scorers)(rows)
        fresh[label] = {str(row["index"]): score for row, score in zip(rows, scores)}
        scored[label] = list(zip(scores, rows))

    if args.scores_out:
        Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.scores_out).write_text(
            json.dumps(fresh, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote scores sidecar to {args.scores_out}")

    print("selection:")
    merged = select_top_k(scored, args.buffer_size)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        for position, (label, score, row) in enumerate(merged):
            sink.write(
                json.dumps(
                    {
                        # Renumbered because load_self_distilled_pool sorts by
                        # index and build_replay_dataset may take a prefix; the
                        # prefix has to be the rows we selected, not whichever
                        # ones happened to be generated first.
                        "index": position,
                        "inputs": row["inputs"],
                        "targets": row["targets"],
                        "gold_targets": row.get("gold_targets", ""),
                        "truncated": bool(row.get("truncated", False)),
                        "opr_score": score,
                        "opr_pool": label,
                        "opr_source_index": row.get("index"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"wrote {len(merged)} rows to {output_path}\n"
        f"  pass --replay_rows {len(merged)} to train.py so the whole buffer is used"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
