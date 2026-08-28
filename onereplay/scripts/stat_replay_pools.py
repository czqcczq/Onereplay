"""Measure the self-distilled replay pools before choosing max_len and the mix.

Adding a second domain to the replay baseline turns two settings that were
harmless on the IF-only line into real decisions, and both of them are decided
by the length distribution of the pools rather than by preference:

  max_len       FLAN's self-distilled answers average 90 tokens, so the 512 the
                IF line trains at never bites. MetaMath answers are several
                hundred tokens. Training truncation keeps the *last* max_len
                tokens (process_dataset.process_glue_myself.tokenizer_to_ids
                slices [-max_len:]), so an over-long row keeps its answer ending
                and its EOS but loses the question. The row stops being a
                question-answer rehearsal and becomes bare continuation of math
                text. What matters is therefore not "does the answer survive"
                but "for how many rows does the prompt survive", which is what
                the budget table below reports.

  mixing ratio  Rows are not the unit the loss is averaged over. On the IF line
                replay at a 50.00% *row* share carried 93.16% of the supervised
                tokens, because Commonsense targets average 7 supervised tokens
                against the pool's 94.8. Mixing a math pool in at equal row
                counts therefore does not give the two domains equal weight;
                C_mix's 0.5/0.5 is an average over *tokens*, so the analogue for
                replay is an equal supervised-token share, and this script
                prints the row ratio that produces it.

It also prints the length ratio between the pools, which is the leading-order
predictor of how far apart mean(F_math) and mean(F_if) will land. The Fisher is
a sum-reduction score, so an example's contribution scales as T^a with a in
[1, 2]; a length ratio r therefore predicts a Fisher ratio somewhere in
[r, r^2]. That is a prediction to check against the mean/max collect_fisher
prints, not a substitute for collecting F.

Reads only the length fields generate_replay_targets already wrote
(prompt_tokens, target_tokens, truncated), so it needs no GPU and no model. Run
it on a login node.

    python -m onereplay.scripts.stat_replay_pools \\
        --pool if=/path/results/replay/flan_selfdistill_20000_seed1.jsonl \\
        --pool math='/path/results/replay/metamath_cmath_30k_*_selfdistill_seed1.jsonl' \\
        --max_lens 512,1024,2048 --mix_at 1024
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

QUANTILES = (50, 90, 95, 99)

# Measured on the IF line and quoted here so the token-share arithmetic below is
# reproducible without rerunning a training job:
#   7.0   mean supervised tokens of a Commonsense170k target
#   24.1  mean supervised tokens F_if actually rests on. Lower than this pool's
#         94.8 because F_if was collected on FLAN's *gold* one-line answers
#         (01/22), not on the self-distilled ones; each F mirrors its own C.
DEFAULT_NEW_TASK_TOKENS = 7.0
DEFAULT_FISHER_REF_TOKENS = 24.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        action="append",
        default=[],
        metavar="LABEL=GLOB",
        help="Repeatable. Quote the glob so the shell does not expand it, e.g. "
        "math='.../metamath_cmath_30k_*_selfdistill_seed1.jsonl'.",
    )
    parser.add_argument(
        "--max_lens",
        type=str,
        default="512,1024,2048",
        help="Training max_len candidates to score the pools against.",
    )
    parser.add_argument(
        "--mix_at",
        type=int,
        default=0,
        help="max_len the mixing table is computed at; 0 = the largest candidate. "
        "The two tables have to agree: truncation caps a row's supervised tokens, "
        "so the token share depends on which max_len is chosen.",
    )
    parser.add_argument(
        "--drop_truncated",
        type=int,
        default=1,
        help="Mirror --replay_drop_truncated: rows the generator cut off have no "
        "stop token and training drops them, so they are excluded from every "
        "statistic that describes what training will see.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--replay_per_batch", type=int, default=4)
    parser.add_argument(
        "--accumulation_size",
        type=int,
        default=128,
        help="Only used to report new-task rows per optimizer update. The replay arm "
        "needs 128 to keep that at the baseline's 64 while half of every "
        "micro-batch is replay.",
    )
    parser.add_argument(
        "--new_task_rows",
        type=int,
        default=168715,
        help="Commonsense170k training rows; sets steps/epoch and pool cycles.",
    )
    parser.add_argument(
        "--new_task_tokens",
        type=float,
        default=DEFAULT_NEW_TASK_TOKENS,
        help="Mean supervised tokens of a new-task row (measured, see module notes).",
    )
    parser.add_argument(
        "--fisher_ref_tokens",
        type=float,
        default=DEFAULT_FISHER_REF_TOKENS,
        help="Mean supervised tokens behind the reference Fisher, used only for the "
        "F_math/F_if magnitude prediction.",
    )
    parser.add_argument(
        "--fisher_pool",
        type=str,
        default="",
        help="Pool label whose length is compared against --fisher_ref_tokens; "
        "empty = the last pool given.",
    )
    parser.add_argument(
        "--fisher_max_len",
        type=int,
        default=2048,
        help="max_len the domain Fisher will be collected at (31's C_math used 2048); "
        "the prediction uses the length distribution capped at this value.",
    )
    parser.add_argument("--json_out", type=str, default="", help="Optional dump of every number.")
    return parser.parse_args()


def quantile(values: list[int], percent: float) -> int:
    """Nearest-rank quantile, matching the diagnostics in 31/32."""

    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(percent / 100 * (len(ordered) - 1))))]


def mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_pool(label: str, pattern: str, drop_truncated: bool) -> dict[str, Any]:
    """Read one pool's length fields and split it the way training will.

    Every row is classified exactly once so the counts add up: a row the
    generator truncated is unusable no matter how long it is, and an empty
    target has no supervised token at all (to_sft_schema filters it out, and a
    micro-batch of such rows would produce a NaN loss).
    """

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(
            f"pool {label}: nothing matched {pattern}\n"
            "  quote the glob so the shell does not expand it, and check the pool was "
            "actually produced (31 for math, 41 for code, 13 for the FLAN pool)"
        )

    prompts: list[int] = []
    targets: list[int] = []
    totals: list[int] = []
    rows = truncated = empty = missing_fields = 0
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows += 1
                if drop_truncated and record.get("truncated"):
                    truncated += 1
                    continue
                if not str(record.get("targets") or "").strip():
                    empty += 1
                    continue
                if "prompt_tokens" not in record or "target_tokens" not in record:
                    missing_fields += 1
                    continue
                prompt_tokens = int(record["prompt_tokens"])
                target_tokens = int(record["target_tokens"])
                prompts.append(prompt_tokens)
                targets.append(target_tokens)
                totals.append(prompt_tokens + target_tokens)

    if missing_fields:
        raise ValueError(
            f"pool {label}: {missing_fields} rows carry no prompt_tokens/target_tokens. "
            "They were not written by generate_replay_targets, so their length is unknown."
        )
    return {
        "label": label,
        "pattern": pattern,
        "files": files,
        "rows": rows,
        "dropped_truncated": truncated,
        "dropped_empty": empty,
        "usable": len(totals),
        "prompt_tokens": prompts,
        "target_tokens": targets,
        "total_tokens": totals,
    }


def budget_row(pool: dict[str, Any], max_len: int) -> dict[str, Any]:
    """Score one pool against one max_len, under left truncation.

    supervised_after is min(target, max_len): the tail slice keeps answer tokens
    first, so an answer longer than the budget fills the whole window and every
    position in it carries loss. prompt_lost counts the rows where the prompt is
    gone entirely, which is the number that decides whether the row is still a
    rehearsal of answering a question.
    """

    prompts = pool["prompt_tokens"]
    targets = pool["target_tokens"]
    totals = pool["total_tokens"]
    usable = max(pool["usable"], 1)

    over = sum(total > max_len for total in totals)
    prompt_gone = sum(target >= max_len for target in targets)
    supervised_after = [min(target, max_len) for target in targets]
    prompt_kept = [
        max(0, min(prompt, max_len - target)) for prompt, target in zip(prompts, targets)
    ]
    prompt_kept_share = mean(
        [kept / prompt for kept, prompt in zip(prompt_kept, prompts) if prompt > 0]
    )
    return {
        "max_len": max_len,
        "over_budget": over,
        "over_budget_share": over / usable,
        "prompt_fully_cut": prompt_gone,
        "prompt_fully_cut_share": prompt_gone / usable,
        "mean_prompt_kept_share": prompt_kept_share,
        "mean_supervised_tokens": mean(supervised_after),
        "median_supervised_tokens": quantile(supervised_after, 50),
    }


def print_pool_report(pool: dict[str, Any]) -> None:
    print(f"---- pool [{pool['label']}] {pool['pattern']}")
    for path in pool["files"]:
        print(f"       file: {Path(path).name}")
    print(
        f"       rows={pool['rows']} usable={pool['usable']} "
        f"dropped: truncated={pool['dropped_truncated']} empty_target={pool['dropped_empty']}"
    )
    for name in ("prompt_tokens", "target_tokens", "total_tokens"):
        values = pool[name]
        quantile_text = " ".join(f"P{q}={quantile(values, q)}" for q in QUANTILES)
        print(
            f"       {name:>13}: mean={mean(values):7.1f} {quantile_text} "
            f"max={max(values) if values else 0}"
        )


def print_budget_table(pools: list[dict[str, Any]], max_lens: list[int]) -> dict[str, list[dict]]:
    print("\n==== max_len budget (training truncation keeps the LAST max_len tokens) ====")
    print(
        f"{'pool':>6} {'max_len':>8} {'over budget':>18} {'prompt fully cut':>18} "
        f"{'prompt kept':>12} {'sup tokens/row':>15}"
    )
    table: dict[str, list[dict]] = {}
    for pool in pools:
        rows = [budget_row(pool, max_len) for max_len in max_lens]
        table[pool["label"]] = rows
        for row in rows:
            print(
                f"{pool['label']:>6} {row['max_len']:>8} "
                f"{row['over_budget']:>7} ({row['over_budget_share']:>6.1%}) "
                f"{row['prompt_fully_cut']:>7} ({row['prompt_fully_cut_share']:>6.1%}) "
                f"{row['mean_prompt_kept_share']:>11.1%} "
                f"{row['mean_supervised_tokens']:>10.1f} (P50 {row['median_supervised_tokens']})"
            )
    print(
        "over budget = row is truncated at all; prompt fully cut = answer alone fills the\n"
        "window, so the question is gone and the row becomes bare continuation. The answer\n"
        "ending and its EOS always survive, so no arm learns 'never terminate'."
    )
    return table


def mixing_candidates(
    stats: dict[str, dict[str, float]],
    labels: list[str],
) -> list[dict[str, Any]]:
    """Three ways to read 'equal weight', with the row counts each implies.

    Only two-pool mixes are enumerated: the row ratio that equalizes supervised
    tokens has no single answer beyond two sources, and every planned arm mixes
    one old-knowledge domain into the IF pool.
    """

    first, second = labels
    available = {label: stats[label]["usable"] for label in labels}
    tokens = {label: stats[label]["mean_supervised_tokens"] for label in labels}

    candidates: list[dict[str, Any]] = []

    # Equal rows: capped by the smaller pool, which is what "17k + 17k" was.
    smaller = min(available.values())
    candidates.append({"name": "equal rows", **{label: smaller for label in labels}})

    # Equal supervised tokens: n_a * L_a = n_b * L_b. Scale up until one side
    # hits its ceiling so the pool stays as large as the ratio allows; a smaller
    # pool cycles more often, and chapter 3 already showed replay memorizes it.
    if tokens[first] > 0 and tokens[second] > 0:
        ratio = tokens[second] / tokens[first]  # rows of `first` per row of `second`
        scale = min(available[first] / ratio, available[second])
        candidates.append(
            {
                "name": "equal supervised tokens",
                first: int(scale * ratio),
                second: int(scale),
            }
        )

    # Both pools whole: the ratio is then whatever the corpora happen to be.
    candidates.append({"name": "full pools", **{label: available[label] for label in labels}})
    return candidates


def print_mixing_table(
    stats: dict[str, dict[str, float]],
    labels: list[str],
    args: argparse.Namespace,
    max_len: int,
) -> list[dict[str, Any]]:
    new_per_batch = args.batch_size - args.replay_per_batch
    if new_per_batch <= 0:
        raise ValueError(
            f"--replay_per_batch {args.replay_per_batch} leaves no new-task row in a "
            f"--batch_size {args.batch_size} micro-batch"
        )
    steps_per_epoch = args.new_task_rows // new_per_batch
    replay_rows_per_epoch = steps_per_epoch * args.replay_per_batch

    print(
        f"\n==== mixing the replay pool at max_len={max_len} "
        f"({new_per_batch} new + {args.replay_per_batch} replay per micro-batch) ===="
    )
    accum_steps = max(args.accumulation_size // args.batch_size, 1)
    new_rows_per_update = accum_steps * new_per_batch
    print(
        f"steps/epoch={steps_per_epoch}  replay rows/epoch={replay_rows_per_epoch}  "
        f"new-task rows/update={accum_steps} x {new_per_batch} = {new_rows_per_update} "
        f"(baseline is 64, set by --accumulation_size {args.accumulation_size})"
    )
    header = (
        f"{'composition':>24} "
        + " ".join(f"{'n_' + label:>10}" for label in labels)
        + f" {'row share':>22} {'token share (replay)':>22} {'replay token share':>19} {'cycles/ep':>10}"
    )
    print(header)

    rows_out: list[dict[str, Any]] = []
    for candidate in mixing_candidates(stats, labels):
        counts = {label: int(candidate[label]) for label in labels}
        pool_size = sum(counts.values())
        if pool_size == 0:
            continue
        row_share = {label: counts[label] / pool_size for label in labels}
        token_weight = {
            label: row_share[label] * stats[label]["mean_supervised_tokens"] for label in labels
        }
        replay_tokens_per_row = sum(token_weight.values())
        replay_token_share_within = {
            label: (token_weight[label] / replay_tokens_per_row if replay_tokens_per_row else 0.0)
            for label in labels
        }
        batch_replay_tokens = args.replay_per_batch * replay_tokens_per_row
        batch_new_tokens = new_per_batch * args.new_task_tokens
        total_tokens = batch_replay_tokens + batch_new_tokens
        replay_share_of_batch = batch_replay_tokens / total_tokens if total_tokens else 0.0
        cycles = replay_rows_per_epoch / pool_size

        print(
            f"{candidate['name']:>24} "
            + " ".join(f"{counts[label]:>10}" for label in labels)
            + " "
            + "/".join(f"{row_share[label]:.2f}" for label in labels).rjust(22)
            + " "
            + "/".join(f"{replay_token_share_within[label]:.2f}" for label in labels).rjust(22)
            + f" {replay_share_of_batch:>18.1%} {cycles:>10.2f}"
        )
        rows_out.append(
            {
                "name": candidate["name"],
                "counts": counts,
                "pool_size": pool_size,
                "row_share": row_share,
                "replay_token_share_within": replay_token_share_within,
                "replay_share_of_batch_tokens": replay_share_of_batch,
                "pool_cycles_per_epoch": cycles,
            }
        )

    print(
        "row share / token share are in the order the pools were passed. 'replay token share'\n"
        "is replay's cut of all supervised tokens in a micro-batch, new task included; the IF\n"
        "line measured 93.16% there and this column is the number to compare against.\n"
        "batch_mix.BatchMixedReplayLoader.describe() prints the same two shares at train time,\n"
        "so the training log can be checked against this table."
    )
    return rows_out


def print_fisher_prediction(
    pool: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    capped = [min(target, args.fisher_max_len) for target in pool["target_tokens"]]
    domain_tokens = mean(capped)
    ratio = domain_tokens / args.fisher_ref_tokens if args.fisher_ref_tokens else 0.0

    print(f"\n==== predicted Fisher magnitude gap for pool [{pool['label']}] ====")
    print(
        f"mean supervised tokens: {domain_tokens:.1f} (capped at --fisher_max_len "
        f"{args.fisher_max_len}) vs reference {args.fisher_ref_tokens:.1f}  =>  r={ratio:.1f}x"
    )
    print(
        f"F is a sum-reduction score, so ||g_n||^2 ~ T_n^a with a in [1, 2]:\n"
        f"  predicted mean(F_{pool['label']}) / mean(F_ref) between {ratio:.0f}x and "
        f"{ratio ** 2:.0f}x"
    )
    print(
        "Check this against the mean/max collect_fisher prints, and read the fitted\n"
        "length_exponent to see where a actually landed. Equal 0.5/0.5 weights are only\n"
        "defensible if the measured ratio comes out near 1."
    )
    return {
        "pool": pool["label"],
        "mean_supervised_tokens": domain_tokens,
        "reference_tokens": args.fisher_ref_tokens,
        "length_ratio": ratio,
        "predicted_fisher_ratio_low": ratio,
        "predicted_fisher_ratio_high": ratio**2,
    }


def main() -> None:
    args = parse_args()
    if not args.pool:
        raise SystemExit("pass at least one --pool LABEL=GLOB")

    pools: list[dict[str, Any]] = []
    print("==== pools ====")
    for spec in args.pool:
        if "=" not in spec:
            raise SystemExit(f"--pool expects LABEL=GLOB, got {spec!r}")
        label, pattern = spec.split("=", 1)
        pool = load_pool(label.strip(), pattern.strip(), args.drop_truncated == 1)
        pools.append(pool)
        print_pool_report(pool)

    max_lens = [int(item) for item in args.max_lens.split(",") if item.strip()]
    budget = print_budget_table(pools, max_lens)

    mix_at = args.mix_at or max(max_lens)
    stats = {
        pool["label"]: {
            "usable": pool["usable"],
            **budget_row(pool, mix_at),
        }
        for pool in pools
    }

    mixing: list[dict[str, Any]] = []
    labels = [pool["label"] for pool in pools]
    if len(labels) == 2:
        mixing = print_mixing_table(stats, labels, args, mix_at)
    else:
        print(
            f"\n(mixing table needs exactly two pools; got {len(labels)}. "
            "Pass the IF pool and one domain pool.)"
        )

    fisher_label = args.fisher_pool or labels[-1]
    fisher_pool = next((pool for pool in pools if pool["label"] == fisher_label), None)
    fisher = print_fisher_prediction(fisher_pool, args) if fisher_pool is not None else {}

    if args.json_out:
        payload = {
            "args": vars(args),
            "pools": [
                {key: value for key, value in pool.items() if not isinstance(value, list)}
                | {"files": pool["files"]}
                for pool in pools
            ],
            "budget": budget,
            "mix_at": mix_at,
            "mixing": mixing,
            "fisher_prediction": fisher,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
