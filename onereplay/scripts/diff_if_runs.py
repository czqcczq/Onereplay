"""Compare two IFEval/IFBench runs at the constraint level.

The summary.json numbers only say a run got worse. They do not say which
constraints moved, and with 344 instructions a 7-point swing is ~26 checks --
few enough that reading them individually is the fastest way to tell a real
capability loss from a scoring artifact.

Usage (on the cluster, inside the venv):
    python test_code/diff_if_runs.py \
        results_part1_if/ifbench/base \
        results_part1_if/ifbench/part1_if_lr5e-5_seed1

Both directories must contain eval_results_loose.jsonl, which the metric
already writes next to summary.json.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def read_results(run_dir: Path, pass_name: str) -> dict[str, dict]:
    path = run_dir / f"eval_results_{pass_name}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            # IFBench reuses the same WildChat prompt under different
            # constraints, so key alone is not unique in every release; pair it
            # with the constraint list to be safe.
            rows[(row.get("key"), tuple(row["instruction_id_list"]))] = row
    return rows


def family_table(rows: dict[str, dict]) -> dict[str, list[int]]:
    """id prefix -> [followed, total]."""

    table: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows.values():
        for iid, ok in zip(row["instruction_id_list"], row["follow_instruction_list"]):
            entry = table[iid.split(":")[0]]
            entry[0] += int(bool(ok))
            entry[1] += 1
    return table


def looks_like_prompt_echo(prompt: str, response: str) -> bool:
    """True when the response opens by parroting the request.

    A base model is a continuation engine: given an instruction it often
    restates it instead of answering. Several IFBench checkers ask for exactly
    that ("repeat the request before answering"), so echoing passes them for
    free -- and an SFT'd model that stops echoing loses those points without
    having lost any ability.
    """

    head = re.sub(r"\s+", " ", response.strip())[:60].lower()
    body = re.sub(r"\s+", " ", prompt.strip()).lower()
    return len(head) >= 20 and head in body


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    before_dir, after_dir = Path(sys.argv[1]), Path(sys.argv[2])

    for pass_name in ("strict", "loose"):
        before = read_results(before_dir, pass_name)
        after = read_results(after_dir, pass_name)
        shared = sorted(set(before) & set(after), key=lambda k: str(k))
        print(f"\n{'=' * 70}\n{pass_name} pass   ({len(shared)} prompts matched)\n{'=' * 70}")

        fb, fa = family_table(before), family_table(after)
        print(f"{'family':<24}{'before':>16}{'after':>16}{'delta':>10}")
        for family in sorted(set(fb) | set(fa)):
            ok_b, tot_b = fb.get(family, [0, 0])
            ok_a, tot_a = fa.get(family, [0, 0])
            acc_b = ok_b / tot_b if tot_b else 0.0
            acc_a = ok_a / tot_a if tot_a else 0.0
            print(
                f"{family:<24}{ok_b:>4}/{tot_b:<4}{acc_b:>7.1%}"
                f"{ok_a:>4}/{tot_a:<4}{acc_a:>7.1%}{acc_a - acc_b:>+10.1%}"
            )

        # Per-constraint regressions, which is where an artifact shows up as a
        # single id losing most of its checks.
        moved: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for key in shared:
            rb, ra = before[key], after[key]
            for iid, ok_b, ok_a in zip(
                rb["instruction_id_list"],
                rb["follow_instruction_list"],
                ra["follow_instruction_list"],
            ):
                if ok_b and not ok_a:
                    moved[iid][0] += 1
                elif ok_a and not ok_b:
                    moved[iid][1] += 1
        print("\nper-constraint flips (pass->fail / fail->pass):")
        for iid, (lost, gained) in sorted(moved.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
            if lost or gained:
                print(f"  {iid:<44} -{lost:<4} +{gained}")

        if pass_name != "loose":
            continue

        # Response shape. A style shift alone can move surface constraints.
        for label, rows in (("before", before), ("after", after)):
            words = sorted(len(r["response"].split()) for r in rows.values())
            echo = sum(looks_like_prompt_echo(r["prompt"], r["response"]) for r in rows.values())
            empty = sum(not r["response"].strip() for r in rows.values())
            if not words:
                continue
            p50 = words[len(words) // 2]
            p90 = words[int(len(words) * 0.9)]
            print(
                f"\n{label:<7} words P50={p50:<6} P90={p90:<6} max={words[-1]:<6}"
                f" prompt-echo={echo}/{len(rows)} empty={empty}"
            )


if __name__ == "__main__":
    main()
