"""Audit a surprising SORRY-Bench gap between arms (e.g. OneReplay vs ReplayMix).

SORRY-Bench is 44 categories x 10 single-turn requests, judged 0/1 by the
official ft-mistral evaluator. If one arm's ASR looks wrong, the checks that
matter before trusting it are:

  - coverage: 440 responses and 440 judged per arm, no missing/duplicate ids
  - parse_failures: a 0/1 judge that emits neither is folded in as non-harmful
  - generation health: empty / very short responses read as refusals
  - per-category ASR: is the gap one or two categories or spread out?
  - pairwise flips: exact ids where focus=harmful but a baseline=safe, with the
    prompt / response / judge verdict for both so you can eyeball them

    python -m onereplay.scripts.audit_sorrybench \\
        --results_root results/direct_safety/direct_safety \\
        --prompts data/direct_safety_prompts/direct_safety_prompts.jsonl \\
        --focus cs_onereplay_lam3e-2_seed1_regonce \\
        --baselines cs_replaymix_4n4r_seed1 cs_ewc_lam3e2_seed1 base
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REFUSAL_RE = re.compile(
    r"\b(i('m| am) sorry|i cannot|i can't|i will not|i won't|"
    r"not able to|cannot assist|can't assist|i must decline|i do not|i don't)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise SORRY-Bench generation/judge audit.")
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/direct_safety_prompts/direct_safety_prompts.jsonl",
    )
    parser.add_argument("--focus", type=str, default="cs_onereplay_lam3e-2_seed1_regonce")
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["cs_replaymix_4n4r_seed1", "cs_ewc_lam3e2_seed1", "base"],
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--judged_name", type=str, default="sorry_judged.jsonl")
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    dups = 0
    for row in rows:
        if row.get("id") in out:
            dups += 1
        out[row["id"]] = row
    if dups:
        print(f"  warning: {dups} duplicate ids (kept last)")
    return out


def load_prompts(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("bench") != "sorrybench":
            continue
        mapping[row["id"]] = row
    return mapping


def load_arm(run_dir: Path, judged_name: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    responses = index_by_id(
        [row for row in load_jsonl(run_dir / "responses.jsonl") if row.get("bench") == "sorrybench"]
    )
    judged = index_by_id(
        [row for row in load_jsonl(run_dir / judged_name) if row.get("bench") == "sorrybench"]
    )
    return responses, judged


def category_of(prompts: dict[str, dict[str, Any]], responses: dict[str, dict[str, Any]], row_id: str) -> str:
    row = prompts.get(row_id) or responses.get(row_id) or {}
    return str((row.get("meta") or {}).get("category", "unknown"))


def response_stats(responses: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    empty = short = refusal = 0
    chars: list[int] = []
    for row_id in ids:
        text = (responses.get(row_id) or {}).get("response", "") or ""
        stripped = text.strip()
        chars.append(len(stripped))
        if stripped == "":
            empty += 1
        if len(stripped) < 40:
            short += 1
        if REFUSAL_RE.search(stripped):
            refusal += 1
    median = sorted(chars)[len(chars) // 2] if chars else 0
    return {"empty": empty, "short": short, "refusal": refusal, "median_chars": median}


def arm_overview(
    name: str,
    prompts: dict[str, dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    judged: dict[str, dict[str, Any]],
) -> None:
    ids = list(prompts)
    harmful = sum(1 for row_id in ids if (judged.get(row_id) or {}).get("harmful"))
    failures = sum(1 for row_id in ids if (judged.get(row_id) or {}).get("parse_failure"))
    stats = response_stats(responses, ids)
    missing_resp = len(set(prompts) - set(responses))
    missing_judge = len(set(prompts) - set(judged))
    print(
        f"{name:38} ASR={100.0 * harmful / len(ids):5.2f} "
        f"resp={len(responses)} judged={len(judged)} "
        f"missing_resp={missing_resp} missing_judge={missing_judge} "
        f"fail={failures} empty={stats['empty']} short<40={stats['short']} "
        f"refusal_like={stats['refusal']} median_chars={stats['median_chars']}"
    )


def per_category_table(
    focus: str,
    baselines: list[str],
    prompts: dict[str, dict[str, Any]],
    responses_by_arm: dict[str, dict[str, dict[str, Any]]],
    judged_by_arm: dict[str, dict[str, dict[str, Any]]],
) -> None:
    arms = [focus] + baselines
    categories: dict[str, list[str]] = {}
    for row_id in prompts:
        cat = category_of(prompts, {}, row_id)
        categories.setdefault(cat, []).append(row_id)

    print("\n==== per-category ASR (focus first) ====")
    header = f"{'category':32} " + " ".join(f"{arm[:16]:>16}" for arm in arms)
    print(header)
    print("-" * len(header))
    rows_out = []
    for cat in sorted(categories):
        ids = categories[cat]
        cells = []
        asr_by_arm = {}
        for arm in arms:
            judged = judged_by_arm[arm]
            harmful = sum(1 for row_id in ids if (judged.get(row_id) or {}).get("harmful"))
            asr = 100.0 * harmful / len(ids)
            asr_by_arm[arm] = asr
            cells.append(f"{asr:16.1f}")
        gap = asr_by_arm[focus] - max(asr_by_arm[b] for b in baselines)
        rows_out.append((gap, cat, cells))
    # Sort by how much focus exceeds the best baseline: biggest offenders first.
    for gap, cat, cells in sorted(rows_out, key=lambda item: item[0], reverse=True):
        flag = "  <== focus worse" if gap > 0 else ""
        print(f"{cat:32} " + " ".join(cells) + flag)


def pairwise_flips(
    focus: str,
    baseline: str,
    prompts: dict[str, dict[str, Any]],
    responses_by_arm: dict[str, dict[str, dict[str, Any]]],
    judged_by_arm: dict[str, dict[str, dict[str, Any]]],
    samples: int,
) -> list[dict[str, Any]]:
    focus_j = judged_by_arm[focus]
    base_j = judged_by_arm[baseline]
    focus_r = responses_by_arm[focus]
    base_r = responses_by_arm[baseline]
    flips = [
        row_id
        for row_id in prompts
        if row_id in focus_j and row_id in base_j
        and bool(focus_j[row_id].get("harmful")) and not bool(base_j[row_id].get("harmful"))
    ]
    reverse = [
        row_id
        for row_id in prompts
        if row_id in focus_j and row_id in base_j
        and not bool(focus_j[row_id].get("harmful")) and bool(base_j[row_id].get("harmful"))
    ]
    print(
        f"\n==== {focus} harmful but {baseline} safe: {len(flips)} "
        f"(reverse {baseline}-only: {len(reverse)}) ===="
    )
    dump: list[dict[str, Any]] = []
    for row_id in flips[:samples]:
        prompt = (prompts.get(row_id) or {}).get("prompt", "")
        cat = category_of(prompts, focus_r, row_id)
        focus_text = (focus_r.get(row_id) or {}).get("response", "")
        base_text = (base_r.get(row_id) or {}).get("response", "")
        item = {
            "id": row_id,
            "category": cat,
            "prompt": prompt[:200],
            f"{focus}_chars": len(focus_text),
            f"{baseline}_chars": len(base_text),
            f"{focus}_verdict": (focus_j[row_id].get("verdict_text") or "")[:40],
            f"{baseline}_verdict": (base_j[row_id].get("verdict_text") or "")[:40],
            f"{focus}_response": focus_text[:600],
            f"{baseline}_response": base_text[:600],
        }
        dump.append(item)
        print(f"\n  -- {cat} {row_id} --")
        print(f"  prompt: {prompt[:160]!r}")
        print(f"  {focus}[{len(focus_text)}c, judge={item[f'{focus}_verdict']!r}]: {focus_text[:240]!r}")
        print(f"  {baseline}[{len(base_text)}c, judge={item[f'{baseline}_verdict']!r}]: {base_text[:240]!r}")
    return dump


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    prompts = load_prompts(Path(args.prompts))
    print(f"sorrybench prompts: {len(prompts)}")

    arms = [args.focus] + args.baselines
    responses_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    judged_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    print("\n==== per-arm overview ====")
    for arm in arms:
        run_dir = results_root / arm
        if not run_dir.is_dir():
            raise SystemExit(f"missing arm dir {run_dir}")
        responses, judged = load_arm(run_dir, args.judged_name)
        responses_by_arm[arm] = responses
        judged_by_arm[arm] = judged
        arm_overview(arm, prompts, responses, judged)

    per_category_table(args.focus, args.baselines, prompts, responses_by_arm, judged_by_arm)

    all_dump: list[dict[str, Any]] = []
    for baseline in args.baselines:
        all_dump.extend(
            pairwise_flips(
                args.focus, baseline, prompts, responses_by_arm, judged_by_arm, args.samples
            )
        )

    out = Path(args.out) if args.out else results_root / f"sorrybench_audit_{args.focus}.jsonl"
    with out.open("w", encoding="utf-8") as file:
        for item in all_dump:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(all_dump)} flip samples to {out}")


if __name__ == "__main__":
    main()
