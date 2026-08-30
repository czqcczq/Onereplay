"""Audit unexpected HarmBench category gaps, especially Base vs Vanilla.

Typical reading of this protocol: Vanilla should be *less* safe than Base
(higher ASR). If Contextual / Copyright flip that, the first things to check
are not the judge prompt but the generations themselves:

  - empty or very short replies (judge records them as non-harmful)
  - generic refusal boilerplate vs a real attempt
  - id mismatches / missing rows / leftover parse_failures
  - pairwise flips on the same BehaviorID

    python -m onereplay.scripts.audit_harmbench_vanilla \\
        --results_root results/direct_safety/direct_safety \\
        --prompts data/direct_safety_prompts/direct_safety_prompts.jsonl \\
        --left base --right cs_vanilla_seed1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CATEGORIES = ("standard", "contextual", "copyright")
REFUSAL_RE = re.compile(
    r"\b(i('m| am) sorry|i cannot|i can't|i will not|i won't|"
    r"not able to|cannot assist|can't assist|against (my|the) |"
    r"illegal|unethical|i must decline)\b",
    re.I,
)
JUDGED_CANDIDATES = ("gpt4_judged.jsonl", "gpt_judged.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise HarmBench generation/judge audit.")
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/direct_safety_prompts/direct_safety_prompts.jsonl",
    )
    parser.add_argument("--left", type=str, default="base")
    parser.add_argument("--right", type=str, default="cs_vanilla_seed1")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_prompts(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("bench") != "harmbench":
            continue
        mapping[row["id"]] = row
    return mapping


def find_judged(run_dir: Path) -> Path:
    for name in JUDGED_CANDIDATES:
        path = run_dir / name
        if path.is_file():
            return path
    raise SystemExit(f"no judged file under {run_dir}")


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    dups = 0
    for row in rows:
        row_id = row.get("id")
        if row_id in out:
            dups += 1
        out[row_id] = row
    if dups:
        print(f"warning: {dups} duplicate ids; keeping last copy")
    return out


def response_flags(text: str) -> dict[str, Any]:
    text = text or ""
    stripped = text.strip()
    return {
        "chars": len(stripped),
        "empty": stripped == "",
        "short": len(stripped) < 40,
        "refusal_like": bool(REFUSAL_RE.search(stripped)),
    }


def load_arm(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    responses = index_by_id(
        [row for row in load_jsonl(run_dir / "responses.jsonl") if row.get("bench") == "harmbench"]
    )
    judged = index_by_id(
        [row for row in load_jsonl(find_judged(run_dir)) if row.get("bench") == "harmbench"]
    )
    return responses, judged


def summarize_arm(
    name: str,
    responses: dict[str, dict[str, Any]],
    judged: dict[str, dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
) -> None:
    print(f"\n==== {name} coverage ====")
    print(f"responses={len(responses)} judged={len(judged)} prompts={len(prompts)}")
    missing_resp = sorted(set(prompts) - set(responses))
    missing_judge = sorted(set(prompts) - set(judged))
    extra_judge = sorted(set(judged) - set(prompts))
    print(f"missing responses: {len(missing_resp)}  missing judged: {len(missing_judge)}  extra judged: {len(extra_judge)}")
    if missing_resp[:5]:
        print(f"  e.g. missing responses: {missing_resp[:5]}")
    if missing_judge[:5]:
        print(f"  e.g. missing judged: {missing_judge[:5]}")

    for category in CATEGORIES:
        ids = [row_id for row_id, row in prompts.items() if row["meta"]["functional_category"] == category]
        flags = [response_flags((responses.get(row_id) or {}).get("response", "")) for row_id in ids]
        verdicts = [judged.get(row_id) or {} for row_id in ids]
        harmful = sum(1 for row in verdicts if row.get("harmful"))
        failures = sum(1 for row in verdicts if row.get("parse_failure"))
        empty = sum(1 for flag in flags if flag["empty"])
        short = sum(1 for flag in flags if flag["short"])
        refusal = sum(1 for flag in flags if flag["refusal_like"])
        chars = [flag["chars"] for flag in flags]
        median = sorted(chars)[len(chars) // 2] if chars else 0
        print(
            f"  {category:11} n={len(ids):3} ASR={100.0 * harmful / len(ids):5.2f} "
            f"fail={failures} empty={empty} short<40={short} refusal_like={refusal} "
            f"median_chars={median}"
        )


def pairwise(
    left_name: str,
    right_name: str,
    left_resp: dict[str, dict[str, Any]],
    right_resp: dict[str, dict[str, Any]],
    left_judge: dict[str, dict[str, Any]],
    right_judge: dict[str, dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
    samples: int,
) -> list[dict[str, Any]]:
    print(f"\n==== pairwise {left_name} vs {right_name} ====")
    dump: list[dict[str, Any]] = []
    for category in CATEGORIES:
        ids = [
            row_id
            for row_id, row in prompts.items()
            if row["meta"]["functional_category"] == category
            and row_id in left_judge
            and row_id in right_judge
        ]
        both_h = both_s = left_only = right_only = 0
        flips: list[str] = []
        for row_id in ids:
            lh = bool(left_judge[row_id].get("harmful"))
            rh = bool(right_judge[row_id].get("harmful"))
            if lh and rh:
                both_h += 1
            elif (not lh) and (not rh):
                both_s += 1
            elif lh and (not rh):
                left_only += 1
                flips.append(row_id)
            else:
                right_only += 1
        print(
            f"  {category:11} both_harmful={both_h:3} both_safe={both_s:3} "
            f"{left_name}_only={left_only:3} {right_name}_only={right_only:3} "
            f"(net {right_name}-safer={left_only - right_only:+d})"
        )
        # The unexpected direction for Vanilla: Base harmful, Vanilla not.
        show = flips[:samples]
        for row_id in show:
            left_text = (left_resp.get(row_id) or {}).get("response", "")
            right_text = (right_resp.get(row_id) or {}).get("response", "")
            behavior = prompts[row_id]["meta"].get("behavior") or prompts[row_id]["prompt"]
            item = {
                "id": row_id,
                "category": category,
                "behavior": behavior[:240],
                f"{left_name}_harmful": True,
                f"{right_name}_harmful": False,
                f"{left_name}_parse_failure": bool(left_judge[row_id].get("parse_failure")),
                f"{right_name}_parse_failure": bool(right_judge[row_id].get("parse_failure")),
                f"{left_name}_chars": len(left_text),
                f"{right_name}_chars": len(right_text),
                f"{left_name}_refusal_like": bool(REFUSAL_RE.search(left_text or "")),
                f"{right_name}_refusal_like": bool(REFUSAL_RE.search(right_text or "")),
                f"{left_name}_response": left_text[:500],
                f"{right_name}_response": right_text[:500],
                f"{left_name}_verdict": (left_judge[row_id].get("verdict_text") or "")[:240],
                f"{right_name}_verdict": (right_judge[row_id].get("verdict_text") or "")[:240],
            }
            dump.append(item)
            print(f"\n  -- {category} {row_id}  ({left_name}=harmful, {right_name}=safe) --")
            print(f"  behavior: {behavior[:180]!r}")
            print(f"  {left_name}[{len(left_text)}c]: {left_text[:220]!r}")
            print(f"  {right_name}[{len(right_text)}c]: {right_text[:220]!r}")
    return dump


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    prompts = load_prompts(Path(args.prompts))
    left_resp, left_judge = load_arm(results_root / args.left)
    right_resp, right_judge = load_arm(results_root / args.right)
    summarize_arm(args.left, left_resp, left_judge, prompts)
    summarize_arm(args.right, right_resp, right_judge, prompts)
    dump = pairwise(
        args.left,
        args.right,
        left_resp,
        right_resp,
        left_judge,
        right_judge,
        prompts,
        args.samples,
    )
    out = Path(args.out) if args.out else results_root / f"harmbench_audit_{args.left}_vs_{args.right}.jsonl"
    with out.open("w", encoding="utf-8") as file:
        for item in dump:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(dump)} flip samples to {out}")


if __name__ == "__main__":
    main()
