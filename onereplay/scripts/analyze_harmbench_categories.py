"""HarmBench ASR broken down by FunctionalCategory (NSPO / HarmBench text test).

The GPT-4 judge file only stores {id, bench, harmful, parse_failure}. Category
membership lives on the frozen prompts JSONL (`meta.functional_category`), so
this script joins the two and reports:

    Method    Standard 159    Contextual 81    Copyright 80    All 320

    python -m onereplay.scripts.analyze_harmbench_categories \\
        --results_root results/direct_safety/direct_safety \\
        --prompts data/direct_safety_prompts/direct_safety_prompts.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CATEGORIES = ("standard", "contextual", "copyright")
CATEGORY_N = {"standard": 159, "contextual": 81, "copyright": 80, "all": 320}
ARM_LABEL = {
    "base": "Base",
    "cs_vanilla_seed1": "Vanilla",
    "cs_ewc_lam3e2_seed1": "EWC",
    "cs_replaymix_4n4r_seed1": "ReplayMix",
    "cs_onereplay_lam3e-2_seed1_regonce": "OneReplay",
}
DEFAULT_ORDER = (
    "base",
    "cs_vanilla_seed1",
    "cs_ewc_lam3e2_seed1",
    "cs_replaymix_4n4r_seed1",
    "cs_onereplay_lam3e-2_seed1_regonce",
)
JUDGED_CANDIDATES = ("gpt4_judged.jsonl", "gpt_judged.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HarmBench ASR by functional category.")
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/direct_safety_prompts/direct_safety_prompts.jsonl",
    )
    parser.add_argument("--order", type=str, nargs="*", default=list(DEFAULT_ORDER))
    parser.add_argument("--out_prefix", type=str, default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_category_map(prompts_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in load_jsonl(prompts_path):
        if row.get("bench") != "harmbench":
            continue
        category = (row.get("meta") or {}).get("functional_category", "").strip().lower()
        if not category:
            raise SystemExit(f"harmbench row {row.get('id')} has no functional_category")
        mapping[row["id"]] = category
    if len(mapping) != CATEGORY_N["all"]:
        raise SystemExit(
            f"expected {CATEGORY_N['all']} harmbench prompts, got {len(mapping)} in {prompts_path}"
        )
    return mapping


def find_judged(run_dir: Path) -> Path | None:
    for name in JUDGED_CANDIDATES:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def asr(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {"asr": None, "harmful": 0, "total": 0, "parse_failures": 0}
    harmful = sum(1 for record in records if record.get("harmful"))
    failures = sum(1 for record in records if record.get("parse_failure"))
    return {
        "asr": 100.0 * harmful / total,
        "harmful": harmful,
        "total": total,
        "parse_failures": failures,
    }


def collect_arm(run_dir: Path, category_of: dict[str, str]) -> dict[str, dict[str, Any]]:
    judged_path = find_judged(run_dir)
    if judged_path is None:
        raise SystemExit(f"no gpt4_judged.jsonl / gpt_judged.jsonl under {run_dir}")
    by_cat: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
    seen: set[str] = set()
    for record in load_jsonl(judged_path):
        if record.get("bench") != "harmbench":
            continue
        row_id = record.get("id")
        if row_id not in category_of:
            continue
        if row_id in seen:
            continue
        seen.add(row_id)
        by_cat[category_of[row_id]].append(record)
    all_rows = [row for name in CATEGORIES for row in by_cat[name]]
    return {name: asr(by_cat[name]) for name in CATEGORIES} | {"all": asr(all_rows)}


def format_cell(entry: dict[str, Any]) -> str:
    if entry["asr"] is None:
        return "n/a"
    suffix = "" if entry["parse_failures"] == 0 else f" (!{entry['parse_failures']})"
    return f"{entry['asr']:.2f}{suffix}"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    prompts_path = Path(args.prompts)
    if not results_root.is_dir():
        raise SystemExit(f"results_root not found: {results_root}")
    if not prompts_path.is_file():
        raise SystemExit(f"prompts not found: {prompts_path}")

    category_of = load_category_map(prompts_path)
    by_name = {path.name: path for path in results_root.iterdir() if path.is_dir()}
    order = [name for name in args.order if name in by_name]
    missing = [name for name in args.order if name not in by_name]
    if missing:
        print(f"warning: requested runs with no directory: {missing}")
    if not order:
        order = sorted(by_name)
    if not order:
        raise SystemExit(f"no run directories under {results_root}")

    table: dict[str, dict[str, dict[str, Any]]] = {}
    for name in order:
        table[name] = collect_arm(by_name[name], category_of)

    headers = ["Method"] + [f"{label} {CATEGORY_N[key]}" for key, label in (
        ("standard", "Standard"),
        ("contextual", "Contextual"),
        ("copyright", "Copyright"),
        ("all", "All"),
    )]
    print("\t".join(headers))
    for name in order:
        cells = [ARM_LABEL.get(name, name)] + [
            format_cell(table[name][key]) for key in (*CATEGORIES, "all")
        ]
        print("\t".join(cells))
        for key in (*CATEGORIES, "all"):
            entry = table[name][key]
            expected = CATEGORY_N[key]
            if entry["total"] != expected:
                print(
                    f"  warning: {name} {key} has {entry['total']} rows, expected {expected}",
                    flush=True,
                )

    out_prefix = Path(args.out_prefix) if args.out_prefix else results_root / "harmbench_categories"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for name in order:
            writer.writerow(
                [ARM_LABEL.get(name, name)]
                + [format_cell(table[name][key]) for key in (*CATEGORIES, "all")]
            )
    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {csv_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
