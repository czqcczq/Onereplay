"""Aggregate direct-safety judgments into the main ASR table (NSPO protocol).

Reads the per-arm judgment files written by the two judges and, for each
benchmark independently, computes

    ASR = harmful_responses / total_responses x 100%

Nothing is merged across benchmarks -- there is no safe/unsafe/refusal
re-labeling. The main table reports AdvBench, HarmBench and SORRY-Bench ASR
side by side, plus their Macro Average (the plain mean of the three). The
reading that matters is per column: how far Vanilla drops from Base, and how
much of that drop each protection method (EWC / ReplayMix / OneReplay) claws
back toward Base.

Layout it expects, one directory per arm:

    <results_root>/<run_name>/gpt4_judged.jsonl    # advbench + harmbench rows
    <results_root>/<run_name>/sorry_judged.jsonl   # sorrybench rows

    python -m onereplay.scripts.analyze_direct_safety \\
        --results_root results/direct_safety/direct_safety \\
        --order base cs_vanilla_seed1 cs_ewc_lam3e2_seed1 \\
                cs_replaymix_4n4r_seed1 cs_onereplay_lam3e-2_seed1_regonce
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BENCHES = ("advbench", "harmbench", "sorrybench")
BENCH_LABEL = {"advbench": "AdvBench", "harmbench": "HarmBench", "sorrybench": "SORRY-Bench"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the direct-safety ASR main table.")
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Directory holding one subdir per run_name (the direct_safety metric dir).",
    )
    parser.add_argument(
        "--order",
        type=str,
        nargs="*",
        default=[],
        help="Run names in table order. Defaults to sorted directory names.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="",
        help="Where to write the table. Defaults to <results_root>/direct_safety_main_table.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def asr_for_bench(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """ASR over one benchmark's verdicts, carrying the parse-failure count."""

    if not records:
        return None
    total = len(records)
    harmful = sum(1 for record in records if record.get("harmful"))
    failures = sum(1 for record in records if record.get("parse_failure"))
    return {
        "asr": 100.0 * harmful / total,
        "harmful": harmful,
        "total": total,
        "parse_failures": failures,
    }


def collect_run(run_dir: Path) -> dict[str, dict[str, Any] | None]:
    """Split an arm's judgments back into the three benchmarks."""

    verdicts = load_jsonl(run_dir / "gpt4_judged.jsonl") + load_jsonl(run_dir / "sorry_judged.jsonl")
    by_bench: dict[str, list[dict[str, Any]]] = {bench: [] for bench in BENCHES}
    for record in verdicts:
        bench = record.get("bench")
        if bench in by_bench:
            by_bench[bench].append(record)
    return {bench: asr_for_bench(by_bench[bench]) for bench in BENCHES}


def macro_average(per_bench: dict[str, dict[str, Any] | None]) -> float | None:
    values = [per_bench[bench]["asr"] for bench in BENCHES if per_bench[bench] is not None]
    if len(values) != len(BENCHES):
        return None
    return sum(values) / len(BENCHES)


def format_cell(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "  n/a"
    suffix = "" if entry["parse_failures"] == 0 else f" (!{entry['parse_failures']})"
    return f"{entry['asr']:5.2f}{suffix}"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_dir():
        raise SystemExit(f"results_root not found: {results_root}")

    run_dirs = [path for path in results_root.iterdir() if path.is_dir()]
    by_name = {path.name: path for path in run_dirs}
    order = args.order or sorted(by_name)
    order = [name for name in order if name in by_name]
    missing = [name for name in (args.order or []) if name not in by_name]
    if missing:
        print(f"warning: requested runs with no directory: {missing}")
    if not order:
        raise SystemExit(f"no run directories under {results_root}")

    table: dict[str, dict[str, Any]] = {}
    for name in order:
        per_bench = collect_run(by_name[name])
        table[name] = {"per_bench": per_bench, "macro": macro_average(per_bench)}

    header = f"{'run_name':<40} " + " ".join(f"{BENCH_LABEL[b]:>14}" for b in BENCHES) + f"{'Macro':>10}"
    lines = [header, "-" * len(header)]
    for name in order:
        cells = " ".join(f"{format_cell(table[name]['per_bench'][b]):>14}" for b in BENCHES)
        macro = table[name]["macro"]
        macro_text = f"{macro:9.2f}" if macro is not None else "      n/a"
        lines.append(f"{name:<40} {cells}{macro_text}")
    report = "\n".join(lines)
    print("\nASR-% (lower is safer); (!n) marks n parse failures folded in as non-harmful\n")
    print(report)

    out_prefix = Path(args.out_prefix) if args.out_prefix else results_root / "direct_safety_main_table"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["run_name"]
            + [f"{BENCH_LABEL[b]}_ASR" for b in BENCHES]
            + [f"{BENCH_LABEL[b]}_parse_failures" for b in BENCHES]
            + ["Macro_Average"]
        )
        for name in order:
            per_bench = table[name]["per_bench"]
            asrs = [
                f"{per_bench[b]['asr']:.4f}" if per_bench[b] else "" for b in BENCHES
            ]
            fails = [
                str(per_bench[b]["parse_failures"]) if per_bench[b] else "" for b in BENCHES
            ]
            macro = table[name]["macro"]
            writer.writerow([name] + asrs + fails + ([f"{macro:.4f}"] if macro is not None else [""]))

    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {csv_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
