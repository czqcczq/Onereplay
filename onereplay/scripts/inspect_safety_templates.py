"""Inspect safety responses per jailbreak template to see why a template failed.

A template with harmful_rate 0 can mean two very different things: the model
robustly refused it, or the model could not follow it at all and produced an
off-topic / degenerate reply that the judge labeled "other". This script breaks
each template down by outcome and prints sampled prompt/response pairs so the
distinction is visible before templates are swapped.

Reads judged.jsonl (has `outcome`) or responses.jsonl (falls back to the
keyword_refusal flag). No GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse the file to inspect, template filter, and sample size."""

    parser = argparse.ArgumentParser(description="Inspect safety responses per template.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="A judged.jsonl or responses.jsonl file (or a glob matching one).",
    )
    parser.add_argument("--track", type=str, default="B")
    parser.add_argument(
        "--template_ids",
        type=str,
        default="",
        help="Comma-separated template_ids to inspect; empty = all in the track.",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--prompt_chars", type=int, default=280)
    parser.add_argument("--response_chars", type=int, default=280)
    return parser.parse_args()


def resolve_file(spec: str) -> Path:
    """Resolve a single file path, expanding a glob to its first match."""

    matches = glob.glob(spec)
    path = Path(matches[0]) if matches else Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"No file matched: {spec!r}")
    return path


def outcome_of(record: dict) -> str:
    """Prefer the judge outcome; fall back to the keyword_refusal flag."""

    if "outcome" in record:
        return record["outcome"]
    return "kw_refusal" if record.get("keyword_refusal") else "kw_other"


def clip(text: str, limit: int) -> str:
    """One-line, length-capped preview of a field."""

    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + " …"


def main() -> None:
    """Group records by template and print outcome mix plus samples."""

    args = parse_args()
    path = resolve_file(args.file)
    wanted = {t.strip() for t in args.template_ids.split(",") if t.strip()}

    by_template: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("track") != args.track:
                continue
            template_id = record.get("template_id", "?")
            if wanted and template_id not in wanted:
                continue
            by_template.setdefault(template_id, []).append(record)

    print(f"file: {path}")
    print(f"track: {args.track}\n")
    for template_id in sorted(by_template):
        records = by_template[template_id]
        counts: dict[str, int] = {}
        length_sum = 0
        for record in records:
            counts[outcome_of(record)] = counts.get(outcome_of(record), 0) + 1
            length_sum += int(record.get("response_tokens", 0))
        n = len(records)
        mix = ", ".join(f"{k}={v} ({v / n:.0%})" for k, v in sorted(counts.items()))
        print(f"=== {template_id}  n={n}  mean_tokens={length_sum / max(n, 1):.1f}")
        print(f"    outcomes: {mix}")
        for record in records[: args.samples]:
            print(f"    - [{outcome_of(record)}] prompt: {clip(record.get('prompt', ''), args.prompt_chars)}")
            print(f"      response: {clip(record.get('response', ''), args.response_chars)}")
        print()


if __name__ == "__main__":
    main()
