"""Self-test: collecting C on the self-distilled corpus covers exactly the rows replay trains on.

"OneReplay + self-distillation" only means something if both routes consume the
same old knowledge. Replay drops a row when the generation hit the token cap
(truncated=1, empty target) or when the schema mapping leaves an empty
instruction/output. collect_cov --require_target 1 has to drop exactly the same
rows, or C would be built from prompts replay never sees and the comparison
would no longer isolate "compress into C" versus "keep the corpus".

This runs on a hand-built corpus that covers every drop reason, so it needs no
GPU and no model weights.

Usage: python -m onereplay.scripts.verify_selfdistill_cov
"""

from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.data.old_knowledge import (  # noqa: E402
    build_collate_fn,
    filter_incomplete_rows,
    limit_dataset,
    load_old_knowledge_dataset,
)
from onereplay.data.replay import load_self_distilled_pool, to_sft_schema  # noqa: E402

# One row per reason a row can be kept or dropped. Schema matches what
# scripts/generate_replay_targets.py writes.
ROWS = [
    {"index": 0, "inputs": "q0", "targets": "a0", "truncated": False, "keep": True},
    {"index": 1, "inputs": "q1", "targets": "", "truncated": True, "keep": False},
    {"index": 2, "inputs": "q2", "targets": "a2", "truncated": False, "keep": True},
    # Prompt was longer than the ceiling, so generation was skipped entirely.
    {"index": 3, "inputs": "q3", "targets": "", "truncated": True, "keep": False},
    # Whitespace-only answer: not flagged truncated, but has no supervised
    # tokens once stripped, so to_sft_schema's non-empty filter drops it.
    {"index": 4, "inputs": "q4", "targets": "   ", "truncated": False, "keep": False},
    # Defensive: an empty prompt would make the user turn empty.
    {"index": 5, "inputs": "", "targets": "a5", "truncated": False, "keep": False},
    {"index": 6, "inputs": "q6", "targets": "a6", "truncated": False, "keep": True},
]


def write_corpus(path: Path) -> None:
    with path.open("w", encoding="utf-8") as sink:
        for row in ROWS:
            record = {key: value for key, value in row.items() if key != "keep"}
            record["gold_targets"] = "gold"
            record["prompt_tokens"] = 8
            record["target_tokens"] = 4
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")


def replay_rows(corpus: Path, cache_dir: str) -> set[tuple[str, str]]:
    """The (prompt, target) pairs the replay baseline actually trains on."""

    args = Namespace(
        replay_self_distill_file=str(corpus),
        replay_cache_dir=cache_dir,
        replay_drop_truncated=1,
        replay_input_column="inputs",
        replay_target_column="targets",
    )
    dataset = to_sft_schema(load_self_distilled_pool(args), args)
    return {(row["instruction"], row["output"]) for row in dataset}


def cov_rows(corpus: Path, cache_dir: str, require_target: int) -> set[tuple[str, str]]:
    """The (prompt, target) pairs collect_cov would forward through the model."""

    args = Namespace(
        dataset_path="",
        dataset_name="",
        dataset_config="",
        dataset_split="train",
        data_files=str(corpus),
        cache_dir=cache_dir,
        streaming=0,
        text_column="",
        input_column="inputs",
        target_column="targets",
        require_target=require_target,
        max_samples=0,
        sample_shuffle=0,
        sample_seed=1,
        shuffle_buffer_size=10,
    )
    dataset = load_old_knowledge_dataset(args)
    if args.require_target == 1:
        dataset = filter_incomplete_rows(dataset, args)
    dataset = limit_dataset(dataset, args)
    return {(row["inputs"].strip(), row["targets"].strip()) for row in dataset}


class StubTokenizer:
    """Just enough surface for build_collate_fn to touch truncation_side."""

    truncation_side = "right"
    chat_template = None


def check_row_sets(corpus: Path, cache_dir: str) -> list[str]:
    failures: list[str] = []
    expected = {(row["inputs"].strip(), row["targets"].strip()) for row in ROWS if row["keep"]}

    replay = replay_rows(corpus, cache_dir)
    if replay != expected:
        failures.append(f"replay kept {sorted(replay)}, expected {sorted(expected)}")

    filtered = cov_rows(corpus, cache_dir, require_target=1)
    if filtered != replay:
        only_cov = sorted(filtered - replay)
        only_replay = sorted(replay - filtered)
        failures.append(
            f"require_target=1 does not match replay; cov-only={only_cov} replay-only={only_replay}"
        )

    # The old default must keep behaving as before, so previously collected C
    # files stay reproducible from the same command line.
    unfiltered = cov_rows(corpus, cache_dir, require_target=0)
    if len(unfiltered) <= len(filtered):
        failures.append(
            f"require_target=0 kept {len(unfiltered)} rows, expected more than the "
            f"{len(filtered)} that survive filtering"
        )
    return failures


def check_truncation_side() -> list[str]:
    failures: list[str] = []

    tokenizer = StubTokenizer()
    build_collate_fn(tokenizer, Namespace(truncation_side="left", max_len=512, use_chat_template=1))
    if tokenizer.truncation_side != "left":
        failures.append(f"--truncation_side left did not apply: {tokenizer.truncation_side}")

    tokenizer = StubTokenizer()
    build_collate_fn(tokenizer, Namespace(truncation_side="", max_len=512, use_chat_template=1))
    if tokenizer.truncation_side != "right":
        failures.append(
            f"empty --truncation_side should leave the tokenizer default, got "
            f"{tokenizer.truncation_side}"
        )
    return failures


def main() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        corpus = Path(workspace) / "selfdistill.jsonl"
        write_corpus(corpus)
        cache_dir = str(Path(workspace) / "cache")

        failures = check_row_sets(corpus, cache_dir) + check_truncation_side()

    kept = sum(1 for row in ROWS if row["keep"])
    print(f"corpus: {len(ROWS)} rows, {kept} usable by both routes")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("OK: C covers exactly the replay rows, and truncation_side is wired through")


if __name__ == "__main__":
    main()
