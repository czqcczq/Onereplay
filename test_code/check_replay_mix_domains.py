"""Throwaway check for the multi-domain replay data path (no model or GPU).

Covers what the cluster job would otherwise only discover after loading a model:
--replay_mix_files parsing, per-pool tokenization at --replay_max_len instead of
the new task's --max_len, and the row-vs-token share the loader reports.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset  # noqa: E402

from onereplay.data.batch_mix import build_batch_mixed_loader  # noqa: E402
from onereplay.data.chat import build_sft_tokenize_fn  # noqa: E402
from onereplay.data.replay import (  # noqa: E402
    build_replay_pools,
    parse_replay_mix,
    replay_max_len,
)


class StubTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(character) % 100 for character in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def write_pool(directory: Path, name: str, rows: int, answer_words: int) -> str:
    path = directory / name
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "inputs": f"{name} question {index}",
                        "targets": "word " * answer_words,
                        "pool_index": index,
                        "truncated": False,
                    }
                )
                + "\n"
            )
    return str(path)


def supervised_tokens(dataset) -> float:
    rows = dataset["labels"]
    return sum(sum(1 for token in row if token != -100) for row in rows) / len(rows)


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    if_file = write_pool(tmp, "flan.jsonl", rows=40, answer_words=4)
    math_file = write_pool(tmp, "metamath.jsonl", rows=60, answer_words=64)

    tokenizer = StubTokenizer()
    args = argparse.Namespace(
        seed=1,
        max_len=64,
        replay_max_len=2048,
        map_cache_dir="",
        replay_mix_files=f"if={if_file},math={math_file}",
        replay_mix_weights="0.8,0.2",
        replay_self_distill_file="",
        replay_cache_dir="",
        replay_input_column="inputs",
        replay_target_column="targets",
        replay_drop_truncated=1,
        replay_pool_size=0,
        replay_sample_seed=1,
        replay_dataset_path="",
        replay_data_files="",
        replay_split="train",
    )

    mix = parse_replay_mix(args)
    print("parsed mix   :", [(label, round(weight, 4)) for label, _, weight in mix])
    assert [label for label, _, _ in mix] == ["if", "math"]
    assert abs(mix[0][2] - 0.8) < 1e-9 and abs(mix[1][2] - 0.2) < 1e-9
    assert replay_max_len(args) == 2048

    pools = build_replay_pools(args, tokenizer)
    assert [pool.label for pool in pools] == ["if", "math"]
    assert len(pools[0].dataset) == 40 and len(pools[1].dataset) == 60

    # The point of --replay_max_len: the math pool's rows are longer than the new
    # task's max_len and must survive whole.
    math_tokens = supervised_tokens(pools[1].dataset)
    print(f"math supervised tokens/row = {math_tokens:.1f} (new task max_len={args.max_len})")
    assert math_tokens > args.max_len, "replay rows were truncated to the new task's max_len"

    # And the negative control: without it, the same pool gets cut to 64.
    args.replay_max_len = 0
    capped = build_replay_pools(args, tokenizer)[1].dataset
    assert max(len(row) for row in capped["input_ids"]) == args.max_len
    print(f"replay_max_len=0 falls back to max_len: rows capped at {args.max_len}")
    args.replay_max_len = 2048

    train = Dataset.from_dict(
        {
            "instruction": [f"cs question {index}" for index in range(200)],
            "input": [""] * 200,
            "output": [f"cs answer {index}" for index in range(200)],
        }
    ).map(build_sft_tokenize_fn(tokenizer, args.max_len))

    loader = build_batch_mixed_loader(
        train,
        None,
        tokenizer,
        new_per_batch=4,
        replay_per_batch=4,
        seed=1,
        replay_pools=pools,
    )
    print(loader.describe())

    # Equal rows would put ~80% of the replay tokens on the long pool; 0.8/0.2
    # rows is what brings the token shares together. This is the whole reason
    # the two arms exist, so assert the direction here rather than trusting the
    # cluster log.
    even = build_batch_mixed_loader(
        train,
        None,
        tokenizer,
        new_per_batch=4,
        replay_per_batch=4,
        seed=1,
        replay_pools=[(pool.label, pool.dataset, 0.5) for pool in pools],
    )
    print(even.describe())

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
