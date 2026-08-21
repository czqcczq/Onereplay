"""Throwaway check for the vanilla replay data path (no model needed)."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset  # noqa: E402

from onereplay.data.chat import build_sft_tokenize_fn  # noqa: E402
from onereplay.data.replay import build_replay_dataset, mix_replay_into_train  # noqa: E402


class StubTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(c) % 100 for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def main():
    tmp = Path(tempfile.mkdtemp())
    flan_dir = tmp / "flan" / "train"
    flan_dir.mkdir(parents=True)
    with (flan_dir / "part.jsonl").open("w", encoding="utf-8") as f:
        for i in range(200):
            # every 10th row has an empty target and must be filtered out
            target = "" if i % 10 == 0 else f"flan answer {i}"
            f.write(json.dumps({"inputs": f"flan question {i}", "targets": target}) + "\n")

    tokenizer = StubTokenizer()
    args = argparse.Namespace(
        seed=1,
        max_len=512,
        map_cache_dir="",
        replay_ratio=0.05,
        replay_dataset_path="",
        replay_data_files=str(flan_dir / "*.jsonl"),
        replay_split="train",
        replay_cache_dir="",
        replay_input_column="inputs",
        replay_target_column="targets",
        replay_pool_size=100,
        replay_sample_seed=1,
    )

    train = Dataset.from_dict(
        {
            "instruction": [f"cs question {i}" for i in range(1000)],
            "input": [""] * 1000,
            "output": [f"cs answer {i}" for i in range(1000)],
            "answer": [f"a{i}" for i in range(1000)],
        }
    ).map(build_sft_tokenize_fn(tokenizer, args.max_len))

    mixed = mix_replay_into_train(args, tokenizer, train)

    print("train columns :", sorted(train.column_names))
    print("mixed columns :", sorted(mixed.column_names))
    print("mixed rows    :", len(mixed))
    assert len(mixed) == 1050, len(mixed)
    assert "answer" not in mixed.column_names, "unshared column should be dropped"
    assert set(mixed.column_names) == {
        "instruction",
        "input",
        "output",
        "input_ids",
        "labels",
        "attention_mask",
    }

    replay_rows = [row for row in mixed if row["instruction"].startswith("flan")]
    print("replay rows   :", len(replay_rows))
    assert len(replay_rows) == 50, len(replay_rows)
    assert all(row["output"] for row in replay_rows), "empty targets leaked through"

    sample = replay_rows[0]
    supervised = [t for t in sample["labels"] if t != -100]
    print("sample instr  :", sample["instruction"])
    print("prompt masked :", sample["labels"].count(-100), "supervised:", len(supervised))
    assert supervised, "replay row has no supervised tokens"
    assert len(sample["labels"]) == len(sample["input_ids"])

    # nesting: a smaller ratio must be a subset of a larger one
    small = build_replay_dataset(args, tokenizer, 20)
    large = build_replay_dataset(args, tokenizer, 50)
    assert small["instruction"] == large["instruction"][:20], "replay subsets are not nested"
    print("nesting check : ok")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
