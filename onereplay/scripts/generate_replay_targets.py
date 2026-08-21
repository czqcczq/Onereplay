"""Stage 1b CLI: rewrite the replay corpus with the base model's own answers.

The vanilla replay baseline trains on FLAN's gold targets, which are bare
one-line answers. For an already instruction-tuned base model that is not a
rehearsal of old knowledge but a new task with a very different output style,
so the baseline drags the model toward FLAN's answer-key format instead of
preserving what it could do. It also starts at a loss around 3.0 while the new
task sits near 0.05, which lets a nominal 10% of rows take over most of the
gradient.

This script replaces every target with the base model's *own* greedy answer to
the same FLAN prompt. Replay then starts near zero loss and only anchors the
model to W0's behavior, which is exactly what the OneReplay penalty
approximates through C. Both routes end up consuming the same information, so
the head-to-head measures how the old knowledge is used rather than which
corpus is being learned.

The pool is shuffled, truncated and schema-mapped exactly like replay.py does,
so row i here is row i there and every subset stays nested inside the pool
that produced C.

Generation length is capped so that prompt + answer + chat template always fit
inside --max_len. A row whose answer would not fit is written back with an
empty target and truncated=1; training drops those rows rather than learning
"never emit a stop token" from a cut-off answer.

Usage: python -m onereplay.scripts.generate_replay_targets [args]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from onereplay.core.modeling import load_causal_lm_and_tokenizer, set_seed  # noqa: E402
from onereplay.data.chat import apply_prompt_template  # noqa: E402
from onereplay.data.replay import load_replay_pool, to_sft_schema  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate base-model targets for self-distillation replay."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)

    # Corpus input. Mirror what 01_collect_cov and the replay baseline used.
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument(
        "--data_files",
        type=str,
        default="",
        help="Glob for the old-knowledge json/jsonl files, e.g. /path/flan/train/*.jsonl",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--input_column", type=str, default="inputs")
    parser.add_argument("--target_column", type=str, default="targets")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=20000,
        help="Rows kept from the shuffled corpus. Match collect_cov's --max_samples.",
    )
    parser.add_argument(
        "--pool_offset",
        type=int,
        default=0,
        help=(
            "Rows skipped at the front of the shuffled corpus before taking "
            "--pool_size. 0 reproduces the training pool. Setting it to the "
            "training pool's size carves a held-out slice out of the same "
            "shuffle, disjoint from every row replay trains on, which is what "
            "the retention probe scores. The written index stays relative to "
            "the slice, so the loader needs no special case."
        ),
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=1,
        help="Must match collect_cov's --sample_seed so the subset stays nested.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Rows to generate, counted from the front of the pool. 0 does the whole pool.",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=384,
        help="Upper bound on the generated answer; the per-batch budget can be lower.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=16,
        help=(
            "Rows whose prompt leaves less room than this inside --max_len are "
            "written back empty instead of getting a stunted answer."
        ),
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="Training sequence budget. Must match the --max_len used for training.",
    )
    parser.add_argument(
        "--length_margin",
        type=int,
        default=8,
        help="Tokens held back from the budget for the trailing template tokens.",
    )

    parser.add_argument(
        "--enable_thinking",
        type=int,
        default=0,
        help=(
            "1 opens Qwen3's thinking block so the distilled target carries a "
            "reasoning trace. Traces are several times longer, so raise "
            "--max_len / --max_new_tokens or most rows come back truncated. "
            "collect_cov must then run with --concat_prompt_target 1, otherwise "
            "the chat template strips <think> back out."
        ),
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--resume",
        type=int,
        default=1,
        help="1 skips rows already present in output_path so a requeued job continues.",
    )
    parser.add_argument("--log_every", type=int, default=20, help="Progress print every N batches.")
    return parser.parse_args()


def build_pool(args: argparse.Namespace):
    """Reproduce replay.py's pool exactly: same loader, shuffle, cut, schema."""

    loader_args = Namespace(
        replay_dataset_path=args.dataset_path,
        replay_data_files=args.data_files,
        replay_split=args.split,
        replay_cache_dir=args.cache_dir,
        replay_input_column=args.input_column,
        replay_target_column=args.target_column,
    )
    pool = load_replay_pool(loader_args)
    pool = pool.shuffle(seed=args.sample_seed)
    start = min(max(args.pool_offset, 0), len(pool))
    end = min(start + args.pool_size, len(pool)) if args.pool_size > 0 else len(pool)
    if start >= end:
        raise ValueError(
            f"--pool_offset {args.pool_offset} leaves no rows: the corpus holds {len(pool)} "
            f"rows and the slice [{start}, {end}) is empty"
        )
    if start > 0:
        print(f"pool slice [{start}, {end}) of {len(pool)} shuffled rows")
    pool = pool.select(range(start, end))
    return to_sft_schema(pool, loader_args)


def read_done_indices(output_path: Path) -> set[int]:
    """Indices already written, so a requeued job does not redo them."""

    if not output_path.is_file():
        return set()
    done: set[int] = set()
    with output_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["index"]))
            except (ValueError, KeyError):
                continue
    return done


def summarize(output_path: Path) -> None:
    """Report how many rows survive into training and how long the answers are."""

    total = 0
    usable = 0
    truncated = 0
    empty = 0
    target_tokens = 0
    with output_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            if record.get("truncated"):
                truncated += 1
            elif not record.get("targets", "").strip():
                empty += 1
            else:
                usable += 1
                target_tokens += int(record.get("target_tokens", 0))
    print(
        f"generated rows      : {total}\n"
        f"usable for training : {usable}\n"
        f"dropped (truncated) : {truncated}\n"
        f"dropped (empty)     : {empty}\n"
        f"mean answer tokens  : {target_tokens / max(usable, 1):.1f}"
    )
    if usable < total * 0.8:
        print(
            "warning: more than 20% of the pool is unusable; raise --max_len or "
            "--max_new_tokens, or generate a larger --pool_size"
        )


def generate_targets(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("loading base model and tokenizer")
    # No extra_config here: names like max_new_tokens would land on the model
    # config and could leak into generation defaults.
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
    )
    model.to(device)
    model.eval()

    dataset = build_pool(args)
    total = len(dataset)
    if args.num_samples > 0:
        total = min(args.num_samples, total)
    print(f"pool holds {len(dataset)} usable rows; generating targets for {total}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done_indices(output_path) if args.resume == 1 else set()
    if done:
        print(f"resuming: {len(done)} rows already in {output_path}")

    todo = [index for index in range(total) if index not in done]
    if not todo:
        print("nothing left to generate")
        summarize(output_path)
        return

    rows = dataset.select(todo)
    instructions = dict(zip(todo, rows["instruction"]))
    gold = dict(zip(todo, rows["output"]))
    prompts = {
        index: apply_prompt_template(
            tokenizer, instruction, enable_thinking=args.enable_thinking == 1
        )
        for index, instruction in instructions.items()
    }
    prompt_tokens = {
        index: len(ids)
        for index, ids in zip(
            todo,
            tokenizer([prompts[index] for index in todo], add_special_tokens=False)["input_ids"],
        )
    }

    def write(sink, index: int, target: str, truncated: bool, answer_tokens: int) -> None:
        sink.write(
            json.dumps(
                {
                    "index": index,
                    "inputs": instructions[index],
                    "targets": target,
                    "gold_targets": gold[index],
                    "truncated": truncated,
                    "prompt_tokens": prompt_tokens[index],
                    "target_tokens": answer_tokens,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    # A prompt this long leaves no room for an answer inside the training
    # budget. Filtering it here rather than at batch time also keeps a handful
    # of pathological FLAN rows from blowing up the generation batch.
    prompt_ceiling = args.max_len - args.min_new_tokens - args.length_margin
    over_long = [index for index in todo if prompt_tokens[index] > prompt_ceiling]
    # Length-sorted batches keep padding low. Every record carries its index,
    # so the file can be written out of order and sorted when it is loaded.
    order = sorted(
        (index for index in todo if prompt_tokens[index] <= prompt_ceiling),
        key=lambda index: prompt_tokens[index],
    )
    if over_long:
        print(f"{len(over_long)} rows have a prompt longer than {prompt_ceiling} tokens; skipping")

    eos_token_id = tokenizer.eos_token_id
    num_batches = (len(order) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    with output_path.open("a", encoding="utf-8") as sink:
        for index in over_long:
            write(sink, index, "", True, 0)
        sink.flush()

        for batch_number, start in enumerate(range(0, len(order), args.batch_size), start=1):
            chunk = order[start : start + args.batch_size]
            encoded = tokenizer(
                [prompts[index] for index in chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            budget = min(
                args.max_new_tokens,
                args.max_len - max(prompt_tokens[index] for index in chunk) - args.length_margin,
            )

            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=budget,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                )
            generated = output_ids[:, encoded["input_ids"].shape[1] :]

            for index, ids in zip(chunk, generated):
                token_ids = ids.tolist()
                # No stop token means the answer ran into the budget. Training
                # on it would teach the model not to end its turn, so drop the
                # text and let the loader filter the row out.
                stopped = eos_token_id in token_ids
                answer_tokens = token_ids.index(eos_token_id) if stopped else len(token_ids)
                text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                write(sink, index, text if stopped else "", not stopped, answer_tokens)
            sink.flush()

            if args.log_every > 0 and batch_number % args.log_every == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / batch_number * (num_batches - batch_number)
                print(
                    f"batch {batch_number}/{num_batches} budget={budget} "
                    f"{elapsed / 60:.1f}min elapsed, ~{remaining / 60:.1f}min left",
                    flush=True,
                )

    print(f"wrote {output_path}")
    summarize(output_path)


def main() -> None:
    generate_targets(parse_args())


if __name__ == "__main__":
    main()
