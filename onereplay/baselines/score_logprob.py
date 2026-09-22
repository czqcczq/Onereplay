"""Add OPR-SC's avg_logprob to a finished self-distillation file, by teacher forcing.

OPR-SC ranks candidates by the mean log-probability of the answer the model
itself produced. scripts/generate_replay_targets.py --record_logprob records it
while decoding; a self-distillation file generated without that flag has the
answers but not the score. Regenerating the whole pool just to get it would
cost hours of autoregressive decoding for a number one forward pass can give.

So this script feeds each row's existing answer back through the same model
behind the same chat prompt and averages the log-probabilities of the answer
tokens. Under the raw convention (log_softmax of the unmodified logits, the
default of generate_replay_targets --logprob_source) this is the same quantity
the decoder would have recorded, up to one caveat: the answer is re-tokenized
from its decoded text, and a decode/encode round trip can occasionally split a
string differently from the ids the model emitted. That moves a row's score by
a few thousandths at most and is recorded as logprob_method in every row.

The span matches generate_replay_targets: the answer tokens plus the stop token
when the answer ended with one (truncated=false), without it when the answer
ran into the budget (truncated=true). Rows with an empty target are copied
through unscored; build_opr_buffer drops them before ranking anyway.

The output is the input file with three fields added per scored row --
avg_logprob, logprob_tokens, logprob_method -- sorted by index, so it can be
handed to build_opr_buffer --pool as it is.

Usage:
    python -m onereplay.baselines.score_logprob \
        --model_dir <dir> --model_name Qwen3-8B \
        --input_path datasets/selfdistill/math_pool_selfdistill_Qwen3-8B.jsonl \
        --output_path results/qwen3-8b/opr_sd/scored_math.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from onereplay.core.modeling import load_causal_lm_and_tokenizer, set_seed  # noqa: E402
from onereplay.data.chat import apply_prompt_template  # noqa: E402

METHOD = "teacher_forced_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-forced avg_logprob for OPR-SC.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--input_column", type=str, default="inputs")
    parser.add_argument("--target_column", type=str, default="targets")
    parser.add_argument(
        "--enable_thinking",
        type=int,
        default=0,
        help=(
            "Must match what the self-distillation was generated with, since it "
            "changes the prompt the answer is conditioned on. 0 is "
            "generate_replay_targets' default."
        ),
    )
    parser.add_argument(
        "--max_tokens_per_batch",
        type=int,
        default=16384,
        help=(
            "Padded tokens per forward pass. The full logits tensor is "
            "batch x length x vocab, so on a 150k vocabulary this is what bounds "
            "memory, not the row count."
        ),
    )
    parser.add_argument("--max_batch_size", type=int, default=16)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: invalid JSON: {error}") from error
    return rows


def done_indices(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    done.add(int(json.loads(line)["index"]))
                except (ValueError, KeyError):
                    continue
    return done


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    if not rows:
        raise SystemExit(f"{input_path} is empty")
    if any("index" not in row for row in rows):
        raise SystemExit(
            f"{input_path} has rows without an index field; resume and the buffer's "
            "provenance both key on it."
        )
    indices = [int(row["index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise SystemExit(
            f"{input_path} repeats index values ({len(indices) - len(set(indices))} "
            "duplicates). Two files concatenated without renumbering, most likely; "
            "the scores sidecar keys on index, so renumber first."
        )

    done = done_indices(output_path) if args.resume == 1 else set()
    if args.resume != 1 and output_path.exists():
        output_path.unlink()
    todo = [row for row in rows if int(row["index"]) not in done]
    print(
        f"{input_path}: {len(rows)} rows, {len(done)} already scored in {output_path}, "
        f"{len(todo)} to go"
    )

    if todo:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_causal_lm_and_tokenizer(
            args.model_dir, args.model_name, args.use_bf16
        )
        model.to(device).eval()
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

        passthrough = []
        items = []
        for row in todo:
            target = str(row.get(args.target_column, "") or "")
            if not target.strip():
                passthrough.append(row)
                continue
            prompt = apply_prompt_template(
                tokenizer,
                str(row.get(args.input_column, "") or ""),
                enable_thinking=args.enable_thinking == 1,
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            # generate_replay_targets averages over the tokens the model emitted,
            # which includes the stop token when there was one.
            if not bool(row.get("truncated", False)):
                answer_ids = answer_ids + [eos]
            if not answer_ids:
                passthrough.append(row)
                continue
            items.append((row, prompt_ids, answer_ids))

        # Longest first, so a memory problem shows up in the first batch rather
        # than hours in.
        items.sort(key=lambda item: len(item[1]) + len(item[2]), reverse=True)
        batches = []
        current = []
        for item in items:
            length = len(item[1]) + len(item[2])
            longest = max([length] + [len(c[1]) + len(c[2]) for c in current])
            if current and (
                len(current) >= args.max_batch_size
                or longest * (len(current) + 1) > args.max_tokens_per_batch
            ):
                batches.append(current)
                current = []
            current.append(item)
        if current:
            batches.append(current)

        start_time = time.time()
        with output_path.open("a", encoding="utf-8") as sink:
            for row in passthrough:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()

            for number, batch in enumerate(batches, 1):
                sequences = [prompt + answer for _, prompt, answer in batch]
                width = max(len(sequence) for sequence in sequences)
                input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
                attention = torch.zeros((len(batch), width), dtype=torch.long)
                for position, sequence in enumerate(sequences):
                    input_ids[position, : len(sequence)] = torch.tensor(sequence)
                    attention[position, : len(sequence)] = 1
                input_ids = input_ids.to(device)
                attention = attention.to(device)

                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention).logits

                for position, (row, prompt, answer) in enumerate(batch):
                    begin, end = len(prompt), len(prompt) + len(answer)
                    # Position t predicts token t+1, so the answer's logits sit
                    # one step to the left of the answer itself.
                    step_logits = logits[position, begin - 1 : end - 1].float()
                    targets = input_ids[position, begin:end]
                    log_probs = torch.log_softmax(step_logits, dim=-1)
                    picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    record = dict(row)
                    record["avg_logprob"] = float(picked.mean())
                    record["logprob_tokens"] = len(answer)
                    record["logprob_method"] = METHOD
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                del logits

                if args.log_every > 0 and number % args.log_every == 0:
                    elapsed = time.time() - start_time
                    remaining = elapsed / number * (len(batches) - number)
                    print(
                        f"batch {number}/{len(batches)} "
                        f"{elapsed / 60:.1f}min elapsed, ~{remaining / 60:.1f}min left",
                        flush=True,
                    )

    # Rewrite in index order once complete, so the file is deterministic no
    # matter how many times the job was requeued, and so ties in the ranking
    # break the same way every time.
    written = load_rows(output_path)
    by_index = {int(row["index"]): row for row in written}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise SystemExit(f"{len(missing)} rows still unscored in {output_path}; rerun to resume")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for index in sorted(by_index):
            sink.write(json.dumps(by_index[index], ensure_ascii=False) + "\n")
    os.replace(temporary, output_path)

    scored = [row["avg_logprob"] for row in by_index.values() if "avg_logprob" in row]
    ordered = sorted(scored)
    print(f"wrote {output_path}: {len(by_index)} rows, {len(scored)} scored")
    if ordered:
        print(
            f"avg_logprob: min {ordered[0]:.4f} / median {ordered[len(ordered) // 2]:.4f} "
            f"/ max {ordered[-1]:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
