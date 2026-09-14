"""Compare decoding policies on the MATH500 rows that greedy decoding ruins.

Greedy decoding is deterministic, so a model that walks into a repeating state
never leaves it. On MATH500 that showed up as more than half of one arm's rows
burning the entire token budget on "216 is 6**3, but 216 is 6**3, but ...",
scoring 0 for lack of a \\boxed span. Raising the budget from 4096 to 8192
changed nothing, which is the signature of non-termination rather than of a
budget that is merely too small.

So before re-running a full evaluation, this answers the cheaper question: is
MATH500 measuring the model or measuring the decoder? It decodes the same
problems several ways and prints one table. Read hit_cap first -- a policy that
does not move hit_cap cannot move the score either. If nothing moves hit_cap,
the ceiling is the model's, not the decoder's, and no amount of eval surgery
will help.

Point --prior_responses at an existing responses.jsonl so the probe spends its
budget on rows that already failed, which is where the policies differ. Without
it the probe just takes the first --limit problems, where most rows terminate
fine and every policy looks the same.

Example:

    python onereplay/scripts/probe_decoding.py \
        --adapter_path /path/to/checkpoints/part1_math_lr5e-5_seed1 \
        --math500_data_path /path/to/math500_test.jsonl \
        --prior_responses /path/to/results/math500/part1_math_lr5e-5_seed1/responses.jsonl \
        --limit 64 --max_new_tokens 8192 --batch_size 64

Keep --limit a multiple of --batch_size. generate() runs until every row in a
batch is done, and these rows are picked precisely because they do not finish,
so a batch costs the full budget whether it holds 64 rows or 4. One saturating
batch per config is the whole bill; a stray second one doubles it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import set_seed  # noqa: E402

from onereplay.eval.generation import batched_generate, configure_decoding  # noqa: E402
from onereplay.eval.metrics.math500 import (  # noqa: E402
    build_prompt,
    extract_answer,
    is_equiv,
    load_json_records,
)
from onereplay.eval.runner import load_eval_model  # noqa: E402

QUESTION_KEYS = ("problem", "question", "prompt", "input")
ANSWER_KEYS = ("answer", "final_answer", "solution", "target", "output")

# stop_on_im_end is separated from the sampling knobs because it is the only
# one that is arguably a bug fix rather than a policy choice: a trained chat
# turn ends '<|im_end|><|endoftext|>', but eos_token on the Qwen3 base
# tokenizer is only '<|endoftext|>'. It gets its own arm so its effect can be
# read independently of the loop-breaking arms.
CONFIGS: dict[str, dict] = {
    "greedy": {},
    "imend": {"stop_on_im_end": True},
    "ngram40": {"no_repeat_ngram_size": 40},
    "ngram40+imend": {"no_repeat_ngram_size": 40, "stop_on_im_end": True},
    "reppen1.05": {"repetition_penalty": 1.05},
    "sample.6": {"do_sample": True, "temperature": 0.6, "top_p": 0.95},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # model_dir/model_name locate the base weights; a full-finetune checkpoint
    # carries its own, so load_eval_model ignores both when --adapter_path
    # points at one. Only a LoRA run needs them.
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B-Base")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="Training output directory: a LoRA adapter, or a full checkpoint "
        "loaded as a model in its own right. Empty probes the base model.",
    )
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--math500_data_path", type=str, required=True)
    parser.add_argument(
        "--prior_responses",
        type=str,
        default="",
        help="responses.jsonl from an earlier run, used to pick which problems "
        "to probe. Strongly recommended; see module docstring.",
    )
    parser.add_argument(
        "--select",
        type=str,
        default="capped",
        choices=["capped", "noanswer", "first"],
        help="capped keeps rows whose earlier response filled the budget, "
        "noanswer keeps rows that produced no \\boxed span (a superset), "
        "first ignores --prior_responses entirely.",
    )
    parser.add_argument(
        "--prior_cap",
        type=int,
        default=0,
        help="max_new_tokens of the earlier run, for --select capped. "
        "0 means the same as --max_new_tokens.",
    )

    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--configs",
        type=str,
        default="greedy,ngram40,reppen1.05,sample.6",
        help=f"Comma-separated subset of: {','.join(CONFIGS)}",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default="",
        help="Optional dump of every response from every config, for eyeballing tails.",
    )
    parser.add_argument("--show_tails", type=int, default=2)
    return parser.parse_args()


def first_existing_key(record: dict, preferred: str, candidates: tuple[str, ...]) -> str:
    if preferred and preferred in record:
        return preferred
    for key in candidates:
        if key in record:
            return key
    return ""


def load_examples(data_path: str) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for record in load_json_records(data_path):
        if not isinstance(record, dict):
            continue
        question_key = first_existing_key(record, "", QUESTION_KEYS)
        answer_key = first_existing_key(record, "", ANSWER_KEYS)
        if not question_key or not answer_key:
            continue
        examples.append(
            {
                "question": str(record[question_key]),
                "answer": str(record[answer_key]),
            }
        )
    return examples


def select_examples(args, examples, tokenizer) -> list[dict[str, str]]:
    """Narrow to the rows worth probing, keeping dataset order."""

    if args.select == "first" or not args.prior_responses:
        return examples[: args.limit]

    cap = args.prior_cap or args.max_new_tokens
    broken: set[str] = set()
    with open(args.prior_responses, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            response = row.get("response", "") or ""
            if args.select == "noanswer":
                hit = not row.get("prediction")
            else:
                length = len(tokenizer(response, add_special_tokens=False)["input_ids"])
                hit = length >= cap - 8
            if hit:
                broken.add(row.get("question", ""))

    picked = [example for example in examples if example["question"] in broken]
    if not picked:
        print(
            f"WARNING: no rows matched --select {args.select} in {args.prior_responses}; "
            "falling back to the first --limit problems",
            flush=True,
        )
        return examples[: args.limit]
    return picked[: args.limit]


def summarize(responses, examples, tokenizer, cap: int) -> dict:
    lengths = [
        len(tokenizer(response, add_special_tokens=False)["input_ids"])
        for response in responses
    ]
    predictions = [extract_answer(response) for response in responses]
    correct = sum(
        int(is_equiv(prediction, example["answer"]))
        for prediction, example in zip(predictions, examples)
    )
    ordered = sorted(lengths)

    def percentile(values, q):
        if not values:
            return 0
        return values[min(len(values) - 1, int(len(values) * q / 100))]

    return {
        "n": len(responses),
        "hit_cap": sum(1 for length in lengths if length >= cap - 8),
        "no_boxed": sum(1 for prediction in predictions if not prediction),
        "correct": correct,
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "max": ordered[-1] if ordered else 0,
    }


def main() -> None:
    args = parse_args()
    names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; pick from {list(CONFIGS)}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_eval_model(
        args.model_dir, args.model_name, use_bf16=args.use_bf16, adapter_path=args.adapter_path
    )
    model.to(device)
    model.eval()

    examples = select_examples(args, load_examples(args.math500_data_path), tokenizer)
    prompts = [build_prompt(example["question"]) for example in examples]
    print(
        f"probing {len(examples)} problems  select={args.select}  "
        f"cap={args.max_new_tokens}  batch={args.batch_size}",
        flush=True,
    )

    dump = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None
    rows = []
    for name in names:
        # Re-seeding per config keeps the sampled arm reproducible and leaves
        # the deterministic arms unaffected.
        set_seed(args.seed)
        configure_decoding(**CONFIGS[name])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        started = time.time()
        responses = batched_generate(
            model,
            tokenizer,
            prompts,
            device,
            args.max_new_tokens,
            args.batch_size,
            log_label=f"probe:{name}",
        )
        elapsed = time.time() - started

        stats = summarize(responses, examples, tokenizer, args.max_new_tokens)
        stats["config"] = name
        stats["seconds"] = round(elapsed, 1)
        if device.type == "cuda":
            stats["peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 1)
        rows.append(stats)

        if dump:
            for example, response in zip(examples, responses):
                dump.write(
                    json.dumps(
                        {
                            "config": name,
                            "question": example["question"],
                            "gold": example["answer"],
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if dump:
        dump.close()

    print("\n==== decoding probe ====")
    header = f"{'config':>14}{'n':>5}{'hit_cap':>9}{'no_boxed':>10}{'acc':>8}{'p50':>7}{'p90':>7}{'max':>7}{'sec':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        n = max(row["n"], 1)
        print(
            f"{row['config']:>14}{row['n']:>5}"
            f"{row['hit_cap'] / n:>8.1%}{row['no_boxed'] / n:>10.1%}"
            f"{row['correct'] / n:>8.1%}"
            f"{row['p50']:>7}{row['p90']:>7}{row['max']:>7}{row['seconds']:>8.0f}"
        )
    print(
        "\nhit_cap is the one to read first. A policy that leaves hit_cap where "
        "greedy put it cannot move the score, and means the ceiling is the "
        "model's rather than the decoder's."
    )


if __name__ == "__main__":
    main()
