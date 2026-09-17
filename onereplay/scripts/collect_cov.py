"""Stage 1 CLI: estimate old-knowledge hidden-state second moments.

Run the frozen base model over an old-knowledge corpus (FLAN by default) and
estimate one matrix per LoRA target layer:

    C_l = E_x[x x^T]

where x is the input hidden state of that layer for one token of the corpus.

--cov_supervision decides which tokens those are:

  assistant_only  (default) only the answer span, the same tokens
                  collect_fisher supervises and the same ones the SFT and replay
                  losses are computed on. This is what "the old task's responses
                  are protected" requires: all four arms then constrain the same
                  positions, and the comparison isolates the weighting matrix.
  all_tokens      every non-padding position, so system, prompt and answer all
                  enter the average in whatever ratio the corpus happens to
                  have. FLAN's one-line targets make this mostly a covariance of
                  prompts; a long-CoT corpus makes it mostly one of answers. This
                  was the only behaviour before the flag existed, so every C
                  collected up to that point is an all_tokens one.

The two are different estimators, not a tuning knob: they have different traces,
so a lambda calibrated against one does not carry over to the other.

The masked span is the answer tokens themselves. The Fisher's loss is shifted by
one (position t scores token t+1), so F's gradient also leans on the last prompt
position while C does not, and C covers the final answer token while F does not.
That is a two-token boundary difference, kept on purpose: C is a statement about
which representations must not drift, not about which logits are scored, and a
mid-stack x feeds every later position through attention anyway.

With --cov_normalization base_output_norm the script instead estimates

    C_l = E_x[(x / ||W_l x||)(x / ||W_l x||)^T]

so the training-time penalty becomes E_x ||DeltaW_l x||^2 / ||W_l x||^2.

Usage: python -m onereplay.scripts.collect_cov [args]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from onereplay.core.covariance import (  # noqa: E402
    register_covariance_hooks,
    save_covariance_payload,
)
from onereplay.core.modeling import (  # noqa: E402
    find_target_linear_module_names,
    load_causal_lm_and_tokenizer,
    set_seed,
)
from onereplay.data.old_knowledge import (  # noqa: E402
    build_collate_fn,
    filter_incomplete_rows,
    fingerprint_pool,
    limit_dataset,
    load_old_knowledge_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse all settings for collecting C from FLAN or another text corpus."""

    parser = argparse.ArgumentParser(description="Collect OneReplay covariance matrices.")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)

    # Dataset input. Use --dataset_path for a dataset saved by datasets.save_to_disk.
    # Use --dataset_name/--dataset_config for a HuggingFace dataset.
    # Use --data_files for local json/jsonl/text files.
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="Muennighoff/flan")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--data_files", type=str, default="")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/weiliu1/huggingface/datasets/cache",
        help="HuggingFace dataset cache directory",
    )
    parser.add_argument(
        "--streaming",
        type=int,
        default=1,
        help="1 streams HF datasets so FLAN does not need to be fully downloaded first",
    )

    # FLAN-like datasets usually have "inputs" and "targets". If your local
    # files have a single field, set --text_column to that field name.
    parser.add_argument("--text_column", type=str, default="")
    parser.add_argument("--input_column", type=str, default="inputs")
    parser.add_argument("--target_column", type=str, default="targets")
    parser.add_argument(
        "--use_chat_template",
        type=int,
        default=1,
        help="1 formats old-knowledge examples with tokenizer.apply_chat_template",
    )
    parser.add_argument(
        "--include_target_in_chat",
        type=int,
        default=1,
        help="1 includes FLAN targets as assistant messages when collecting C",
    )
    parser.add_argument(
        "--cov_supervision",
        type=str,
        choices=["all_tokens", "assistant_only"],
        default="assistant_only",
        help=(
            "Which token positions enter C. assistant_only (default) restricts C "
            "to the answer span, matching what collect_fisher and the SFT/replay "
            "losses are computed on, and needs --include_target_in_chat 1. "
            "all_tokens averages over every non-padding token instead, mixing "
            "prompt and answer in whatever ratio the corpus has; it is what every "
            "C collected before this flag existed used. The two have different "
            "traces, so a lambda calibrated on one does not transfer."
        ),
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Optional system message inserted before each FLAN example",
    )
    parser.add_argument(
        "--enable_thinking",
        type=int,
        default=0,
        help="1 renders the prompt with Qwen3's thinking block open; must match "
        "the self-distillation setting so C sees the sequence that was generated",
    )
    parser.add_argument(
        "--concat_prompt_target",
        type=int,
        default=0,
        help="1 builds the text as generation_prompt + raw target instead of "
        "re-rendering the assistant turn. Required with --enable_thinking 1, "
        "because the chat template strips <think> out of assistant messages",
    )
    parser.add_argument(
        "--debug_print_examples",
        type=int,
        default=0,
        help="Print this many fully rendered texts before collecting, to verify "
        "what C actually sees (e.g. that <think> survived)",
    )

    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument(
        "--sample_shuffle",
        type=int,
        default=1,
        help=(
            "1 shuffles the corpus with --sample_seed before taking max_samples, "
            "so the subset is a reproducible random sample instead of the first N rows. "
            "0 keeps the original order."
        ),
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=1,
        help="Seed for the reproducible subset shuffle when --sample_shuffle 1.",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        choices=["uniform", "balanced"],
        default="uniform",
        help=(
            "uniform (default) samples rows uniformly, reproducing the corpus's raw "
            "task mixture. balanced draws a per-task quota with FLAN's capped-"
            "proportional weighting min(N_i, --mixing_rate_max), so C is not "
            "dominated by whichever tasks own the most rows. balanced needs a "
            "--task_column and a map-style dataset."
        ),
    )
    parser.add_argument(
        "--task_column",
        type=str,
        default="task",
        help="Column naming each row's task, used only by --sample_strategy balanced.",
    )
    parser.add_argument(
        "--mixing_rate_max",
        type=int,
        default=3000,
        help=(
            "FLAN's mixing rate maximum for --sample_strategy balanced: a task's "
            "weight is min(N_i, this), so tasks at or above it are equally weighted."
        ),
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10000,
        help="Approximate-shuffle buffer size used only for streaming datasets.",
    )
    parser.add_argument(
        "--require_target",
        type=int,
        default=0,
        help=(
            "1 drops rows with an empty input or target before sampling. Needed for a "
            "self-distilled corpus, where a prompt whose generation hit the token cap is "
            "stored with an empty target and is dropped by the replay loader too; without "
            "this, C would cover rows replay never trains on."
        ),
    )
    parser.add_argument(
        "--require_target_column",
        type=str,
        default="",
        help=(
            "Column --require_target checks for emptiness. Empty means --target_column, "
            "which is right whenever the two are the same corpus. Set it to targets while "
            "--target_column is gold_targets to run the gold ablation on the self-distilled "
            "file's exact row set: gold is filled in on truncated rows too, so filtering on "
            "it would give the gold arm extra rows and confound target source with pool size."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument(
        "--truncation_side",
        type=str,
        choices=["", "left", "right"],
        default="",
        help=(
            "Which end to cut when a rendered example exceeds --max_len. Empty keeps the "
            "tokenizer default (right, i.e. the assistant answer is dropped first). "
            "Training truncates on the left, so pass left to match it whenever the "
            "targets are long enough to overflow."
        ),
    )
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument(
        "--cov_normalization",
        type=str,
        choices=["none", "base_output_norm"],
        default="none",
        help=(
            "none collects E[x x^T]. base_output_norm collects "
            "E[(x / ||W x||)(x / ||W x||)^T] for a relative-error penalty."
        ),
    )
    parser.add_argument(
        "--cov_norm_eps",
        type=float,
        default=1e-6,
        help="Lower bound for ||W x|| when --cov_normalization base_output_norm is used.",
    )
    parser.add_argument(
        "--cov_accum_device",
        type=str,
        choices=["cpu", "device"],
        default="cpu",
        help=(
            "Where the running X^T X sums live. cpu (default) copies every "
            "batch's contribution to host memory, which costs one transfer per "
            "batch per layer and scales with the size of C rather than with the "
            "batch: covering all seven projections of an 8B model is 33.75 GiB "
            "of PCIe traffic per batch. device keeps the sums beside the "
            "activations, removing those transfers, but needs the whole C "
            "resident in GPU memory alongside the model."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./mycode/onereplay/flan_qwen3_qv_cov.pt",
    )
    return parser.parse_args()


def collect_covariances(args: argparse.Namespace) -> None:
    """Run the full collection stage and write normalized C matrices to disk."""

    if args.cov_supervision == "assistant_only" and args.include_target_in_chat != 1:
        raise ValueError(
            "--cov_supervision assistant_only needs --include_target_in_chat 1: with no "
            "assistant turn every position would be masked out and C would come out zero"
        )

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Stage 1: loading base model and tokenizer")
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
    )
    model.to(device)
    model.eval()

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    target_module_names = find_target_linear_module_names(model, target_modules)
    if not target_module_names:
        raise ValueError(f"No target Linear modules found for: {target_modules}")
    print(f"Stage 1: collecting C for {len(target_module_names)} target modules")

    dataset = load_old_knowledge_dataset(args)
    if args.require_target == 1:
        # Before limit_dataset, so --max_samples counts usable rows.
        dataset = filter_incomplete_rows(dataset, args)
    dataset = limit_dataset(dataset, args)
    if args.sample_strategy == "balanced":
        print(
            f"Stage 1: taking a seed={args.sample_seed} task-balanced sample "
            f"(mixing_rate_max={args.mixing_rate_max}) of {args.max_samples} rows"
        )
    elif args.sample_shuffle == 1:
        print(
            f"Stage 1: taking a seed={args.sample_seed} random sample of "
            f"{'all' if args.max_samples <= 0 else args.max_samples} rows"
        )
    pool_rows, pool_hash = fingerprint_pool(dataset, args)
    print(f"Stage 1: pool rows={pool_rows} fingerprint={pool_hash}")
    print("  collect_fisher must print the same value, or C and F saw different rows")

    if args.debug_print_examples > 0:
        from onereplay.data.old_knowledge import example_to_model_text

        for row in range(min(args.debug_print_examples, len(dataset))):
            rendered = example_to_model_text(dataset[row], tokenizer, args)
            print(f"---- rendered example {row} ({len(rendered)} chars) ----")
            print(rendered)
            print(f"---- contains <think>: {'<think>' in rendered} ----")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer, args),
    )

    token_mask_holder: dict[str, torch.Tensor | None] = {"token_mask": None}
    cov_sums, counts, handles = register_covariance_hooks(
        model,
        target_module_names,
        token_mask_holder,
        args,
    )

    assistant_only = args.cov_supervision == "assistant_only"
    if args.cov_normalization == "base_output_norm":
        print(
            "Stage 1: forwarding old-knowledge data and accumulating "
            "normalized X^T X with x' = x / max(||W x||, eps)"
        )
    else:
        print("Stage 1: forwarding old-knowledge data and accumulating X^T X")
    print(f"Stage 1: supervision = {args.cov_supervision}")

    total_tokens = 0
    supervised_tokens = 0
    zero_supervision_rows = 0
    prompt_mismatches = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                total_tokens += int(attention_mask.sum())
            # The model still reads the whole sequence either way: the answer is
            # conditioned on its prompt, so the prompt has to be forwarded even
            # when it is excluded from C. Only the hook's mask narrows.
            if assistant_only:
                supervision_mask = batch["supervision_mask"]
                per_row = supervision_mask.sum(dim=1)
                supervised_tokens += int(per_row.sum())
                zero_supervision_rows += int((per_row == 0).sum())
                prompt_mismatches += int(batch["prompt_prefix_mismatches"])
                token_mask_holder["token_mask"] = supervision_mask
            else:
                token_mask_holder["token_mask"] = attention_mask
            model_inputs = {
                key: value
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            model(**model_inputs)
            if step % 50 == 0:
                print(f"  processed batches: {step}")

    for handle in handles:
        handle.remove()

    if not assistant_only:
        supervised_tokens = total_tokens
    print(
        f"Stage 1: C rests on {supervised_tokens} tokens, "
        f"{supervised_tokens / max(total_tokens, 1):.1%} of the {total_tokens} "
        "non-padding tokens in the pool"
    )
    if assistant_only:
        if zero_supervision_rows:
            # Same failure the Fisher run reports: a right-side cut removes the
            # answer, and the row is then forwarded while contributing nothing.
            print(
                f"  warning: {zero_supervision_rows} rows contributed no token. Either "
                "truncation ate the answer (--truncation_side should be left) or the "
                "row had an empty target"
            )
        if prompt_mismatches:
            print(
                f"  warning: {prompt_mismatches} rows where the prompt render was not a "
                "clean token prefix of the full render; their masks fall back to the "
                "shared prefix"
            )
        print(
            "  trace(C) is now on a different scale than an all_tokens run: re-sweep "
            "lambda instead of reusing the old grid"
        )

    # pop instead of iterating: with --cov_accum_device device the sums are the
    # single largest allocation of the run, and normalizing them into a second
    # dict would briefly hold two full copies of C. Dropping each sum as soon as
    # its mean lands on the host keeps the peak at one copy plus one layer.
    covariances = {}
    for module_name in list(cov_sums.keys()):
        cov_sum = cov_sums.pop(module_name)
        covariances[module_name] = (cov_sum / max(counts[module_name], 1)).cpu()
        del cov_sum

    metadata = {
        "model_name": args.model_name,
        "pool_rows": pool_rows,
        "pool_fingerprint": pool_hash,
        "target_modules": target_modules,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_path": args.dataset_path,
        "data_files": args.data_files,
        "dataset_split": args.dataset_split,
        "cache_dir": args.cache_dir,
        "streaming": args.streaming,
        "text_column": args.text_column,
        "input_column": args.input_column,
        "target_column": args.target_column,
        "use_chat_template": args.use_chat_template,
        "include_target_in_chat": args.include_target_in_chat,
        "cov_supervision": args.cov_supervision,
        "supervised_tokens": supervised_tokens,
        "total_tokens": total_tokens,
        "zero_supervision_rows": zero_supervision_rows,
        "prompt_prefix_mismatches": prompt_mismatches,
        "system_prompt": args.system_prompt,
        "require_target": args.require_target,
        "require_target_column": args.require_target_column,
        "truncation_side": args.truncation_side or getattr(tokenizer, "truncation_side", ""),
        "max_samples": args.max_samples,
        "sample_shuffle": args.sample_shuffle,
        "sample_seed": args.sample_seed,
        "sample_strategy": args.sample_strategy,
        "task_column": args.task_column,
        "mixing_rate_max": args.mixing_rate_max,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "max_len": args.max_len,
        "cov_normalization": args.cov_normalization,
        "cov_norm_eps": args.cov_norm_eps,
        "cov_accum_device": args.cov_accum_device,
    }
    save_covariance_payload(args.output_path, covariances, counts, metadata)
    print(f"Stage 1 done: saved covariance file to {args.output_path}")


def main() -> None:
    collect_covariances(parse_args())


if __name__ == "__main__":
    main()
