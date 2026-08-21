"""Stage 1b CLI: estimate the diagonal empirical Fisher of the base weights.

Run the frozen base model over the same old-knowledge rows collect_cov used and
estimate one matrix per target layer:

    F_l = (1/N) sum_n (d L_n / d W_l)^2
    L_n = -sum_{t in assistant} log p(y_t | x, y_<t)

This is the EWC counterpart of Stage 1. The dataset flags mirror collect_cov's
one for one and must be passed identically, because "EWC and OneReplay consumed
the same old knowledge" is the fairness constraint the whole comparison rests
on. The script prints a hash of the selected rows so that claim is checkable
rather than assumed.

Three defaults differ from collect_cov on purpose:

  --use_bf16 0    C is accumulated from activations that are widened to fp32 in
                  the hook, but a gradient is already rounded by the time it
                  reaches us. bf16 has fp32's exponent range so nothing
                  underflows, yet its 8-bit mantissa still puts a ~1% noise
                  floor on every squared gradient. fp32 removes the question
                  for a one-off job that has hours of headroom.
  batch_size 1    The Fisher needs per-example gradients: squaring a
                  batch-summed gradient computes (sum_n g_n)^2, not
                  sum_n g_n^2. This is not configurable.
  labels          Only assistant tokens are supervised, so F measures response
                  sensitivity rather than prompt reconstruction.

Usage: python -m onereplay.scripts.collect_fisher [args]
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
from torch.utils.data import DataLoader  # noqa: E402

from onereplay.core.fisher import (  # noqa: E402
    DiagonalFisherAccumulator,
    build_supervised_collate_fn,
    fisher_summary,
    length_weighting_report,
    save_fisher_payload,
    select_target_weights,
    sequence_sum_nll,
)
from onereplay.core.modeling import (  # noqa: E402
    find_target_linear_module_names,
    load_causal_lm_and_tokenizer,
    set_seed,
)
from onereplay.data.old_knowledge import (  # noqa: E402
    filter_incomplete_rows,
    fingerprint_pool,
    limit_dataset,
    load_old_knowledge_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse all settings for estimating F from FLAN or another text corpus."""

    parser = argparse.ArgumentParser(description="Collect diagonal empirical Fisher for EWC.")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument(
        "--use_bf16",
        type=int,
        default=0,
        help=(
            "0 keeps the model in fp32 so gradients are not rounded to bf16's 8-bit "
            "mantissa before being squared. Widening afterwards protects the sum but "
            "cannot recover precision the backward pass already lost."
        ),
    )

    # Dataset input. These must match collect_cov's values exactly; the pool
    # fingerprint printed below is what proves they did.
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
    parser.add_argument("--streaming", type=int, default=1)

    parser.add_argument("--text_column", type=str, default="")
    parser.add_argument("--input_column", type=str, default="inputs")
    parser.add_argument("--target_column", type=str, default="targets")
    parser.add_argument("--use_chat_template", type=int, default=1)
    parser.add_argument(
        "--include_target_in_chat",
        type=int,
        default=1,
        help=(
            "Must be 1 for Fisher collection: with no assistant turn there is no "
            "supervised token and every score would be zero."
        ),
    )
    parser.add_argument("--system_prompt", type=str, default="")

    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--sample_shuffle", type=int, default=1)
    parser.add_argument("--sample_seed", type=int, default=1)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--require_target", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument(
        "--truncation_side",
        type=str,
        choices=["", "left", "right"],
        default="",
        help=(
            "Leave empty to match collect_cov, which uses the tokenizer default. Right "
            "truncation cuts the answer first, so an overflowing row can end up with no "
            "supervised token; that is reported as zero_supervision_rows rather than "
            "silently dropped, because dropping would change which rows F saw relative "
            "to C."
        ),
    )
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    parser.add_argument(
        "--fisher_device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help=(
            "Where the fp32 accumulator lives. q_proj,v_proj on Qwen3-1.7B needs about "
            "0.7 GiB and belongs on the GPU. The eight-projection full scope needs "
            "about 7 GiB on top of an fp32 model and its gradients, so use cpu there."
        ),
    )
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument(
        "--output_path",
        type=str,
        default="./mycode/onereplay/flan_qwen3_qv_fisher.pt",
    )
    return parser.parse_args()


def collect_fisher(args: argparse.Namespace) -> None:
    """Run the full Fisher estimation stage and write F to disk."""

    if args.include_target_in_chat != 1:
        raise ValueError(
            "--include_target_in_chat 0 leaves no assistant turn to supervise, so every "
            "per-example score would be zero and F would come out empty"
        )

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    accumulator_device = device if args.fisher_device == "cuda" else torch.device("cpu")

    print("Stage 1b: loading base model and tokenizer")
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
    )
    model.to(device)
    # eval() disables dropout so two runs on the same row give the same score,
    # and use_cache off keeps generation-time buffers out of a backward pass.
    model.eval()
    model.config.use_cache = False

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    target_module_names = find_target_linear_module_names(model, target_modules)
    if not target_module_names:
        raise ValueError(f"No target Linear modules found for: {target_modules}")
    target_weights = select_target_weights(model, target_module_names)
    print(f"Stage 1b: estimating F for {len(target_weights)} target modules")

    dataset = load_old_knowledge_dataset(args)
    if args.require_target == 1:
        dataset = filter_incomplete_rows(dataset, args)
    dataset = limit_dataset(dataset, args)

    pool_rows, pool_hash = fingerprint_pool(dataset, args)
    print(f"Stage 1b: pool rows={pool_rows} fingerprint={pool_hash}")
    print("  this must equal the fingerprint collect_cov printed for the same flags")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=build_supervised_collate_fn(tokenizer, args),
    )
    accumulator = DiagonalFisherAccumulator(target_weights, device=accumulator_device)
    print(
        f"Stage 1b: accumulator on {accumulator_device} "
        f"({accumulator.memory_bytes() / 1024**3:.3f} GiB resident)"
    )

    prompt_mismatches = 0
    start_time = time.time()
    for step, batch in enumerate(dataloader, start=1):
        prompt_mismatches += 1 - int(batch["prompt_matched"])
        loss, supervised_tokens = sequence_sum_nll(
            model,
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )
        if supervised_tokens == 0:
            # Truncation ate the answer, or the row had no target. Its score is
            # genuinely zero; counting it keeps N equal to the row count C saw.
            accumulator.add_empty()
            continue
        model.zero_grad(set_to_none=True)
        loss.backward()
        accumulator.add_example(supervised_tokens)

        if args.log_every > 0 and step % args.log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"  examples: {step} ({elapsed / step * 1000:.0f}ms/example, "
                f"{elapsed / 60:.1f}min elapsed)",
                flush=True,
            )
    model.zero_grad(set_to_none=True)

    fishers = accumulator.finalize()
    report = length_weighting_report(accumulator.supervised_tokens, accumulator.grad_sq_norms)
    scale = fisher_summary(fishers)

    print("Stage 1b: estimator diagnostics")
    print(json.dumps(report, indent=2))
    print(
        "  length_exponent is the fitted a in ||g_n||^2 ~ T_n^a. Uncorrelated per-token "
        "gradients give a=1, perfectly aligned ones give a=2."
    )
    print(
        f"  effective_sample_size {report['effective_sample_size']:.0f} of "
        f"{report['examples']:.0f} rows ({report['ess_ratio']:.3f}); this is how many "
        "examples F actually rests on."
    )
    if prompt_mismatches:
        print(
            f"  warning: {prompt_mismatches} rows where the prompt render was not a clean "
            "token prefix of the full render; their masks fall back to the shared prefix"
        )
    print(f"Stage 1b: F mean magnitude {scale['mean']:.3e}, max {scale['max']:.3e}")
    print(
        "  lambda for EWC scales inversely with this; the OneReplay grid does not "
        "transfer. Start the sweep near 1 / mean and widen by decades."
    )

    counts = {name: accumulator.num_examples for name in fishers}
    metadata = {
        "estimator": "assistant_only_sequence_sum_diagonal_empirical_fisher",
        "loss_reduction": "sum",
        "supervision": "assistant_only",
        "per_example_backward": True,
        "num_examples": accumulator.num_examples,
        "pool_rows": pool_rows,
        "pool_fingerprint": pool_hash,
        "model_name": args.model_name,
        "use_bf16": args.use_bf16,
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
        "system_prompt": args.system_prompt,
        "require_target": args.require_target,
        "truncation_side": args.truncation_side or getattr(tokenizer, "truncation_side", ""),
        "max_samples": args.max_samples,
        "sample_shuffle": args.sample_shuffle,
        "sample_seed": args.sample_seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "max_len": args.max_len,
        "prompt_prefix_mismatches": prompt_mismatches,
        "length_weighting": report,
        "fisher_scale": scale,
    }
    save_fisher_payload(args.output_path, fishers, counts, metadata)
    print(f"Stage 1b done: saved Fisher file to {args.output_path}")


def main() -> None:
    collect_fisher(parse_args())


if __name__ == "__main__":
    main()
