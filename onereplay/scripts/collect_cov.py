"""Stage 1 CLI: estimate old-knowledge hidden-state second moments.

Run the frozen base model over an old-knowledge corpus (FLAN by default) and
estimate one matrix per LoRA target layer:

    C_l = E_x[x x^T]

where x is the input hidden state of that layer for one non-padding token.

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
        "--system_prompt",
        type=str,
        default="",
        help="Optional system message inserted before each FLAN example",
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
        "--output_path",
        type=str,
        default="./mycode/onereplay/flan_qwen3_qv_cov.pt",
    )
    return parser.parse_args()


def collect_covariances(args: argparse.Namespace) -> None:
    """Run the full collection stage and write normalized C matrices to disk."""

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
    if args.sample_shuffle == 1:
        print(
            f"Stage 1: taking a seed={args.sample_seed} random sample of "
            f"{'all' if args.max_samples <= 0 else args.max_samples} rows"
        )
    pool_rows, pool_hash = fingerprint_pool(dataset, args)
    print(f"Stage 1: pool rows={pool_rows} fingerprint={pool_hash}")
    print("  collect_fisher must print the same value, or C and F saw different rows")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer, args),
    )

    attention_holder: dict[str, torch.Tensor | None] = {"attention_mask": None}
    cov_sums, counts, handles = register_covariance_hooks(
        model,
        target_module_names,
        attention_holder,
        args,
    )

    if args.cov_normalization == "base_output_norm":
        print(
            "Stage 1: forwarding old-knowledge data and accumulating "
            "normalized X^T X with x' = x / max(||W x||, eps)"
        )
    else:
        print("Stage 1: forwarding old-knowledge data and accumulating X^T X")
    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            attention_holder["attention_mask"] = batch.get("attention_mask")
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

    covariances = {}
    for module_name, cov_sum in cov_sums.items():
        covariances[module_name] = cov_sum / max(counts[module_name], 1)

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
        "system_prompt": args.system_prompt,
        "require_target": args.require_target,
        "truncation_side": args.truncation_side or getattr(tokenizer, "truncation_side", ""),
        "max_samples": args.max_samples,
        "sample_shuffle": args.sample_shuffle,
        "sample_seed": args.sample_seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "max_len": args.max_len,
        "cov_normalization": args.cov_normalization,
        "cov_norm_eps": args.cov_norm_eps,
    }
    save_covariance_payload(args.output_path, covariances, counts, metadata)
    print(f"Stage 1 done: saved covariance file to {args.output_path}")


def main() -> None:
    collect_covariances(parse_args())


if __name__ == "__main__":
    main()
