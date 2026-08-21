"""Stage 1 of OneReplay: estimate old-knowledge hidden-state second moments.

This script runs the frozen base model on an old-knowledge corpus, such as FLAN
Collection, and estimates one matrix per LoRA target layer:

    C_l = E_x[x x^T]

where x is the input hidden state of that layer for one non-padding token.
The matrices are saved once and reused during LoRA fine-tuning.

For the relative-error version of OneReplay, use:

    --cov_normalization base_output_norm

Then the script estimates:

    C_l = E_x[(x / ||W_l x||) (x / ||W_l x||)^T]
        = E_x[x x^T / ||W_l x||^2]

so the training-time penalty becomes E_x ||DeltaW_l x||^2 / ||W_l x||^2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import (  # noqa: E402
    find_target_linear_module_names,
    load_causal_lm_and_tokenizer,
    save_covariance_payload,
    set_seed,
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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=512)
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


def load_old_knowledge_dataset(args: argparse.Namespace):
    """Load the old-knowledge corpus from disk, local files, or HuggingFace."""

    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)
        return dataset[args.dataset_split] if args.dataset_split in dataset else dataset

    if args.data_files:
        suffix = Path(args.data_files).suffix.lower()
        file_format = "text" if suffix == ".txt" else "json"
        dataset = load_dataset(
            file_format,
            data_files=args.data_files,
            split=args.dataset_split,
            cache_dir=args.cache_dir,
        )
        return dataset

    config = args.dataset_config if args.dataset_config else None
    return load_dataset(
        args.dataset_name,
        config,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        streaming=bool(args.streaming),
    )


def limit_dataset(dataset, args: argparse.Namespace):
    """Take a reproducible subset of a map-style Dataset or streaming IterableDataset.

    With --sample_shuffle 1 (default), the dataset is shuffled with
    --sample_seed before taking --max_samples, so the subset is a reproducible
    random sample of the corpus rather than its first N rows. Growing
    max_samples with the same seed keeps smaller subsets nested in larger ones
    for map-style datasets.

    Streaming datasets only support approximate buffer shuffling; the result is
    still deterministic for a fixed seed and buffer size.
    """

    max_samples = args.max_samples
    shuffle = args.sample_shuffle == 1

    if hasattr(dataset, "select"):
        if shuffle:
            dataset = dataset.shuffle(seed=args.sample_seed)
        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset

    if hasattr(dataset, "take"):
        if shuffle:
            dataset = dataset.shuffle(
                seed=args.sample_seed,
                buffer_size=args.shuffle_buffer_size,
            )
        if max_samples > 0:
            dataset = dataset.take(max_samples)
        return dataset

    return dataset


def example_to_plain_text(example: dict[str, Any], args: argparse.Namespace) -> str:
    """Convert one FLAN-style row into plain text.

    This is kept for ablations. The default path below uses chat template,
    because Qwen instruction tuning and your BoolQ data are chat-formatted.
    """

    if args.text_column and args.text_column in example:
        return str(example[args.text_column])

    pieces: list[str] = []
    if args.input_column in example and example[args.input_column] is not None:
        pieces.append(str(example[args.input_column]))
    if args.target_column in example and example[args.target_column] is not None:
        pieces.append(str(example[args.target_column]))

    if pieces:
        return "\n".join(piece.strip() for piece in pieces if piece.strip())

    # Fallback for unknown schemas: join string-like fields so the script still
    # works with many json/jsonl datasets without code edits.
    for value in example.values():
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())
    return "\n".join(pieces)


def example_to_messages(example: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    """Convert one FLAN row into chat messages.

    FLAN has an instruction/input field and a target field. We map them to:

        user:      inputs
        assistant: targets

    This makes the old-knowledge hidden states live in the same chat-template
    distribution as the later LoRA fine-tuning data.
    """

    user_content = ""
    if args.text_column and args.text_column in example:
        user_content = str(example[args.text_column]).strip()
    elif args.input_column in example and example[args.input_column] is not None:
        user_content = str(example[args.input_column]).strip()
    else:
        user_content = example_to_plain_text(example, args).strip()

    messages: list[dict[str, str]] = []
    if args.system_prompt.strip():
        messages.append({"role": "system", "content": args.system_prompt.strip()})
    messages.append({"role": "user", "content": user_content})

    target = ""
    if args.target_column in example and example[args.target_column] is not None:
        target = str(example[args.target_column]).strip()
    if args.include_target_in_chat == 1 and target:
        messages.append({"role": "assistant", "content": target})

    return messages


def example_to_model_text(example: dict[str, Any], tokenizer, args: argparse.Namespace) -> str:
    """Build the exact text that will be tokenized and forwarded through model.

    When use_chat_template=1, tokenizer.apply_chat_template inserts model-
    specific role tokens such as Qwen's user/assistant markers. If a tokenizer
    has no chat template, we fall back to plain text with a clear error-free path
    so the script still works for non-chat base models.
    """

    if args.use_chat_template != 1:
        return example_to_plain_text(example, args)

    messages = example_to_messages(example, args)
    if getattr(tokenizer, "chat_template", None) is None:
        return example_to_plain_text(example, args)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_collate_fn(tokenizer, args: argparse.Namespace):
    """Create a DataLoader collator that tokenizes text and pads each batch."""

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [example_to_model_text(example, tokenizer, args) for example in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_len,
            padding=True,
            return_tensors="pt",
            # Chat templates already include model-specific special tokens.
            # Plain-text ablations still use normal tokenizer special tokens.
            add_special_tokens=args.use_chat_template != 1,
        )
        return tokenized

    return collate_fn


def make_covariance_hook(
    module_name: str,
    cov_sums: dict[str, torch.Tensor],
    counts: dict[str, int],
    attention_holder: dict[str, torch.Tensor | None],
    cov_normalization: str,
    cov_norm_eps: float,
):
    """Build a forward pre-hook that accumulates X^T X for one target layer.

    The hook sees the input hidden states of a Linear layer before W or LoRA is
    applied. This is exactly the x in DeltaW x.

    With cov_normalization="none", the collected matrix is E[x x^T].

    With cov_normalization="base_output_norm", each token vector is replaced by:

        x' = x / max(||W x||_2, cov_norm_eps)

    and the collected matrix is E[x' x'^T]. This makes the later penalty
    measure relative output perturbation ||DeltaW x||^2 / ||W x||^2 instead of
    absolute output perturbation.
    """

    def hook(_module, inputs, output):
        hidden_states = inputs[0].detach()
        base_outputs = output.detach()
        if hidden_states.dim() == 2:
            flat_x = hidden_states
            flat_y = base_outputs
            flat_mask = None
        else:
            batch, seq_len, hidden_dim = hidden_states.shape
            flat_x = hidden_states.reshape(batch * seq_len, hidden_dim)
            flat_y = base_outputs.reshape(batch * seq_len, base_outputs.shape[-1])
            attention_mask = attention_holder.get("attention_mask")
            flat_mask = None
            if attention_mask is not None and attention_mask.shape[:2] == (batch, seq_len):
                flat_mask = attention_mask.reshape(batch * seq_len).bool().to(flat_x.device)

        if flat_mask is not None:
            flat_x = flat_x[flat_mask]
            flat_y = flat_y[flat_mask]
        if flat_x.numel() == 0:
            return

        flat_x = flat_x.float()
        if cov_normalization == "base_output_norm":
            # This is a forward hook, so output is already the frozen base
            # layer value W x. Reusing it avoids an extra large matrix multiply
            # for every target module.
            denom = flat_y.float().norm(dim=-1).clamp_min(float(cov_norm_eps)).unsqueeze(-1)
            flat_x = flat_x / denom

        xtx = flat_x.T @ flat_x
        if module_name not in cov_sums:
            cov_sums[module_name] = xtx.cpu()
            counts[module_name] = int(flat_x.shape[0])
        else:
            cov_sums[module_name] += xtx.cpu()
            counts[module_name] += int(flat_x.shape[0])

    return hook


def register_covariance_hooks(
    model,
    target_module_names: list[str],
    attention_holder: dict[str, torch.Tensor | None],
    args: argparse.Namespace,
):
    """Attach hooks to every target Linear layer and return hook handles."""

    cov_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []
    module_dict = dict(model.named_modules())

    for module_name in target_module_names:
        module = module_dict[module_name]
        hook = make_covariance_hook(
            module_name,
            cov_sums,
            counts,
            attention_holder,
            args.cov_normalization,
            args.cov_norm_eps,
        )
        handles.append(module.register_forward_hook(hook))

    return cov_sums, counts, handles


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
    dataset = limit_dataset(dataset, args)
    if args.sample_shuffle == 1:
        print(
            f"Stage 1: taking a seed={args.sample_seed} random sample of "
            f"{'all' if args.max_samples <= 0 else args.max_samples} rows"
        )
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
        "target_modules": target_modules,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_path": args.dataset_path,
        "data_files": args.data_files,
        "dataset_split": args.dataset_split,
        "cache_dir": args.cache_dir,
        "streaming": args.streaming,
        "use_chat_template": args.use_chat_template,
        "include_target_in_chat": args.include_target_in_chat,
        "system_prompt": args.system_prompt,
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
    args = parse_args()
    collect_covariances(args)


if __name__ == "__main__":
    main()
