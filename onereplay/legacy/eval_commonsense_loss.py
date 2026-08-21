"""Evaluate Commonsense170k validation SFT loss for a base model or LoRA adapter.

This script reuses the same chat formatting, train/validation split, and label
masking as train_commonsense_lora.py. It is useful for adding the pre-finetune
base model as a comparable baseline without running any training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import load_causal_lm_and_tokenizer, set_seed  # noqa: E402
from onereplay.legacy.train_commonsense_lora import (  # noqa: E402
    evaluate_loss,
    load_and_prepare_dataset,
)
from process_dataset.process_glue_myself import build_loader  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse evaluation settings shared with the formal Commonsense170k runs."""

    parser = argparse.ArgumentParser(description="Evaluate Commonsense170k SFT loss.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/weiliu1/huggingface/datasets/commonsense_170k",
    )
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--map_cache_dir", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    """Load model, build the same validation split, compute loss, and save JSON."""

    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load the frozen base model first. If an adapter is provided, attach it
    # afterwards so the validation loss matches the trained LoRA run.
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(device)
    model.eval()

    # Reuse the exact Commonsense170k preprocessing from training. The training
    # split is built too because HuggingFace's train_test_split returns both
    # parts together, but only the validation loader is evaluated.
    start_time = time.time()
    _, valid_dataset = load_and_prepare_dataset(args, tokenizer)
    valid_loader = build_loader(valid_dataset, tokenizer, batch_size=args.batch_size, train=False)
    val_loss = evaluate_loss(model, valid_loader, device)

    record = {
        "run_name": args.run_name or (Path(args.adapter_path).name if args.adapter_path else "base"),
        "adapter_path": args.adapter_path,
        "seed": args.seed,
        "max_val_samples": args.max_val_samples,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "val_loss": val_loss,
        "elapsed_sec": time.time() - start_time,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
