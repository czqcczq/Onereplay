"""Train LoRA/OneReplay on Commonsense170k and save reusable adapters.

Commonsense170k is an instruction-tuning dataset with fields:

  instruction: user instruction
  input: optional extra input
  output: target assistant response
  answer: short normalized answer, not used for SFT loss

This script converts each example into the model's chat template and trains only
on the assistant answer tokens, matching the masking strategy used in the BoolQ
prototype.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import (  # noqa: E402
    load_causal_lm_and_tokenizer,
    load_covariance_file,
    lora_covariance_regularizer,
    move_covariances_to_device,
    set_seed,
    to_identity_covariances,
)
from process_dataset.process_glue_myself import build_loader, tokenizer_to_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse model, dataset, LoRA, OneReplay, and saving settings."""

    parser = argparse.ArgumentParser(description="Commonsense170k OneReplay LoRA training")
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
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument(
        "--map_cache_dir",
        type=str,
        default="",
        help="Writable directory for HuggingFace map cache files.",
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_size", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=500)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    parser.add_argument("--cov_path", type=str, default="mycode/onereplay/results/cov_flan_chat_10k_qv.pt")
    parser.add_argument("--replay_lambda", type=float, default=0.0)
    parser.add_argument("--normalize_replay_by_layers", type=int, default=1)
    parser.add_argument("--measure_replay_when_lambda_zero", type=int, default=1)
    parser.add_argument(
        "--identity_cov",
        type=int,
        default=0,
        help=(
            "1 replaces each loaded C with an identity matrix, turning the "
            "penalty into plain L2 on DeltaW (||DeltaW||_F^2). Ablation control "
            "to isolate the effect of the old-knowledge covariance structure. "
            "cov_path is still required, only for layer keys and shapes."
        ),
    )

    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="mycode/onereplay/results/adapters/commonsense_lora")
    parser.add_argument("--metrics_path", type=str, default="")
    return parser.parse_args()


def append_jsonl(path: str, record: dict) -> None:
    """Append one JSON record for restart-friendly experiment logging."""

    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def apply_train_template(tokenizer, instruction: str, input_text: str, output_text: str) -> tuple[str, str]:
    """Return full training text and prompt-only text for label masking.

    full_text contains both user and assistant messages.
    prompt_text ends at the assistant generation point and excludes the answer.
    tokenizer_to_ids then masks prompt_text tokens with -100.
    """

    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content = f"{user_content}\n\nInput:\n{input_text.strip()}"

    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text.strip()},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    ).rstrip()
    if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
        full_text += tokenizer.eos_token

    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return full_text, prompt_text


def load_and_prepare_dataset(args: argparse.Namespace, tokenizer):
    """Load Commonsense170k, split train/validation, and add tokenized fields."""

    dataset_dict = load_from_disk(args.dataset_path)
    dataset = dataset_dict["train"] if "train" in dataset_dict else dataset_dict
    split = dataset.train_test_split(test_size=args.val_fraction, seed=args.seed, shuffle=True)
    train_dataset = split["train"]
    valid_dataset = split["test"]

    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_val_samples > 0:
        valid_dataset = valid_dataset.select(range(min(args.max_val_samples, len(valid_dataset))))

    def add_tokenized_fields(example):
        full_text, prompt_text = apply_train_template(
            tokenizer,
            example["instruction"],
            example.get("input", ""),
            example["output"],
        )
        tokenized = tokenizer_to_ids(
            tokenizer=tokenizer,
            text=full_text,
            prompt_text=prompt_text,
            max_length=args.max_len,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
        }

    map_cache_dir = getattr(args, "map_cache_dir", "")
    if map_cache_dir:
        # In managed/sandboxed runs the dataset directory can be read-only.
        # Direct HuggingFace map cache files to an explicit writable location.
        cache_dir = Path(map_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_dataset = train_dataset.map(
            add_tokenized_fields,
            load_from_cache_file=False,
            cache_file_name=str(cache_dir / "commonsense_train_tokenized.arrow"),
        )
        valid_dataset = valid_dataset.map(
            add_tokenized_fields,
            load_from_cache_file=False,
            cache_file_name=str(cache_dir / "commonsense_valid_tokenized.arrow"),
        )
    else:
        train_dataset = train_dataset.map(add_tokenized_fields)
        valid_dataset = valid_dataset.map(add_tokenized_fields)
    return train_dataset, valid_dataset


def train_one_epoch(model, train_loader, optimizer, device, covariances, args):
    """Train one epoch with optional OneReplay regularization."""

    accumulation_steps = max(args.accumulation_size // args.batch_size, 1)
    model.train()
    optimizer.zero_grad()
    total_task_loss = 0.0
    total_replay_reg = 0.0
    total_samples = 0

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        num_samples = input_ids.shape[0]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        task_loss = outputs.loss
        if covariances:
            replay_reg, replay_stats = lora_covariance_regularizer(
                model,
                covariances,
                normalize_by_layers=bool(args.normalize_replay_by_layers),
            )
        else:
            replay_reg = torch.zeros((), device=device, dtype=torch.float32)
            replay_stats = {"used_layers": 0.0, "missing_layers": 0.0}

        loss = task_loss + args.replay_lambda * replay_reg
        (loss / accumulation_steps).backward()

        if step % accumulation_steps == 0 or step == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        total_task_loss += float(task_loss.detach().cpu()) * num_samples
        total_replay_reg += float(replay_reg.detach().cpu()) * num_samples
        total_samples += num_samples
        if step == 1:
            print(
                "OneReplay regularizer uses "
                f"{int(replay_stats['used_layers'])} LoRA layers; "
                f"missing C for {int(replay_stats['missing_layers'])} layers"
            )
        if args.log_every > 0 and (step % args.log_every == 0 or step == len(train_loader)):
            print(
                f"step {step}/{len(train_loader)} "
                f"task_loss={float(task_loss.detach().cpu()):.6f} "
                f"replay_reg={float(replay_reg.detach().cpu()):.6e}",
                flush=True,
            )

    return total_task_loss / total_samples, total_replay_reg / total_samples


def evaluate_loss(model, valid_loader, device) -> float:
    """Compute validation SFT loss on held-out Commonsense170k examples."""

    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += float(outputs.loss.detach().cpu())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def main() -> None:
    args = parse_args()
    print("the file is " + str(Path(__file__).resolve()))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for attr, value in sorted(vars(args).items()):
        print(f"\t{attr.upper()}={value}")
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
    )
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.to(device)
    model.print_trainable_parameters()

    should_load_covariances = (
        args.replay_lambda > 0
        or (args.measure_replay_when_lambda_zero == 1 and Path(args.cov_path).exists())
    )
    covariances = load_covariance_file(args.cov_path) if should_load_covariances else {}
    if covariances and args.identity_cov == 1:
        covariances = to_identity_covariances(covariances)
        print(f"identity_cov=1: using identity C (L2 on DeltaW) for {len(covariances)} layers")
    if covariances:
        covariances = move_covariances_to_device(covariances, device=device, dtype=torch.float32)
        print(f"loaded {len(covariances)} covariance matrices from {args.cov_path}")

    print("loading and tokenizing Commonsense170k")
    train_dataset, valid_dataset = load_and_prepare_dataset(args, tokenizer)
    train_loader = build_loader(train_dataset, tokenizer, batch_size=args.batch_size, train=True)
    valid_loader = build_loader(valid_dataset, tokenizer, batch_size=args.batch_size, train=False)

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, replay_reg = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            covariances,
            args,
        )
        val_loss = evaluate_loss(model, valid_loader, device)
        elapsed = time.time() - start_time
        record = {
            "epoch": epoch + 1,
            "seed": args.seed,
            "train_task_loss": train_loss,
            "train_replay_reg": replay_reg,
            "train_lambda_reg": args.replay_lambda * replay_reg,
            "val_loss": val_loss,
            "elapsed_sec": elapsed,
            "replay_lambda": args.replay_lambda,
            "identity_cov": args.identity_cov,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
        }
        print(json.dumps(record, ensure_ascii=False, indent=2))
        append_jsonl(args.metrics_path, record)

    if args.save == 1:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print(f"saved final LoRA adapter to {args.save_path}")


if __name__ == "__main__":
    main()
