"""Stage 2 of OneReplay: LoRA fine-tuning with a covariance regularizer.

This is intentionally close to mycode/standard_ft/vanilla_lora.py so you can
compare the two files. The new part is:

    loss = task_loss + replay_lambda * mean_l tr(DeltaW_l C_l DeltaW_l^T)

where each C_l was collected by collect_flan_cov.py.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from onereplay.core import (  # noqa: E402
    load_causal_lm_and_tokenizer,
    load_covariance_file,
    lora_covariance_regularizer,
    set_seed,
)
from process_dataset.process_glue_myself import build_loader, tokenizer_to_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse vanilla LoRA settings plus OneReplay regularization settings."""

    parser = argparse.ArgumentParser(description="OneReplay LoRA fine-tuning")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save", type=int, default=0, help="1 means save final LoRA adapter")
    parser.add_argument("--save_path", type=str, default="./mycode/onereplay/final_adapter")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/weiliu1/huggingface/datasets/boolq_qwen3_1.7b/",
    )
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=256)

    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_size", type=int, default=128)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    parser.add_argument(
        "--cov_path",
        type=str,
        default="./mycode/onereplay/flan_qwen3_qv_cov.pt",
        help="Path produced by collect_flan_cov.py",
    )
    parser.add_argument(
        "--replay_lambda",
        type=float,
        default=1e-4,
        help="Weight for tr(DeltaW C DeltaW^T)",
    )
    parser.add_argument(
        "--normalize_replay_by_layers",
        type=int,
        default=1,
        help="1 averages the regularizer over LoRA layers",
    )
    parser.add_argument(
        "--measure_replay_when_lambda_zero",
        type=int,
        default=1,
        help="1 measures replay_reg for vanilla LoRA when cov_path exists, without adding it to loss",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="",
        help="Optional jsonl file for epoch metrics",
    )
    return parser.parse_args()


def print_args(args: argparse.Namespace) -> None:
    """Print arguments once so experiment logs are self-contained."""

    for attr, value in sorted(vars(args).items()):
        print(f"\t{attr.upper()}={value}")


def load_boolq_dataset(args: argparse.Namespace):
    """Load the same jsonl BoolQ files used by vanilla_lora.py."""

    return load_dataset(
        "json",
        data_files={
            "train": str(Path(args.dataset_dir) / "boolq_train.jsonl"),
            "validation": str(Path(args.dataset_dir) / "boolq_validation.jsonl"),
        },
    )


def tokenize_dataset(dataset, tokenizer, args: argparse.Namespace):
    """Add input_ids, labels, and attention_mask fields for causal LM training."""

    def add_tokenized_fields(example):
        tokenized = tokenizer_to_ids(
            tokenizer=tokenizer,
            text=example["text"],
            prompt_text=example["prompt_text"],
            max_length=args.max_len,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
        }

    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_val_samples > 0:
        valid_dataset = valid_dataset.select(range(min(args.max_val_samples, len(valid_dataset))))

    return train_dataset.map(add_tokenized_fields), valid_dataset.map(add_tokenized_fields)


def append_jsonl(path: str, record: dict) -> None:
    """Append one metrics record so long-running experiments are recoverable."""

    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_one_training_example(train_loader, tokenizer) -> None:
    """Show one tokenized sample so label masking can be inspected in logs."""

    batch = next(iter(train_loader))
    print("==============================")
    print("here is an example of the training data sample")
    print(tokenizer.decode(batch["input_ids"][0]))
    print(f"input_ids={batch['input_ids'][0]}")
    print(f"labels={batch['labels'][0]}")
    print(f"attention_mask={batch['attention_mask'][0]}")
    print("==============================")


def find_answer_positions(labels: torch.Tensor) -> torch.Tensor:
    """Return the first supervised label position for each sample.

    In this dataset, prompt labels are -100 and the answer tokens are normal
    token ids. The first non--100 label is the position where the model should
    predict True/False.
    """

    answer_positions = []
    for row in labels:
        non_ignored = torch.nonzero(row != -100, as_tuple=False).flatten()
        if len(non_ignored) == 0:
            answer_positions.append(torch.tensor(row.shape[0] - 1, device=row.device))
        else:
            answer_positions.append(non_ignored[0])
    return torch.stack(answer_positions)


def compare_ground_truth(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
) -> tuple[int, int, list[str]]:
    """Compare predicted answer tokens with the first supervised label token."""

    right_answers = 0
    text_answers: list[str] = []
    answer_positions = find_answer_positions(labels)

    for i in range(predictions.size(0)):
        answer_token = tokenizer.decode(predictions[i]).strip().lower()
        gold_token = tokenizer.decode(labels[i, answer_positions[i]]).strip().lower()
        text_answers.append(answer_token)
        if answer_token == gold_token:
            right_answers += 1

    return right_answers, predictions.size(0), text_answers


def train_one_epoch(
    model,
    train_dataloader,
    optimizer,
    device,
    covariances: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[float, float]:
    """Train one epoch and return average task loss and replay regularizer.

    The task loss is divided by accumulation_steps before backward so the
    effective gradient scale is stable when gradient accumulation is used.
    """

    accumulation_steps = max(args.accumulation_size // args.batch_size, 1)
    model.train()
    optimizer.zero_grad()

    total_task_loss = 0.0
    total_replay_reg = 0.0
    total_samples = 0

    for step, batch in enumerate(train_dataloader, start=1):
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

        if step % accumulation_steps == 0 or step == len(train_dataloader):
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

    return total_task_loss / total_samples, total_replay_reg / total_samples


def evaluate(model, val_dataloader, tokenizer, device) -> tuple[float, float, list[str]]:
    """Evaluate loss and True/False token accuracy on the validation set."""

    model.eval()
    total_eval_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_predictions: list[str] = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_eval_loss += float(outputs.loss.detach().cpu())

            answer_positions = find_answer_positions(labels)
            # Causal LM logits at position p-1 predict the token at position p.
            logit_positions = torch.clamp(answer_positions - 1, min=0)
            batch_indices = torch.arange(input_ids.shape[0], device=device)
            answer_logits = outputs.logits[batch_indices, logit_positions, :]
            predictions = torch.argmax(answer_logits, dim=-1)

            right, total, text_predictions = compare_ground_truth(predictions, labels, tokenizer)
            correct_predictions += right
            total_predictions += total
            all_predictions += text_predictions

    avg_eval_loss = total_eval_loss / len(val_dataloader)
    accuracy = correct_predictions / max(total_predictions, 1)
    return avg_eval_loss, accuracy, all_predictions


def main() -> None:
    args = parse_args()
    print("the file is " + str(Path(__file__).resolve()))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print_args(args)
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Stage 2: loading base model, tokenizer, and LoRA adapter")
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
    )

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    should_load_covariances = (
        args.replay_lambda > 0
        or (args.measure_replay_when_lambda_zero == 1 and Path(args.cov_path).exists())
    )
    if should_load_covariances:
        print("Stage 2: loading precomputed old-knowledge covariance matrices")
        covariances = load_covariance_file(args.cov_path)
        print(f"loaded {len(covariances)} covariance matrices from {args.cov_path}")
    else:
        print("Stage 2: replay_lambda=0, running vanilla LoRA baseline")
        covariances = {}

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
    )

    print("Stage 2: loading and tokenizing downstream fine-tuning data")
    dataset = load_boolq_dataset(args)
    train_dataset, valid_dataset = tokenize_dataset(dataset, tokenizer, args)
    train_loader = build_loader(train_dataset, tokenizer, batch_size=args.batch_size, train=True)
    valid_loader = build_loader(valid_dataset, tokenizer, batch_size=args.batch_size, train=False)
    print_one_training_example(train_loader, tokenizer)

    best_dev_acc = 0.0
    best_dev_epoch = 0

    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        print(f"training epoch={epoch + 1}")
        start_time = time.time()
        train_loss, replay_reg = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            covariances,
            args,
        )
        train_time = time.time() - start_time
        print(
            f"training epoch {epoch + 1} done, "
            f"time={train_time:.2f}, task_loss={train_loss:.4f}, "
            f"replay_reg={replay_reg:.4f}, lambda_reg={args.replay_lambda * replay_reg:.4f}"
        )

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"Peak memory usage: {peak:.2f} MB")

        eval_start = time.time()
        val_loss, val_acc, _ = evaluate(model, valid_loader, tokenizer, device)
        print(
            f"validation epoch {epoch + 1} done, "
            f"time={time.time() - eval_start:.2f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_dev_acc:
            best_dev_acc = val_acc
            best_dev_epoch = epoch + 1

        append_jsonl(
            args.metrics_path,
            {
                "epoch": epoch + 1,
                "seed": args.seed,
                "model_name": args.model_name,
                "target_modules": args.target_modules,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "replay_lambda": args.replay_lambda,
                "cov_path": args.cov_path if args.replay_lambda > 0 else "",
                "train_task_loss": train_loss,
                "train_replay_reg": replay_reg,
                "train_lambda_reg": args.replay_lambda * replay_reg,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_dev_acc": best_dev_acc,
                "best_dev_epoch": best_dev_epoch,
            },
        )

    print("--------")
    print(f"the best dev acc is {best_dev_acc:.4f}, epoch={best_dev_epoch}")

    if args.save == 1:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print(f"saved final LoRA adapter to {args.save_path}")


if __name__ == "__main__":
    main()
