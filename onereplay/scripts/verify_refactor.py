"""Parity check: legacy training loop vs refactored SFTTrainer.

Runs a handful of steps twice with the same seed — once through the original
train_commonsense_lora.train_one_epoch, once through trainers.sft.SFTTrainer —
and compares train_task_loss and replay_reg. Identical numbers mean the
refactor only moved code.

Run with --lora_dropout 0 (default here) so dropout noise cannot mask a
real difference.

    python -m onereplay.scripts.verify_refactor \
      --dataset_path ... --cov_path ... --max_train_samples 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from onereplay.core.modeling import (  # noqa: E402
    build_lora_model,
    load_causal_lm_and_tokenizer,
    set_seed,
)
from onereplay.core.regularizer import ReplayRegularizer  # noqa: E402
from onereplay.data.chat import build_loader  # noqa: E402
from onereplay.data.commonsense import load_and_prepare_dataset  # noqa: E402
from onereplay.trainers.sft import SFTTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the OneReplay refactor is numerics-neutral")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--cov_path", type=str, required=True)
    parser.add_argument("--map_cache_dir", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=64)
    parser.add_argument("--max_val_samples", type=int, default=8)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_size", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--replay_lambda", type=float, default=1e-4)
    parser.add_argument("--normalize_replay_by_layers", type=int, default=1)
    parser.add_argument("--identity_cov", type=int, default=0)
    parser.add_argument("--measure_replay_when_lambda_zero", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def build_fresh_model(args: argparse.Namespace, device):
    """Rebuild the base model + LoRA from the same seed so both runs start equal."""

    set_seed(args.seed)
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir, args.model_name, args.use_bf16, args
    )
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    model = build_lora_model(
        model,
        target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from onereplay.legacy import train_commonsense_lora as legacy

    # ---- legacy path -------------------------------------------------------
    model, tokenizer = build_fresh_model(args, device)
    train_dataset, _ = load_and_prepare_dataset(args, tokenizer)
    from onereplay.core.covariance import (
        load_covariance_file,
        move_covariances_to_device,
        to_identity_covariances,
    )

    covariances = load_covariance_file(args.cov_path)
    if args.identity_cov == 1:
        covariances = to_identity_covariances(covariances)
    covariances = move_covariances_to_device(covariances, device=device, dtype=torch.float32)

    set_seed(args.seed)
    legacy_loader = build_loader(
        train_dataset, tokenizer, batch_size=args.batch_size, train=True
    )
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=0.0
    )
    legacy_loss, legacy_reg = legacy.train_one_epoch(
        model, legacy_loader, optimizer, device, covariances, args
    )

    # ---- refactored path ---------------------------------------------------
    model, tokenizer = build_fresh_model(args, device)
    regularizer = ReplayRegularizer.from_path(
        args.cov_path,
        device=device,
        identity=args.identity_cov == 1,
        normalize_by_layers=bool(args.normalize_replay_by_layers),
    )
    set_seed(args.seed)
    new_loader = build_loader(train_dataset, tokenizer, batch_size=args.batch_size, train=True)
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=0.0
    )
    trainer = SFTTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        regularizer=regularizer,
        replay_lambda=args.replay_lambda,
        batch_size=args.batch_size,
        accumulation_size=args.accumulation_size,
        log_every=args.log_every,
    )
    new_loss, new_reg = trainer.train_one_epoch(new_loader)

    loss_delta = abs(legacy_loss - new_loss)
    reg_delta = abs(legacy_reg - new_reg)
    print(f"legacy  task_loss={legacy_loss:.10f} replay_reg={legacy_reg:.10e}")
    print(f"refactor task_loss={new_loss:.10f} replay_reg={new_reg:.10e}")
    print(f"delta   task_loss={loss_delta:.3e} replay_reg={reg_delta:.3e}")

    rel_reg = reg_delta / max(abs(legacy_reg), 1e-12)
    if loss_delta <= args.tolerance and rel_reg <= args.tolerance:
        print("PARITY OK: refactor is numerically equivalent")
    else:
        raise SystemExit(
            f"PARITY FAILED: task_loss delta {loss_delta:.3e}, replay_reg rel delta {rel_reg:.3e}"
        )


if __name__ == "__main__":
    main()
