"""Commonsense170k held-out SFT loss metric."""

from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from onereplay.data.chat import build_loader
from onereplay.data.commonsense import load_and_prepare_dataset


class CommonsenseLossMetric:
    name = "commonsense"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        args = Namespace(
            dataset_path=cfg.get("dataset_path", ""),
            seed=int(cfg.get("seed", 1)),
            max_train_samples=int(cfg.get("max_train_samples", 0)),
            max_val_samples=int(cfg.get("max_val_samples", 1000)),
            val_fraction=float(cfg.get("val_fraction", 0.01)),
            max_len=int(cfg.get("max_len", 512)),
            map_cache_dir=cfg.get("map_cache_dir", ""),
        )
        batch_size = int(cfg.get("batch_size", 16))
        start_time = time.time()
        _, valid_dataset = load_and_prepare_dataset(args, tokenizer)
        valid_loader = build_loader(valid_dataset, tokenizer, batch_size=batch_size, train=False)

        model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                total_loss += float(outputs.loss.detach().cpu())
                total_batches += 1
        val_loss = total_loss / max(total_batches, 1)

        summary = {
            "run_name": cfg.get("run_name", "base"),
            "adapter_path": cfg.get("adapter_path", ""),
            "seed": args.seed,
            "max_val_samples": args.max_val_samples,
            "max_len": args.max_len,
            "batch_size": batch_size,
            "val_loss": val_loss,
            "elapsed_sec": time.time() - start_time,
        }
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
