"""Supervised fine-tuning trainer with OneReplay regularization."""

from __future__ import annotations

from typing import Any

import torch

from onereplay.trainers.base import BaseTrainer


class SFTTrainer(BaseTrainer):
    """Task loss is standard causal LM cross-entropy on assistant tokens."""

    def compute_task_loss(self, prepared: dict[str, Any]) -> torch.Tensor:
        input_ids = prepared["input_ids"].to(self.device)
        attention_mask = prepared["attention_mask"].to(self.device)
        labels = prepared["labels"].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss
