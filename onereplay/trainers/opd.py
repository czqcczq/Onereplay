"""On-policy distillation trainer with OneReplay regularization.

The student samples its own completions, the teacher scores those same tokens
with frozen logits, and the task loss is a per-token reverse KL restricted to
completion tokens:

    loss = KL(student || teacher) + lambda * tr(DeltaW C DeltaW^T)

BaseTrainer adds the regularizer, so it stays outside any no_grad block.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from onereplay.data.chat import apply_prompt_template
from onereplay.trainers.base import BaseTrainer


class OPDTrainer(BaseTrainer):
    """On-policy distillation: student rollout scored by a frozen teacher."""

    def __init__(
        self,
        teacher_model: nn.Module,
        tokenizer,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        kl_temperature: float = 1.0,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.kl_temperature = kl_temperature
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)
        self.teacher_device = next(self.teacher_model.parameters()).device

    def profiled_devices(self) -> list[Any]:
        """The teacher's card counts towards OPD's memory cost too."""

        return [self.device, self.teacher_device]

    def rollout(self, prompt_texts: list[str]) -> dict[str, torch.Tensor]:
        """Student on-policy sampling. Replace this method to plug in vLLM."""

        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        prompt_mask = encoded["attention_mask"].to(self.device)
        prompt_width = input_ids.shape[1]

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if was_training:
            self.model.train()

        completion = generated[:, prompt_width:]
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        # Keep tokens up to and including the first EOS; everything after is padding.
        is_eos = completion == eos_id if eos_id is not None else torch.zeros_like(completion).bool()
        after_eos = is_eos.cumsum(dim=1) - is_eos.int() > 0
        completion_mask = (~after_eos).to(prompt_mask.dtype)
        if pad_id is not None and eos_id is not None and pad_id != eos_id:
            completion_mask = completion_mask * (completion != pad_id).to(prompt_mask.dtype)

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        labels = generated.clone()
        labels[:, :prompt_width] = -100
        labels[:, prompt_width:] = torch.where(
            completion_mask.bool(), completion, torch.full_like(completion, -100)
        )

        return {
            "input_ids": generated,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Render prompts from the batch and run the student rollout."""

        if "prompt_text" in batch:
            prompt_texts = list(batch["prompt_text"])
        elif "instruction" in batch:
            instructions = batch["instruction"]
            inputs = batch.get("input") or [""] * len(instructions)
            prompt_texts = []
            for instruction, input_text in zip(instructions, inputs):
                user_content = str(instruction).strip()
                if input_text and str(input_text).strip():
                    user_content = f"{user_content}\n\nInput:\n{str(input_text).strip()}"
                prompt_texts.append(apply_prompt_template(self.tokenizer, user_content))
        else:
            raise ValueError(
                "OPDTrainer needs 'instruction' (with optional 'input') or 'prompt_text' "
                "in the batch; use data.chat.build_opd_loader."
            )

        return self.rollout(prompt_texts)

    def compute_task_loss(self, prepared: dict[str, Any]) -> torch.Tensor:
        """Per-token reverse KL between student and teacher on completion tokens."""

        input_ids = prepared["input_ids"].to(self.device)
        attention_mask = prepared["attention_mask"].to(self.device)
        labels = prepared["labels"].to(self.device)

        # Only completion tokens carry loss. Selecting them before the softmax
        # keeps the float32 vocab-sized tensors off the prompt region, which
        # otherwise dominates memory at long sequence lengths.
        loss_mask = labels[:, 1:] != -100
        if not bool(loss_mask.any()):
            return torch.zeros((), device=self.device, dtype=torch.float32)

        student_logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits[:, :-1, :][loss_mask]
        with torch.no_grad():
            teacher_logits = self.teacher_model(
                input_ids=input_ids.to(self.teacher_device),
                attention_mask=attention_mask.to(self.teacher_device),
            ).logits[:, :-1, :][loss_mask.to(self.teacher_device)].to(self.device)

        temperature = self.kl_temperature
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
        token_kl = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        return token_kl.mean()
