"""Model and tokenizer loading helpers shared by collection, training, and eval."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    """Fix common random seeds so repeated runs are easier to compare."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def join_model_path(model_dir: str, model_name: str) -> str:
    """Build a model path robustly, whether model_dir ends with '/' or not."""

    return os.path.join(model_dir, model_name)


def load_causal_lm_and_tokenizer(
    model_dir: str,
    model_name: str,
    use_bf16: int,
    extra_config: Any | None = None,
):
    """Load the base causal LM and tokenizer used by both stages.

    extra_config is usually argparse.Namespace. We copy its fields into the
    HuggingFace config to preserve the behavior of your vanilla_lora.py.
    """

    model_weight = join_model_path(model_dir, model_name)
    config = AutoConfig.from_pretrained(model_weight)
    if extra_config is not None:
        for key, value in vars(extra_config).items():
            if not hasattr(config, key):
                setattr(config, key, value)

    tokenizer = AutoTokenizer.from_pretrained(model_weight, padding_side="left")
    dtype = torch.bfloat16 if use_bf16 == 1 else None
    model = AutoModelForCausalLM.from_pretrained(
        model_weight,
        config=config,
        torch_dtype=dtype,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer


def find_target_linear_module_names(
    model: nn.Module,
    target_modules: list[str],
) -> list[str]:
    """Find base-model Linear modules whose final name matches LoRA targets.

    For Qwen-style models, target_modules is often ["q_proj", "v_proj"].
    The returned names are full module names such as
    "model.layers.0.self_attn.q_proj"; these names are used as keys for C.
    """

    names: list[str] = []
    target_set = set(target_modules)
    for name, module in model.named_modules():
        short_name = name.rsplit(".", 1)[-1]
        if short_name in target_set and isinstance(module, nn.Linear):
            names.append(name)
    return names


def print_trainable_parameters(model: nn.Module) -> None:
    """Mirror PEFT's print_trainable_parameters for a plain (unwrapped) model."""

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    percentage = 100.0 * trainable / max(total, 1)
    print(
        f"trainable params: {trainable:,} || all params: {total:,} "
        f"|| trainable%: {percentage:.4f}"
    )


def snapshot_reference_weights(
    model: nn.Module,
    covariances: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Freeze W0 for every Linear layer that has a covariance matrix.

    Full fine-tuning needs DeltaW = W - W0, so the initial weights have to
    survive the whole run. Only layers with a C are snapshotted, which makes
    this dictionary the definition of the penalty's coverage.

    Tied weights (Qwen3's small checkpoints share embed_tokens with lm_head)
    would otherwise be counted twice and silently double that layer's lambda.
    embed_tokens is an nn.Embedding so it never matches here, but the storage
    check also guards against any other aliasing in the checkpoint.
    """

    from onereplay.core.regularizer import lookup_covariance

    references: dict[str, torch.Tensor] = {}
    seen_storage: set[int] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        _, covariance = lookup_covariance(covariances, module_name)
        if covariance is None:
            continue
        pointer = module.weight.data_ptr()
        if pointer in seen_storage:
            continue
        seen_storage.add(pointer)
        references[module_name] = module.weight.detach().clone()
    return references


def build_lora_model(
    model: nn.Module,
    target_modules: list[str],
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> nn.Module:
    """Wrap a base causal LM with a PEFT LoRA adapter for causal LM training."""

    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def attach_adapter(model: nn.Module, adapter_path: str) -> nn.Module:
    """Attach a saved LoRA adapter to a base model for evaluation."""

    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)


def is_lora_adapter_dir(path: str) -> bool:
    """True when a training output holds a PEFT adapter rather than full weights.

    LoRA runs save a small adapter that has to be attached to the base model;
    full fine-tuning runs save a complete checkpoint that loads on its own.
    PEFT always writes adapter_config.json, so its presence separates the two.
    """

    return (Path(path) / "adapter_config.json").is_file()
