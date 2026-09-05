"""Model and tokenizer loading helpers shared by collection, training, and eval."""

# =============================================================================
# PORT NOTE 文件状态：部分搬运
#
# onereplay/core/modeling.py 的逐字节副本，只加注释、未改代码行。校验：
#     diff onereplay/core/modeling.py con-pretrain/onereplay_port/core/modeling.py
#
# 只有一个函数是真要搬的：snapshot_reference_weights。它的**调用时机**是这次移植里
# 第二大的静默风险（第一是 regularizer.py 的 lookup_covariance），先读那条批注。
# 其余函数或被 LitGPT 自己的机制替代（建模、tokenizer），或属于 LoRA（删）。
# =============================================================================

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


# PORT NOTE [REPLACE] LitGPT 自己建模，这个函数整体不要。
#
# 对应物在 pretrain.py L200-224：
#     with fabric.init_module(empty_init=True): model = GPT(config)  L200-201
#     initialize_weights(...)                                        L203
#     tie_embeddings                                                  L205-206
#     torch.compile                                                   L213
#     fabric.setup                                                    L214
#     optimizer                                                       L216-218
#     dataloaders                                                     L220-221
#     fabric.load_raw(initial_checkpoint_dir / "lit_model.pth")       L223-224
# 注意最后一行：**权重是最后才到位的**，前面一直是随机初始化。这决定了 W0 快照点。
#
# use_bf16 这个参数也没有对应物：精度由 Fabric 的 precision 统一管，默认 "bf16-mixed"
# （utils.py L363-380），不再是模型自己的 torch_dtype。
# tokenizer 在 CPT 里只用于离线数据准备，训练时 pretrain.py 只吃 token id，不需要它。
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


# PORT NOTE [MODIFY] 逻辑（按末段名匹配 + isinstance nn.Linear）通用，只要换目标名。
#
# Pythia/GPT-NeoX 在 LitGPT 里的四个目标短名：["qkv", "proj", "fc"]。
# 注意 attn.proj 和 mlp.proj **短名相同**，按末段名匹配会同时命中两者——这里恰好是想要的
# （两个都是目标层），但如果以后想只保护其中一个，就必须改成匹配完整后缀而非末段名。
# 另外 GptNeoxMLP 是 fc/proj 两个 Linear（model.py L804-815），LLaMAMLP 是
# fc_1/fc_2/proj 三个（L818-831）；config.py 默认 mlp_class_name="GptNeoxMLP"，
# 所以 Pythia 走两个。选别的模型前先确认 mlp_class_name。
#
# 还要决定 lm_head 收不收。HF 侧 Qwen3-1.7B 与 embed_tokens 共享权重，LitGPT 默认不 tie
# （pretrain.py L205 只在 train.tie_embeddings 为真时才 tie，而 args.py L34 默认 None）。
# 也就是说同一个模型在两个框架里 lm_head 是不是独立参数**可能不一样**，这会改变目标层数、
# 进而改变 lambda。移植时显式决定并记录，不要依赖默认值。
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


# PORT NOTE [MODIFY] ★ 函数体几乎不用改，但**调用时机**必须钉死，否则静默算错。
#
# 必须在 pretrain.py L223-224 的 fabric.load_raw 之**后**调用。
# 在那之前模型是 initialize_weights（L203）产生的随机权重，此时快照会把随机初始化当成 W0，
# 于是惩罚变成"把模型拉回随机初始化"——完全反向的目标。不会报错、不会 NaN，loss 只是变差，
# 而"CPT 加了正则效果变差"是个完全合理的实验结论，所以这个 bug 极难从结果上察觉。
# 建议直接加断言：快照后抽一层，检查它和 lit_model.pth 里对应张量数值相等。
#
# 另外两点：
#   - 名字前缀。L213 的 torch.compile 已经把 named_modules() 的名字加上 `_orig_mod.`，
#     而这里用 named_modules() 的名字作为 references 的键、下游 resolve_full_layers 也用
#     同一套名字来匹配，两边一致就自洽。但它还要和 C 的键对上（lookup_covariance），
#     所以前缀处理只能有一处、且三者共用。
#   - data_ptr 去重（L127-130）与 train.tie_embeddings 相互作用。LitGPT 的 tie 写法是
#     `model.transformer.wte.weight = model.lm_head.weight`（L206），共享同一 storage，
#     data_ptr 相同，去重逻辑正好挡住重复计数。wte 是 nn.Embedding 本来就不匹配 nn.Linear，
#     但保留这个检查是对的。
#   - dtype。bf16-mixed 下 module.weight 是 fp32，所以这里 clone 出来的 W0 也是 fp32，
#     快照开销从 3.20 GiB 变成 6.41 GiB。这是换精度的真实代价，要写进成本表；
#     同时它意味着 resolve_full_layers 绝不能再 cast（否则又是两份），见那条批注。
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


# PORT NOTE [DELETE] 从这里到文件末尾（原文件 L135-175：build_lora_model、attach_adapter、
# is_lora_adapter_dir）全是 LoRA/PEFT，CPT 全参数训练下删掉。
# 顺带删掉文件顶部 transformers 的 import 和 print_trainable_parameters
# （LitGPT 用 utils.num_parameters，pretrain.py L211 已经在打了）。
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
