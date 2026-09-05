"""Covariance matrix load/save, identity ablation, and collection hooks."""

# =============================================================================
# PORT NOTE 文件状态：整体搬运
#
# onereplay/core/covariance.py 的逐字节副本，只加注释、未改代码行。校验：
#     diff onereplay/core/covariance.py con-pretrain/onereplay_port/core/covariance.py
#
# 这是全套代码里最与框架无关的一份，数学不用动。要决策的只有命名和 dtype 两件事，
# 逐个函数见下面的批注。
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def load_covariance_file(path: str) -> dict[str, torch.Tensor]:
    """Load C matrices from disk and return only the covariance dictionary."""

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "covariances" in payload:
        return payload["covariances"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported covariance file format: {path}")


# PORT NOTE [KEEP] 消融臂原样保留。
#
# 这一臂在 CPT 上比在 SFT 上更重要：inspect_cov_layers 已经查出
# layers.2.mlp.down_proj 99.995% 的迹集中在单个方向、层间保护强度差 6.13e+04 倍。
# 如果真 C 相对 identity C 没有优势，机制性结论就立不住，这个臂必须跑。
#
# 只有 docstring 措辞要改（说的是 "LoRA update"，全参下应是 W - W0），代码不用改。
def to_identity_covariances(
    covariances: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Replace every C_l with an identity matrix of the same size.

    This is the key ablation control for OneReplay. It turns the penalty
    tr(DeltaW C DeltaW^T) into tr(DeltaW DeltaW^T) = ||DeltaW||_F^2, i.e. plain
    L2 shrinkage on the LoRA update with no old-knowledge structure. Comparing
    real C against this identity control isolates whether the benefit comes
    from the covariance directions or merely from shrinking DeltaW.

    Module-name keys are preserved so lookup_covariance still matches layers.
    """

    identity: dict[str, torch.Tensor] = {}
    for name, covariance in covariances.items():
        dim = covariance.shape[-1]
        identity[name] = torch.eye(dim, dtype=covariance.dtype)
    return identity


# PORT NOTE [MODIFY] C 常驻的 dtype 要对着 bf16-mixed 重新决定。
#
# fp32 默认值原本是纯 bf16 训练下的必需：权重是 bf16，C 得当那个 fp32 操作数，
# 否则 2*DeltaW*C 会被舍掉。LitGPT 默认 precision="bf16-mixed"，主权重本来就是
# fp32，所以 fp32 的 C 不再承担精度作用，纯粹变成显存选择——而且值得测：C 全程常驻
# sum_l(d_in_l^2) * 4 字节，是本方法最大的固定开销，改 bf16 能砍一半。
#
# 但不要单独改这里。这里选什么 dtype，必须和 regularizer.py 的 resolve_full_layers
# 一起看：那边刻意不再做 cast，就是为了避免 W0 以两种精度各存一份（之前那 ~6.4 GiB
# 显存谜团的来源）。两个旋钮、一条不变量：大张量的 fp32 副本只能有一份。
#
# FSDP 下还有一个这个签名表达不了的问题：C 按整个模块存，但每张卡只持有该模块权重的
# dim-0 分片。要么每卡存全量 C（简单、浪费，就是现在单卡的做法），要么把 C 切到与本地
# 分片对齐。这个决定放在 regularizer 里做，但它会在这个函数上显形。
def move_covariances_to_device(
    covariances: dict[str, torch.Tensor],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Move all C matrices once before training starts.

    The OneReplay penalty reads every target layer's covariance at every
    optimization step. Keeping C on CPU would repeatedly copy large matrices to
    GPU inside the training loop, so this helper pays that transfer cost once.
    """

    return {
        name: covariance.to(device=device, dtype=dtype)
        for name, covariance in covariances.items()
    }


def save_covariance_payload(
    output_path: str,
    covariances: dict[str, torch.Tensor],
    counts: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    """Save C matrices, token counts, and collection settings together."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "covariances": {key: value.cpu() for key, value in covariances.items()},
        "counts": counts,
        "metadata": metadata,
    }
    torch.save(payload, output_path)


# PORT NOTE [MODIFY] mask 分支在定长打包下会失效，而它的失效方式是危险的、不是无害的。
#
# LitGPT 预训练管线把文档打包成定长块、完全没有 padding（见 data/text_files.py 的
# TokensLoader），所以没有 attention_mask 可放进 attention_holder。此时 hook 走
# flat_mask = None、统计所有位置——这恰好是对的，因为每个位置都是真 token。
#
# 危险在这一行的形状守卫：
#     if attention_mask is not None and attention_mask.shape[:2] == (batch, seq_len)
# SFT 下它是有效的保险，因为残留的旧 mask 通常 seq_len 不同、会被挡掉。定长打包下每个
# batch 形状完全一样，残留或被错误填入的 mask 能通过守卫，于是静默地屏蔽掉错误的 token，
# C 在一个位置子集上被采集，不报错也不告警。
#
# 所以移植要求不是"删掉这个分支"，而是"证明 holder 是空的"：要么在新采集器里断言首次
# 调用时 attention_holder.get("attention_mask") is None，要么直接把 holder 参数从签名
# 里去掉，让这个问题不可能出现。倾向后者——一个"一旦可达就会污染 C"的死分支比没有分支更糟。
#
# 新采集器另外两个小点：
#   - base_output_norm 把 `output` 当作冻结的基线值 W x。采集必须在刚加载完的权重上、
#     eval + no_grad 里做，不能在任何 optimizer step 之后，否则归一化分母对的是漂移后的权重。
#   - 采集时不要 torch.compile。LitGPT 默认编译（pretrain.py L213），而编译后的子模块上挂
#     forward hook 正是会 graph break 或被跳过的情形。采集是独立脚本，保持 eager。
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


# PORT NOTE [MODIFY] C 的键就是你 hook 的那个模型的名字，两套名字空间没有一个字符串是重合的。
#
# 拿 HF 名字去 hook LitGPT 的 GPT，这里 module_dict[module_name] 会直接 KeyError——这个失效
# 至少是响的。静默的那个在下游 lookup_covariance（regularizer.py），见那边的批注。
#
# 两条出路，选哪条取决于基座模型：
#
# 方案 A（推荐）：直接在 LitGPT 的 GPT 模块上采集，时机是 convert_hf_checkpoint +
# fabric.load_raw 之后。C 从出生就是 LitGPT 命名，不存在可能写错的映射代码。这也是唯一能让
# 采集与训练共用同一条数据管线的方案，而这一点很关键，见 scripts/collect_cov.py 的批注。
#
# 方案 B：继续在 HF 模型上采集再重映射键名。Pythia/GPT-NeoX 是 1:1、没有融合错配
# （convert_hf_checkpoint.py L35-57）：
#     gpt_neox.layers.N.attention.query_key_value -> transformer.h.N.attn.qkv
#     gpt_neox.layers.N.attention.dense           -> transformer.h.N.attn.proj
#     gpt_neox.layers.N.mlp.dense_h_to_4h         -> transformer.h.N.mlp.fc
#     gpt_neox.layers.N.mlp.dense_4h_to_h         -> transformer.h.N.mlp.proj
# 每个 block 四个目标 Linear，两边数量一致，所以 lambda 不需要重新换算。
#
# 关于 qkv 融合这个具体疑问：HF Pythia 本来就把 q/k/v 融进一个叫 query_key_value 的 Linear，
# LitGPT 转换时调 qkv_reassemble（convert_hf_checkpoint.py L726-744），它是沿
# dim 0 也就是**输出行**做 chunk/cat。C 是对层**输入**求的 d_in x d_in 矩阵，输出行的置换
# 完全不影响 C。也就是说在 HF 的 query_key_value 上采的 C 可以逐比特直接用到 attn.qkv 上，
# 不用重采、不用置换。
#
# 这也正是这里该选 Pythia 而不是 Qwen3 的理由。Qwen3 在 HF 里是 q_proj/k_proj/v_proj 三个
# 独立 Linear，而 LitGPT 融成一个 attn.qkv，于是 197 个目标模块变成 141 个，所有依赖层数的量
# 都跟着变——包括 lambda。见 ReplayRegularizer.normalize_by_layers 的批注。
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
