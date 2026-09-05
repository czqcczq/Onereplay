"""OneReplay covariance regularizer (framework-agnostic).

The math used by OneReplay is one formula regardless of how the model is
adapted:

    regularizer = E_x ||DeltaW x||_2^2 = tr(DeltaW C DeltaW^T)

where C = E_x[x x^T] is estimated once from old-knowledge data. Only the
source of DeltaW differs between the two training modes:

  LoRA          y = W x + scale * B A x, so DeltaW = scale * B A and the trace
                collapses to scale^2 * tr((B^T B)(A C A^T)), which touches only
                rank x rank matrices.
  full finetune DeltaW = W - W0 against a frozen snapshot of the initial
                weights, evaluated as sum((DeltaW C) * DeltaW).

Both paths compute the same quantity; the LoRA form is an algebraic shortcut
that exploits the low-rank factorization, not a different penalty.

If C is collected with x' = x / ||W x||, the same training code instead
computes a relative-error penalty:

    E_x ||DeltaW x||_2^2 / ||W x||_2^2
"""

# =============================================================================
# PORT NOTE 文件状态：搬运，但要砍掉将近一半
#
# onereplay/core/regularizer.py 的逐字节副本，只加注释、未改代码行。校验：
#     diff onereplay/core/regularizer.py con-pretrain/onereplay_port/core/regularizer.py
#
# CPT 是全参数训练，所以这里所有 LoRA 和 EWC 路径都是死代码。按下列清单删（行号是**原文件**
# 的，不含这些注释）：
#     get_lora_weight_matrices     L32-55
#     lora_covariance_regularizer  L98-155
#     _check_fisher_shape          L321-336
#     lora_fisher_regularizer      L339-395
#     full_fisher_regularizer      L398-447
#     EWCRegularizer               L625-716
# 716 行里删掉约 340 行。留下的是四个全参函数，加上剪掉 LoRA 分支的 ReplayRegularizer。
#
# 数学部分一行都不用改。这个文件里所有真实风险都是**命名**或**精度**风险，其中最要命的那个是静默的：
#
#   - lookup_covariance 返回 (None, None) 不会抛异常。它只是把层名记进 missing_layers，
#     该层惩罚变成 0，训练照常全速跑、loss 曲线干净。名字空间彻底对不上时，每层 R 都是 0，
#     整个 run 看起来和不加正则的 baseline 一模一样。这个文件先读那条批注。
#   - normalize_by_layers 会除以层数，所以 lambda 只在"相对某个层数"的意义下有定义。
#     q/k/v 一融合，层数变，lambda 就得跟着换算。
#   - 这里有几处 docstring 在论证 ".grad 是 bf16"，那在纯 bf16 训练下成立，在 LitGPT 默认的
#     bf16-mixed 下不成立。代码依然正确，只是它声明的约束放松了。不要不加检查地照搬那套推理，
#     更不要为了迎合旧推理去"修"代码。
# =============================================================================

from __future__ import annotations

import torch
from torch import nn


# PORT NOTE [DELETE] 仅 LoRA 用。CPT 全参数训练，新项目不依赖 PEFT。
# 与 lora_covariance_regularizer 一起删。
def get_lora_weight_matrices(module: nn.Module, adapter_name: str = "default"):
    """Return A, B, and scale from one PEFT LoRA-wrapped linear module.

    PEFT stores A and B inside ModuleDicts. For a linear layer:
      A: [rank, d_in]
      B: [d_out, rank]
      scale: lora_alpha / rank
    """

    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return None
    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        return None

    lora_a = module.lora_A[adapter_name].weight
    lora_b = module.lora_B[adapter_name].weight

    if hasattr(module, "scaling") and adapter_name in module.scaling:
        scale = float(module.scaling[adapter_name])
    else:
        rank = lora_a.shape[0]
        alpha = module.lora_alpha[adapter_name]
        scale = float(alpha) / float(rank)
    return lora_a, lora_b, scale


# PORT NOTE [MODIFY] 函数保留，但要剥的前缀换一个。
#
# LitGPT 没有 PEFT，`base_model.model.` 永远不会出现。但 pretrain.py L213 的
# `model = torch.compile(model)` 会包一层 OptimizedModule，之后 named_modules() 里所有
# 名字都带 `_orig_mod.` 前缀。改成剥这个前缀即可，逻辑同构：
#     prefixes = ("_orig_mod.",)
#
# fabric.setup 在 FSDP 下可能再包一层，视 strategy 而定。移植时别猜——在 pretrain.py
# L214 之后打一行 next(iter(model.named_modules()))[0] 看实际前缀，按实际的写。
def strip_peft_prefix(module_name: str) -> str:
    """Convert PEFT module names back to base-model style names.

    Example:
      base_model.model.model.layers.0.self_attn.q_proj
    becomes:
      model.layers.0.self_attn.q_proj
    """

    prefixes = (
        "base_model.model.",
        "base_model.",
    )
    for prefix in prefixes:
        if module_name.startswith(prefix):
            return module_name[len(prefix) :]
    return module_name


# PORT NOTE [MODIFY] ★ 全套代码里最危险的一处。整个文件先读这条。
#
# 失效是静默的：查不到只返回 (None, None)，调用方（full_covariance_regularizer L188、
# resolve_full_layers L252）把层名记进 missing_layers 就 continue。不抛异常、不告警。
# 名字空间彻底对不上时，每一层都走 continue，R 恒等于 0，训练全速跑完、loss 曲线干净，
# 结果和不加正则的 baseline 完全一致——而你会以为自己跑的是 onereplay 臂。
#
# 现在的兜底是后缀匹配（L91），HF/PEFT 下靠它恰好能命中，本质是运气：`_orig_mod.` 前缀
# 也会因为后缀匹配而"碰巧"命中。但 LitGPT 名字里 attn.proj 和 mlp.proj 都以 `proj` 结尾，
# 后缀匹配在这种命名下更容易撞出多个候选，而 len(matches) == 1 不满足时它同样静默返回 None。
#
# 移植时改三件事：
#   1. 前缀处理显式化（见 strip_peft_prefix），不要靠后缀匹配兜。
#   2. 如果走方案 B（在 HF 上采 C），HF->LitGPT 的映射表放在这里。
#   3. **必须加硬失败**：在 resolve_full_layers 解析完之后，断言 missing 为空、
#      或 used_layers 等于预期的目标层数，不满足就 raise。这是整个移植里唯一不能省的断言。
#      正式跑之前先跑一次 lambda 很大的 smoke run，确认 loss 明显被拉动——R 恒为 0 的话
#      loss 不会有任何反应，这是最便宜的验证。
def lookup_covariance(
    covariances: dict[str, torch.Tensor],
    lora_module_name: str,
) -> tuple[str, torch.Tensor] | tuple[None, None]:
    """Find the covariance matrix that corresponds to one LoRA module.

    The exact module name can differ before and after PEFT wraps the model.
    We first try the stripped exact name, then suffix matching.
    """

    base_name = strip_peft_prefix(lora_module_name)
    if base_name in covariances:
        return base_name, covariances[base_name]

    matches = [key for key in covariances if base_name.endswith(key) or key.endswith(base_name)]
    if len(matches) == 1:
        key = matches[0]
        return key, covariances[key]
    return None, None


# PORT NOTE [DELETE] LoRA 专用的低秩捷径，全参数训练下用不到。
# 数学上它和 full_covariance_regularizer 是同一个惩罚，只是利用了 DeltaW = scale*B@A 的
# 分解把迹收缩到 rank x rank。CPT 没有这个分解，删。
def lora_covariance_regularizer(
    model: nn.Module,
    covariances: dict[str, torch.Tensor],
    adapter_name: str = "default",
    normalize_by_layers: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum_l tr(DeltaW_l C_l DeltaW_l^T) for all LoRA layers.

    We avoid explicitly building the full DeltaW = scale * B @ A. Instead:

        tr((scale * B A) C (scale * B A)^T)
        = scale^2 * tr((B^T B) (A C A^T))

    The matrices inside the trace are only rank x rank, so this is much cheaper
    than multiplying a full d_out x d_in DeltaW.
    """

    total_reg: torch.Tensor | None = None
    used_layers = 0
    missing_layers: list[str] = []

    for module_name, module in model.named_modules():
        weights = get_lora_weight_matrices(module, adapter_name=adapter_name)
        if weights is None:
            continue

        lora_a, lora_b, scale = weights
        cov_key, covariance = lookup_covariance(covariances, module_name)
        if covariance is None:
            missing_layers.append(module_name)
            continue

        covariance = covariance.to(device=lora_a.device, dtype=torch.float32)
        a = lora_a.float()
        b = lora_b.float()

        # A C A^T measures how much the LoRA input projection responds to old
        # hidden-state directions. B^T B then weights that response by the LoRA
        # output projection.
        a_c_a_t = a @ covariance @ a.T
        b_t_b = b.T @ b
        layer_reg = (scale**2) * torch.sum(b_t_b * a_c_a_t.T)

        total_reg = layer_reg if total_reg is None else total_reg + layer_reg
        used_layers += 1

    if total_reg is None:
        device = next(model.parameters()).device
        total_reg = torch.zeros((), device=device, dtype=torch.float32)

    if normalize_by_layers and used_layers > 0:
        total_reg = total_reg / used_layers

    stats = {
        "used_layers": float(used_layers),
        "missing_layers": float(len(missing_layers)),
    }
    return total_reg, stats


# PORT NOTE [KEEP] 核心惩罚，autograd 臂。代码原样可用。
#
# 保留它的理由不是性能（analytic 路径更快），而是等价性校验：两条实现算同一个量，
# 换框架后要重新验一次 analytic == autograd，这是最直接的回归测试。
#
# 移植后要留意的是它读的是 module.weight。bf16-mixed 下 Fabric 保留 fp32 主权重，
# 所以 L192-193 的 .float() 变成 no-op，DeltaW 是真 fp32——这正是之前担心
# "bf16 只有 8 位尾数、小 ΔW 被吃掉"的那个问题自然消失的地方，不需要为此改代码。
#
# 唯一要确认的是它别被 torch.compile 吞进图里：这个函数在 fabric.backward 之外调用，
# 走的是 eager，没问题；但如果哪天挪进 forward 就要重新评估。
def full_covariance_regularizer(
    model: nn.Module,
    covariances: dict[str, torch.Tensor],
    reference_weights: dict[str, torch.Tensor],
    normalize_by_layers: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum_l tr(DeltaW_l C_l DeltaW_l^T) for full fine-tuning.

    DeltaW = W - W0 against the frozen snapshot in reference_weights. Without a
    low-rank factorization to exploit, the trace is evaluated directly:

        tr(DeltaW C DeltaW^T) = sum((DeltaW @ C) * DeltaW)

    Only layers present in reference_weights are visited, so the snapshot
    doubles as the definition of which layers the penalty covers.
    """

    total_reg: torch.Tensor | None = None
    used_layers = 0
    missing_layers: list[str] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        reference = reference_weights.get(module_name)
        if reference is None:
            continue

        _, covariance = lookup_covariance(covariances, module_name)
        if covariance is None:
            missing_layers.append(module_name)
            continue

        weight = module.weight
        covariance = covariance.to(device=weight.device, dtype=torch.float32)
        delta = weight.float() - reference.to(device=weight.device, dtype=torch.float32)
        layer_reg = torch.sum((delta @ covariance) * delta)

        total_reg = layer_reg if total_reg is None else total_reg + layer_reg
        used_layers += 1

    if total_reg is None:
        device = next(model.parameters()).device
        total_reg = torch.zeros((), device=device, dtype=torch.float32)

    if normalize_by_layers and used_layers > 0:
        total_reg = total_reg / used_layers

    stats = {
        "used_layers": float(used_layers),
        "missing_layers": float(len(missing_layers)),
    }
    return total_reg, stats


# PORT NOTE [KEEP，但 FSDP 多卡时要重做] 单卡原样可用。
#
# docstring 里那段"只解析 device、绝不解析 dtype"的推理是上次修显存 bug 的成果，别改回去：
# 在这里 cast 到 fp32 会让 W0 以 fp32(6.41 GiB) + bf16(3.20 GiB) 两份常驻，导致 analytic
# 路径峰值反而比它要替代的 autograd 路径高 2.06 GiB。这条约束在 bf16-mixed 下依然要守，
# 只是原因变了：那时 W0 快照本身就该存 fp32（主权重是 fp32），别再存第二份。
#
# 缓存时机：self._layers 是第一次调用时懒解析的，也就是第一个 optimizer step。那时权重
# 早已 load_raw 完成，所以时机是安全的——前提是 W0 快照本身在 load_raw 之后做，
# 见 modeling.py 的 snapshot_reference_weights 批注。
#
# FSDP 下必须改：每张卡上 module.weight 只是 dim-0 的本地分片，而 C 是整模块的
# d_in x d_in。d_in 是**输入维**、不是被切的那一维，所以 delta @ covariance 的形状恰好
# 还能对上（[本地行数, d_in] @ [d_in, d_in]），每卡算自己那部分行的惩罚，梯度也只写本地
# 分片——数学上是自动可分的，不需要 all-reduce C。要 all-reduce 的只有用于日志的标量 R。
# 但这依赖 weight 是 DTensor 时 .to()/@ 的行为，移植时先用 2 卡小模型验一次形状和数值。
def resolve_full_layers(
    model: nn.Module,
    covariances: dict[str, torch.Tensor],
    reference_weights: dict[str, torch.Tensor],
) -> tuple[list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]], list[str]]:
    """Pair every regularized Linear with its C and its frozen W0, once.

    full_covariance_regularizer re-walks named_modules() and re-runs the suffix
    matching in lookup_covariance on every call. That is a few milliseconds of
    Python per optimizer step, invisible next to the fp32 matmuls it precedes but
    not next to the analytic path, which is an order of magnitude cheaper. The
    pairing cannot change during a run -- C is frozen, W0 is frozen, and the
    module tree is fixed once the model is built -- so it is resolved once here
    and reused.

    Only the device is resolved here, never the dtype. Casting to fp32 at this
    point looks free, since full_covariance_grad_ needs fp32 operands anyway, but
    the converted tensor would then be cached for the whole run beside the bf16
    snapshot it came from, leaving W0 resident twice. On Qwen3-1.7B that was 6.41
    GiB of fp32 on top of 3.20 GiB of bf16, which made the analytic path peak 2.06
    GiB *higher* than the autograd path it exists to be cheaper than, and
    reference_memory_bytes could not see the second copy, so the difference
    surfaced in the cost tables as unexplained "temporary" memory.

    full_covariance_grad_ casts per layer instead. That is exact either way -- the
    snapshot is cloned from a bf16 model, so widening it recovers no information --
    and it keeps one layer of temporaries alive rather than 197.
    """

    layers: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]] = []
    missing: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        reference = reference_weights.get(module_name)
        if reference is None:
            continue
        _, covariance = lookup_covariance(covariances, module_name)
        if covariance is None:
            missing.append(module_name)
            continue
        weight = module.weight
        layers.append(
            (
                module_name,
                weight,
                covariance.to(device=weight.device),
                reference.to(device=weight.device),
            )
        )
    return layers, missing


# PORT NOTE [KEEP] analytic 注入，代码原样可用；这是效率优势的来源，别退回 autograd。
#
# 之前的疑问"analytic 优化会不会和 bf16-mixed 必须用的 autocast 冲突"——不冲突，两者作用在
# 不同环节。autocast 管的是 forward 里算 logits 时的算子精度；这个函数在 forward/backward
# 之外，用 no_grad 直接读权重、写 .grad，从来不进 autocast 的作用域。bf16-mixed 换过来
# 之后这个函数一行都不用改。
#
# 反而变简单了：L314 的 `delta_c.to(weight.grad.dtype)` 在 bf16-mixed 下是 fp32->fp32
# 的 no-op，docstring 里"只有空 buffer 能保住 1% 量级的项"那条约束随之消失。
# 也就是说注入位置（窗口开头 vs 中间）不再敏感，见 accumulate_grad 的批注。
#
# allow_tf32 那段推理继续有效，与框架无关。
@torch.no_grad()
def full_covariance_grad_(
    layers: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]],
    scale: float,
    allow_tf32: bool = True,
    compute_dtype: torch.dtype = torch.float32,
) -> float:
    """Add scale * dR/dW straight into .grad and return the un-normalized R.

    R = sum_l tr(DeltaW_l C_l DeltaW_l^T) has an analytic gradient,

        dR/dDeltaW = DeltaW (C + C^T) = 2 DeltaW C    for symmetric C,

    and DeltaW C is exactly the product the forward pass already forms. Letting
    autograd rediscover it costs a second matmul of the same size, and keeps every
    layer's fp32 DeltaW alive in the graph until backward returns -- measured at
    4.35 GiB of temporaries for the 197 covered layers on Qwen3-1.7B. Writing the
    gradient directly halves the arithmetic and lets each layer's temporaries die
    immediately, so the peak is one layer instead of all of them.

    That last sentence holds only while the caller hands over the snapshot in its
    stored dtype; resolve_full_layers documents the cached fp32 copy that used to
    defeat it.

    R reads only weights, and weights are frozen across a gradient-accumulation
    window, so injecting once per window accumulates the same gradient as adding
    lambda*R to any single micro-batch's loss. The caller injects at the start of
    the window, matching where the autograd path adds it; see accumulate_grad for
    why that position is not free in bf16.

    C is stored fp32 and is not modified; allow_tf32 only lets the matmul round its
    inputs to 11 mantissa bits inside the tensor core. That is a ~5e-4 relative
    perturbation of C against a 7x throughput difference on H100, and it sits an
    order of magnitude below the 5e-3 penalty-ratio gap that two different
    sampling strategies of the *same* replay corpus already produce.
    """

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    total = 0.0
    try:
        for _, weight, covariance, reference in layers:
            delta = weight.to(compute_dtype) - reference.to(compute_dtype)
            delta_c = delta @ covariance.to(compute_dtype)
            total += float((delta_c * delta).sum())
            if scale != 0.0:
                if weight.grad is None:
                    weight.grad = torch.zeros_like(weight)
                weight.grad.add_(delta_c.to(weight.grad.dtype), alpha=2.0 * scale)
            del delta, delta_c
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return total


# PORT NOTE [DELETE] 从这里到 full_fisher_regularizer 结尾（原文件 L321-447）整段是 EWC
# 基线，属于旧实验的对照组，不随 onereplay 搬进 LitGPT。
#
# 如果新实验还要 EWC 基线，那是独立的一份 PR：EWC 的 F 是对角的、按 param 存，
# 和 C 按模块存的 d_in x d_in 结构不同，混在一个文件里就是 _check_fisher_shape 存在的原因。
# 在 LitGPT 上重做时应该分开成两个文件，别再复制这个耦合。
def _check_fisher_shape(module_name: str, fisher: torch.Tensor, delta: torch.Tensor) -> None:
    """Fail loudly when a covariance file was handed to the EWC path.

    Both files are dicts of tensors keyed by module name, so passing one where
    the other is expected costs nothing at load time and only shows up as a
    shape error deep inside a matmul, or worse, as a silently different penalty
    on a square layer. C is d_in x d_in and F is d_out x d_in, which separates
    them for every layer where the projection is not square.
    """

    if fisher.shape != delta.shape:
        raise ValueError(
            f"Fisher for {module_name} has shape {tuple(fisher.shape)} but DeltaW is "
            f"{tuple(delta.shape)}. A covariance file is d_in x d_in while a Fisher file "
            "is d_out x d_in, so check that --fisher_path is not pointing at a cov_*.pt."
        )


def lora_fisher_regularizer(
    model: nn.Module,
    fishers: dict[str, torch.Tensor],
    adapter_name: str = "default",
    normalize_by_layers: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum_l sum_ij F_l,ij (DeltaW_l,ij)^2 for all LoRA layers.

    Unlike the covariance penalty, this one does not collapse into rank x rank
    matrices. tr(DeltaW C DeltaW^T) contracts C against the row space of DeltaW,
    which the low-rank factorization can exploit; an elementwise weighting pairs
    every entry of F with a distinct entry of DeltaW, so DeltaW = scale * B A has
    to be materialized. That costs O(d_out * d_in * rank) per layer, which is the
    same order as the A C A^T contraction on the covariance path, so the two
    baselines stay comparable in per-step time.

    Penalizing the synthesized update rather than A and B separately is
    deliberate: the entries of DeltaW depend on the product of the two factors,
    so per-factor Fisher matrices would measure importance in coordinates that
    have no meaning on their own.
    """

    total_reg: torch.Tensor | None = None
    used_layers = 0
    missing_layers: list[str] = []

    for module_name, module in model.named_modules():
        weights = get_lora_weight_matrices(module, adapter_name=adapter_name)
        if weights is None:
            continue

        lora_a, lora_b, scale = weights
        _, fisher = lookup_covariance(fishers, module_name)
        if fisher is None:
            missing_layers.append(module_name)
            continue

        delta = scale * (lora_b.float() @ lora_a.float())
        fisher = fisher.to(device=delta.device, dtype=torch.float32)
        _check_fisher_shape(module_name, fisher, delta)
        layer_reg = torch.sum(fisher * delta * delta)

        total_reg = layer_reg if total_reg is None else total_reg + layer_reg
        used_layers += 1

    if total_reg is None:
        device = next(model.parameters()).device
        total_reg = torch.zeros((), device=device, dtype=torch.float32)

    if normalize_by_layers and used_layers > 0:
        total_reg = total_reg / used_layers

    stats = {
        "used_layers": float(used_layers),
        "missing_layers": float(len(missing_layers)),
    }
    return total_reg, stats


def full_fisher_regularizer(
    model: nn.Module,
    fishers: dict[str, torch.Tensor],
    reference_weights: dict[str, torch.Tensor],
    normalize_by_layers: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum_l sum_ij F_l,ij (W - W0)_ij^2 for full fine-tuning.

    Same penalty as the LoRA path with DeltaW read from a frozen W0 snapshot
    instead of the adapter factors, mirroring how full_covariance_regularizer
    relates to lora_covariance_regularizer.
    """

    total_reg: torch.Tensor | None = None
    used_layers = 0
    missing_layers: list[str] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        reference = reference_weights.get(module_name)
        if reference is None:
            continue

        _, fisher = lookup_covariance(fishers, module_name)
        if fisher is None:
            missing_layers.append(module_name)
            continue

        weight = module.weight
        delta = weight.float() - reference.to(device=weight.device, dtype=torch.float32)
        fisher = fisher.to(device=weight.device, dtype=torch.float32)
        _check_fisher_shape(module_name, fisher, delta)
        layer_reg = torch.sum(fisher * delta * delta)

        total_reg = layer_reg if total_reg is None else total_reg + layer_reg
        used_layers += 1

    if total_reg is None:
        device = next(model.parameters()).device
        total_reg = torch.zeros((), device=device, dtype=torch.float32)

    if normalize_by_layers and used_layers > 0:
        total_reg = total_reg / used_layers

    stats = {
        "used_layers": float(used_layers),
        "missing_layers": float(len(missing_layers)),
    }
    return total_reg, stats


# PORT NOTE [MODIFY] 类保留，剪掉 LoRA 分支。这是 pretrain.py 唯一需要持有的对象。
#
# 要改的地方：
#   - __init__/from_path 去掉 adapter_name；reference_weights 从"可选、None 表示走 LoRA"
#     变成必填，__call__ L610 的 if/else 二选一随之消失。
#   - from_path L497 的 `from onereplay.core.covariance import ...` 改成新项目的包路径。
#   - reg_impl 默认值建议直接改成 "analytic"，autograd 只在等价性校验时用。
#
# ★ normalize_by_layers 是 lambda 换算的关键，单独说：
#   惩罚除以 used_layers，所以 lambda 的物理含义是"每层平均惩罚的权重"，只在给定层数下有定义。
#   Qwen3-1.7B 在 HF 下 197 个目标 Linear，进 LitGPT 因 q/k/v 融合变成 141 个，要保持等效
#   惩罚强度就得乘 141/197 = 0.7157。选 Pythia 则两边都是 4*n_layer、1:1，不用换算——
#   这是选 Pythia 的另一个理由。
#   移植后第一件事是把 used_layers 打出来，和预期层数对上；对不上就是 lookup_covariance
#   在静默丢层。
class ReplayRegularizer:
    """Hold C on device and compute the OneReplay penalty each step.

    A thin wrapper so trainers and future verl integrations can call a single
    object without reloading C. When reference_weights is set the full
    fine-tuning path is used, otherwise the LoRA path. The underlying pure
    functions are unchanged.
    """

    def __init__(
        self,
        covariances: dict[str, torch.Tensor],
        adapter_name: str = "default",
        normalize_by_layers: bool = True,
        reference_weights: dict[str, torch.Tensor] | None = None,
        reg_impl: str = "autograd",
        allow_tf32: bool = True,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.covariances = covariances
        self.adapter_name = adapter_name
        self.normalize_by_layers = normalize_by_layers
        self.reference_weights = reference_weights
        # "autograd" builds the penalty into the loss graph and lets backward
        # derive dR/dW; "analytic" writes 2*lambda*DeltaW C into .grad itself.
        # Both are the same penalty. The flag exists so the equivalence check can
        # flip one argument instead of one git version, the same reason
        # --reg_once_per_update kept its 0 branch.
        self.reg_impl = reg_impl
        self.allow_tf32 = allow_tf32
        self.compute_dtype = compute_dtype
        self._layers: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]] | None = None
        self._missing: list[str] = []

    @classmethod
    def from_path(
        cls,
        path: str,
        device: torch.device | str,
        identity: bool = False,
        normalize_by_layers: bool = True,
        adapter_name: str = "default",
        dtype: torch.dtype = torch.float32,
        reg_impl: str = "autograd",
        allow_tf32: bool = True,
        compute_dtype: torch.dtype = torch.float32,
    ) -> "ReplayRegularizer":
        from onereplay.core.covariance import (
            load_covariance_file,
            move_covariances_to_device,
            to_identity_covariances,
        )

        covariances = load_covariance_file(path)
        if identity:
            covariances = to_identity_covariances(covariances)
        covariances = move_covariances_to_device(covariances, device=device, dtype=dtype)
        return cls(
            covariances=covariances,
            adapter_name=adapter_name,
            normalize_by_layers=normalize_by_layers,
            reg_impl=reg_impl,
            allow_tf32=allow_tf32,
            compute_dtype=compute_dtype,
        )

    def set_reference_weights(self, reference_weights: dict[str, torch.Tensor]) -> None:
        """Switch to the full fine-tuning path using a frozen W0 snapshot."""

        self.reference_weights = reference_weights
        self._layers = None

    @property
    def injects_grad(self) -> bool:
        """Whether the trainer must call accumulate_grad instead of adding to the loss.

        Only the full fine-tuning path is covered. The LoRA path already collapses
        to rank x rank matrices and costs ~11 ms per update, 1.6% of a step, so
        there is nothing there worth a second code path.
        """

        return self.reg_impl == "analytic" and self.reference_weights is not None

    # PORT NOTE [MODIFY] 代码保留，docstring 的约束要改；调用时机映射见 trainers/base.py 批注。
    #
    # docstring 说"必须在累积窗口**开头**、第一个 micro-batch 的 backward 之前调用，因为
    # .grad 是 bf16、只有空 buffer 才保得住 ~1% 量级的项"。bf16-mixed 下 .grad 是 fp32，
    # 这条约束消失，窗口内任何位置注入都等价（DeltaW 在窗口内不变）。
    # 结论：注入位置不再是正确性问题，只剩一个记账问题——见下面 clip_gradients 那条。
    #
    # 在 LitGPT 里的落点：pretrain.py L351 算出 is_accumulating 之后、L352 的
    # `with fabric.no_backward_sync(...)` 之前，条件写
    #     if state["iter_num"] % accum_iters == 1:
    # 注意 iter_num 在 L345 已经先自增，所以窗口开头是余 1、不是余 0，很容易差一。
    # 更稳的写法是别去对齐窗口开头，改成在 L359 `if not is_accumulating:` 分支里、
    # clip_gradients 之前注入一次——语义等价，且不依赖对 iter_num 自增顺序的推理。
    #
    # ★ 真正的新问题：pretrain.py L360 有 fabric.clip_gradients(max_norm=train.max_norm)，
    # 你现在的 trainers/base.py 里**没有对应物**。惩罚梯度会一起参与裁剪，等于 lambda 被
    # 隐式缩放了一个随 step 变化的系数，而且 onereplay 臂和 baseline 臂的裁剪系数还不一样。
    # 三个选项，必须显式选一个并写进论文：
    #   (a) 注入放在 clip 之后、optimizer.step() 之前 —— 惩罚不被裁剪，语义最干净；
    #   (b) 注入放在 clip 之前 —— 与 LitGPT 默认行为一致，但要记录裁剪触发频率；
    #   (c) max_norm 设成极大值关掉裁剪 —— 所有臂统一，但偏离 LitGPT 默认配方。
    # 我倾向 (a)：成本对比的前提是各臂只有惩罚项这一个差异，(b) 会引入第二个差异。
    def accumulate_grad(self, model: nn.Module, replay_lambda: float) -> tuple[float, dict]:
        """Add lambda * dR/dW into .grad and return the reported R.

        Call this once per optimizer step, at the *start* of the accumulation
        window, before the first micro-batch's backward. Anywhere in the window
        gives the same DeltaW, but .grad is bf16 under full fine-tuning and only an
        empty buffer preserves a term running at ~1% of the task gradient. That is
        also where the autograd path adds it, which keeps the two comparable.

        The returned value is normalized the same way the autograd path normalizes
        it, so the logged replay_reg stays comparable across implementations.
        """

        if self._layers is None:
            self._layers, self._missing = resolve_full_layers(
                model, self.covariances, self.reference_weights or {}
            )
            print(
                f"analytic regularizer resolved {len(self._layers)} layers; "
                f"no penalty matrix for {len(self._missing)} layers; "
                f"tf32={int(self.allow_tf32)} dtype={self.compute_dtype}"
            )

        used = len(self._layers)
        divisor = used if (self.normalize_by_layers and used > 0) else 1
        total = full_covariance_grad_(
            self._layers,
            scale=replay_lambda / divisor,
            allow_tf32=self.allow_tf32,
            compute_dtype=self.compute_dtype,
        )
        stats = {"used_layers": float(used), "missing_layers": float(len(self._missing))}
        return total / divisor, stats

    def layer_keys(self) -> dict[str, torch.Tensor]:
        """Per-layer matrices, used as the definition of the penalty's coverage.

        snapshot_reference_weights only needs a dict keyed by module name to
        decide which layers to freeze W0 for, so both regularizers expose their
        matrices under one name and the caller stays agnostic.
        """

        return self.covariances

    # PORT NOTE [MODIFY] 单卡口径正确；FSDP 下这两个 *_memory_bytes 的口径要重新定义。
    #
    # 多卡时"C 占多少显存"取决于上面 resolve_full_layers 那条批注里选的方案：每卡全量 C 时
    # 这个数是每卡真实占用；C 按分片切时它会高估。成本表要报的是**每卡峰值**，因为那才是
    # "能不能塞进一张卡"的决定量，别报全局求和后的数。
    #
    # reference_memory_bytes 的 docstring 那句"这就是快照的全部开销，成立的前提是
    # resolve_full_layers 缓存的是原张量而非 dtype 转换后的副本"要保留——它正是上次
    # 显存谜团的成因说明，是这个方法可信的唯一依据。
    def memory_bytes(self) -> int:
        """Bytes the covariance matrices occupy wherever they were moved.

        This is OneReplay's main fixed memory overhead over vanilla LoRA: one
        d_in x d_in matrix per target layer, resident for the whole run.
        """

        return sum(
            covariance.numel() * covariance.element_size()
            for covariance in self.covariances.values()
        )

    def reference_memory_bytes(self) -> int:
        """Bytes the W0 snapshot occupies; zero on the LoRA path.

        Full fine-tuning needs DeltaW = W - W0, so the initial weights of every
        regularized layer stay resident. LoRA reads DeltaW straight from B and
        A and pays nothing here.

        This is the entire snapshot cost on both reg_impl paths, which is true
        only because resolve_full_layers caches these very tensors instead of
        dtype-converted copies of them. A cast there would be invisible to this
        method, and the cost tables would carry it as temporary memory.
        """

        if not self.reference_weights:
            return 0
        return sum(
            reference.numel() * reference.element_size()
            for reference in self.reference_weights.values()
        )

    def __call__(self, model: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
        if self.reference_weights is not None:
            return full_covariance_regularizer(
                model,
                self.covariances,
                self.reference_weights,
                normalize_by_layers=self.normalize_by_layers,
            )
        return lora_covariance_regularizer(
            model,
            self.covariances,
            adapter_name=self.adapter_name,
            normalize_by_layers=self.normalize_by_layers,
        )


# PORT NOTE [DELETE] EWC 基线整类删除（原文件 L625-716），理由同 _check_fisher_shape 批注。
class EWCRegularizer:
    """Hold F on device and compute the EWC penalty each step.

    The regularization baseline OneReplay is measured against. Same interface as
    ReplayRegularizer, so the trainer, the metrics schema and the lambda
    plumbing are shared and the only thing that changes between the two runs is
    which matrix weights DeltaW.

    F is estimated once on the frozen pre-trained weights and never updated.
    That is not a simplification of the method: EWC accumulates Fisher matrices
    across *task boundaries*, and this setting has exactly one, from the
    instruction-tuned base model to the new task. Keeping F fixed also keeps the
    comparison against C one-dimensional, since C is likewise estimated once and
    frozen; an F that adapted during training would confound "Fisher versus
    input covariance" with "adaptive versus static".
    """

    def __init__(
        self,
        fishers: dict[str, torch.Tensor],
        adapter_name: str = "default",
        normalize_by_layers: bool = True,
        reference_weights: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.fishers = fishers
        self.adapter_name = adapter_name
        self.normalize_by_layers = normalize_by_layers
        self.reference_weights = reference_weights

    @classmethod
    def from_path(
        cls,
        path: str,
        device: torch.device | str,
        normalize_by_layers: bool = True,
        adapter_name: str = "default",
        dtype: torch.dtype = torch.float32,
    ) -> "EWCRegularizer":
        from onereplay.core.fisher import load_fisher_file, move_fishers_to_device

        fishers = load_fisher_file(path)
        fishers = move_fishers_to_device(fishers, device=device, dtype=dtype)
        return cls(
            fishers=fishers,
            adapter_name=adapter_name,
            normalize_by_layers=normalize_by_layers,
        )

    def set_reference_weights(self, reference_weights: dict[str, torch.Tensor]) -> None:
        """Switch to the full fine-tuning path using a frozen W0 snapshot."""

        self.reference_weights = reference_weights

    def layer_keys(self) -> dict[str, torch.Tensor]:
        return self.fishers

    def memory_bytes(self) -> int:
        """Bytes the Fisher matrices occupy wherever they were moved.

        One d_out x d_in matrix per target layer. For attention projections
        under grouped-query attention this is smaller than the covariance path's
        d_in x d_in, so EWC's resident cost is not what separates the two
        methods; the collection stage is, since C needs forward passes only
        while F needs a backward pass per example.
        """

        return sum(fisher.numel() * fisher.element_size() for fisher in self.fishers.values())

    def reference_memory_bytes(self) -> int:
        """Bytes the W0 snapshot occupies; zero on the LoRA path."""

        if not self.reference_weights:
            return 0
        return sum(
            reference.numel() * reference.element_size()
            for reference in self.reference_weights.values()
        )

    def __call__(self, model: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
        if self.reference_weights is not None:
            return full_fisher_regularizer(
                model,
                self.fishers,
                self.reference_weights,
                normalize_by_layers=self.normalize_by_layers,
            )
        return lora_fisher_regularizer(
            model,
            self.fishers,
            adapter_name=self.adapter_name,
            normalize_by_layers=self.normalize_by_layers,
        )
