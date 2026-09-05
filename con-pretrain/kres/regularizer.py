"""惩罚项 R = (λ/N) Σ_l tr(ΔW_l C_l ΔW_lᵀ) 及其 analytic 梯度注入。

ΔW = W - W₀，W₀ 是冻结的快照。梯度有闭式解：

    dR/dΔW = ΔW (C + Cᵀ) = 2 ΔW C          （C 对称）

而 ΔW C 正是算 R 时已经形成的那个乘积。让 autograd 去重新发现它要多一次同尺寸矩阵乘，
而且会把每层的 fp32 ΔW 挂在图上直到 backward 返回。所以默认走 analytic：直接写 .grad，
算术量减半、每层的临时张量立刻可回收，峰值是一层而不是全部 110 层。
autograd 那条路径保留下来只为等价性校验——两条实现算同一个量，换框架后必须重验一次。

四条不变量，每条对应一个「不报错但结论是错的」失效：

1. **名字前缀处理只有一处。** `pretrain.py` L213 的 `torch.compile` 会给
   `named_modules()` 的名字加 `_orig_mod.`，L214 的 `fabric.setup` 再加
   `_forward_module.`，而 C 的键是采集时的干净 LitGPT 名。三者（C 的键、W₀ 快照的键、
   解析时的匹配）必须共用同一套规范化，见 `canonical_name`。

2. **解析不全就 raise，绝不 continue。** 原实现查不到只记进 missing_layers 然后
   continue，于是惩罚恒为 0、训练全速跑完、loss 曲线干净，结果和 baseline 完全一致。
   而且不允许后缀兜底匹配：`proj` 这个末段名在本模型里出现 44 次
   （attn.proj 22 + mlp.proj 22），后缀匹配必然撞出多个候选后静默返回 None。

3. **W₀ 快照必须在权重就位之后拍。** `pretrain.py` 的装配顺序是
   `GPT(config)` → `initialize_weights`(L203) → ... → `load_raw`(L223)，**权重最后才到位**。
   在 L223 之前快照就会把随机初始化当成 W₀，惩罚变成「把模型拉回随机初始化」，方向完全
   反掉，而"CPT 加了正则效果变差"是个看起来完全合理的实验结论。
   `verify_snapshot` 拿 lit_model.pth 逐层比对，这是唯一能拦住它的手段。

4. **不做任何 dtype cast 缓存。** `resolve_layers` 只解析 device、绝不解析 dtype。
   在那里 cast 到 fp32 看着免费，但转换后的张量会被缓存一整个 run、和它来源的快照并存，
   于是 W₀ 常驻两份——这正是上次那个显存谜团（fp32 6.41 GiB 叠在 bf16 3.20 GiB 上，
   让 analytic 路径的峰值反而比它要替代的 autograd 路径高 2.06 GiB）的成因。
   不变量：大张量的 fp32 副本只能有一份。

λ 的语义：惩罚除以 N（实际解析到的层数），所以 λ 是「每层平均惩罚的权重」，只在给定
层数下有定义。本基座 N=110（22 block × 5），SFT 侧的 LoRA 只覆盖 q/v 两个，λ 要换算。
除以 N 只修正总体尺度，**完全不触及层间的相对权重**——inspect_cov 报的那个 mean_diag
层间跨度对此毫无反应，那是另一个问题。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

WRAPPER_SEGMENTS = frozenset({"_orig_mod", "_forward_module", "module"})
"""包装器插进模块名开头的那些段。

`_orig_mod` 来自 torch.compile（pretrain.py L213），`_forward_module` 来自
fabric.setup（L214），`module` 来自 DDP。它们会嵌套，比如
`_forward_module._orig_mod.transformer.h.0.attn.qkv`，所以要循环剥。
"""


def canonical_name(module_name: str) -> str:
    """把包装后的模块名还原成采集时的干净 LitGPT 名。

    这是整个项目里**唯一**一处前缀处理。C 的键、W₀ 快照的键、解析时的匹配全都过这个
    函数，三者才能自洽。原实现靠 `lookup_covariance` 的后缀匹配兜底，那在 LitGPT 命名
    下是不能用的——见模块 docstring 第 2 条。
    """
    parts = module_name.split(".")
    start = 0
    while start < len(parts) and parts[start] in WRAPPER_SEGMENTS:
        start += 1
    return ".".join(parts[start:])


def iter_linears(model: nn.Module):
    """遍历所有 nn.Linear，产出 (规范名, 模块)。"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield canonical_name(name), module


def snapshot_reference_weights(
    model: nn.Module,
    target_names: list[str],
) -> dict[str, torch.Tensor]:
    """给每个目标层冻结 W₀。

    **调用时机是这个函数的正确性前提**，见模块 docstring 第 3 条：必须在
    `pretrain.py` L223-224 的 `fabric.load_raw` 之后。拍完立刻用 `verify_snapshot`
    对着 checkpoint 验一遍。

    按 data_ptr 去重：`train.tie_embeddings` 让 `transformer.wte.weight` 和
    `lm_head.weight` 共享同一份 storage（L206 直接赋值），两边都算就等于把那一层的 λ
    悄悄翻倍。wte 是 nn.Embedding 本来就不匹配 nn.Linear，但这个检查还能挡住
    checkpoint 里任何其他形式的别名。

    dtype 保持权重的原样。bf16-mixed 下 `module.weight` 是 fp32，所以快照也是 fp32
    （0.4b 基座上 1.41 GB）。这是换精度的真实代价，要写进成本表。
    """
    wanted = set(target_names)
    references: dict[str, torch.Tensor] = {}
    seen_storage: set[int] = set()

    for name, module in iter_linears(model):
        if name not in wanted:
            continue
        pointer = module.weight.data_ptr()
        if pointer in seen_storage:
            continue
        seen_storage.add(pointer)
        references[name] = module.weight.detach().clone()

    missing = wanted - set(references)
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(wanted)} 个目标层在模型里找不到（前几个：{sorted(missing)[:5]}）。"
            "模块树和采集 C 时的不一致，或者前缀规范化没覆盖当前的包装方式"
        )
    return references


def verify_snapshot(
    references: dict[str, torch.Tensor],
    checkpoint_path: str | Path,
    rtol: float = 0.0,
) -> None:
    """拿 lit_model.pth 逐层验证 W₀ 快照，拍早了就 raise。

    这是防「把随机初始化当 W₀」的唯一手段。那个 bug 不报错、不 NaN，只是让惩罚方向
    完全反掉，而「加了正则效果变差」是个完全合理的结论，从结果上根本看不出来。

    默认 rtol=0，即要求逐比特相等：快照是从刚 load_raw 完的权重上 clone 的，中间没有
    任何算术，本来就该逐比特相同。放宽它只会让这条断言失去意义。

    用 mmap 读 checkpoint，避免为了做一次比对再把 1.5 GB 读进内存。
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"验证 W₀ 需要 checkpoint，但 {path} 不存在")

    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    problems: list[str] = []
    for name, reference in references.items():
        key = f"{name}.weight"
        if key not in state:
            problems.append(f"{key} 不在 checkpoint 里")
            continue
        expected = state[key]
        got = reference.detach().cpu()
        if got.shape != expected.shape:
            problems.append(f"{key} 形状 {tuple(got.shape)} != {tuple(expected.shape)}")
            continue
        expected = expected.to(got.dtype)
        if rtol == 0.0:
            if not torch.equal(got, expected):
                diff = float((got.float() - expected.float()).abs().max())
                problems.append(f"{key} 与 checkpoint 不逐比特相等（最大差 {diff:.3e}）")
        elif not torch.allclose(got, expected, rtol=rtol, atol=0.0):
            diff = float((got.float() - expected.float()).abs().max())
            problems.append(f"{key} 超出 rtol={rtol}（最大差 {diff:.3e}）")

        if len(problems) >= 5:
            break

    if problems:
        raise RuntimeError(
            "W₀ 快照与 checkpoint 不符，拒绝继续。最可能的原因是快照拍在了 "
            "pretrain.py L223 的 fabric.load_raw **之前**，那样 W₀ 是 initialize_weights "
            "产生的随机权重，惩罚会把模型往随机初始化拉、方向完全反掉：\n  "
            + "\n  ".join(problems)
        )


def load_reference_weights(
    checkpoint_path: str | Path,
    target_names: list[str],
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """直接从基座 checkpoint 读 W₀，不看当前模型里是什么。

    续跑必须走这条路。`--resume` 与 `--initial_checkpoint_dir` 是互斥的
    （pretrain.py 的 validate_args），所以续跑时 `load_raw` 根本不执行，权重是
    `fabric.load(resume, state)` 恢复的**中途状态**。此时从模型上快照就会把已经漂移
    的权重当成 W₀，惩罚从「留在基座附近」变成「留在当下别动」——λ 越大越像是学不动，
    而不是像遗忘更少。6 小时 walltime 下单臂要续跑 5 次，这个错误会在每一段重新发生。

    `verify_snapshot` 能拦住这种情况（它对着基座文件逐比特比），但拦住之后续跑就没法
    继续了。所以正确的做法不是"拍完再验"，而是首段和各续跑段都从同一个文件读 W₀。

    直接搬到目标 device 并定好 dtype，是为了让 `resolve_layers` 里的 `.to()` 成为
    空操作。否则 W₀ 会同时存在 CPU 和 GPU 两份——这与模块 docstring 第 4 条是同一条
    不变量，只是跨的是设备而不是精度。
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"读 W₀ 需要基座 checkpoint，但 {path} 不存在")

    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    references: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for name in target_names:
        key = f"{name}.weight"
        if key not in state:
            missing.append(key)
            continue
        tensor = state[key]
        references[name] = tensor.to(device=device, dtype=dtype or tensor.dtype).clone()

    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(target_names)} 个目标层在 {path.name} 里找不到"
            f"（前几个：{missing[:5]}）。C 和这个 checkpoint 不是同一个模型"
        )
    return references


def resolve_layers(
    model: nn.Module,
    covariances: dict[str, torch.Tensor],
    reference_weights: dict[str, torch.Tensor],
    expected_layers: int | None = None,
) -> list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]]:
    """把每个被正则的 Linear 和它的 C、W₀ 配好，只做一次。

    配对在一个 run 里不会变（C 冻结、W₀ 冻结、模块树建好后固定），所以解析一次复用。

    **只解析 device，绝不解析 dtype**，理由见模块 docstring 第 4 条。
    `full_covariance_grad_` 逐层现场 cast——两种做法数值上完全等价，但那样活着的临时
    张量是一层而不是 110 层。

    `expected_layers` 不为 None 时做硬断言。这是整套实现里唯一不能省的断言：名字空间
    对不上时每层都解析失败，惩罚恒为 0，整个 run 等于 baseline。
    """
    layers: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]] = []
    missing_cov: list[str] = []

    for name, module in iter_linears(model):
        reference = reference_weights.get(name)
        if reference is None:
            continue
        covariance = covariances.get(name)  # 全名精确匹配，不做后缀兜底
        if covariance is None:
            missing_cov.append(name)
            continue
        weight = module.weight
        layers.append(
            (
                name,
                weight,
                covariance.to(device=weight.device),
                reference.to(device=weight.device),
            )
        )

    problems: list[str] = []
    if missing_cov:
        problems.append(
            f"{len(missing_cov)} 个层有 W₀ 但没有 C（前几个：{missing_cov[:5]}）"
        )
    unresolved = set(reference_weights) - {n for n, *_ in layers} - set(missing_cov)
    if unresolved:
        problems.append(
            f"{len(unresolved)} 个层有 W₀ 但在模型里找不到（前几个：{sorted(unresolved)[:5]}）"
        )
    if expected_layers is not None and len(layers) != expected_layers:
        problems.append(f"解析到 {len(layers)} 层，预期 {expected_layers} 层")

    if problems:
        raise RuntimeError(
            "正则器解析失败，拒绝继续——惩罚会静默变成 0，整个 run 等于 baseline "
            "而 loss 曲线看不出任何异常：\n  " + "\n  ".join(problems)
        )
    if not layers:
        raise RuntimeError("一层都没解析到，惩罚恒为 0")

    return layers


def full_covariance_regularizer(
    layers: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]],
    normalize_by_layers: bool = True,
) -> torch.Tensor:
    """autograd 路径：返回可微的 R，交给 backward 去求 dR/dW。

    保留它不是为了性能（analytic 更快），而是为了等价性校验——两条实现算同一个量，
    换框架后要重新验一次 analytic == autograd，这是最直接的回归测试。

        tr(ΔW C ΔWᵀ) = sum((ΔW @ C) * ΔW)

    读的是 module.weight。bf16-mixed 下 Fabric 保留 fp32 主权重，所以 .float() 是
    no-op、ΔW 是真 fp32——之前担心的「bf16 只有 8 位尾数、小 ΔW 被吃掉」在这个精度
    设置下本来就不成立。
    """
    total: torch.Tensor | None = None
    for _, weight, covariance, reference in layers:
        delta = weight.float() - reference.to(device=weight.device, dtype=torch.float32)
        layer_reg = torch.sum((delta @ covariance.to(torch.float32)) * delta)
        total = layer_reg if total is None else total + layer_reg

    assert total is not None  # resolve_layers 已经保证 layers 非空
    if normalize_by_layers:
        total = total / len(layers)
    return total


@torch.no_grad()
def full_covariance_grad_(
    layers: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]],
    scale: float,
    allow_tf32: bool = True,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[float, float]:
    """把 scale · dR/dW 直接加进 .grad。返回 (未归一化的 R, 注入梯度的范数)。

    返回注入梯度的范数是为了回答「惩罚相对 LM 梯度有多大」这个问题——它决定了
    `fabric.clip_gradients` 到底有没有实际影响，也是 λ 标定的原始依据。

    这个函数在 forward/backward 之外、用 no_grad 直接读权重写 .grad，从来不进
    autocast 的作用域，所以和 bf16-mixed 不冲突。

    C 存的是 fp32 且不被修改；allow_tf32 只让矩阵乘在 tensor core 内部把输入舍到
    11 位尾数，那是对 C 的约 5e-4 相对扰动，换来 H100 上 7 倍吞吐，比同一个 replay
    语料换两种采样策略就能产生的 5e-3 惩罚比差异还低一个数量级。
    """
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    total = 0.0
    grad_sq = 0.0
    try:
        for _, weight, covariance, reference in layers:
            delta = weight.to(compute_dtype) - reference.to(compute_dtype)
            delta_c = delta @ covariance.to(compute_dtype)
            total += float((delta_c * delta).sum())
            if scale != 0.0:
                if weight.grad is None:
                    weight.grad = torch.zeros_like(weight)
                update = delta_c.to(weight.grad.dtype) * (2.0 * scale)
                weight.grad.add_(update)
                grad_sq += float(update.float().pow(2).sum())
                del update
            del delta, delta_c
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return total, grad_sq**0.5


def global_grad_norm(model: nn.Module) -> float:
    """所有参数梯度的全局 L2 范数。

    口径要和 `fabric.clip_gradients` 一致：它算的是**全部参数**的全局范数，不只是被
    正则的那 110 层。所以判断裁剪会不会触发必须用这个数，而不是惩罚梯度自己的范数。
    """
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum())
    return total**0.5


class ReplayRegularizer:
    """持有 C、每个 optimizer step 算一次惩罚。这是 pretrain.py 唯一需要拿着的对象。"""

    def __init__(
        self,
        covariances: dict[str, torch.Tensor],
        reference_weights: dict[str, torch.Tensor],
        expected_layers: int | None = None,
        normalize_by_layers: bool = True,
        reg_impl: str = "analytic",
        allow_tf32: bool = True,
        compute_dtype: torch.dtype = torch.float32,
        log_grad_norms: bool = True,
    ) -> None:
        if reg_impl not in ("analytic", "autograd"):
            raise ValueError(f"reg_impl 只能是 analytic 或 autograd，收到 {reg_impl!r}")
        self.covariances = covariances
        self.reference_weights = reference_weights
        self.expected_layers = expected_layers
        self.normalize_by_layers = normalize_by_layers
        # 两条路径算的是同一个惩罚。留 autograd 是为了等价性校验能只翻一个参数，
        # 而不是翻一个 git 版本
        self.reg_impl = reg_impl
        self.allow_tf32 = allow_tf32
        self.compute_dtype = compute_dtype
        self.log_grad_norms = log_grad_norms
        self._layers = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        model: nn.Module,
        target_names: list[str],
        checkpoint_path: str | Path,
        device: torch.device | str,
        identity: bool = False,
        dtype: torch.dtype = torch.float32,
        resuming: bool = False,
        module_shapes: dict[str, int] | None = None,
        **kwargs,
    ) -> "ReplayRegularizer":
        """加载 C、取 W₀、验一致性，一次做完。

        **首段必须在 fabric.load_raw 之后调用。** `verify_snapshot` 会拦住拍早了的
        情况，但那是最后一道防线，不是可以依赖的日常路径。

        `resuming=True` 时改从 `checkpoint_path` 读 W₀，而不是从模型上快照。续跑时
        模型里是漂移过的中途权重，从它快照就等于把 W₀ 挪到当下，惩罚方向随之改变；
        详见 `load_reference_weights`。首段仍然走快照 + 逐比特验证，因为那条路径能
        额外证明「模型确实收到了基座权重」，是比读文件更强的检查。
        """
        from kres.covariance import (
            assert_covers,
            load_covariance_file,
            move_covariances_to_device,
            to_identity_covariances,
        )

        covariances = load_covariance_file(path)
        # module_shapes 一起传：C 的维度不符会在第一个 optimizer step 才形状报错，
        # 提前拦掉，顺带能抓出"C 是在别的模型上采的"
        assert_covers(covariances, target_names, module_shapes)
        if identity:
            covariances = to_identity_covariances(covariances)
        covariances = move_covariances_to_device(covariances, device=device, dtype=dtype)

        if resuming:
            probe = next(iter(model.parameters()))
            references = load_reference_weights(
                checkpoint_path, target_names, device=probe.device, dtype=probe.dtype
            )
        else:
            references = snapshot_reference_weights(model, target_names)
            verify_snapshot(references, checkpoint_path)

        return cls(
            covariances=covariances,
            reference_weights=references,
            expected_layers=len(target_names),
            **kwargs,
        )

    @property
    def injects_grad(self) -> bool:
        """True 时训练循环要调 accumulate_grad，而不是把 R 加到 loss 上。"""
        return self.reg_impl == "analytic"

    def layers(self, model: nn.Module):
        """懒解析并缓存配对。第一次调用时权重早已就位，所以时机是安全的。"""
        if self._layers is None:
            self._layers = resolve_layers(
                model, self.covariances, self.reference_weights, self.expected_layers
            )
            print(f"kres 正则器解析到 {len(self._layers)} 层"
                  f"（预期 {self.expected_layers}）；reg_impl={self.reg_impl} "
                  f"tf32={int(self.allow_tf32)} dtype={self.compute_dtype}")
        return self._layers

    def accumulate_grad(self, model: nn.Module, replay_lambda: float) -> tuple[float, dict]:
        """把 λ·dR/dW 加进 .grad，返回 (归一化后的 R, 统计量)。

        **每个 optimizer step 只调一次。** R 只依赖 W 和 C，而 W 在一个梯度累积窗口内
        根本不变（optimizer.step 只在窗口末尾执行），所以窗口内它是个常量。每个
        micro-batch 都调的话，累积下来就是 `gradient_accumulation_iters` 倍的 λ，而这个
        错误不报任何异常。这个倍数由 global_batch_size 决定，本项目待定在 125~128 之间
        （global_batch_size 500 或 512、micro 4、单卡），成本表和 λ 都依赖它，必须钉死。

        落点在 pretrain.py L359 的 `if not is_accumulating:` 分支内、L360 的
        clip_gradients 之前。裁剪是对整个梯度向量做一次标量缩放、方向不变，所以在它
        之前注入能精确保住 LM 与惩罚的比例，也就是 λ 的语义；反过来先裁再注入，比例会
        被 1/c₀ 放大，λ 变成逐步波动的量。统计量里带上三个范数，用来事后确认裁剪到底
        有没有触发。
        """
        layers = self.layers(model)
        divisor = len(layers) if self.normalize_by_layers else 1

        lm_norm = global_grad_norm(model) if self.log_grad_norms else float("nan")
        total, reg_grad_norm = full_covariance_grad_(
            layers,
            scale=replay_lambda / divisor,
            allow_tf32=self.allow_tf32,
            compute_dtype=self.compute_dtype,
        )
        stats = {
            "used_layers": float(len(layers)),
            "lm_grad_norm": lm_norm,
            "reg_grad_norm": reg_grad_norm,
            "total_grad_norm": global_grad_norm(model) if self.log_grad_norms else float("nan"),
        }
        return total / divisor, stats

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """autograd 路径：返回可微的 R，调用方自己乘 λ 加到 loss 上。"""
        return full_covariance_regularizer(
            self.layers(model), normalize_by_layers=self.normalize_by_layers
        )

    def memory_bytes(self) -> int:
        """C 占的字节数。这是本方法相对 baseline 最大的固定显存开销。"""
        return sum(c.numel() * c.element_size() for c in self.covariances.values())

    def reference_memory_bytes(self) -> int:
        """W₀ 快照占的字节数。

        这就是快照的**全部**开销，成立的前提是 `resolve_layers` 缓存的是原张量而不是
        dtype 转换后的副本。那里做一次 cast 对这个方法是不可见的，成本表就会把它记成
        来源不明的临时显存——上次那个显存谜团就是这么来的。
        """
        return sum(r.numel() * r.element_size() for r in self.reference_weights.values())
