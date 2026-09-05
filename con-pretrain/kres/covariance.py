"""C = E[x xᵀ] 的目标层发现、采集、存取。

x 是目标 Linear 的**输入** hidden state，也就是惩罚 tr(ΔW C ΔWᵀ) 里的那个 x。
C 的形状是 d_in × d_in，只由 in_features 决定，与 out_features 无关。

这个文件守着四条不变量，每一条对应一个「不报错但结论是错的」失效：

1. **C 的键是 LitGPT 的模块全名**，因为采集就在 LitGPT 的 GPT 上做。HF 名
   （`model.layers.0.self_attn.q_proj`）与 LitGPT 名（`transformer.h.0.attn.qkv`）
   零字符串重合，互为后缀的判定是 False，所以不存在「靠后缀兜底碰巧命中」这条路。

2. **查不到必须硬失败。** 原实现的 `lookup_covariance` 查不到只返回 (None, None)，
   调用方把层名记进 missing_layers 就 continue，于是惩罚恒为 0、训练全速跑完、
   loss 曲线干净，结果和不加正则的 baseline 完全一致。见 `assert_covers`。
   顺带：`proj` 这个末段名字在本模型里出现 44 次（attn.proj 22 + mlp.proj 22），
   后缀匹配在这套命名下必然撞出多个候选，所以只允许全名精确匹配。

3. **不接受 torch.compile 过的模型。** 编译会把模块名加上 `_orig_mod.` 前缀，而且
   编译后的子模块上挂 forward hook 正是会 graph break 或被静默跳过的情形。
   采集是独立脚本，保持 eager，见 `find_target_linears` 里的前缀检查。

4. **累加一律 fp32。** XᵀX 要在 1e9 量级的 token 上累加，bf16 累加会严重掉精度。

与 onereplay 原实现的三处有意偏离（都记在这里，方便和那边对账）：

- 用 `register_forward_pre_hook` 而不是 `register_forward_hook`。原实现要 `output`
  是为了 base_output_norm 那条归一化路径；本项目定的是 cov_normalization=none，
  即 C = E[x xᵀ]，不需要 output，pre-hook 也就不必持有它。
- 删掉了 `attention_holder` 参数，而不是留着传 None。定长打包语料没有 padding，
  每个位置都是真 token，本来就该统计全部位置。但原实现里那个
  `attention_mask.shape[:2] == (batch, seq_len)` 的形状守卫在定长打包下**每个 batch
  形状都一样**，残留或被错误填入的 mask 能通过守卫、静默屏蔽掉一部分 token，C 就在
  一个位置子集上被采集，不报错也不告警。参数不存在，这个失效就不可能发生。
- 累加在 GPU 上做，结束时一次性搬回 CPU。原实现每个 hook 调用都 `.cpu()`，那是
  每 batch 110 次、总计 1.67 GB 的设备间拷贝，在 3 万个 batch 的规模下不可接受。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

BLOCK_PREFIX = "transformer.h."
"""LitGPT 的 transformer block 前缀。目标层只从 block 内部取。"""

COMPILE_PREFIX = "_orig_mod."


def find_target_linears(
    model: nn.Module,
    n_layer: int,
    include_lm_head: bool = False,
) -> list[str]:
    """列出目标 Linear 的模块全名，并对模块树做结构性校验。

    不硬编码「110 个」这种数字：层数、每 block 几个 Linear 都从模块树上点出来，再
    检查它们自洽（每个 block 的后缀集合一致、block 数等于 n_layer）。换基座或改
    mlp_class 时这里会自己报错，而不是悄悄给出一个错的目标集。

    `include_lm_head` 默认关闭。lm_head 与 transformer.wte 共享同一份 storage
    （tie_embeddings 在 pretrain.py L206 直接赋值），惩罚它等于同时惩罚输入嵌入表，
    而它的 C 是末层 hidden state 的二阶矩——用输出侧的统计量约束输入侧的嵌入矩阵，
    概念上是混乱的。留这个开关是为了后面把「保护输出头」当成一条消融来跑。
    """
    linears = [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]

    compiled = [n for n in linears if n.startswith(COMPILE_PREFIX)]
    if compiled:
        raise RuntimeError(
            f"模型被 torch.compile 过（{len(compiled)} 个模块带 {COMPILE_PREFIX!r} 前缀）。"
            "采集必须在 eager 模型上做：编译后的子模块上挂 forward hook 会 graph break "
            "或被静默跳过，而且模块名会多一层前缀、和训练时的键对不上。"
        )

    return target_names_from_linears(linears, n_layer, include_lm_head)


def target_names_from_linears(
    linear_names: list[str],
    n_layer: int,
    include_lm_head: bool = False,
) -> list[str]:
    """`find_target_linears` 的纯名字版本，供训练侧调用。

    拆出来是因为训练时的模型必然是 `torch.compile` + `fabric.setup` 包装过的，
    上面那个函数会因为 compile 前缀直接 raise——那条检查是为采集准备的（编译后挂
    hook 会失效），对只读权重的训练侧不适用。调用方先用
    `regularizer.canonical_name` 把名字规范化，再交给这里做同样的结构性校验。

    这样「目标层清单」在采集侧和训练侧走的是同一套推导，两边不会各算一份。
    """
    per_block: dict[int, list[str]] = defaultdict(list)
    outside: list[str] = []
    for name in linear_names:
        if not name.startswith(BLOCK_PREFIX):
            outside.append(name)
            continue
        index, suffix = name[len(BLOCK_PREFIX) :].split(".", 1)
        per_block[int(index)].append(suffix)

    if not per_block:
        raise RuntimeError(f"在 {BLOCK_PREFIX!r} 下没找到任何 nn.Linear，模块树和预期不符")
    if len(per_block) != n_layer:
        raise RuntimeError(f"block 数 {len(per_block)} != config.n_layer {n_layer}")

    suffix_sets = {tuple(sorted(v)) for v in per_block.values()}
    if len(suffix_sets) != 1:
        raise RuntimeError(f"各 block 的 Linear 后缀集合不一致：{sorted(suffix_sets)}")

    suffixes = sorted(per_block[min(per_block)])
    names = [f"{BLOCK_PREFIX}{i}.{s}" for i in range(n_layer) for s in suffixes]

    if include_lm_head:
        if "lm_head" not in outside:
            raise RuntimeError(f"要求纳入 lm_head，但 block 外的 Linear 是 {outside}")
        names.append("lm_head")

    return names


def assert_covers(
    covariances: dict[str, torch.Tensor],
    target_names: list[str],
    module_shapes: dict[str, int] | None = None,
) -> None:
    """断言 C 完整覆盖目标层，且维度对得上。

    这是整套实现里唯一不能省的断言。名字空间对不上时每一层都查不到，惩罚恒为 0，
    整个 run 等于 baseline，而 loss 曲线看起来完全正常——没有这条断言，这种失效
    只能靠「和 baseline 结果一模一样」这种事后观察发现，而那时候机时已经烧掉了。

    维度也一起查：C 是 d_in × d_in，如果哪一层的 C 是在别的模型上采的、维度不符，
    后面 ΔW @ C 会直接形状报错，但那要等到第一个 optimizer step；这里提前拦掉。
    """
    missing = [n for n in target_names if n not in covariances]
    extra = [n for n in covariances if n not in target_names]

    problems: list[str] = []
    if missing:
        shown = missing[:5]
        problems.append(
            f"{len(missing)}/{len(target_names)} 个目标层没有 C（前几个：{shown}）"
        )
    if extra:
        problems.append(f"C 里有 {len(extra)} 个不在目标层清单里的键（前几个：{extra[:5]}）")

    if module_shapes:
        for name in target_names:
            if name not in covariances:
                continue
            cov = covariances[name]
            d_in = module_shapes.get(name)
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                problems.append(f"{name} 的 C 形状 {tuple(cov.shape)} 不是方阵")
            elif d_in is not None and cov.shape[0] != d_in:
                problems.append(f"{name} 的 C 是 {cov.shape[0]}×{cov.shape[0]}，但 d_in={d_in}")

    if problems:
        raise RuntimeError(
            "C 与目标层不匹配，拒绝继续——惩罚会静默变成 0，整个 run 等于 baseline：\n  "
            + "\n  ".join(problems)
        )


def module_in_features(model: nn.Module, target_names: list[str]) -> dict[str, int]:
    """取每个目标层的 in_features，用于 assert_covers 的维度校验。

    按规范名建表，所以采集侧（裸模型）和训练侧（torch.compile + fabric.setup 包装过）
    都能用同一份 `target_names`。前缀处理统一走 `regularizer.canonical_name`，这是
    整个项目里唯一一处——名字规范化有第二份实现，就迟早会有两份不一致的键集。
    """
    from kres.regularizer import canonical_name

    table = {canonical_name(n): m for n, m in model.named_modules()}
    out: dict[str, int] = {}
    for name in target_names:
        mod = table.get(name)
        if not isinstance(mod, nn.Linear):
            raise RuntimeError(f"{name} 不是 nn.Linear（实际 {type(mod).__name__}）")
        out[name] = mod.in_features
    return out


def make_covariance_hook(
    module_name: str,
    cov_sums: dict[str, torch.Tensor],
    counts: dict[str, int],
) -> Callable:
    """造一个 forward pre-hook，累加单层的 XᵀX 和 token 计数。

    定长打包语料没有 padding，每个位置都是真 token，所以统计全部位置。这里不接受
    任何 mask，理由见模块 docstring。
    """

    def hook(_module: nn.Module, inputs: tuple) -> None:
        hidden = inputs[0].detach()
        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        if flat.numel() == 0:
            return

        # addmm_ 原地累加，避免每 batch 为 d_in×d_in 的中间结果再分配一次显存
        if module_name not in cov_sums:
            cov_sums[module_name] = flat.T @ flat
            counts[module_name] = int(flat.shape[0])
        else:
            cov_sums[module_name].addmm_(flat.T, flat)
            counts[module_name] += int(flat.shape[0])

    return hook


def register_covariance_hooks(
    model: nn.Module,
    target_names: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, int], list]:
    """给每个目标层挂 hook。返回 (累加器, token 计数, handle 列表)。

    累加器留在 hook 看到的那个设备上（也就是模型所在的设备），结束时由
    `finalize` 一次性搬回 CPU。
    """
    table = dict(model.named_modules())
    cov_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    for name in target_names:
        module = table.get(name)
        if not isinstance(module, nn.Linear):
            raise RuntimeError(f"目标层 {name} 不存在或不是 nn.Linear")
        handles.append(module.register_forward_pre_hook(make_covariance_hook(name, cov_sums, counts)))

    return cov_sums, counts, handles


def finalize(
    cov_sums: dict[str, torch.Tensor],
    counts: dict[str, int],
) -> dict[str, torch.Tensor]:
    """把累加的 XᵀX 除成 E[x xᵀ]，搬回 CPU fp32。

    顺手对称化：XᵀX 在数学上对称，但浮点累加会让它偏离对称约 1e-7 量级，而下游要
    做特征值分解（谱诊断）和 ΔW C ΔWᵀ，非对称会让 eigvalsh 这类只读下三角的算子
    给出和上三角不一致的结果。代价只是一次加法。
    """
    out: dict[str, torch.Tensor] = {}
    for name, total in cov_sums.items():
        cov = (total / max(counts[name], 1)).cpu()
        out[name] = 0.5 * (cov + cov.T)
    return out


def save_covariance_payload(
    output_path: str | Path,
    covariances: dict[str, torch.Tensor],
    counts: dict[str, int],
    metadata: dict[str, Any],
) -> Path:
    """C、token 计数、采集设置存在同一个文件里。

    metadata 不是可选的装饰：几周后拿到一堆 .pt，要靠它分清哪个 C 是用哪份数据、
    多少 token、哪个 checkpoint 采的。缺了它就没法回答「这条臂的 C 是对的吗」。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "covariances": {k: v.cpu() for k, v in covariances.items()},
            "counts": counts,
            "metadata": metadata,
        },
        path,
    )
    return path


def load_covariance_file(path: str | Path) -> dict[str, torch.Tensor]:
    """只取 C 本身。带 metadata 的新格式和裸 dict 的旧格式都能读。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "covariances" in payload:
        return payload["covariances"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"不认识的协方差文件格式：{path}")


def load_covariance_payload(path: str | Path) -> dict[str, Any]:
    """连 counts / metadata 一起取，谱诊断和成本表要用。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "covariances" not in payload:
        raise ValueError(f"{path} 不是带 metadata 的协方差文件")
    return payload


def to_identity_covariances(covariances: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """把每个 C 换成同尺寸单位矩阵，这是最关键的那条消融。

    惩罚 tr(ΔW C ΔWᵀ) 退化成 tr(ΔW ΔWᵀ) = ‖ΔW‖_F²，也就是不带任何旧知识结构的
    L2 收缩。真 C 相对它没有优势，机制性结论就立不住，所以这条臂必须跑——尤其是
    在 inspect_cov_layers 已经查出层间保护强度差 6e4 倍的情况下。

    键原样保留，这样 assert_covers 仍然能匹配。
    """
    return {name: torch.eye(cov.shape[-1], dtype=cov.dtype) for name, cov in covariances.items()}


def move_covariances_to_device(
    covariances: dict[str, torch.Tensor],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """训练开始前一次性搬到设备上。

    惩罚每个 optimizer step 都要读全部目标层的 C，留在 CPU 就是每步把 1.67 GB
    拷过去。dtype 默认 fp32：bf16-mixed 下主权重本来就是 fp32，所以这里的 fp32
    不再承担精度作用（那是纯 bf16 训练的要求），但 ΔW 在训练早期非常小，bf16 的 C
    会给一个小量引入相对误差。换 bf16 能把 1.67 GB 砍到 0.83 GB，属于可测的优化项。
    """
    return {name: cov.to(device=device, dtype=dtype) for name, cov in covariances.items()}
