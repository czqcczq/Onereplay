"""把一次 CPT 造成的遗忘按参数分区归因：C 管得到的那些层，和 C 管不到的其余参数。

c_4b 的实测结果是一对矛盾的数：惩罚在**自己的目标上**已经饱和（|CΔW| 相对 vanilla
压到 1/13，R 降 58%，而且 |CΔW| 的绝对值在往下走），遗忘却只追回 8%。加大 λ 救不了
这个——按这个兑换斜率，把 R 压到 0（等于把被正则的 110 层冻死、塑性全部放弃）也只有
14% 上限，而 replay_4b 是 72%。

所以瓶颈不在 λ，在于「C 约束的量」和「实际遗忘」之间的相关性。有两种互斥的解释，
修法完全不同：

  1. 遗忘主要来自**未被正则的参数**——lm_head、embedding、LayerNorm、bias。C 是带着
     `include_lm_head=False` 采的，惩罚逐比特碰不到它们（两条臂这部分的 ‖ΔW‖ 分别是
     77.80 和 77.96，差 0.2%，就是随机噪声）。这是个硬天花板，λ 调到多大都跨不过去，
     修法是把 C 扩到这些层。
  2. 遗忘主要来自**被正则的 110 层**，但落在 C 的低特征值方向上。惩罚确实把漂移旋转
     进了 C 的零空间（|CΔW| 掉 13 倍而 ‖ΔW‖ 只掉 1.7%），如果遗忘没跟着掉，说明零空间
     里照样在产生遗忘——那是激活二阶矩这个曲率代理本身选错了，得换估计方式。

区分办法是拼接。拿基座 W₀ 和跑完的检查点 W，按分区各取一半拼出两个混合模型：

    仅正则层漂移   = 被正则的 110 层取 W，其余取 W₀
    仅未正则层漂移 = 被正则的 110 层取 W₀，其余取 W

各测一次 probe，相对基座涨了多少就是那一部分单独的贡献。

**同时测 val，这一半同样重要。** 「哪部分参数在承担新域学习」决定了扩 C 的代价：要是
新域的收益也主要来自 lm_head，那把它约束住就是拿塑性换遗忘，和 replay 没有本质区别，
方法的卖点（几乎零开销）就没了。

两部分的贡献之和一般**不等于**完整漂移的贡献——loss 不是参数的线性函数，拆开就有交叉
项。脚本把完整模型也测一遍，差额直接打出来标成交互项，不藏进任何一边。交互项和两个
主项同量级时，这个归因本身就不该被当结论用，所以它必须显式出现在表里。

用法：

    python -m kres.attribute_forgetting \\
        --base       model/<基座>/lit_model.pth \\
        --checkpoint out/vanilla/final/lit_model.pth \\
        --cov        cov/arm_4b.pt

评估集、批大小、迭代数默认与 03_cpt_train.pbs 一致（probe 100 个 iter、micro_batch 8），
所以「完整漂移」那一行应当与训练日志收尾那次的 probe loss 对得上——对不上就说明这里的
数据流或精度和训练时不是一回事，后面三行也就不能信。这条对照是白送的，别跳过。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import lightning as L
import torch

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.pretrain import validate
from litgpt.utils import get_default_supported_precision

from kres.covariance import load_covariance_file
from kres.replay_data import NO_REPLAY, ReplayMixedData

WEIGHT_SUFFIX = ".weight"


def load_state(path: Path) -> dict[str, torch.Tensor]:
    """读权重。训练中途存的检查点把权重包在 "model" 里，基座是裸的 state_dict。"""
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return state


def split_keys(keys, cov_modules: set[str]) -> tuple[set[str], set[str]]:
    """按 C 覆盖的模块名把参数分成两堆。

    C 里存的键是**模块名**（`transformer.h.0.attn.qkv`），权重键多一个 `.weight`。
    这个错位如果搞反，两堆会整个对调，而结果表照样打得出来、只是结论完全相反。
    """
    regularized = {
        k for k in keys if k.endswith(WEIGHT_SUFFIX) and k[: -len(WEIGHT_SUFFIX)] in cov_modules
    }
    return regularized, set(keys) - regularized


def merge(base: dict, drifted: dict, take_drifted: set[str]) -> dict[str, torch.Tensor]:
    """`take_drifted` 里的键取漂移后的值，其余取基座。键集以基座为准。"""
    return {k: (drifted[k] if k in take_drifted else base[k]).clone() for k in base}


def unwrap(model):
    """取出 fabric 包装下的原始 module，load_state_dict 要打在它身上。"""
    return getattr(model, "_forward_module", model)


def make_loaders(fabric: L.Fabric, data: ReplayMixedData) -> dict[str, torch.utils.data.DataLoader]:
    """每个变体都重建一次 loader。

    不复用是有意的：litdata 的 StreamingDataset 自己记 epoch 与游标，同一个 loader 连续
    迭代四次未必每次都从头给同一批数据。真发生了的话四个变体吃的是不同样本，表里的差异
    全是数据差异，而且**看不出来**——所有数字都在合理范围内。重建的代价是几百毫秒。
    """
    loaders = data.eval_dataloaders()
    names = list(loaders)
    prepared = fabric.setup_dataloaders(*loaders.values())
    if len(names) == 1:
        prepared = [prepared]
    return dict(zip(names, prepared))


def evaluate(fabric, model, data, state_dict, max_iters: int) -> dict[str, float]:
    unwrap(model).load_state_dict(state_dict, strict=True)
    loaders = make_loaders(fabric, data)
    return {
        name: validate(fabric, model, dl, max_iters=max_iters, verbose=False).item()
        for name, dl in loaders.items()
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", type=Path, required=True, help="基座 lit_model.pth（W₀）")
    ap.add_argument("--checkpoint", type=Path, required=True, help="跑完的检查点（W），一般是 vanilla 的 final")
    ap.add_argument("--cov", type=Path, required=True, help="C 文件，只用它的键集来划分区")
    ap.add_argument("--plan", type=Path, default=Path("data/chunks/fineweb_edu/replay_plan.json"))
    ap.add_argument("--new-data", type=Path, default=Path("data/chunks/biomed/train"))
    ap.add_argument("--val-data", type=Path, default=Path("data/chunks/biomed/val"))
    ap.add_argument("--probe-data", type=Path, default=Path("data/chunks/fineweb_edu/probe"))
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-iters", type=int, default=100, help="与 03_cpt_train.pbs 的 EVAL_ITERS 一致")
    ap.add_argument("--precision", type=str, default=None, help="默认与训练一致（bf16-mixed）")
    args = ap.parse_args(argv)

    model_dir = args.base.parent
    for path in (args.base, args.checkpoint, args.cov):
        if not path.is_file():
            raise SystemExit(f"找不到 {path}")

    base_sd = load_state(args.base)
    drift_sd = load_state(args.checkpoint)
    missing = [k for k in base_sd if k not in drift_sd]
    if missing:
        raise SystemExit(
            f"检查点里缺 {len(missing)} 个基座有的键（前几个：{missing[:5]}）。"
            f"两个文件不是同一个模型"
        )

    cov_modules = set(load_covariance_file(args.cov))
    reg_keys, unreg_keys = split_keys(base_sd, cov_modules)
    if len(reg_keys) != len(cov_modules):
        raise SystemExit(
            f"C 有 {len(cov_modules)} 个模块，但只有 {len(reg_keys)} 个能在权重里对上。"
            f"C 和这个模型不匹配"
        )

    def sq_norm(keys):
        return sum(float((drift_sd[k].float() - base_sd[k].float()).pow(2).sum()) for k in keys)

    reg_sq, unreg_sq = sq_norm(reg_keys), sq_norm(unreg_keys)
    total_sq = reg_sq + unreg_sq
    print(f"基座    {args.base}")
    print(f"检查点  {args.checkpoint}")
    print(f"C       {args.cov}")
    print(
        f"\n分区：被正则 {len(reg_keys)} 个张量（‖ΔW‖={reg_sq ** 0.5:.3f}，"
        f"占 ΔW² 的 {reg_sq / total_sq:.1%}）"
        f" / 未正则 {len(unreg_keys)} 个（‖ΔW‖={unreg_sq ** 0.5:.3f}，{unreg_sq / total_sq:.1%}）"
    )
    sample = sorted(unreg_keys)[:6]
    print(f"未正则的都是些什么：{sample}{' ...' if len(unreg_keys) > 6 else ''}")

    precision = args.precision or get_default_supported_precision(training=True)
    fabric = L.Fabric(devices=1, precision=precision)
    fabric.launch()

    config = Config.from_file(model_dir / "model_config.yaml")
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    model = fabric.setup(model)

    data = ReplayMixedData(
        plan_path=args.plan,
        arm=NO_REPLAY,
        new_data_path=args.new_data,
        val_data_path=args.val_data,
        probe_data_path=args.probe_data,
    )
    data.connect(tokenizer=None, batch_size=args.micro_batch, max_seq_length=model.max_seq_length)
    data.prepare_data()
    data.setup()

    # 顺序里基座出现两次：第一次当基准，最后一次核对四轮评估之间数据流没有漂移。
    # 这条自检比它省下的时间值钱——loader 状态污染是这个脚本唯一会静默出错的地方
    variants = [
        ("基座 W₀", set()),
        ("仅正则层漂移", reg_keys),
        ("仅未正则层漂移", unreg_keys),
        ("完整漂移", set(base_sd)),
        ("基座 W₀（复测）", set()),
    ]
    results: list[tuple[str, dict[str, float]]] = []
    for name, take in variants:
        losses = evaluate(fabric, model, data, merge(base_sd, drift_sd, take), args.max_iters)
        results.append((name, losses))
        fabric.print(f"  {name} 评完：" + "  ".join(f"{k} {v:.4f}" for k, v in losses.items()))

    ref = dict(results[0][1])
    drift_check = max(abs(results[-1][1][k] - ref[k]) for k in ref)
    if drift_check > 1e-4:
        raise SystemExit(
            f"两次评基座差了 {drift_check:.2e}，说明四轮评估吃的不是同一批数据，"
            f"下面的归因不成立。检查 eval loader 是不是带了状态"
        )
    results = results[:-1]

    full = dict(results[-1][1])
    print(f"\n{'变体':<18}{'probe':>9}{'ppl':>9}{'遗忘':>9}{'占比':>8}   {'val':>9}{'ppl':>9}{'学到':>9}")
    print("-" * 84)
    for name, losses in results:
        p, v = losses["probe"], losses["val"]
        forget = p - ref["probe"]
        share = forget / (full["probe"] - ref["probe"]) if full["probe"] != ref["probe"] else float("nan")
        print(
            f"{name:<18}{p:>9.4f}{math.exp(p):>9.3f}{forget:>+9.4f}{share:>8.1%}   "
            f"{v:>9.4f}{math.exp(v):>9.3f}{ref['val'] - v:>+9.4f}"
        )

    by_name = {n: d for n, d in results}
    for metric, label in (("probe", "遗忘"), ("val", "新域收益")):
        a = by_name["仅正则层漂移"][metric] - ref[metric]
        b = by_name["仅未正则层漂移"][metric] - ref[metric]
        t = full[metric] - ref[metric]
        print(f"\n{label}分解：被正则 {a:+.4f} + 未正则 {b:+.4f} + 交互 {t - a - b:+.4f} = {t:+.4f}")

    print(
        "\n怎么读：\n"
        "  「占比」大的那一边就是遗忘的主要来源。未正则那边占大头 → C 覆盖不全是硬天花板，\n"
        "  λ 调多大都没用，该做的是把 C 扩到 lm_head/embedding；被正则那边占大头 → C 覆盖\n"
        "  到位却没拦住，那是激活二阶矩这个曲率代理不合适，该换估计方式而不是扩范围。\n"
        "  交互项与两个主项同量级时这套加法就不成立，只能当定性参考。\n"
        "  最后一列同样要看：扩 C 的代价等于放弃那部分参数承担的新域收益。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
