"""把一次 CPT 造成的遗忘按参数分区归因：C 管得到的那些层，和 C 管不到的其余参数。

c_4b 的实测结果是一对矛盾的数：惩罚在**自己的目标上**已经饱和（|CΔW| 相对 vanilla
压到 1/13，R 降 58%，而且 |CΔW| 的绝对值在往下走），遗忘却只追回 8%。加大 λ 救不了
这个——按这个兑换斜率，把 R 压到 0（等于把被正则的 110 层冻死、塑性全部放弃）也只有
14% 上限，而 replay_4b 是 72%。

所以瓶颈不在 λ，在于「C 约束的量」和「实际遗忘」之间的相关性。有两种互斥的解释，
修法完全不同：

  1. 遗忘主要来自**未被正则的参数**——lm_head、词嵌入、LayerNorm、bias（201 个张量，
     占 ΔW² 的 24.3%）。C 是带着 `include_lm_head=False` 采的，惩罚逐比特碰不到它们
     （两条臂这部分的 ‖ΔW‖ 差 0.2%，就是噪声）。这是个硬天花板，λ 调到多大都跨不过
     去，修法是把 C 扩到这些层。
  2. 遗忘主要来自**被正则的 110 层**（22 个 block × 5 个大矩阵，占 ΔW² 的 75.7%），
     但落在 C 的低特征值方向上。惩罚确实把漂移旋转进了 C 的零空间（|CΔW| 掉 13 倍而
     ‖ΔW‖ 只掉 1.7%），如果遗忘没跟着掉，说明零空间里照样在产生遗忘——那是激活二阶矩
     这个曲率代理本身选错了，得换估计方式。

两种模式，回答的是同一个问题，但只有第二种在这个模型上成立。

`--mode swap`（**已证伪，保留只为复现**）
    硬拼接：一个分区取训练后的权重、另一个取基座。实测交互项 −0.3680 和最大主项
    +0.3662 同量级反号，加法归因被完全吃掉。症状很明确：「仅正则层漂移」的 probe
    2.6030 比完整漂移的 2.4130 **还差**，「仅未正则层漂移」的 val 2.5288 比基座
    2.4875 **还差**——只动一部分参数不可能比全动更糟，除非拼出来的模型本身是坏的。
    原因是参数共适应：W 是一整套互相匹配的配置，换回一半，剩下一半就失去了配合
    对象。这两行测到的是「拼坏了有多难受」，不是「这部分造成了多少遗忘」。

`--mode sweep`（默认）
    不走到那么远的地方。从**真实的最终模型**出发，只把一个分区的漂移按 α 缩回去：

        W(α) = W₀ + ΔW_其余 + α·ΔW_该分区

    α=1 是真实模型，α=0 才退化成上面那个缝合怪。看 α→1 附近的斜率：那里离真实解
    很近，共适应几乎没被破坏，而斜率的含义恰恰是我们要的反事实——「当初要是把这个
    分区的漂移压下去一成，遗忘能少多少」，这正是正则器在做的事。整条曲线还能顺带
    显示共适应从哪个 α 开始发作（曲线掉头往上的地方）。

    真正的判据是**兑换率** = 少忘的量 ÷ 付出的新域代价。两个分区谁的兑换率高，就该
    往谁身上加约束。这个比较是同量纲、同模型、同数据的，比两条 20 小时的臂更干净。

用法：

    python -m kres.attribute_forgetting \\
        --base       model/<基座>/lit_model.pth \\
        --checkpoint out/vanilla/final/lit_model.pth \\
        --cov        cov/arm_4b.pt

评估集、批大小、迭代数默认与 03_cpt_train.pbs 一致（100 个 iter、micro_batch 8），
所以 α=1 那一行应当与训练日志收尾那次的 probe/val 对得上——对不上就说明这里的数据流或
精度和训练时不是一回事，整张表也就不能信。这条对照是白送的，别跳过。
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
DEFAULT_ALPHAS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]


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
    调用方要拿 `len(reg) == len(cov_modules)` 兜住这一条。
    """
    regularized = {
        k for k in keys if k.endswith(WEIGHT_SUFFIX) and k[: -len(WEIGHT_SUFFIX)] in cov_modules
    }
    return regularized, set(keys) - regularized


def blend(base: dict, drifted: dict, scaled: set[str], alpha: float) -> dict[str, torch.Tensor]:
    """`scaled` 里的键取 W₀ + α·ΔW，其余键取完整漂移后的值。键集以基座为准。

    α=1 和 α=0 走整数分支而不是算 `base + α·(drift-base)`：后者在 α=1 时会因为一次
    加减法引入 1 ulp 的偏差，让「真实模型」那一行和训练日志差在最后一位。这条对照是
    整张表可信度的锚点，不能让它被浮点噪声弄脏。
    """
    out = {}
    for k in base:
        if k not in scaled:
            out[k] = drifted[k].clone()
        elif alpha == 1.0:
            out[k] = drifted[k].clone()
        elif alpha == 0.0:
            out[k] = base[k].clone()
        else:
            b = base[k].float()
            out[k] = b + alpha * (drifted[k].float() - b)
    return out


def merge(base: dict, drifted: dict, take_drifted: set[str]) -> dict[str, torch.Tensor]:
    """硬拼接：`take_drifted` 里的键取漂移后的值，其余取基座。"""
    return {k: (drifted[k] if k in take_drifted else base[k]).clone() for k in base}


def unwrap(model):
    """取出 fabric 包装下的原始 module，load_state_dict 要打在它身上。"""
    return getattr(model, "_forward_module", model)


def make_loaders(fabric: L.Fabric, data: ReplayMixedData) -> dict[str, torch.utils.data.DataLoader]:
    """每个变体都重建一次 loader。

    不复用是有意的：litdata 的 StreamingDataset 自己记 epoch 与游标，同一个 loader 连续
    迭代十几次未必每次都从头给同一批数据。真发生了的话各变体吃的是不同样本，表里的差异
    全是数据差异，而且**看不出来**——所有数字都在合理范围内。重建的代价是几百毫秒。
    """
    loaders = data.eval_dataloaders()
    names = list(loaders)
    prepared = fabric.setup_dataloaders(*loaders.values())
    if len(names) == 1:
        prepared = [prepared]
    return dict(zip(names, prepared))


def fmt_row(label: str, losses: dict, ref: dict) -> str:
    p, v = losses["probe"], losses["val"]
    return (
        f"{label:<16}{p:>9.4f}{math.exp(p):>9.3f}{p - ref['probe']:>+9.4f}   "
        f"{v:>9.4f}{math.exp(v):>9.3f}{ref['val'] - v:>+9.4f}"
    )


HEADER = f"{'':<16}{'probe':>9}{'ppl':>9}{'遗忘':>9}   {'val':>9}{'ppl':>9}{'新域收益':>9}"


def run_swap(evaluate, base_sd, drift_sd, reg_keys, unreg_keys, ref):
    variants = [
        ("仅正则层漂移", reg_keys),
        ("仅未正则层漂移", unreg_keys),
        ("完整漂移", set(base_sd)),
    ]
    rows = [(n, evaluate(merge(base_sd, drift_sd, k))) for n, k in variants]
    full = dict(rows[-1][1])

    print("\n" + HEADER)
    print("-" * 80)
    print(fmt_row("基座 W₀", ref, ref))
    for name, losses in rows:
        print(fmt_row(name, losses, ref))

    by_name = {n: d for n, d in rows}
    for metric, label in (("probe", "遗忘"), ("val", "新域收益")):
        a = by_name["仅正则层漂移"][metric] - ref[metric]
        b = by_name["仅未正则层漂移"][metric] - ref[metric]
        t = full[metric] - ref[metric]
        print(f"\n{label}分解：被正则 {a:+.4f} + 未正则 {b:+.4f} + 交互 {t - a - b:+.4f} = {t:+.4f}")
    print(
        "\n  ⚠ 交互项与主项同量级时这套加法不成立（实测就是这样）。硬拼接会把训练出来的\n"
        "    自洽配置打散，测到的是「拼坏了」而不是「这部分造成的遗忘」。用 --mode sweep。"
    )


def run_sweep(evaluate, base_sd, drift_sd, partitions, alphas, ref):
    """每个分区扫一条回滚曲线，其余参数始终保持完整漂移。"""
    # α=1 与扫哪个分区无关，都是真实模型。评一次就够，也保证两条曲线共用同一个锚点
    full = evaluate(blend(base_sd, drift_sd, set(), 1.0))
    print("\n" + HEADER)
    print("-" * 80)
    print(fmt_row("基座 W₀", ref, ref))
    print(fmt_row("真实模型 α=1", full, ref))

    slopes = {}
    for label, keys in partitions.items():
        print(f"\n=== 回滚「{label}」（{len(keys)} 个张量），其余保持完整漂移 ===")
        print(HEADER)
        print("-" * 80)
        curve = {}
        for a in alphas:
            curve[a] = full if a == 1.0 else evaluate(blend(base_sd, drift_sd, keys, a))
            print(fmt_row(f"α = {a:.2f}", curve[a], ref))
        near = max((a for a in alphas if a < 1.0), default=None)
        if near is not None:
            d_forget = curve[near]["probe"] - full["probe"]
            d_plastic = curve[near]["val"] - full["val"]
            slopes[label] = (near, d_forget, d_plastic)

    if not slopes:
        return
    print("\n局部兑换率（从 α=1 回滚一小步，即把该分区的漂移压下去一部分）")
    print("-" * 80)
    print(f"{'分区':<16}{'回滚到':>8}{'遗忘变化':>12}{'新域代价':>12}{'兑换率':>12}")
    for label, (near, d_forget, d_plastic) in slopes.items():
        rate = (-d_forget / d_plastic) if d_plastic > 1e-6 else float("nan")
        print(f"{label:<16}{near:>8.2f}{d_forget:>+12.4f}{d_plastic:>+12.4f}{rate:>12.2f}")
    print(
        "\n  「遗忘变化」为负 = 少忘了；「新域代价」为正 = 新域变差了。兑换率 = 前者绝对值 ÷ 后者，\n"
        "  越高越划算。两个分区的兑换率之比就是「该往谁身上加约束」的答案——它同量纲、同模型、\n"
        "  同一批数据，比再跑两条 20 小时的臂干净得多。\n"
        "  兑换率是 nan 说明回滚这个分区并没有让新域变差，那这个分区的漂移是纯亏损，直接约束它。\n"
        "  曲线在小 α 处掉头往上，是共适应开始发作，那一段的数不能用来外推。"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", type=Path, required=True, help="基座 lit_model.pth（W₀）")
    ap.add_argument("--checkpoint", type=Path, required=True, help="跑完的检查点（W），一般是 vanilla 的 final")
    ap.add_argument("--cov", type=Path, required=True, help="C 文件，只用它的键集来划分区")
    ap.add_argument("--mode", choices=("sweep", "swap"), default="sweep", help="swap 已证伪，只为复现保留")
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS, help="回滚系数，必须含 1.0")
    ap.add_argument("--plan", type=Path, default=Path("data/chunks/fineweb_edu/replay_plan.json"))
    ap.add_argument("--new-data", type=Path, default=Path("data/chunks/biomed/train"))
    ap.add_argument("--val-data", type=Path, default=Path("data/chunks/biomed/val"))
    ap.add_argument("--probe-data", type=Path, default=Path("data/chunks/fineweb_edu/probe"))
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-iters", type=int, default=100, help="与 03_cpt_train.pbs 的 EVAL_ITERS 一致")
    ap.add_argument("--precision", type=str, default=None, help="默认与训练一致（bf16-mixed）")
    args = ap.parse_args(argv)

    alphas = sorted(set(args.alphas), reverse=True)
    if args.mode == "sweep" and 1.0 not in alphas:
        raise SystemExit("--alphas 必须包含 1.0：整张表的斜率和锚点都以它为基准")

    model_dir = args.base.parent
    for path in (args.base, args.checkpoint, args.cov):
        if not path.is_file():
            raise SystemExit(f"找不到 {path}")

    base_sd = load_state(args.base)
    drift_sd = load_state(args.checkpoint)
    missing = [k for k in base_sd if k not in drift_sd]
    if missing:
        raise SystemExit(
            f"检查点里缺 {len(missing)} 个基座有的键（前几个：{missing[:5]}）。两个文件不是同一个模型"
        )

    cov_modules = set(load_covariance_file(args.cov))
    reg_keys, unreg_keys = split_keys(base_sd, cov_modules)
    if len(reg_keys) != len(cov_modules):
        raise SystemExit(
            f"C 有 {len(cov_modules)} 个模块，但只有 {len(reg_keys)} 个能在权重里对上。C 和这个模型不匹配"
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
    print(f"未正则的都是些什么：{sorted(unreg_keys)[:6]}{' ...' if len(unreg_keys) > 6 else ''}")

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

    def evaluate(state_dict) -> dict[str, float]:
        unwrap(model).load_state_dict(state_dict, strict=True)
        return {
            name: validate(fabric, model, dl, max_iters=args.max_iters, verbose=False).item()
            for name, dl in make_loaders(fabric, data).items()
        }

    # 基座评两次，头尾各一次。这条自检比它花的一分钟值钱——loader 带状态是这个脚本唯一
    # 会静默出错的地方，那种错所有数字都落在合理范围内，看不出来
    ref = evaluate(merge(base_sd, drift_sd, set()))
    fabric.print(f"  基座 W₀：" + "  ".join(f"{k} {v:.4f}" for k, v in ref.items()))

    if args.mode == "swap":
        run_swap(evaluate, base_sd, drift_sd, reg_keys, unreg_keys, ref)
    else:
        partitions = {
            f"被正则 {len(reg_keys)} 个": reg_keys,
            f"未正则 {len(unreg_keys)} 个": unreg_keys,
        }
        run_sweep(evaluate, base_sd, drift_sd, partitions, alphas, ref)

    recheck = evaluate(merge(base_sd, drift_sd, set()))
    drift_check = max(abs(recheck[k] - ref[k]) for k in ref)
    if drift_check > 1e-4:
        raise SystemExit(
            f"\n两次评基座差了 {drift_check:.2e}，说明各轮评估吃的不是同一批数据，上面的表不成立。"
            f"检查 eval loader 是不是带了状态"
        )
    print(f"\n自检通过：基座复测与首测最大差 {drift_check:.2e}（各变体吃的是同一批数据）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
