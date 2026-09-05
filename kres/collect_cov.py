"""Stage 1：在冻结的基座上采集 C = E[x xᵀ]。

    python -m kres.collect_cov --chunks <段目录> --output <out.pt> --cov-tokens 5e8

一次只采一个 replay 段（fineweb_edu/seg00、seg01 ...）。某条臂要读多个段时，对每段各采
一次，再用 `kres.mix_covariances` 按 token 计数加权合并——那个合并与「直接在这几段的并集
上采一次」严格等价，但 3 条嵌套的臂只需 3 次采集而不是 6 次。

段目录同时就是那条臂 replay 用的目录，这是设计要求：本方法要论证「用旧数据估出来的 C
可以替代重放这些旧数据」，两条臂必须面对同一批旧数据。别手写目录名，从
`data/chunks/fineweb_edu/replay_plan.json` 里取。

核心原则：**C 必须走和训练完全同一条数据管线。**
这不是洁癖。C = E[x xᵀ] 是对输入分布的估计，采集时和训练时看到的分布只要不一致，
C 就在保护一个模型实际不会遇到的方向。SFT 那边的病态 C（inspect_cov_layers 查出
layers.2.mlp.down_proj 99.995% 的迹集中在单个方向、层间保护强度差 6.13e4 倍）最可能
的成因正是这个：left padding + chat 模板让 pad / BOS / 模板分隔符这类高度重复的 token
在位置统计里占比极高，而它们的 hidden state 几乎是同一个向量，多个近乎相同的 x 叠出来
的 XᵀX 自然接近 rank-1。

所以这里直接用 `litgpt.data.LitData`——就是 `pretrain.py` L220 的 get_dataloaders
用的那个 data module，同一个 StreamingDataset、同一个 TokensLoader、同一个 block_size。
定长打包语料没有 padding、没有模板 token，那个成因会自然消失，但前提是采集器也用打包
数据。继续用旧管线采 C、用打包数据训练，只会把问题藏得更深。

采完立刻跑 `python -m kres.inspect_cov`，别等训练。如果新 C 的层间差距还是 1e4 量级，
那就不是管线问题，得回到 C 的定义本身。

开销量级（0.4b 基座、seq_len 4096）：XᵀX 的 FLOPs 和模型前向大致同量级，所以采集
约等于「以两倍前向成本扫一遍语料」。fp32 累加在 H100 上约 0.2 s/batch(micro=4)，
5e8 token 约 2 小时。C 的累加器常驻 sum(d_in²) × 4 字节 = 1.67 GB。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kres.covariance import (
    assert_covers,
    find_target_linears,
    finalize,
    module_in_features,
    register_covariance_hooks,
    save_covariance_payload,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "model" / "open-sci-ref-v0.02-0.4b-fineweb-edu-1.4t-300B-4096"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="采集 OneReplay 的协方差矩阵 C")
    p.add_argument("--chunks", required=True, help="一个 replay 段的 litdata chunk 目录（fineweb_edu/seg00 ...）")
    p.add_argument("--output", required=True, help="C 的输出路径（.pt）")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL), help="含 model_config.yaml 与 lit_model.pth")
    p.add_argument(
        "--cov-tokens",
        type=float,
        default=5e8,
        help="采多少 token 后停止。定长打包下每 batch 贡献 micro_batch×seq_len 个 token，"
             "比 SFT 高一到两个数量级，SFT 的样本数不能直接沿用。这个值需要靠谱扫描来定",
    )
    p.add_argument("--micro-batch-size", type=int, default=4, help="与训练一致，默认 4")
    p.add_argument("--seq-len", type=int, default=4096, help="与训练一致，默认 4096")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42, help="dataloader 的 shuffle 种子，决定采到哪批样本")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--forward-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="前向的激活精度。float32 是对冻结基座的忠实求值、可复现；bfloat16 与训练时"
             "（bf16-mixed）看到的激活精度一致但更快。无论选哪个，XᵀX 一律 fp32 累加",
    )
    p.add_argument("--include-lm-head", action="store_true", help="把 lm_head 也纳入目标层（消融用）")
    p.add_argument("--log-every", type=int, default=50, help="每多少 batch 打一行进度")
    p.add_argument("--max-batches", type=int, default=0, help=">0 时只跑这么多 batch（自测用）")
    return p.parse_args()


def load_frozen_model(args: argparse.Namespace):
    """构造 eager 的 GPT 并载入 lit_model.pth。

    **不要 torch.compile。** 编译会给模块名加 `_orig_mod.` 前缀、而且编译后的子模块上
    挂 forward hook 正是会 graph break 或被静默跳过的情形。covariance.find_target_linears
    对此有硬检查。
    """
    from litgpt.config import Config
    from litgpt.model import GPT

    model_dir = Path(args.model_dir)
    config = Config.from_file(model_dir / "model_config.yaml")
    config.block_size = max(config.block_size, args.seq_len)

    model = GPT(config)
    checkpoint = model_dir / "lit_model.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    # strict=True：键少一个都要报错。C 采在一个权重不完整的模型上是无声的灾难
    model.load_state_dict(state, strict=True)
    model.max_seq_length = args.seq_len

    dtype = getattr(torch, args.forward_dtype)
    model = model.to(device=args.device, dtype=dtype)
    model.eval()
    return model, config


def build_dataloader(args: argparse.Namespace):
    """用训练那条管线取 dataloader，见模块 docstring。"""
    from litgpt.data import LitData

    data = LitData(data_path=args.chunks, num_workers=args.num_workers, seed=args.seed)
    data.connect(tokenizer=None, batch_size=args.micro_batch_size, max_seq_length=args.seq_len)
    return data.train_dataloader()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"设备 {args.device}，前向 dtype {args.forward_dtype}")
    model, config = load_frozen_model(args)
    print(f"基座 {config.name}：n_layer={config.n_layer} n_embd={config.n_embd}")

    target_names = find_target_linears(model, config.n_layer, include_lm_head=args.include_lm_head)
    shapes = module_in_features(model, target_names)
    cov_bytes = sum(d * d for d in shapes.values()) * 4
    print(f"目标层 {len(target_names)} 个（{len(target_names) // config.n_layer} 个/block × {config.n_layer}）"
          f"{'，含 lm_head' if args.include_lm_head else ''}")
    print(f"C 累加器常驻 {cov_bytes / 1e9:.2f} GB (fp32)")

    dataloader = build_dataloader(args)
    cov_sums, counts, handles = register_covariance_hooks(model, target_names)

    target_tokens = int(args.cov_tokens)
    tokens_per_batch = args.micro_batch_size * args.seq_len
    print(f"目标 {target_tokens:,} token，每 batch {tokens_per_batch:,} → 约 "
          f"{max(1, target_tokens // tokens_per_batch):,} 个 batch")

    seen_tokens = 0
    batches = 0
    started = time.time()
    try:
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[:, : args.seq_len].contiguous().long().to(args.device)
                model(input_ids)
                seen_tokens += input_ids.numel()
                batches += 1

                if batches % args.log_every == 0:
                    elapsed = time.time() - started
                    print(f"  batch {batches:,}  token {seen_tokens:,}  "
                          f"{elapsed / batches:.3f} s/batch  已用 {elapsed / 60:.1f} 分钟")

                if seen_tokens >= target_tokens:
                    break
                if args.max_batches and batches >= args.max_batches:
                    break
    finally:
        for handle in handles:
            handle.remove()

    if not batches:
        raise RuntimeError(f"{args.chunks} 一个 batch 都没产出——chunk 目录是空的或 block_size 不匹配")

    # 所有目标层看到的位置数必须完全相同：它们在同一次前向里被同一批 token 驱动。
    # 不相同说明某层的 hook 漏挂了或被跳过（比如模型里有条件分支绕过了某个 Linear）
    distinct = sorted(set(counts.values()))
    if len(distinct) != 1:
        raise RuntimeError(f"各层的 token 计数不一致：{distinct[:5]}...，说明有 hook 没被调用到")

    covariances = finalize(cov_sums, counts)
    assert_covers(covariances, target_names, shapes)

    elapsed = time.time() - started
    metadata = {
        "model_name": config.name,
        "model_dir": str(args.model_dir),
        "chunks": str(args.chunks),
        "cov_normalization": "none",  # C = E[x xᵀ]，与 SFT 侧定义一致
        "seq_len": args.seq_len,
        "micro_batch_size": args.micro_batch_size,
        "forward_dtype": args.forward_dtype,
        "seed": args.seed,
        "include_lm_head": args.include_lm_head,
        "target_layers": len(target_names),
        "batches": batches,
        "tokens_seen": seen_tokens,
        "tokens_per_layer": distinct[0],
        "elapsed_seconds": round(elapsed, 1),
    }
    path = save_covariance_payload(args.output, covariances, counts, metadata)

    print(f"\n完成：{batches:,} batch / {seen_tokens:,} token / {elapsed / 60:.1f} 分钟")
    print(f"C 写入 {path}（{path.stat().st_size / 1e9:.2f} GB）")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("\n下一步：python -m kres.inspect_cov --cov " + str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
