"""Stage 1 CLI: estimate old-knowledge hidden-state second moments.

Run the frozen base model over an old-knowledge corpus (FLAN by default) and
estimate one matrix per LoRA target layer:

    C_l = E_x[x x^T]

where x is the input hidden state of that layer for one non-padding token.

With --cov_normalization base_output_norm the script instead estimates

    C_l = E_x[(x / ||W_l x||)(x / ||W_l x||)^T]

so the training-time penalty becomes E_x ||DeltaW_l x||^2 / ||W_l x||^2.

Usage: python -m onereplay.scripts.collect_cov [args]
"""

# =============================================================================
# PORT NOTE 文件状态：[REFERENCE-ONLY] 不搬这个文件，逻辑重写
#
# onereplay/scripts/collect_cov.py 的逐字节副本，只加注释、未改代码行。校验：
#     diff onereplay/scripts/collect_cov.py con-pretrain/onereplay_port/scripts/collect_cov.py
#
# 这是 stage 1 入口。骨架照抄、数据源整段重写，详见下面 collect_covariances 上的批注
# ——那条批注是本次移植里最需要想清楚的一条，因为它同时是病态 C 的可能解法。
#
# 顶部 docstring 里"x is the input hidden state of that layer for one non-padding token"
# 这句在预训练打包语料下要改：没有 padding，是"每个位置"。
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from onereplay.core.covariance import (  # noqa: E402
    register_covariance_hooks,
    save_covariance_payload,
)
from onereplay.core.modeling import (  # noqa: E402
    find_target_linear_module_names,
    load_causal_lm_and_tokenizer,
    set_seed,
)
from onereplay.data.old_knowledge import (  # noqa: E402
    build_collate_fn,
    filter_incomplete_rows,
    fingerprint_pool,
    limit_dataset,
    load_old_knowledge_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse all settings for collecting C from FLAN or another text corpus."""

    parser = argparse.ArgumentParser(description="Collect OneReplay covariance matrices.")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)

    # Dataset input. Use --dataset_path for a dataset saved by datasets.save_to_disk.
    # Use --dataset_name/--dataset_config for a HuggingFace dataset.
    # Use --data_files for local json/jsonl/text files.
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="Muennighoff/flan")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--data_files", type=str, default="")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/weiliu1/huggingface/datasets/cache",
        help="HuggingFace dataset cache directory",
    )
    parser.add_argument(
        "--streaming",
        type=int,
        default=1,
        help="1 streams HF datasets so FLAN does not need to be fully downloaded first",
    )

    # FLAN-like datasets usually have "inputs" and "targets". If your local
    # files have a single field, set --text_column to that field name.
    parser.add_argument("--text_column", type=str, default="")
    parser.add_argument("--input_column", type=str, default="inputs")
    parser.add_argument("--target_column", type=str, default="targets")
    parser.add_argument(
        "--use_chat_template",
        type=int,
        default=1,
        help="1 formats old-knowledge examples with tokenizer.apply_chat_template",
    )
    parser.add_argument(
        "--include_target_in_chat",
        type=int,
        default=1,
        help="1 includes FLAN targets as assistant messages when collecting C",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Optional system message inserted before each FLAN example",
    )
    parser.add_argument(
        "--enable_thinking",
        type=int,
        default=0,
        help="1 renders the prompt with Qwen3's thinking block open; must match "
        "the self-distillation setting so C sees the sequence that was generated",
    )
    parser.add_argument(
        "--concat_prompt_target",
        type=int,
        default=0,
        help="1 builds the text as generation_prompt + raw target instead of "
        "re-rendering the assistant turn. Required with --enable_thinking 1, "
        "because the chat template strips <think> out of assistant messages",
    )
    parser.add_argument(
        "--debug_print_examples",
        type=int,
        default=0,
        help="Print this many fully rendered texts before collecting, to verify "
        "what C actually sees (e.g. that <think> survived)",
    )

    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument(
        "--sample_shuffle",
        type=int,
        default=1,
        help=(
            "1 shuffles the corpus with --sample_seed before taking max_samples, "
            "so the subset is a reproducible random sample instead of the first N rows. "
            "0 keeps the original order."
        ),
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=1,
        help="Seed for the reproducible subset shuffle when --sample_shuffle 1.",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        choices=["uniform", "balanced"],
        default="uniform",
        help=(
            "uniform (default) samples rows uniformly, reproducing the corpus's raw "
            "task mixture. balanced draws a per-task quota with FLAN's capped-"
            "proportional weighting min(N_i, --mixing_rate_max), so C is not "
            "dominated by whichever tasks own the most rows. balanced needs a "
            "--task_column and a map-style dataset."
        ),
    )
    parser.add_argument(
        "--task_column",
        type=str,
        default="task",
        help="Column naming each row's task, used only by --sample_strategy balanced.",
    )
    parser.add_argument(
        "--mixing_rate_max",
        type=int,
        default=3000,
        help=(
            "FLAN's mixing rate maximum for --sample_strategy balanced: a task's "
            "weight is min(N_i, this), so tasks at or above it are equally weighted."
        ),
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10000,
        help="Approximate-shuffle buffer size used only for streaming datasets.",
    )
    parser.add_argument(
        "--require_target",
        type=int,
        default=0,
        help=(
            "1 drops rows with an empty input or target before sampling. Needed for a "
            "self-distilled corpus, where a prompt whose generation hit the token cap is "
            "stored with an empty target and is dropped by the replay loader too; without "
            "this, C would cover rows replay never trains on."
        ),
    )
    parser.add_argument(
        "--require_target_column",
        type=str,
        default="",
        help=(
            "Column --require_target checks for emptiness. Empty means --target_column, "
            "which is right whenever the two are the same corpus. Set it to targets while "
            "--target_column is gold_targets to run the gold ablation on the self-distilled "
            "file's exact row set: gold is filled in on truncated rows too, so filtering on "
            "it would give the gold arm extra rows and confound target source with pool size."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument(
        "--truncation_side",
        type=str,
        choices=["", "left", "right"],
        default="",
        help=(
            "Which end to cut when a rendered example exceeds --max_len. Empty keeps the "
            "tokenizer default (right, i.e. the assistant answer is dropped first). "
            "Training truncates on the left, so pass left to match it whenever the "
            "targets are long enough to overflow."
        ),
    )
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument(
        "--cov_normalization",
        type=str,
        choices=["none", "base_output_norm"],
        default="none",
        help=(
            "none collects E[x x^T]. base_output_norm collects "
            "E[(x / ||W x||)(x / ||W x||)^T] for a relative-error penalty."
        ),
    )
    parser.add_argument(
        "--cov_norm_eps",
        type=float,
        default=1e-6,
        help="Lower bound for ||W x|| when --cov_normalization base_output_norm is used.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./mycode/onereplay/flan_qwen3_qv_cov.pt",
    )
    return parser.parse_args()


# PORT NOTE [REFERENCE-ONLY] ★ 这是新项目里唯一需要**重写**（而不是改）的组件。
#
# 骨架结构原样可用，照着抄：
#     加载模型 -> find_target_linear_module_names -> register_covariance_hooks
#     -> no_grad 前向遍历数据 -> cov_sum / count -> save_covariance_payload
# 换掉的只有中间那段"数据从哪来"（L262-294：load_old_knowledge_dataset / limit_dataset /
# DataLoader + build_collate_fn），以及模型换成 LitGPT 的 GPT。
#
# ★ 核心原则：C 必须走**和训练完全同一条**数据管线。
# 这不是洁癖，而是那个病态 C 最可能的成因。C = E[x x^T] 是对"输入分布"的估计，训练时模型
# 见到的分布和采集时见到的分布只要不一致，C 就在保护一个模型实际不会遇到的方向。
# 现在的 SFT 管线（left padding + chat 模板 + 短样本）让 pad/BOS/模板分隔符这类高度重复的
# token 在位置统计里占比极高，而这些 token 的 hidden state 几乎是同一个向量——多个近乎相同
# 的 x 叠出来的 X^T X 自然接近 rank-1。这与 inspect_cov_layers 查到的
# "layers.2.mlp.down_proj 99.995% 的迹在单个方向"高度吻合。
# 定长打包的预训练管线没有 padding、没有模板 token，这个成因会自然消失——但**前提是采集器
# 也用打包数据**。如果继续用旧管线采 C、用打包数据训练，问题只会被藏得更深。
#
# 具体怎么改：
#   1. 数据源换成读 litdata chunk，也就是 pretrain.py L220 的 get_dataloaders 用的那份
#      （见 litgpt/data/text_files.py 与 lit_data.py）。理想做法是直接复用 LitGPT 的
#      DataModule 拿 train_dataloader，采集脚本只负责挂 hook 和遍历。
#   2. batch 只有 input_ids，没有 attention_mask 也没有 position_ids。
#      L315-319 那个 model_inputs 过滤要改成 model(input_ids)；
#      L314 给 attention_holder 赋值的这行**直接删掉**，理由见 core/covariance.py 里
#      make_covariance_hook 的批注（残留 mask 会通过形状守卫、静默污染 C）。
#   3. 模型换成 GPT(config) + load_raw 后的权重，且**不要 torch.compile**（hook 会失效）。
#      顺带 C 的键就是 LitGPT 命名，省掉整个映射层。
#   4. 采样量。counts 现在按 token 数累加，定长打包下每 batch 贡献
#      batch_size * seq_len 个 token，比 SFT 多一到两个数量级，达到同样的估计精度所需的
#      batch 数少很多。先扫一下"用多少 token 采 C"对 C 谱的影响再定，别沿用 SFT 的数值。
#
# 采完立刻跑 inspect_cov_layers 看谱，别等训练。如果新 C 的层间差距还是 1e4 量级，
# 那就不是管线问题，得回到 C 的定义本身（层归一化、或按 trace 归一化每层的 C）。
def collect_covariances(args: argparse.Namespace) -> None:
    """Run the full collection stage and write normalized C matrices to disk."""

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Stage 1: loading base model and tokenizer")
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
    )
    model.to(device)
    model.eval()

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    target_module_names = find_target_linear_module_names(model, target_modules)
    if not target_module_names:
        raise ValueError(f"No target Linear modules found for: {target_modules}")
    print(f"Stage 1: collecting C for {len(target_module_names)} target modules")

    dataset = load_old_knowledge_dataset(args)
    if args.require_target == 1:
        # Before limit_dataset, so --max_samples counts usable rows.
        dataset = filter_incomplete_rows(dataset, args)
    dataset = limit_dataset(dataset, args)
    if args.sample_strategy == "balanced":
        print(
            f"Stage 1: taking a seed={args.sample_seed} task-balanced sample "
            f"(mixing_rate_max={args.mixing_rate_max}) of {args.max_samples} rows"
        )
    elif args.sample_shuffle == 1:
        print(
            f"Stage 1: taking a seed={args.sample_seed} random sample of "
            f"{'all' if args.max_samples <= 0 else args.max_samples} rows"
        )
    pool_rows, pool_hash = fingerprint_pool(dataset, args)
    print(f"Stage 1: pool rows={pool_rows} fingerprint={pool_hash}")
    print("  collect_fisher must print the same value, or C and F saw different rows")

    if args.debug_print_examples > 0:
        from onereplay.data.old_knowledge import example_to_model_text

        for row in range(min(args.debug_print_examples, len(dataset))):
            rendered = example_to_model_text(dataset[row], tokenizer, args)
            print(f"---- rendered example {row} ({len(rendered)} chars) ----")
            print(rendered)
            print(f"---- contains <think>: {'<think>' in rendered} ----")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer, args),
    )

    attention_holder: dict[str, torch.Tensor | None] = {"attention_mask": None}
    cov_sums, counts, handles = register_covariance_hooks(
        model,
        target_module_names,
        attention_holder,
        args,
    )

    if args.cov_normalization == "base_output_norm":
        print(
            "Stage 1: forwarding old-knowledge data and accumulating "
            "normalized X^T X with x' = x / max(||W x||, eps)"
        )
    else:
        print("Stage 1: forwarding old-knowledge data and accumulating X^T X")
    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            attention_holder["attention_mask"] = batch.get("attention_mask")
            model_inputs = {
                key: value
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            model(**model_inputs)
            if step % 50 == 0:
                print(f"  processed batches: {step}")

    for handle in handles:
        handle.remove()

    covariances = {}
    for module_name, cov_sum in cov_sums.items():
        covariances[module_name] = cov_sum / max(counts[module_name], 1)

    metadata = {
        "model_name": args.model_name,
        "pool_rows": pool_rows,
        "pool_fingerprint": pool_hash,
        "target_modules": target_modules,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_path": args.dataset_path,
        "data_files": args.data_files,
        "dataset_split": args.dataset_split,
        "cache_dir": args.cache_dir,
        "streaming": args.streaming,
        "text_column": args.text_column,
        "input_column": args.input_column,
        "target_column": args.target_column,
        "use_chat_template": args.use_chat_template,
        "include_target_in_chat": args.include_target_in_chat,
        "system_prompt": args.system_prompt,
        "require_target": args.require_target,
        "require_target_column": args.require_target_column,
        "truncation_side": args.truncation_side or getattr(tokenizer, "truncation_side", ""),
        "max_samples": args.max_samples,
        "sample_shuffle": args.sample_shuffle,
        "sample_seed": args.sample_seed,
        "sample_strategy": args.sample_strategy,
        "task_column": args.task_column,
        "mixing_rate_max": args.mixing_rate_max,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "max_len": args.max_len,
        "cov_normalization": args.cov_normalization,
        "cov_norm_eps": args.cov_norm_eps,
    }
    save_covariance_payload(args.output_path, covariances, counts, metadata)
    print(f"Stage 1 done: saved covariance file to {args.output_path}")


def main() -> None:
    collect_covariances(parse_args())


if __name__ == "__main__":
    main()
