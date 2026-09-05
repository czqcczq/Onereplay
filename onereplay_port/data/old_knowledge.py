"""Old-knowledge corpus loading (FLAN) for covariance collection."""

# =============================================================================
# PORT NOTE 文件状态：[REFERENCE-ONLY] 整个文件都不搬
#
# onereplay/data/old_knowledge.py 的逐字节副本，只加注释、未改代码行。校验：
#     diff onereplay/data/old_knowledge.py con-pretrain/onereplay_port/data/old_knowledge.py
#
# 这份文件是整套代码里与 SFT 绑定最深的一份，440 行几乎没有一行能直接用。放进来只为一个
# 目的：说明"旧知识语料"这个概念在 CPT 下变成了什么，以及哪些旧概念**没有**对应物。
#
# 概念对照：
#   SFT 时                                CPT 时
#   ------------------------------------  ------------------------------------
#   FLAN（第一阶段的代理语料）              第一阶段预训练语料本身（真实的、不是代理的）
#   一条 example = 一个 prompt+answer      一个定长 token 块，跨文档
#   chat 模板渲染                          无，纯文本
#   left padding 到 batch 最长              无 padding，定长打包
#   按 task 配额均衡采样                    按 domain token 占比采样
#
# 第一行就值得强调：SFT 时 FLAN 只是第一阶段能力的**代理**，C 估的是代理分布；CPT 时你能拿到
# 真实的第一阶段语料（这正是选 Pythia+Pile / OLMo+Dolma 这类"预训练数据公开"模型的全部理由），
# C 估的是真实分布。这是方法论上的一次实质提升，论文里应该讲出来。
#
# 需要新写的东西很少，因为 LitGPT 的数据准备已经做了大部分：
#   - 用 litgpt/data/text_files.py 或 prepare_slimpajama.py 那套 optimize() 把第一阶段语料
#     预处理成 litdata chunk；
#   - 采集器直接读这份 chunk（见 scripts/collect_cov.py 的批注）。
#   - 唯一要自己写的是"从第一阶段语料里抽一个子集用于采 C"，那只是对 chunk 做截断/抽样，
#     不需要这个文件里的任何函数。
#
# 下面各函数的批注只标"有没有对应物"，不用细读实现。
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, load_from_disk


def load_old_knowledge_dataset(args: argparse.Namespace):
    """Load the old-knowledge corpus from disk, local files, or HuggingFace."""

    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)
        return dataset[args.dataset_split] if args.dataset_split in dataset else dataset

    if args.data_files:
        suffix = Path(args.data_files).suffix.lower()
        file_format = "text" if suffix == ".txt" else "json"
        dataset = load_dataset(
            file_format,
            data_files=args.data_files,
            split=args.dataset_split,
            cache_dir=args.cache_dir,
        )
        return dataset

    config = args.dataset_config if args.dataset_config else None
    return load_dataset(
        args.dataset_name,
        config,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        streaming=bool(args.streaming),
    )


def filter_incomplete_rows(dataset, args: argparse.Namespace):
    """Drop rows whose input or target is empty.

    Needed to keep C on exactly the rows replay trains on. The self-distilled
    corpus stores an empty target for every prompt whose generation hit the
    token cap (2565 of 20000 at max_new_tokens=384), and the replay loader drops
    those rows via --replay_drop_truncated plus to_sft_schema's non-empty
    filter. Without the same filter here, C would still absorb the prompt-side
    activations of rows replay never sees.

    --require_target_column decouples "which column decides the row survives"
    from "which column is read as the answer". They are the same by default.
    They differ in exactly one case: the gold-target ablation, which reads
    gold_targets out of the self-distilled file so that the prompts, the row
    order and the pool cut all stay bit-identical and only the assistant text
    changes. Gold is present on every row including the truncated ones, so
    filtering on gold_targets would silently hand the gold arm ~13% more rows
    and turn a target-source ablation into a pool-size one.
    """

    input_column = args.input_column
    target_column = getattr(args, "require_target_column", "") or args.target_column

    def is_complete(example: dict[str, Any]) -> bool:
        target = example.get(target_column)
        if not (target and str(target).strip()):
            return False
        if args.text_column:
            source = example.get(args.text_column)
        else:
            source = example.get(input_column)
        return bool(source and str(source).strip())

    before = len(dataset) if hasattr(dataset, "__len__") else None
    dataset = dataset.filter(is_complete)
    if before is not None:
        after = len(dataset)
        print(
            f"Stage 1: require_target dropped {before - after} of {before} rows "
            f"with an empty {target_column} or {args.text_column or input_column}"
        )
    else:
        print(f"Stage 1: require_target filtering a streaming dataset on {target_column}")
    return dataset


def allocate_task_quota(
    counts: dict[str, int],
    total: int,
    cap: int,
) -> tuple[dict[str, int], int]:
    """Split a row budget across tasks by FLAN's capped-proportional weight.

    Each task i is weighted by w_i = min(N_i, cap), so any task with at least
    `cap` rows gets equal weight and only smaller ones are down-weighted. This is
    exactly FLAN's examples-proportional mixing with a mixing rate maximum
    (Wei et al. 2021): the target share is w_i / sum_j w_j.

    The continuous target is turned into integers with the largest-remainder
    method: floor every quota, then hand the leftover rows one at a time to the
    tasks with the largest fractional part. Allocation never exceeds a task's own
    row count, so the subset can be drawn without replacement; if clamping frees
    up rows they are redistributed to tasks that still have capacity. Ordering is
    deterministic (fractional part desc, then task name), so the split is
    reproducible for a fixed corpus.
    """

    weights = {task: min(count, cap) for task, count in counts.items()}
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("all task weights are zero; check the task column")

    exact = {task: total * weights[task] / weight_sum for task in counts}
    quota = {task: min(int(exact[task]), counts[task]) for task in counts}
    allocated = sum(quota.values())

    order = sorted(counts, key=lambda task: (-(exact[task] - int(exact[task])), task))
    while allocated < total:
        progressed = False
        for task in order:
            if allocated >= total:
                break
            if quota[task] < counts[task]:
                quota[task] += 1
                allocated += 1
                progressed = True
        if not progressed:
            # Every task is at capacity: the corpus has fewer than `total` rows.
            break
    return quota, weight_sum


def _task_counts_and_indices(
    dataset,
    task_column: str,
    chunk: int = 100_000,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """One pass over the task column, returning per-task counts and row indices."""

    counts: dict[str, int] = {}
    indices: dict[str, list[int]] = {}
    total = len(dataset)
    for start in range(0, total, chunk):
        values = dataset[start : min(start + chunk, total)][task_column]
        for offset, task in enumerate(values):
            key = str(task)
            counts[key] = counts.get(key, 0) + 1
            indices.setdefault(key, []).append(start + offset)
    return counts, indices


# PORT NOTE [REFERENCE-ONLY] 无对应概念。
# allocate_task_quota / _task_counts_and_indices / balanced_task_subset 这一整套
# （原文件 L89-203，对应 --sample_strategy balanced）是按 FLAN 的 task 字段配额抽样。
# 预训练语料没有 task 标签，最接近的类比是 domain（web/code/arxiv/...）配比，
# 而那个配比应该直接取第一阶段的真实 token 占比，用 mix_covariances.py 的权重表达，
# 不需要在采样阶段做配额。整段丢弃。
def balanced_task_subset(dataset, args: argparse.Namespace):
    """Draw a task-balanced subset so C is not dominated by the largest tasks.

    Unlike the uniform path, which reproduces the corpus's raw row mixture, this
    allocates a per-task quota with allocate_task_quota and then draws that many
    rows from each task without replacement. Each task's draw uses an independent
    RNG seeded by (sample_seed, task name), so growing the corpus of one task
    does not reshuffle the others, and the whole selection is reproducible.

    Only the map-style path is supported: a streaming IterableDataset cannot be
    indexed per task without consuming it.
    """

    if not hasattr(dataset, "select"):
        raise ValueError(
            "balanced sampling needs a map-style dataset; pass --streaming 0 or "
            "point --data_files at local json/jsonl files"
        )
    task_column = getattr(args, "task_column", "task")
    if task_column not in dataset.column_names:
        raise ValueError(
            f"balanced sampling needs a {task_column!r} column; found {dataset.column_names}"
        )

    total = args.max_samples
    if total is None or total <= 0:
        raise ValueError("balanced sampling needs --max_samples > 0")
    total = min(total, len(dataset))
    cap = int(getattr(args, "mixing_rate_max", 3000))
    seed = int(getattr(args, "sample_seed", 1))

    counts, indices = _task_counts_and_indices(dataset, task_column)
    quota, weight_sum = allocate_task_quota(counts, total, cap)

    selected: list[int] = []
    for task in sorted(counts):
        rows = list(indices[task])
        random.Random(f"{seed}:{task}").shuffle(rows)
        selected.extend(rows[: quota[task]])
    # Mix tasks so batching and any debug printing see an interleaved order. This
    # does not affect C, which is an order-invariant sum over tokens.
    random.Random(seed).shuffle(selected)

    print(
        f"balanced sampling: {len(selected)} rows over {len(counts)} tasks "
        f"(cap={cap}, seed={seed}, weight_sum={weight_sum})"
    )
    return dataset.select(selected)


def limit_dataset(dataset, args: argparse.Namespace):
    """Take a reproducible subset of a map-style Dataset or streaming IterableDataset.

    --sample_strategy uniform (default) shuffles with --sample_seed and takes the
    first --max_samples rows, so the subset reproduces the corpus's raw mixture.
    Growing max_samples with the same seed keeps smaller subsets nested in larger
    ones for map-style datasets. Streaming datasets only support approximate
    buffer shuffling; the result is still deterministic for a fixed seed and
    buffer size.

    --sample_strategy balanced instead draws a per-task quota (FLAN's capped-
    proportional weighting), so C is not dominated by whichever tasks own the
    most rows. See balanced_task_subset.
    """

    if getattr(args, "sample_strategy", "uniform") == "balanced":
        return balanced_task_subset(dataset, args)

    max_samples = args.max_samples
    shuffle = args.sample_shuffle == 1

    if hasattr(dataset, "select"):
        if shuffle:
            dataset = dataset.shuffle(seed=args.sample_seed)
        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset

    if hasattr(dataset, "take"):
        if shuffle:
            dataset = dataset.shuffle(
                seed=args.sample_seed,
                buffer_size=args.shuffle_buffer_size,
            )
        if max_samples > 0:
            dataset = dataset.take(max_samples)
        return dataset

    return dataset


# PORT NOTE [REFERENCE-ONLY] 概念要保留，实现要重写——这是本文件唯一值得带走的想法。
#
# 它给"C 是在哪批数据上采的"生成一个可复现的指纹，写进 metadata。CPT 下同样需要：
# 记录 litdata chunk 的路径、chunk 索引范围、消耗的 token 数、以及 tokenizer 版本。
# 没有这个，几个月后没人能说清某个 C 文件对应哪份语料，而 C 文件是会被反复复用的。
# 实现上对 chunk 的文件名+大小做哈希就够，不需要遍历样本。
def fingerprint_pool(dataset, args: argparse.Namespace) -> tuple[int, str]:
    """Hash the selected rows so two stages can prove they saw the same corpus.

    C and F have to be estimated on the same old-knowledge rows or the OneReplay
    versus EWC comparison is confounded by which data each method got. The
    selection is deterministic on the map-style path (a full shuffle at a fixed
    seed followed by select), but "should be deterministic" and "was
    deterministic" are different claims, and a divergence caused by a library
    upgrade or an edited flag would not raise anything. Hashing the rows turns
    the assumption into a value both stages print and store.

    Streaming datasets are skipped rather than consumed: iterating an
    IterableDataset here would drain the very iterator the caller is about to
    forward through the model.
    """

    if not hasattr(dataset, "select"):
        return 0, "streaming-not-fingerprinted"

    digest = hashlib.sha256()
    rows = 0
    for example in dataset:
        if args.text_column:
            source = example.get(args.text_column)
        else:
            source = example.get(args.input_column)
        digest.update(str(source or "").encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(example.get(args.target_column) or "").encode("utf-8"))
        digest.update(b"\x01")
        rows += 1
    return rows, digest.hexdigest()


# PORT NOTE [REFERENCE-ONLY] 从这里到文件末尾（原文件 L279-440）全是渲染与 tokenize，
# 在 CPT 下整段被 LitGPT 的离线数据准备取代，不搬。
#
# 只有一点必须带走，是个陷阱：build_collate_fn（L413）用 tokenizer(..., padding=True)
# 做 batch 内 padding，而 tokenizer 在 modeling.py L52/L63 被设成 padding_side="left"。
# 于是每个 batch 的短样本前面都是一串 pad token，它们的 hidden state 几乎完全相同——
# 这些近乎重复的 x 叠进 X^T X，直接把 C 推向 rank-1。
# 现在的 hook 有 mask 分支会把 pad 位置滤掉（core/covariance.py L119-121），所以理论上没吃到
# pad；但这也正说明"C 到底统计了哪些位置"完全依赖那个 mask，而那个 mask 在新管线里会消失。
# 这是必须逐个确认 mask 分支的原因，别把它当成无害的死代码。
#
# 关于别人提醒的 padding 方案：LitGPT 预训练侧不做 padding，走的是定长打包
# （data/text_files.py L51-105 的 optimize + TokensLoader），所以"动态 padding 省时间"
# 这个建议在这里不适用——打包比动态 padding 更省，没有一个 pad token。
# 但打包有个代价：TokensLoader 只按固定长度切 token 流，**不隔离 attention**，
# 同一个块里跨文档的 token 能互相看见。LitGPT 没有实现 block-diagonal mask
# （model.py L149-153 训练路径直接把 mask 置 None）。
# 好消息是 model.py L170 的 block(x, cos, sin, mask, ...) 说明 mask 参数已经一路通到底，
# 要加隔离只需给 GPT.forward 补一个参数并把文档边界传进去。
# 是否要做取决于基座模型自己怎么预训练的：Pythia/Pile 就是不隔离的定长打包，
# 那么保持不隔离才是与第一阶段一致的选择，反而不该改。先查基座的预训练配方再决定。
# max_seq_length 同理，跟基座对齐（Pythia 是 2048），不要自己挑 4k/8k。
def example_to_plain_text(example: dict[str, Any], args: argparse.Namespace) -> str:
    """Convert one FLAN-style row into plain text.

    This is kept for ablations. The default path below uses chat template,
    because Qwen instruction tuning and your BoolQ data are chat-formatted.
    """

    if args.text_column and args.text_column in example:
        return str(example[args.text_column])

    pieces: list[str] = []
    if args.input_column in example and example[args.input_column] is not None:
        pieces.append(str(example[args.input_column]))
    if args.target_column in example and example[args.target_column] is not None:
        pieces.append(str(example[args.target_column]))

    if pieces:
        return "\n".join(piece.strip() for piece in pieces if piece.strip())

    # Fallback for unknown schemas: join string-like fields so the script still
    # works with many json/jsonl datasets without code edits.
    for value in example.values():
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())
    return "\n".join(pieces)


def example_to_messages(example: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    """Convert one FLAN row into chat messages.

    FLAN has an instruction/input field and a target field. We map them to:

        user:      inputs
        assistant: targets

    This makes the old-knowledge hidden states live in the same chat-template
    distribution as the later LoRA fine-tuning data.
    """

    user_content = ""
    if args.text_column and args.text_column in example:
        user_content = str(example[args.text_column]).strip()
    elif args.input_column in example and example[args.input_column] is not None:
        user_content = str(example[args.input_column]).strip()
    else:
        user_content = example_to_plain_text(example, args).strip()

    messages: list[dict[str, str]] = []
    if args.system_prompt.strip():
        messages.append({"role": "system", "content": args.system_prompt.strip()})
    messages.append({"role": "user", "content": user_content})

    target = ""
    if args.target_column in example and example[args.target_column] is not None:
        target = str(example[args.target_column]).strip()
    if args.include_target_in_chat == 1 and target:
        messages.append({"role": "assistant", "content": target})

    return messages


def example_to_model_text(example: dict[str, Any], tokenizer, args: argparse.Namespace) -> str:
    """Build the exact text that will be tokenized and forwarded through model.

    When use_chat_template=1, tokenizer.apply_chat_template inserts model-
    specific role tokens such as Qwen's user/assistant markers. If a tokenizer
    has no chat template, we fall back to plain text with a clear error-free path
    so the script still works for non-chat base models.
    """

    if args.use_chat_template != 1:
        return example_to_plain_text(example, args)

    if getattr(tokenizer, "chat_template", None) is None:
        return example_to_plain_text(example, args)

    # Qwen3's template drops <think>...</think> out of assistant turns, which
    # would silently discard a self-distilled reasoning trace exactly where C is
    # supposed to see it. Concatenating the generation prompt with the raw target
    # reproduces the sequence the model actually produced instead of re-rendering
    # it, so the trace survives verbatim.
    if getattr(args, "concat_prompt_target", 0) == 1 and args.include_target_in_chat == 1:
        target = str(example.get(args.target_column) or "").strip()
        if target:
            text = f"{example_to_prompt_text(example, tokenizer, args)}{target}"
            eos = tokenizer.eos_token or ""
            return text if not eos or text.endswith(eos) else text + eos

    return tokenizer.apply_chat_template(
        example_to_messages(example, args),
        tokenize=False,
        add_generation_prompt=False,
    )


def example_to_prompt_text(example: dict[str, Any], tokenizer, args: argparse.Namespace) -> str:
    """Render the same row up to the point where the assistant would answer.

    Pairs with example_to_model_text for Fisher collection: the token prefix the
    two renders share is exactly the span that must not be supervised, because
    the Fisher of an instruction-following model should measure the sensitivity
    of its responses, not of the prompts it was handed.

    Callers compare token ids rather than trusting this string's length, so a
    chat template that inserts extra tokens into the generation prompt degrades
    into a shorter mask instead of a misaligned one.
    """

    if args.use_chat_template != 1 or getattr(tokenizer, "chat_template", None) is None:
        # Plain-text ablation. example_to_plain_text joins the input and the
        # target with a newline, so the prompt is everything before that join.
        if args.text_column and args.text_column in example:
            return str(example[args.text_column]).strip()
        source = example.get(args.input_column)
        return f"{str(source or '').strip()}\n"

    messages = [
        message for message in example_to_messages(example, args) if message["role"] != "assistant"
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=getattr(args, "enable_thinking", 0) == 1,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_collate_fn(tokenizer, args: argparse.Namespace):
    """Create a DataLoader collator that tokenizes text and pads each batch."""

    # Truncation drops the assistant turn by default (transformers truncates on
    # the right), while training keeps it (`full_input_ids[-max_length:]` in
    # tokenizer_to_ids drops the prompt head instead). With FLAN's one-line gold
    # targets almost nothing overflows, so the mismatch never mattered; with
    # self-distilled answers up to 384 tokens it decides whether C sees the
    # answers at all. Set --truncation_side left to match training.
    truncation_side = getattr(args, "truncation_side", "")
    if truncation_side:
        tokenizer.truncation_side = truncation_side

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [example_to_model_text(example, tokenizer, args) for example in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_len,
            padding=True,
            return_tensors="pt",
            # Chat templates already include model-specific special tokens.
            # Plain-text ablations still use normal tokenizer special tokens.
            add_special_tokens=args.use_chat_template != 1,
        )
        return tokenized

    return collate_fn
