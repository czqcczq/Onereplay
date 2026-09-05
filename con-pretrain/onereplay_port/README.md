# OneReplay → LitGPT 移植：只复制与批注

这个目录是 `onereplay/` 里与全参数协方差路径相关的 8 个文件的**逐字节副本**。

**代码一行都没有改**，只插入了注释。这是可验证的：

```bash
# 应当只显示注释行的增加，不出现任何代码行的修改
diff -r onereplay/core            con-pretrain/onereplay_port/core
diff    onereplay/mix_covariances.py   con-pretrain/onereplay_port/mix_covariances.py
diff    onereplay/trainers/base.py     con-pretrain/onereplay_port/trainers/base.py
diff    onereplay/scripts/collect_cov.py con-pretrain/onereplay_port/scripts/collect_cov.py
diff    onereplay/data/old_knowledge.py  con-pretrain/onereplay_port/data/old_knowledge.py

# 列出全部待改点
rg "PORT NOTE" con-pretrain/onereplay_port/
```

目录结构镜像 `onereplay/`，方便逐文件对照原件。这里的代码**不可运行**（import 路径还是旧的），
它只是一份带批注的阅读材料。

## 批注标记

| 标记 | 含义 |
| --- | --- |
| `[KEEP]` | 原样可用，不需要改 |
| `[MODIFY]` | 需要改动，批注写明改什么、为什么 |
| `[DELETE]` | LoRA / EWC 相关，正式移植时删掉 |
| `[REPLACE]` | 功能保留但实现要换成 LitGPT 的方式 |
| `[REFERENCE-ONLY]` | 这个文件不搬，逻辑要在 LitGPT 里重写；批注指出对应的 LitGPT 位置 |

每个文件顶部有一个 `PORT NOTE 文件状态` 块，说明这份文件整体的去向。

## 文件去向

真的要搬（5 个）：

- `core/covariance.py` — C 的加载/保存/identity 消融/采集 hook。整体搬运，改命名与 dtype。
- `core/regularizer.py` — 惩罚项与 analytic 梯度注入。搬运，但 716 行里要删掉约 340 行的 LoRA/EWC。
- `core/modeling.py` — 只有 `snapshot_reference_weights` 要搬，其余被 LitGPT 的建模流程取代。
- `core/profiling.py` — 时间/显存计量。整体搬运，这是 LitGPT 给不了的东西。
- `mix_covariances.py` — C 的加权混合。整体搬运，只改 import 路径。

仅供对照（3 个）：

- `trainers/base.py` — 注入时序的参照。批注里有一张逐行对应到 `litgpt/pretrain.py` 的表。
- `scripts/collect_cov.py` — stage 1 入口。骨架照抄、数据源重写。
- `data/old_knowledge.py` — SFT 语料渲染，整份丢弃；批注只解释概念对照。

## 三个必须先处理的风险

按危险程度排序，都是**静默失效**——不报错、不 NaN、loss 曲线正常，但结果是错的：

1. **`lookup_covariance` 查不到不报错**（`core/regularizer.py`）。HF 与 LitGPT 的模块名没有一个字符串重合，查不到只会记进 `missing_layers`，惩罚变 0，整个 run 等于 baseline。移植时必须加硬断言。
2. **`snapshot_reference_weights` 的调用时机**（`core/modeling.py`）。必须在 `pretrain.py` L223-224 的 `fabric.load_raw` 之后；早于它就会把随机初始化当成 W₀，惩罚方向完全反掉，而"加了正则效果变差"是个看起来合理的结论。
3. **`fabric.clip_gradients` 没有对应物**（`trainers/base.py`）。`pretrain.py` L360 的裁剪一定是开着的（L62 默认 `max_norm=1.0`），惩罚梯度会一起被裁，等于 λ 被隐式缩放，且各臂系数不同。

另有两个不是 bug 但会影响结论的点：λ 需随目标层数换算（q/k/v 融合导致 197→141，选 Pythia 可避免），
以及 `gradient_accumulation_iters` 从 SFT 的 8 变成 128，惩罚开销被摊薄 16 倍——成本表必须写明这个值。

## LitGPT 阅读顺序

每一步回答一个具体问题，后一步的问题由前一步引出。

1. **装配顺序** — `litgpt/pretrain.py` L44-230。
   正则器在哪构造、W₀ 在哪快照。注意 L200-224 的顺序：`GPT(config)` → `initialize_weights` → `tie_embeddings`(L205) → `torch.compile`(L213) → `fabric.setup`(L214) → optimizer(L216) → dataloaders(L220) → `load_raw`(L223)。**权重最后才到位**，这决定快照点。

2. **注入点** — 同文件 `fit()` L336-375，重点 L348-362。
   惩罚加在哪。`is_accumulating`(L351) 的语义和你现在的 `window_open` **相反**；L360 的 `clip_gradients` 是新出现的东西。

3. **累积算术** — `litgpt/args.py` L8-73。
   窗口有多大。`gradient_accumulation_iters`(L56-60) = `global_batch_size // (devices*num_nodes) // micro_batch_size`；`pretrain.py` L59-60 默认 512/4，单卡即 **128**。

4. **模块树与 mask 路径** — `litgpt/model.py` L85-181。
   层叫什么、能不能隔离 attention。L152 训练路径硬置 `mask = None`；L170 的 `block(x, cos, sin, mask, ...)` 说明管道已通到底，只差 `GPT.forward` 加一个参数。

5. **qkv 融合与 MLP 命名** — 同文件 L390-400（`self.qkv` 是单个 Linear）、L804-831（`GptNeoxMLP` 是 `fc`/`proj`，`LLaMAMLP` 是 `fc_1`/`fc_2`/`proj`）。
   C 的键怎么映射。`config.py` 默认 `mlp_class_name="GptNeoxMLP"`，Pythia 走两个 MLP Linear。

6. **数据管线** — `litgpt/data/text_files.py` 全文，重点 L51-105（`optimize` + `TokensLoader`）和 L135-139（`encode(text, bos=True, eos=False)`）。
   一个 batch 长什么样、BOS 在哪、为什么没有文档边界信息。可对照 `lit_data.py` 和 `prepare_slimpajama.py`。

7. **命名映射的权威来源** — `litgpt/scripts/convert_hf_checkpoint.py` L35-57（GPT-NeoX/Pythia，**1:1 无融合错配**）、L726-744（`qkv_reassemble` 沿 dim 0 置换输出行，**不影响 C**）、L641-707（Qwen3）。
   比手写的名字表可靠。

8. **精度与 optimizer** — `litgpt/utils.py` L363-380（默认 `bf16-mixed`，fp32 主权重 + autocast）、L636-668（`instantiate_torch_optimizer` 始终传 `model.parameters()`，**不支持 param groups**；要分组得改 `pretrain.py` L217）。
