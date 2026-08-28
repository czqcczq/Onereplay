# 遗忘保护：数学线与代码线

本文档记录同一套方法（OneReplay，激活协方差正则）在**两个能力域**上的保护效果：先做的数学线（MATH-500 / GSM8K），后做的代码线（HumanEval / MBPP）。两条线共用同一个新任务（Commonsense170k）与同一套训练超参，**唯一变量是正则项里的 C**，因此可以并排读。

| 线 | 评测产出 | 作业 |
|---|---|---|
| 数学线 | 2026-08-24，配对诊断补于 08-26 | `33_math_eval_base_vanilla.pbs`（base+vanilla）、`32_math_train_eval.pbs`（cmix） |
| 代码线 | 2026-08-28 | `39_code_eval_base_vanilla.pbs`（base+vanilla）、`41_code_data_cov.pbs`（采 C_code）、`42_code_train_eval.pbs`（cif+cifcode） |

数学线的配对诊断工具是 `onereplay/scripts/diagnose_math_pairs.py`，按 `question` 配对各 run 的 `responses.jsonl`，做逐题 McNemar、换抽取口径重打分、并按 baseline 是否写出格式标记把增益切开。代码线的失败分桶工具是 `onereplay/scripts/analyze_code_failures.py`。

---

## 结论提要

### 代码线

- **代码线上遗忘真实发生了，且被保护住了。** vanilla 相对 base 在 MBPP 掉 3.80pp（p=0.0319，显著）；cifcode（C_if+C_code）把差距收到 −2.20pp，HumanEval 从 −6.10pp 收到 −3.05pp。
- **坏掉的是输出结构，不是算法。** MBPP 上 vanilla 的语法错误从 base 的 0.4%（2 题）暴涨到 8.6%（43 题，约 21×），而断言失败几乎不动（37.0% → 36.4%）。cifcode 把语法错误压回 0.0%。
- **C_code 是必要的，这是代码线最强的发现。** HumanEval 上纯 C_if（cif）不但没保护住，反而是四个 run 里最差的——42.68%，比 vanilla 还低 6.1pp，语法错误飙到 15.9%（26 题里 23 题是丢了 `def` 头的裸函数体）。加上 C_code 后回到 51.83%、语法错误 2.4%。
- **待复核**：cif 在 HumanEval 上的这一格反常，需先确认评的是正确的 balanced C_if adapter（见第 7 节）。
- **pass@1 不是主量具**，量程不够；判据要靠 40 号代码 CE 探针的漂移倍数与 bootstrap 配对差。

### 数学线

- **数学解题能力没有被遗忘。** 在 vanilla 正常作答的子集上，OneReplay 与 vanilla 是 72.6% vs 72.2%（MATH-500，p=0.905）与 77.9% vs 80.1%（GSM8K，p=0.094）。三个 run 在 MATH-500 上两两对比全部不显著，其中一个 p 恰为 1.0000。
- **被遗忘、也被保护的是输出格式与终止行为。** vanilla 在 GSM8K 上有 22.4% 的题完全不写 `####`，OneReplay 是 0.2%，与 base 持平。严格口径下 vanilla 从 75.44% 塌到 45.41%。
- **GSM8K 判分器的兜底把这 30 个点的损伤压缩成了 1.67pp。** 只读 `summary.json` 完全看不到格式崩溃。
- **"保护后超过 base"不成立。** MATH-500 上 p=1.0000，GSM8K 上 p=0.1477，两个都不显著。正确的表述是"恢复到与 base 不可区分"。
- **已知缺陷**：GSM8K 的 base/vanilla 跑在 cap=1792，cmix 跑在 4096，尚未补评。见第 3.4 节。

### 两条线的对照

数学线上"要保护的能力"（数学解题）**从未被破坏**，所以那批数据无法回答 C_math 有没有价值。代码线上"要保护的能力"（写出可执行代码）**确实被破坏了**，因此代码线才是第一条能真正检验混合 C 的实验线。

---

## 1. 配置

### 1.1 共同部分

Qwen3-1.7B + LoRA（r=8, α=16, `q_proj,v_proj`），新任务 Commonsense170k，3 epoch，lr 1e-4，micro-batch 8，accum 64，max_len 512，seed=1，`reg_once_per_update=1`，λ=3e-2。所有 run 之间**只差正则项里的 C**。

评测一律 greedy（`do_sample=False`）、`enable_thinking=False`、零 few-shot、单轮 chat。

### 1.2 run 清单

| 简称 | run 名 | 正则 / C | 用于 |
|---|---|---|---|
| base | — | 未训练 | 两线 |
| vanilla | `cs_vanilla_seed1` | 无 | 两线 |
| cmix（if+math） | `cs_onereplay_ifmath_lam3e-2_seed1_regonce` | C_mix = 0.5·C_if + 0.5·C_math | 数学线 |
| cif（纯 C_if） | `cs_onereplay_balanced_lam3e-2_seed1_regonce` | 仅 C_if | 代码线 |
| cifcode（if+code） | `cs_onereplay_ifcode_lam3e-2_seed1_regonce` | C_mix = 0.5·C_if + 0.5·C_code | 代码线 |

各 C 的来源：

| C | 语料 | 文件 |
|---|---|---|
| C_if | FLAN 20k | `cov_flan_chat_20k_qv.pt` |
| C_math | MetaMath 自蒸馏 30k（按 `original_question` 去重、每题一个增广视角） | `cov_math_metamath30k_qv.pt` |
| C_code | Magicoder OSS-Instruct 20k Python 自蒸馏 | `cov_code_magicoder20k_qv.pt` |
| C_mix(if+math) | 等权凸组合（0.5 / 0.5） | `cov_if_math_..._mix_qv.pt` |
| C_mix(if+code) | 等权凸组合（0.5 / 0.5） | `cov_if_code_magicoder20k_mix_qv.pt` |

混合走 `onereplay/mix_covariances.py` 做协方差层面的线性组合，**不是把两份语料混在一起重采**。两条线的权重都是 `W_IF=0.5` 加 `W_MATH`/`W_CODE=0.5` 且带 `--normalize_weights 1`（31 号 / 41 号作业），所以是**凸组合而非两个矩阵相加**：C_mix 的 trace 是两者的加权平均，λ=3e-2 与纯 C_if 的 run 同强度量级。

C_code 池的构建（`prepare_magicoder_ccode.py`）：Magicoder OSS-Instruct-75K 过滤 Python、要求带 ``` 代码围栏、去重、对 HumanEval/MBPP 做 8-gram 重叠污染剔除、`max_prompt_tokens=1024`，随机抽 20k。自蒸馏用 `MAXLEN=2048 / MAX_NEW_TOKENS=1280`（100 条长度诊断确认截断率 0，target P99=1113、prompt+target max=1504）。

### 1.3 判分口径

| 指标 | 题数 | prompt 要求 | 抽取 | 兜底 |
|---|---:|---|---|---|
| MATH-500 | 500 | 结尾给 `\boxed{...}` | 最后一个 `\boxed`，Hendrycks `is_equiv` | **无**，抽不到即记错 |
| GSM8K | 1319 | 结尾给 `#### <数字>` | `####` 后最后一段数字 | **有**，无标记时取全文最后一个数字 |
| HumanEval | 164 | 补全函数体 | 首个 `\ndef ` 处截断，沙箱执行 `check(candidate)` | 无 |
| MBPP（test） | 500 | 写出函数 | 同上，沙箱跑 assert 列表 | 无 |

GSM8K 的兜底差异是理解数学线全部结果的关键。代码线判分器带黑名单且会在首个 `\ndef ` 处截断，**绝对分数低于公开报告值，只在本文档内做组间对比，不要外引**。代码评测预算 `code_max_new_tokens=512`、`timeout=10s`，四个 run 同预算。

---

# 第一部分：结果数据

## 2. 代码线：HumanEval / MBPP

### 2.1 pass@1

| run | HumanEval | 95% CI | 通过数 | MBPP | 95% CI | 通过数 |
|---|---:|---|---:|---:|---|---:|
| base | 54.88% | [47.24, 62.30] | 90/164 | 43.00% | [38.73, 47.38] | 215/500 |
| vanilla | 48.78% | [41.25, 56.37] | 80/164 | 39.20% | [35.02, 43.55] | 196/500 |
| cif | 42.68% | [35.37, 50.34] | 70/164 | 39.20% | [35.02, 43.55] | 196/500 |
| cifcode | 51.83% | [44.23, 59.35] | 85/164 | 40.80% | [36.58, 45.16] | 204/500 |

相对 base 的差（pp）：

| run | HumanEval | MBPP |
|---|---:|---:|
| vanilla | −6.10 | −3.80 |
| cif | **−12.20** | −3.80 |
| cifcode | **−3.05** | **−2.20** |

### 2.2 配对 McNemar（vs base）

| 对比 | 指标 | Δpass@1 | b（赢） | c（输） | discord | p |
|---|---|---:|---:|---:|---:|---:|
| vanilla vs base | HumanEval | −6.10pp | 10 | 20 | 30 | 0.0987 |
| vanilla vs base | MBPP | −3.80pp | 26 | 45 | 71 | **0.0319** |

cif / cifcode 对 base、以及 cif↔cifcode 的配对检验**本轮未产出**，见第 7 节。

### 2.3 失败原因分布（占全部题目的比例）

**HumanEval（164 题）**

| run | blocked | timeout | 语法错误 | 断言失败 | 名字/属性错误 | 类型/取值错误 | 其他 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.0% | 0.0% | 0.6% | 34.1% | 5.5% | 4.9% | 0.0% |
| vanilla | 0.0% | 0.6% | 1.8% | 39.0% | 6.7% | 3.0% | 0.0% |
| cif | 0.0% | 0.6% | **15.9%** | 32.3% | 3.0% | 5.5% | 0.0% |
| cifcode | 0.0% | 0.6% | 2.4% | 35.4% | 4.9% | 4.9% | 0.0% |

**MBPP（500 题）**

| run | blocked | timeout | 语法错误 | 断言失败 | 名字/属性错误 | 类型/取值错误 | 其他 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.0% | 0.2% | 0.4% | 37.0% | 11.6% | 7.2% | 0.6% |
| vanilla | 0.0% | 0.2% | **8.6%** | 36.4% | 5.8% | 9.8% | 0.0% |
| cif | 3.2% | 1.0% | 0.0% | 32.2% | 17.6% | 6.2% | 0.6% |
| cifcode | 0.8% | 0.8% | 0.0% | 35.2% | 15.6% | 6.0% | 0.8% |

换算成题数（便于对照）：

| 桶 | 指标 | base | vanilla | cif | cifcode |
|---|---|---:|---:|---:|---:|
| 语法错误 | HumanEval | 1 | 3 | **26** | 4 |
| 语法错误 | MBPP | 2 | **43** | 0 | 0 |
| 断言失败 | HumanEval | 56 | 64 | 53 | 58 |
| 断言失败 | MBPP | 185 | 182 | 161 | 176 |

### 2.4 语法错误成因（HumanEval，只看语法错误那些题）

| run | 语法错误题数 | 纯散文无代码 | 残留 markdown | 只剩单行片段 | 有代码但无函数定义 | 有函数定义但语法坏了 | 空输出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cif | 26 | 0 | 0 | **17** | **6** | 3 | 0 |
| cifcode | 4 | 0 | 0 | 2 | 0 | 2 | 0 |

前四类属于"代码本身可能是好的，丢的是『只输出代码』这条指令的服从度"；最后的"有函数定义但语法坏了"才是真写错。**cif 的 26 题里 23 题（88%）落在服从度那一侧。**

cif 的样例（截 240 字符）：

```
[HumanEval/2] return number - math.floor(number)
[HumanEval/4] mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)
[HumanEval/7] return [s for s in strings if substring in s]
```

三条都是**合法的函数体，但没有 `def` 头**——模型把签名丢了。

---

## 3. 数学线：MATH-500 / GSM8K

`production` 是评测器实际使用的口径，即 `summary.json` 里的数字。`lenient` 给 MATH-500 加上"无 boxed 时取最后一个数字"的兜底，`strict` 要求 GSM8K 的数字必须跟在 `####` 之后。MATH-500 的 strict 与 production 同义（都只认 boxed），列出仅为对齐格式。

### 3.1 准确率与配对检验

**MATH-500（500 题）**

| run | production | 95% CI | lenient | 正确数 |
|---|---:|---|---:|---:|
| base | 69.20% | [65.02, 73.09] | 69.40% | 346 |
| vanilla | 68.40% | [64.20, 72.32] | 68.40% | 342 |
| cmix | 69.40% | [65.23, 73.28] | 69.40% | 347 |

| 对比 | Δ | b（赢） | c（输） | discord | p |
|---|---:|---:|---:|---:|---:|
| vanilla vs base | −0.80pp | 38 | 42 | 80 | 0.7376 |
| cmix vs base | +0.20pp | 33 | 32 | 65 | **1.0000** |
| cmix vs vanilla | +1.00pp | 41 | 36 | 77 | 0.6488 |

**GSM8K（1319 题）**

| run | production | 95% CI | strict | 正确数 |
|---|---:|---|---:|---:|
| base | 77.10% | [74.76, 79.29] | 76.27% | 1017 |
| vanilla | 75.44% | [73.04, 77.68] | **45.41%** | 995 |
| cmix | 78.70% | [76.40, 80.82] | 77.86% | 1038 |

| 对比 | Δ | b（赢） | c（输） | discord | p |
|---|---:|---:|---:|---:|---:|
| vanilla vs base | −1.67pp | 105 | 127 | 232 | 0.1678 |
| cmix vs base | +1.59pp | 106 | 85 | 191 | 0.1477 |
| cmix vs vanilla | +3.26pp | 132 | 89 | 221 | **0.0046** |

### 3.2 抽取健康度

| 指标 | run | 抽不出答案 | 缺格式标记 | 回复长度 P99（字符） |
|---|---|---:|---:|---:|
| MATH-500 | base | 26（5.2%） | 26（5.2%） | 10,633 |
| MATH-500 | vanilla | 30（6.0%） | 29（5.8%） | 14,876 |
| MATH-500 | cmix | 20（4.0%） | 19（3.8%） | 13,011 |
| GSM8K | base | 0（0.0%） | 3（0.2%） | 1,134 |
| GSM8K | vanilla | 1（0.1%） | **295（22.4%）** | **4,785** |
| GSM8K | cmix | 0（0.0%） | 3（0.2%） | 1,117 |

### 3.3 增益来源分解

把全部题目按 **vanilla 是否写出格式标记**切成两个分区，在每个分区里单独做配对比较。

**MATH-500（cmix vs vanilla，总净差 +5 题）**

| 分区 | 题数 | vanilla | cmix | b | c | 净 | 占总 Δ | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla 写出 `\boxed` | 471 | 72.6% | 72.2% | 34 | 36 | −2 | −40% | 0.9050 |
| vanilla 未写 `\boxed` | 29 | 0.0% | 24.1% | 7 | 0 | +7 | 140% | 0.0156 |

**GSM8K（cmix vs vanilla，总净差 +43 题）**

| 分区 | 题数 | vanilla | cmix | b | c | 净 | 占总 Δ | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla 写出 `####` | 1024 | 77.9% | 80.1% | 90 | 68 | +22 | 51% | 0.0945 |
| vanilla 未写 `####` | 295 | 66.8% | 73.9% | 42 | 21 | +21 | 49% | **0.0111** |

### 3.4 已知缺陷：GSM8K 的解码预算不一致

`summary.json` 不记录 `MATH_MAX_NEW_TOKENS`，但截断的 run 会把最长回复钉在预算上，所以逐题文件的 token max 就能反推当时用的是多少。

| 指标 | run | token P50/P90/P99/max | 顶到自身 max 的题数 | 字符 max | mtime |
|---|---|---|---:|---:|---|
| MATH-500 | base | 未测 | 未测 | 15,897 | 08-24 07:51 |
| MATH-500 | vanilla | 528/2209/4096/**4096** | 32（6.4%） | 16,820 | 08-24 11:54 |
| MATH-500 | cmix | 562/1835/4096/**4096** | 22（4.4%） | 18,805 | 08-24 16:21 |
| GSM8K | base | 123/218/399/**1792** | 未用新口径测 | 5,842 | 08-24 01:07 |
| GSM8K | vanilla | 132/234/1792/**1792** | 16（1.2%） | 7,663 | 08-24 01:07 |
| GSM8K | cmix | 123/214/450/**4096** | 3（0.2%） | 16,667 | 08-24 19:17 |

---

# 第二部分：分析

## 4. 代码线分析

### 4.1 损伤集中在输出结构，算法没坏

MBPP 上 vanilla 相对 base 掉的 3.80pp（p=0.0319，是本文档里唯一一个 vanilla 对 base 的显著退化），几乎完全由语法错误桶承担：**2 题 → 43 题，约 21 倍**；同期断言失败从 37.0% 到 36.4%，纹丝不动。

这个分布形状的含义是明确的：模型**仍然会写算法**，坏的是"把代码按要求的形式吐出来"这件事。cifcode 把 MBPP 语法错误从 vanilla 的 8.6% 压回 **0.0%**（低于 base 的 0.4%），断言失败桶保持在与 base 相当的水平——**这是"保住输出结构"这一主张最直接的机制证据。**

需要说清的是两个基准的损伤位置不同：**MBPP 的语法桶是被 vanilla 炸开的**（0.4% → 8.6%），而 **HumanEval 的语法桶 vanilla 几乎没动**（0.6% → 1.8%），那里炸开的是 cif（15.9%，见 4.2）。所以"cifcode 修好了语法桶"这句话在 MBPP 上是对 vanilla 而言，在 HumanEval 上是对 cif 而言。

这与数学线第 5.2 节的现象是同一件事：GSM8K 上 22.4% 的题不写 `####`、MATH-500 上截断率上升、IFEval −11.46pp、Multi-IF −18.00pp、代码上语法错误 21 倍——**五处都在说"指令服从与终止行为被 fine-tune 破坏了"。**

### 4.2 纯 C_if 在 HumanEval 上反而最差，这是 C_code 必要性的直接证据

代码线最有信息量的一格：

| run | HumanEval pass@1 | HumanEval 语法错误 |
|---|---:|---:|
| base | 54.88% | 0.6%（1 题） |
| vanilla | 48.78% | 1.8%（3 题） |
| **cif** | **42.68%** | **15.9%（26 题）** |
| cifcode | 51.83% | 2.4%（4 题） |

**纯 C_if 不但没保护住 HumanEval，反而比不加保护的 vanilla 还差 6.1pp。** 机制在 2.4 节的成因表里：26 道语法错误中 17 道"只剩单行片段"、6 道"有代码但无函数定义"，即模型输出了合法的函数体却丢掉了 `def` 头。这是一种很特定的服从度失效——C_if 把模型往"FLAN 式的简短直接回答"方向拉，恰好与 HumanEval"请补全并输出完整函数"的要求冲突。

加入 C_code 之后这一桶从 26 题塌到 4 题（其中仅 2 题是真写坏），pass@1 回到 51.83%。**cifcode 相对 cif 在 HumanEval 上 +9.15pp、在 MBPP 上 +1.60pp。**

MBPP 上则是另一幅图景：cif 和 cifcode 都把语法错误清零，但 cif 的 pass@1 并没有比 vanilla 高——清零省下来的题没有变成通过，而是迁移到了别的失败桶。以 base 为基准看更清楚：

| MBPP 桶 | base | vanilla | cif | cifcode |
|---|---:|---:|---:|---:|
| 语法错误 | 0.4% | **8.6%** | 0.0% | 0.0% |
| 名字/属性错误 | 11.6% | 5.8% | **17.6%** | 15.6% |
| blocked | 0.0% | 0.0% | **3.2%** | 0.8% |

vanilla 的名字/属性错误（5.8%）反而低于 base，是因为它的失败被吸进了语法桶——题在更早的阶段就挂了，跑不到报 `NameError` 那一步。cif 与 cifcode 把语法桶清零之后，这些题得以执行，于是失败重新暴露在名字/属性桶里，双双高于 base。**cifcode 并非没有这个副作用，只是更轻**（15.6% vs 17.6%，blocked 0.8% vs 3.2%），净效果是 pass@1 高出 cif 1.60pp。

结论：**在代码域，光有 C_if 不够，C_code 提供了 C_if 不覆盖的方向。** 这正是数学线无法回答的那个问题（见 4.3 与第 6 节）。

### 4.3 量具限制：pass@1 不足以定案

按 42 号作业文件头预登记的判据，pass@1 的量程本就不够：MBPP 实测 vanilla 掉 3.80pp，而预期能救回的量级约 2.28pp，落在噪声边缘。HumanEval 只有 164 题，配对可分辨下限约 `1.96×√(discord率/n)`，只能读出十几个点以上的差异——vanilla vs base 的 −6.10pp 就已经不显著（p=0.0987）。

因此：

- **主量具是 40 号代码 CE 探针**，看 cifcode 把漂移倍数压回多少，以及 bootstrap 里 cif vs cifcode 的配对差是否显著。
- 本轮**没有产出** cif / cifcode 对 base 的 McNemar，也没有 cif↔cifcode 的配对检验。上面 4.2 的 +9.15pp 目前只是点估计，未做显著性。
- **cif 在 HumanEval 上的反常需要先排除 adapter 张冠李戴。** Hopper 上纯 C_if 是 `cs_onereplay_balanced_...` 那一份（不是 32 号默认的 RWTH 命名），名字对不上时 `eval_one` 只打一行"缺 adapter"就静默跳过。下"C_if 单独对代码有害"这个结论之前，必须确认评的是对的那份权重。

---

## 5. 数学线分析

### 5.1 数学解题能力没有发生可测的遗忘

MATH-500 三个两两对比全部不显著。`discord` 说明两个模型在 13%–16% 的题上给出不同结果，净差却只有 ±1~4 题——这是纯抖动。

**这个指标的分辨率是 3.5pp**（`1.96×√(0.16/500)`），而要测的遗忘量是 0.8pp。要让 0.8pp 达到 80% 功效需要约 20,000 道题。**MATH-500 在当前配置下不可能给出显著结果，这是实验设计的问题，不是结果的问题。**

GSM8K 上唯一显著的是 cmix vs vanilla（+3.26pp，p=0.0046）。与 base 的两个对比都不显著，所以能写的句子是"OneReplay 相对 vanilla 显著恢复了 GSM8K 上的损失，恢复到与 base 不可区分的水平"，**不能写"超过 base"**。分辨率是 2.26pp（`1.96×√(0.176/1319)`），遗忘量 1.67pp 与 cmix 相对 base 的 1.59pp 都在门槛以下；要让 1.67pp 达到 80% 功效需要约 5,000 道题。

### 5.2 被破坏又被保护的是格式与终止行为

**vanilla 在 GSM8K 上有 22.4% 的题完全不写 `####`，而 base 和 cmix 都是 0.2%。** 这不是数学出错，是不再服从"结尾用 `#### <数字>` 收束"这条指令。中位长度三者接近（322/383/339 字符），炸开的只有尾部：vanilla 的 P99 是 base 与 cmix 的 4 倍。也就是说在大约五分之一的题上，vanilla 进入了跑题或复读，再没回到要求的收尾。

`strict` 与 `production` 相差 30.03pp，即 **396 道题的分完全来自兜底**：295 道没有标记，另约 100 道写了 `####` 但后面没跟上数字（多半刚写完分隔符就被截断）。GSM8K 的答案通常就是推理里的最后一个数，所以兜底在很多题上碰巧抓对了，**把 30 个点的格式崩溃压缩成了 1.67pp 的准确率下降**。

这样反而多出一个方法论上的观察：**带宽松兜底的判分器会把 30 个点的格式崩溃压缩成看起来像噪声的 1.67pp。** 只看 `summary.json` 的人永远发现不了。

### 5.3 增益来源分解与方法论限制

**MATH-500：** 在 vanilla 正常作答的 471 道题上，两者是 72.6% vs 72.2%，p=0.905，方向甚至是负的。那 +1.00pp 全部来自 vanilla 彻底没给答案的 29 道题里 cmix 抢回的 7 道。vanilla 在该分区是 0.0%，那是判分规则决定的（无 boxed 必错），所以那一档根本不是数学指标。**这条是数学线最干净的证据——两个 run 同预算，无混淆。**

**GSM8K：** 增益大致对半开，不是纯格式效应。但两点要注意：干净的那一档本身不显著（+2.2pp，p=0.0945），信号强度与 MATH-500 的 471 道题一致——正方向、测不出来；而显著的那一档正好压在 3.4 节的预算缺陷上。

**预算缺陷的影响：** MATH-500 三个 run 同为 4096，可比。base 没有跑 token 化，但它的字符 max 15,897 与另两个 run 的 16,820 / 18,805 同量级（cmix 已确认对应 4096 token），而 1792 token 只能对应 7 千字符上下；加上 no-answer 率从 1792 时代的 11.6% 降到 5.2%，可以确认它已按 `33_math_eval_base_vanilla.pbs` 文件头的步骤归档重评过。

**GSM8K 的 base/vanilla 停在 1792，cmix 是 4096，不可比。** 影响上界：vanilla 比 cmix 多截断约 13 题（0.99pp）。按 `probe_math500_budget.py` 的经验（放开预算后仅约 10% 的截断样本转正），期望影响只有 1–2 题。但这个偏差落点很糟——被截断的回复写不出结尾的 `####`，所以那 16 题几乎必然全在"无 `####`"那个分区里，而该分区净差只有 +21。最坏情况下剔掉这 16 道，净差退到 +5，p 从 0.0111 回到 0.5 量级。**这个显著性经不起补评，补完再引用。**

**切分的方法论限制：** 这个切分是**看过数据之后、按 baseline 的行为**做的，不是预注册的。两个分区难度并不相同——无标记那档双方都更低（66.8/73.9 vs 77.9/80.1），说明 vanilla 恰恰在难题上失控。分区级的 p 值应当按描述性数字读，没有做多重比较校正。真正可引用的是"有标记"分区，因为它是 baseline 行为正常的子集。

---

## 6. 两条线合起来看

| | 数学线 | 代码线 |
|---|---|---|
| 目标能力被破坏了吗 | **否**（干净分区 72.6→72.2、77.9→80.1，均不显著） | **是**（MBPP −3.80pp，p=0.0319） |
| 破坏形式 | 格式与终止行为（缺 `####` 22.4%、截断率↑） | 输出结构（语法错误 21×） |
| 保护有效吗 | 格式层面有效（22.4% → 0.2%），能力层面无从判断 | 有效（语法错误回 0，pass@1 收回大半） |
| 能否检验混合 C 的价值 | **不能**——要保护的东西从未坏过 | **能**——且已给出 C_code 必要的初步证据 |

数学线的定位应改为「**用数学基准测出的格式与终止行为保持**」，双口径并排报告：`production` / 干净分区说明数学能力未受损，`no-marker` + `strict` + 截断率说明格式行为被摧毁又被救回。`C_math` / `C_mix` 相对纯 `C_if` 有没有价值，那批数据无法回答——混合 C 想要保护的那个东西（数学能力）从未被破坏过。

**代码线补上了这个缺口。** 它同时满足两个条件：目标能力确实退化（MBPP 显著），且有纯 C_if 对照（cif）与混合 C（cifcode）并排。目前的点估计指向"C_code 必要"，但要正式成立还缺 4.3 列出的三件事。

---

## 7. 数据缺口

### 代码线

| 项 | 状态 |
|---|---|
| 40 号代码 CE 探针（base/vanilla/cif/cifcode） | **未做，阻塞主结论**。pass@1 量程不够，判据在探针的漂移倍数与 bootstrap 配对差 |
| cif / cifcode 对 base 的 McNemar、cif↔cifcode 配对检验 | 未做。4.2 的 +9.15pp 目前只是点估计 |
| 复核 cif 用的是正确的 balanced C_if adapter | **未做，阻塞 4.2 的结论**。名字对不上时 `eval_one` 会静默跳过 |
| 纯 C_code run（`cs_onereplay_code_lam3e-2_seed1_regonce`） | 未做。42 号里 `TRAIN_KINDS="code mix"` 即可补，用于拆开 C_if 与 C_code 各自的贡献 |
| cifcode 的 IF 侧（IFEval / Multi-IF / commonsense） | 未做。42 号 `WITH_IF=1`。C_mix 的主张是两边同时保住，为代码牺牲 IF 不算成功 |
| seed 2 / 3 | 未做 |

### 数学线

| 项 | 状态 |
|---|---|
| GSM8K 的 base/vanilla at 4096 | **未做，阻塞 3.3 节 GSM8K 分区的显著性结论**；现为 1792，与 cmix 的 4096 不可比 |
| 纯 C_if run 的数学评测 | 未做。没有它就无法判断混入 C_math 是否带来任何变化 |
| 纯 C_math run（`cs_onereplay_math_lam3e-2_seed1_regonce`）的数学评测 | 未做 |
| MATH-500 base 的 token 长度 / 截断率 | 未测（诊断只对 vanilla 与 cmix 加了 `--model_path`） |
| AMC（83 题）/ AIME（60 题） | **放弃。** 非 thinking + greedy 下 Qwen3-1.7B 在这两个集上接近地板，样本量又只有几十，任何方法差异都不可分辨 |
| 能让数学遗忘真实发生的配置 | 未做。候选：全参微调（`11_train_full_baselines.slurm`）或 LoRA 扩到 all-linear |
| 数学侧的 held-out 探针（CE 口径） | 未做。基建已有（`27_probe_heldout_curve.slurm`），MetaMath 自蒸馏池已在 31 号产出，切一个池外分片即可 |

---

## 附 A：复现诊断

**数学线配对诊断**（只读 `responses.jsonl`，不需要 GPU；`--model_path` 只用来算 token 长度，可省略）：

```bash
R=/scratch/weiliu87/student/czq/Onereplay/results

# vs base
python -m onereplay.scripts.diagnose_math_pairs \
  --results_root $R --metrics math500,gsm8k \
  --runs base,cs_vanilla_seed1,cs_onereplay_ifmath_lam3e-2_seed1_regonce \
  --baseline base --cap 4096 --model_path $REPO_ROOT/models/Qwen3-1.7B

# vs vanilla（增益来源分解看这个）
python -m onereplay.scripts.diagnose_math_pairs \
  --results_root $R --metrics math500,gsm8k \
  --runs cs_vanilla_seed1,cs_onereplay_ifmath_lam3e-2_seed1_regonce \
  --baseline cs_vanilla_seed1 --cap 4096 --model_path $REPO_ROOT/models/Qwen3-1.7B
```

**代码线失败分桶**：

```bash
python -m onereplay.scripts.analyze_code_failures \
  --results_root $R --metrics humaneval,mbpp \
  --runs base,cs_vanilla_seed1,cs_onereplay_balanced_lam3e-2_seed1_regonce,cs_onereplay_ifcode_lam3e-2_seed1_regonce \
  --baseline base
```

---

## 附 B：原始 summary.json

路径统一省略前缀 `/scratch/weiliu87/student/czq/Onereplay`。

### HumanEval

```json
{
  "metric": "humaneval", "run_name": "base",
  "num_examples": 164, "passed": 90, "pass_at_1": 0.5487804878048781
}
{
  "metric": "humaneval", "run_name": "cs_vanilla_seed1",
  "num_examples": 164, "passed": 80, "pass_at_1": 0.4878048780487805
}
{
  "metric": "humaneval", "run_name": "cs_onereplay_balanced_lam3e-2_seed1_regonce",
  "num_examples": 164, "passed": 70, "pass_at_1": 0.4268292682926829
}
{
  "metric": "humaneval", "run_name": "cs_onereplay_ifcode_lam3e-2_seed1_regonce",
  "adapter_path": ".../adapters/cs_onereplay_ifcode_lam3e-2_seed1_regonce",
  "num_examples": 164, "passed": 85, "pass_at_1": 0.5182926829268293
}
```

### MBPP

```json
{
  "metric": "mbpp", "run_name": "base",
  "num_examples": 500, "passed": 215, "pass_at_1": 0.43
}
{
  "metric": "mbpp", "run_name": "cs_vanilla_seed1",
  "num_examples": 500, "passed": 196, "pass_at_1": 0.392
}
{
  "metric": "mbpp", "run_name": "cs_onereplay_balanced_lam3e-2_seed1_regonce",
  "num_examples": 500, "passed": 196, "pass_at_1": 0.392
}
{
  "metric": "mbpp", "run_name": "cs_onereplay_ifcode_lam3e-2_seed1_regonce",
  "adapter_path": ".../adapters/cs_onereplay_ifcode_lam3e-2_seed1_regonce",
  "num_examples": 500, "passed": 204, "pass_at_1": 0.408
}
```

### MATH-500

```json
{
  "run_name": "base",
  "adapter_path": "",
  "data_path": ".../datasets/math/math500_test.jsonl",
  "num_examples": 500, "num_scored": 500, "correct": 346, "accuracy": 0.692
}
{
  "run_name": "cs_vanilla_seed1",
  "adapter_path": ".../adapters/cs_vanilla_seed1",
  "num_examples": 500, "num_scored": 500, "correct": 342, "accuracy": 0.684
}
{
  "run_name": "cs_onereplay_ifmath_lam3e-2_seed1_regonce",
  "adapter_path": ".../adapters/cs_onereplay_ifmath_lam3e-2_seed1_regonce",
  "num_examples": 500, "num_scored": 500, "correct": 347, "accuracy": 0.694
}
```

### GSM8K

```json
{
  "run_name": "base",
  "adapter_path": "",
  "data_path": ".../datasets/math/gsm8k_test.jsonl",
  "num_examples": 1319, "num_scored": 1319, "correct": 1017, "accuracy": 0.7710386656557998
}
{
  "run_name": "cs_vanilla_seed1",
  "adapter_path": ".../adapters/cs_vanilla_seed1",
  "num_examples": 1319, "num_scored": 1319, "correct": 995, "accuracy": 0.7543593631539045
}
{
  "run_name": "cs_onereplay_ifmath_lam3e-2_seed1_regonce",
  "adapter_path": ".../adapters/cs_onereplay_ifmath_lam3e-2_seed1_regonce",
  "num_examples": 1319, "num_scored": 1319, "correct": 1038, "accuracy": 0.7869598180439727
}
```
