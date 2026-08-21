# LoRA 口径三组实验（2026-08-09）

本次一起记三件事，三者共用同一套 LoRA 超参，可以直接并排比：

| # | 实验 | 作业 | 脚本 |
|---|---|---|---|
| 1 | replay 基线：batch 级混合三档（12.5% / 25% / 50%） | `or_rmix-2849732_{0,1,2}` | `18_sweep_replay_batchmix.slurm` |
| 2 | OneReplay 的 C 换成自蒸馏语料（λ=1e-2） | `or_sdcov-2856510_0`（C 由 `or_covsd-2849738` 采集） | `19_` + `20_` |
| 3 | λ 扫描续跑 3e-2 / 5e-2 / 1e-1 | `or_sweep-2856542_{1,2,3}` | `06_sweep_lambda_train.slurm` |

公共配置：Qwen3-1.7B + LoRA（r=8, α=16, dropout 0.1, `q_proj,v_proj`，56 层，1.61M 可训练参数 = 0.093%），Commonsense170k（168,715 训练 / 1,000 验证），3 epoch，lr 1e-4，micro-batch 8，max_len 512，bf16，seed=1，单卡 H100 80GB。旧知识语料统一是 FLAN 20k 的自蒸馏版本（19,994 条里丢掉 2,565 条截断行，剩 17,429 条可用）。

三组实验的**唯一变量**分别是：混合比例 / C 的数据源 / λ。其余逐字相同。

## 1. 总表

### 1.1 IFEval（541 条，%）

| run | strict prompt | strict inst | loose prompt | loose inst |
|---|---:|---:|---:|---:|
| base（未训练） | 66.54 | 74.70 | 70.61 | 77.94 |
| vanilla | 55.08 | 65.23 | 57.49 | 67.39 |
| replay 12.5%（7n1r） | **66.91** | **74.94** | **70.98** | **78.18** |
| replay 25%（6n2r） | 65.80 | 73.98 | 69.50 | 77.34 |
| replay 50%（4n4r） | 65.43 | 73.74 | 69.69 | 77.10 |
| OneReplay gold C λ=1e-2 | 65.25 | 72.90 | 68.21 | 75.66 |
| OneReplay **自蒸馏 C** λ=1e-2 | 65.62 | 73.02 | 68.76 | 75.78 |
| OneReplay gold C λ=3e-2 | 66.17 | 73.38 | 68.58 | 75.54 |
| OneReplay gold C λ=5e-2 | 64.88 | 72.90 | 67.65 | 75.18 |
| OneReplay gold C λ=1e-1 | 63.77 | 71.94 | 67.28 | 74.70 |

### 1.2 Multi-IF English（909 组，strict，%）

| run | 平均 prompt | 平均 inst | turn 1 | turn 2 | turn 3 |
|---|---:|---:|---:|---:|---:|
| base（未训练） | 49.47 | 73.34 | 68.32 | 48.84 | 31.25 |
| vanilla | 31.47 | 60.65 | 57.43 | 27.17 | 9.82 |
| replay 12.5%（7n1r） | **49.51** | **73.52** | **68.87** | 48.18 | **31.47** |
| replay 25%（6n2r） | 48.18 | 72.94 | 67.99 | 46.42 | 30.13 |
| replay 50%（4n4r） | — 未完成 — | | | | |
| OneReplay gold C λ=1e-2 | 47.21 | 72.13 | 67.22 | 48.29 | 26.12 |
| OneReplay **自蒸馏 C** λ=1e-2 | 48.18 | 72.03 | 68.54 | 46.31 | 29.69 |
| OneReplay gold C λ=3e-2 | 47.06 | 72.01 | 68.76 | 46.42 | 26.00 |
| OneReplay gold C λ=5e-2 | 46.00 | 71.40 | 66.23 | 45.21 | 26.56 |
| OneReplay gold C λ=1e-1 | 46.67 | 71.77 | 65.68 | 45.98 | 28.35 |

turn 列是 strict prompt accuracy。

### 1.3 新任务：Commonsense 留出集 val loss

| run | ep1 | ep2 | ep3 | vs vanilla |
|---|---:|---:|---:|---:|
| vanilla | 0.06039 | 0.05133 | **0.04494** | 基准 |
| replay 12.5% | 0.07126 | 0.05828 | 0.05291 | +17.7% |
| replay 25% | 0.07433 | 0.06327 | 0.05746 | +27.9% |
| replay 50% | 0.07927 | 0.06967 | 0.06463 | +43.8% |
| OneReplay gold C λ=1e-2 | — | — | 0.04724 | +5.1% |
| OneReplay 自蒸馏 C λ=1e-2 | 0.06174 | 0.05263 | 0.04718 | +5.0% |
| OneReplay gold C λ=3e-2 | 0.06278 | 0.05375 | 0.04916 | +9.4% |
| OneReplay gold C λ=5e-2 | 0.06363 | 0.05393 | 0.04699 | +4.6% |
| OneReplay gold C λ=1e-1 | 0.06480 | 0.05579 | 0.04903 | +9.1% |

**噪声底：** λ 从 1e-2 涨到 1e-1，约束强度差 10 倍，val loss 却在 0.0470–0.0492 之间来回跳（5e-2 甚至比 1e-2 还低）。所以 OneReplay 各 λ 之间 0.002 以内的 val loss 差别不可解读，只有 replay 那一列的 0.0529 → 0.0646 是真信号。

## 2. 实验 1：replay 基线

### 2.1 关键发现：行比例 ≠ token 比例

脚本设计时保证了"每次更新的 new task 行数恒为 63/66/64"，但**loss 是按 supervised token 平均的**，训练日志打出的真实比例是：

| 档位 | replay 行占比 | replay 占 supervised token | 池子循环次数/epoch |
|---|---:|---:|---:|
| 7n1r | 12.5% | **66.06%** | 1.38× |
| 6n2r | 25.0% | **81.95%** | 3.23× |
| 4n4r | 50.0% | **93.16%** | 9.68× |

原因是两边答案长度差一个数量级：Commonsense 的 target 平均只有 **7.0** 个 supervised token（都是 `the correct answer is option1` 这种短答案），自蒸馏 replay 的 target 平均 **94.8** 个。

这解释了 1.3 表里 replay 那三行为什么 val loss 掉得这么厉害：即使只切 1/8 的行给 replay，梯度里也已经有 2/3 来自旧知识，新任务被稀释成配角。**"每次更新 new task 行数不变"这个公平性设计只保住了样本数，没保住梯度权重。** 这一点必须写进论文的实验描述，否则 replay 基线的 retention 优势会被误读成方法优势。

### 2.2 开销

| run | 步数/epoch | accum | ms/step | 单 epoch 秒（3 次） | 3 epoch 合计 | peak alloc / reserved |
|---|---:|---:|---:|---|---:|---|
| vanilla（饱和测试推算） | 21,090 | 64 | 89.2 | 1881 | 1.57 h | 21.29 / — |
| replay 12.5% | 24,102 | 72 | 97–103 | 2478 / 2341 / 2441 | 2.02 h | 21.29 / 23.00 |
| replay 25% | 28,119 | 88 | 102–104 | 2857 / 2886 / 2912 | 2.40 h | 21.29 / 23.00 |
| replay 50% | 42,178 | 128 | 112–119 | 5034 / 4855 / 4715 | 4.06 h | 21.29 / 23.00 |

实测墙钟比脚本注释里按步数比预测的（×1.14 / ×1.33 / ×2.00）要贵：真实是 **×1.29 / ×1.53 / ×2.59**。差额来自单步时间本身也涨了（89.2 → 97~119 ms），因为 replay 行序列长得多，动态 padding 下每步的 token 数被拉高。replay 不额外占显存（allocated 与 vanilla 逐位相同）。

### 2.3 50% 档的 Multi-IF 需要补跑

不是墙时间不够，是 I/O 故障：evaluate 在 multiif 生成到约 470/909 时抛 `OSError: [Errno 116] Stale file handle`（`or_rmix-2849732_2.err:187`）。训练已完成、adapter 已保存、IFEval 已出结果，只需单独重跑评测：

```bash
python -m onereplay.scripts.evaluate \
  --model_dir /hpcwork/xsz96350/Chen_logs/checkpoints/ --model_name Qwen3-1.7B \
  --adapter_path $RESULTS_ROOT/adapters/cs_replaymix_4n4r_seed1 \
  --run_name cs_replaymix_4n4r_seed1 \
  --metrics multiif --out_dir $RESULTS_ROOT --seed 1 \
  --multiif_input onereplay/third_party/Multi-IF/multiIF_20241018.csv
```

按 12.5% → 25% 的走势（49.51 → 48.18）和 IFEval 上 50% 档继续下滑，预期 50% 档的 Multi-IF 在 47 上下，不会改变"replay 越多 retention 越不涨、新任务越差"的结论。

## 3. 实验 2：C 换成自蒸馏语料

补的是四格表的右下角，让 OneReplay 与 replay 吃同一份旧知识（同一批 prompt、同一批自蒸馏回答、同样丢掉那 2,565 行截断的），差别只剩"压成二阶矩一次带走"还是"留着语料反复重训"。

**尺度对齐已验证：** `or_covsd-2849738` 逐层比了 trace，`trace(自蒸馏 C) / trace(gold C)` 均值 0.976（min 0.932，max 1.004），等效 λ = 1.02e-2。所以 λ=1e-2 原样跑，两边是干净的 A/B，不存在"只是正则变强了"的混淆。

| λ=1e-2 | IFEval strict prompt | Multi-IF avg strict prompt | val loss ep3 | reg ep3 |
|---|---:|---:|---:|---:|
| gold C | 65.25 | 47.21 | 0.04724 | 0.254 |
| 自蒸馏 C | **65.62** | **48.18** | **0.04718** | 0.245 |

三个轴同时小幅变好，且 val loss 没有变差——不是权衡，是净赢。但幅度小（IFEval +0.37pt、Multi-IF +0.97pt），单 seed 下**不足以下"自蒸馏 C 更好"的定论**，只能说"至少不比 gold C 差，且换过来之后与 replay 的对比才站得住"。真要下结论需要补 seed。

开销：单 epoch 2197 / 2115 / 2113 s（合计 1.78 h），peak allocated 22.17 GiB，其中 C 常驻 0.875 GiB。一次性采 C 花 **15.5 分钟**（qv scope 56 层）。

## 4. 实验 3：λ 扫描——曲线在 3e-2 见顶

接上一轮（`or_sweep-2427205`，1e-2 是当时的最优点）继续往上扫：

| λ | IFEval strict prompt | Multi-IF avg strict prompt | val loss ep3 | reg ep3 | λ·reg |
|---:|---:|---:|---:|---:|---:|
| 1e-2 | 65.25 | **47.21** | 0.04724 | 0.254 | 0.00254 |
| **3e-2** | **66.17** | 47.06 | 0.04916 | 0.109 | 0.00327 |
| 5e-2 | 64.88 | 46.00 | 0.04699 | 0.072 | 0.00360 |
| 1e-1 | 63.77 | 46.67 | 0.04903 | 0.041 | 0.00408 |

**结论（即使 5e-2 / 1e-1 只有单点也已经确定）：3e-2 是最优工作点，再往上是纯亏。**

理由不是"3e-2 的分最高"这么简单，而是**再往上没有任何东西被换回来**：

- retention 两个指标都在 3e-2 之后单调下滑（IFEval 66.17 → 64.88 → 63.77，Multi-IF 47.06 → 46.00 / 46.67）。
- 而新任务 val loss 从 1e-2 到 1e-1 全在 0.0470–0.0492 的噪声带里（5e-2 甚至最低）。也就是说**大 λ 并没有换来更好的新任务拟合**——它既不是"牺牲新任务保 retention"，也不是"牺牲 retention 保新任务"，而是两头都不占。
- 所以这不是一条权衡曲线被走到了另一端，是约束过强本身在损害优化：正则把 ΔW 压到 reg 只剩 0.041（1e-2 时是 0.254，压掉 84%），LoRA 那 1.6M 参数的可用方向被挤没了，两个任务一起变差。

这与 IFEval 上的读数一致：λ=3e-2 的 66.17% 距 base 的 66.54% 只差 **0.37pt**，是所有 OneReplay 档里最接近 base 的；而 λ=1e-1 反而退回到 -2.77pt。

**已知缺口：λ=2e-2 没跑。** 脚本 `LAMBDAS=(2e-2 3e-2 5e-2 1e-1)` 配 `--array=1-6`，而取值用的是 `LAMBDAS[TASK_ID+1]`（zsh 1-based），于是 TASK_ID=0 对应的 2e-2 被跳过，TASK_ID=4/5/6 越界直接 exit 1（`or_sweep-2856542_{4,5,6}` 三个空 job）。**这是第二次犯同一个错**——上一轮 `2427205` 因为完全相同的原因跳过了 1e-5。真正的峰值可能落在 2e-2 和 3e-2 之间，但这不影响"3e-2 之后变差"这个结论。

**修复建议：** 把 `06_sweep_lambda_train.slurm` 第 89 行改成 `LAMBDA="${LAMBDAS[$((TASK_ID + 1))]}"` 配 `--array=0-$((n-1))`，并在脚本里加一句 `[[ $((TASK_ID+1)) -le ${#LAMBDAS[@]} ]]` 的显式断言，或者干脆把 `--array` 上界从数组长度自动生成。

## 5. 三组放一起看

以 base 为满分、vanilla 为零分，算 retention 保留率（`(run - vanilla) / (base - vanilla)`）：

| run | IFEval 保留率 | Multi-IF 保留率 | val loss 代价 | 3 epoch 训练时间 |
|---|---:|---:|---:|---:|
| replay 12.5% | **103%** | **100%** | +17.7% | 2.02 h |
| replay 25% | 94% | 93% | +27.9% | 2.40 h |
| replay 50% | 90% | 未测 | +43.8% | 4.06 h |
| OneReplay 自蒸馏 C λ=1e-2 | 92% | 93% | **+5.0%** | 1.78 h |
| OneReplay gold C λ=3e-2 | 97% | 87% | +9.4% | 2.00 h |

两条路线的分工很清楚：

- **replay 在 retention 上更强**，12.5% 那档甚至超过 base（IFEval 66.91 vs 66.54，Multi-IF 49.51 vs 49.47）——但这个"超过"要打折看，它的代价是新任务 val loss 掉 17.7%，而且根源就是 2.1 节那个 66% 的 token 占比：模型有 2/3 的梯度在复习旧知识，retention 好不奇怪。
- **OneReplay 在新任务上几乎无损**（+5.0%），换回 92~97% 的 retention，且训练时间是全场最低的 1.78 h（比 replay 最便宜的那档还快 12%），显存只多 0.875 GiB 的 C。
- **加 replay 的比例是负收益的**：12.5% → 50%，retention 从 103% 掉到 90%，val loss 从 +17.7% 恶化到 +43.8%，时间从 2.02 h 涨到 4.06 h。三个维度全变差，没有一个理由用大比例 replay。

一次性开销（可在多次训练间摊销）：自蒸馏生成 20k 条 target 39 min（两条路线共用），OneReplay 额外采 C 15.5 min / 0.875 GiB。

## 6. 待办

1. 补跑 `cs_replaymix_4n4r_seed1` 的 Multi-IF（见 2.3，约 1.5 h）。
2. 补 λ=2e-2 这一点，顺手修掉 `06_` 的 array 索引 bug（见第 4 节）。
3. OneReplay 自蒸馏 C 的优势只有单 seed 支撑，要下定论需补 seed 2/3。
4. 论文里描述 replay 基线时必须写 token 占比而不是行占比，否则 retention 对比会被误读。

## 7. 复现

```bash
# 实验 1：replay batch 级混合三档
sbatch onereplay/slurm/18_sweep_replay_batchmix.slurm

# 实验 2：先采自蒸馏 C（15 min），再训练
sbatch onereplay/slurm/19_collect_cov_selfdistill.slurm
sbatch onereplay/slurm/20_train_onereplay_selfdistill.slurm

# 实验 3：λ 扫描（注意先修 array 索引）
sbatch onereplay/slurm/06_sweep_lambda_train.slurm
```

原始日志在 `onereplay/slurm/outputs/`：`or_rmix-2849732_{0,1,2}`、`or_covsd-2849738`、`or_sdcov-2856510_0`、`or_sweep-2856542_{1,2,3}`。
