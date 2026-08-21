# OPD / Vanilla Replay / 成本基准 汇总（2026-08-03）

本轮共三组作业，均已正常完成：

| 作业 | Job ID | 内容 |
|---|---|---|
| OPD 训练+评测 | 2649408 | `cs_opd_lam0_seed1`、`cs_opd_lam1e-4_seed1`（teacher=Qwen3-8B，student=Qwen3-1.7B） |
| Replay 扫描 | 2649411 (array 0-2) | `cs_replay_r0.01/0.05/0.10_seed1`（vanilla replay，FLAN 数据混入） |
| 成本基准 | 2649412 | `bench_vanilla`、`bench_onereplay_lam1e-2`（含 profile 运行） |

公共配置：Qwen3-1.7B + LoRA (r=8, α=16, q_proj/v_proj)，Commonsense170k，seed=1。**本页所有 run 一律是 LoRA（可训练参数 0.093%），没有全量微调；表中「SFT」指训练目标是交叉熵，不是全量 SFT。**
注意 OPD 与 replay 的训练配置不同：

- **OPD**：1 epoch 上限 2000 步（batch 4 × 2000 = 8000 样本，约为一个 epoch 的 4.7%），lr=1e-5，蒸馏 teacher=Qwen3-8B。
- **Replay**：3 epochs 全量（约 168.7k 新任务样本 + 按 ratio 混入 FLAN replay），lr=1e-4，与之前 vanilla / OneReplay λ 扫描一致。

## 主结果：retention（IFEval + Multi-IF English）

Base / Vanilla / OneReplay λ=1e-2 来自之前的运行（见 `当前结果总结.md`），其余为本轮新结果。

### IFEval（541 prompts）

| 模型 | strict prompt | strict instr | loose prompt | loose instr |
|---|---:|---:|---:|---:|
| Base（不训练） | 66.54% | 74.70% | 70.61% | 77.94% |
| Vanilla SFT（3ep） | 55.08% | 65.23% | 57.49% | 67.39% |
| OneReplay λ=1e-2（3ep） | 65.25% | 72.90% | 68.21% | 75.66% |
| **OPD λ=0**（2000 步） | 65.25% | 73.38% | 69.13% | 76.74% |
| **OPD λ=1e-4**（2000 步） | **66.54%** | **74.46%** | **70.61%** | **77.70%** |
| **Replay r=0.01**（3ep） | 46.95% | 58.75% | 48.43% | 60.19% |
| **Replay r=0.05**（3ep） | 42.51% | 55.40% | 44.18% | 56.47% |
| **Replay r=0.10**（3ep） | 39.74% | 53.60% | 40.85% | 54.68% |

### Multi-IF English（909 conversations，3 turns）

| 模型 | avg strict prompt | avg strict instr | turn1 strict prompt | turn2 | turn3 |
|---|---:|---:|---:|---:|---:|
| Base | 49.47% | 73.34% | 68.32% | 48.84% | 31.25% |
| Vanilla LoRA | 31.47% | 60.65% | 57.43% | 27.17% | 9.82% |
| OneReplay λ=1e-2 | 47.21% | 72.13% | 67.22% | 48.29% | 26.12% |
| **OPD λ=0** | 48.11% | 72.32% | 66.89% | 47.08% | 30.36% |
| **OPD λ=1e-4** | **48.51%** | **72.67%** | 67.77% | 47.74% | 30.02% |
| **Replay r=0.01** | 23.64% | 53.29% | 47.85% | 17.60% | 5.47% |
| **Replay r=0.05** | 21.96% | 50.67% | 43.23% | 17.05% | 5.58% |
| **Replay r=0.10** | 19.31% | 48.29% | 38.50% | 14.63% | 4.80% |

## 新任务拟合（Commonsense val loss）

| 模型 | 训练末期 val loss | 独立 commonsense 评测 val loss |
|---|---:|---:|
| OPD λ=0 | 0.9462（蒸馏目标） | 10.62 |
| OPD λ=1e-4 | 0.9581（蒸馏目标） | 10.64 |
| Replay r=0.01 | 0.0464 | — |
| Replay r=0.05 | 0.0494 | — |
| Replay r=0.10 | 0.0486 | — |
| （参考）OneReplay λ 扫描 3ep | 0.045–0.047 | — |

**注意**：OPD 的两个 val loss 不可与 SFT 系列直接比较。训练期 val loss 是对 teacher 输出的蒸馏损失；独立评测的 10.6 是在 gold 标签上的 CE，说明 OPD 学生模仿的是 teacher 的自由生成，而非 Commonsense 数据集的目标答案格式。OPD 的新任务效果需要用生成式准确率（而非 CE loss）另行评测才有意义。

## 训练成本

> **口径声明：本页所有 7 个 run 都是 LoRA，没有任何全量微调。**
> Qwen3-1.7B + LoRA(r=8, α=16, 仅 q_proj/v_proj)，可训练参数 1,605,632 / 1,722,180,608 = **0.093%**（七个 run 的日志逐一核对，数值完全一致）。
> 日志里的 `PARADIGM=sft` 指的是**训练目标**是交叉熵 SFT，`PARADIGM=opd` 指目标是对 teacher 的蒸馏——两者都跑在同一套 LoRA 上。表中写「SFT」时一律指 **LoRA-SFT**，不要读成全量 SFT。OPD 的 teacher（Qwen3-8B）是完整模型，但全程冻结、只做推理，不参与训练。

成本数据来自四个作业，**口径不同**：

- **bench 2649412 + replay 2649411**（单卡 H100，`batch=8 / accum=64`，全量 1 epoch = 21090 步）
- **OPD 2649408**（双卡 H100，`batch=4 / accum=32`，只跑 2000 步 = 8000 样本）——每步样本数只有前者一半，单步时间**不可**与上面几行直接相比
- **OPD 真 vanilla 2832597**（2026-08-08 补做，双卡 H100，与 2649408 逐项同配置，唯一差别是 `--measure_replay_when_lambda_zero 0`）——这才是 OPD 的干净成本基线，见下文 OPD 内部对比
- **batch 饱和测试 2831163**（2026-08-08，单卡 H100 独占，纯 vanilla，`batch=8/16` 各吃 6400 样本）——见下文「batch 饱和」小节与 `2026-08-08_batch_size饱和测试.md`

**相对开销一律以「不加正则」为分母**，即正则让训练变慢的比例。**报相对百分比时必须标注 batch**：正则的绝对开销是与 batch 无关的常数（9–11 ms/step），所以同一个方法在 b8 是 +10%、b32 是 +2.3%、OPD b4 是 +5.3%，只看百分比会误以为是三个不同结论。

### 单步时间（ms/step）

| run | 适配 | 目标 | batch×accum | ms/step | vs 稳态基准 88.0 ms |
|---|---|---|---|---:|---:|
| bench_vanilla（基准） | LoRA r=8 | SFT CE | 8×64 | 97.9（全程均值，**含污染**） | — |
| └ 同 run 的稳态值 | LoRA r=8 | SFT CE | 8×64 | **88.0**（step 3500–17500） | 基准 |
| bench_onereplay λ=1e-2 | LoRA r=8 | SFT CE | 8×64 | 96.9 | **+10.1%** |
| cs_replay r=0.01 | LoRA r=8 | SFT CE | 8×64 | 89.0 | +1.1% |
| cs_replay r=0.05 | LoRA r=8 | SFT CE | 8×64 | 91.1 | +3.5% |
| cs_replay r=0.10 | LoRA r=8 | SFT CE | 8×64 | 105.8 | +20.2% |
| （复核）独占单卡 vanilla | LoRA r=8 | SFT CE | 8×64 | **89.2**（2831163） | 与稳态 88.0 吻合 |

OPD 那三条 run 是 `batch=4 / accum=32` 的双卡口径，单步时间（真 vanilla 309.5 / λ=0 319.2 / λ=1e-4 310.3 ms）**不能**与上表并列，它们的同口径对比见下面「OPD 内部对比」一节。

**基准 run 的后段被污染，必须用稳态值 88.0 ms 而不是 97.9 ms。** `bench_vanilla` 打印的是累计平均，step 3500–17500 一直稳定在 88 ms，之后一路爬到 98 ms；反推可知最后 3590 步耗时约 524 s，瞬时约 146 ms/step，是前段的 1.66 倍，凭空给基准加了约 208 s。同作业的 OneReplay run 则从 step 1500 到 21090 全程 97 ms，没有任何漂移。用被污染的 97.9 ms 当分母，会得到「加了正则反而快 1.0%」这种物理上不可能的结论——正则是纯增量计算，真值必然 ≥ 0。

两笔开销恰好互相抵消，这就是 -1.0% 的来源：基准若全程 88 ms 应为 1856 s，实际 2064.6 s（多 208 s）；OneReplay 每步多 8.9 ms × 21090 步 = 多 187 s（1856+187≈2043 s，正是实测值）。208−187 = 21 s，与实测差值 2064.6−2043.2 = 21.4 s 吻合。

**正则的真实开销约 +10%，独立证据有两条**：稳态比 88.0 → 96.9 ms 是 +10.1%；profile 的 300 步短跑（同条件、独立作业段）是 27.9 s → 31.0 s，+11.0%。另外原先记的「replay_reg 仅占 4.1%」低估了——那只是**前向** phase，正则的反向被计进了 `backward`（15.59 → 17.23 s，+1.64 s），前向 1.27 + 反向 1.64 = 2.91 s，占 27.9 s 的 10.4%，与上面两个数一致。

**用 88.0 ms 当基准后，整张表才自洽**：传统 replay 的开销随 ratio 单调上升（+1.1% → +3.5% → +20.2%），而原先那三个负值（-9.1% / -7.0%）本就说不通。

**这 +10% 是实现开销，不是方法的固有代价。** 正则每步约 11 GFLOP，而任务前反向约 42 TFLOP，占比仅 0.03%，理论上应当看不见。开销出在 CPU 侧：`lora_covariance_regularizer` 每步重新遍历整个模型的 `named_modules()`（几百个 module），对每个命中的 LoRA 层调 `lookup_covariance` 做字符串 `endswith` 匹配（56 层 × 56 个 key ≈ 3000 次字符串比较），再 launch 56 组小 kernel。

但**优化空间没有原先估的那么大**：profile 把这 9–11 ms 拆成前向 4.2 ms + 反向 5.5 ms，缓存 (module, C) 配对只能消掉前向那半，反向那 5.5 ms 是 autograd 逐层求梯度的 launch 开销，缓存动不了它——要一起压下去，得把 56 层的独立小 GEMM 合并成 batched 操作。**在此之前，对外不应宣称「零开销」。**

### 显存（峰值）

| run | 适配 | 卡数 | peak allocated | peak reserved | 其中 C 矩阵 |
|---|---|---|---:|---:|---:|
| bench_vanilla | LoRA r=8 | 1 | 21.29 GiB | 23.16 GiB | 0 |
| bench_onereplay λ=1e-2 | LoRA r=8 | 1 | 22.17 GiB | 24.03 GiB | 0.875 GiB |
| cs_replay r=0.01 | LoRA r=8 | 1 | 21.29 GiB | 23.00 GiB | 0 |
| cs_replay r=0.05 | LoRA r=8 | 1 | 21.29 GiB | 23.02 GiB | 0 |
| cs_replay r=0.10 | LoRA r=8 | 1 | 21.29 GiB | 23.00 GiB | 0 |
| cs_opd λ=0 | LoRA r=8 | 2 | 28.44 GiB（student 12.12 + teacher 16.33） | 29.73 GiB | 0.875 GiB |
| cs_opd λ=1e-4 | LoRA r=8 | 2 | 28.44 GiB（同上，逐位相同） | 29.73 GiB | 0.875 GiB |

- **显存这一栏的结论是可靠的**（不受上面的计时污染影响）：OneReplay 的全部显存代价就是 C 常驻的 0.875 GiB，peak +0.88 GiB，且与 λ 取值无关。
- **传统 replay 显存 ±0**：它只是往训练集里多混数据，不增加常驻张量。
- **OPD 的显存代价在 teacher 卡**（16.33 GiB）；student 卡反而只有 12.12 GiB，因为它的 batch 是 4 而非 8。C 矩阵那 0.875 GiB 在 OPD 里更不构成压力。

### OPD 内部对比：真 vanilla vs 带正则（2832597 + 2649408）

**先说结论：正则在 OPD 里的开销是 +9.8 ms/step（全程口径 +3.2%）或 +10.6 ms/step（稳态口径 +5.3%），显存 +0.881 GiB 全在 student 卡上。**

原先这里用 `λ=1e-4 vs λ=0` 做对比，得到「加了正则反而快 2.8%」——**那组对比是无效的**，原因在配置和代码两层：

1. 两个 run 都设了 `MEASURE_REPLAY_WHEN_LAMBDA_ZERO=1`，λ=0 那条同样加载 C（日志里都有 `loaded 56 covariance matrices ... 0.875 GiB resident`）并每步求值——`replay_reg` 非零（0.106→0.374），只是 `train_lambda_reg=0.0` 没进 loss。
2. `onereplay/trainers/base.py` 写的是 `loss = task_loss + self.replay_lambda * replay_reg`。λ=0 时这一项仍是带 `grad_fn` 的张量，反向照样穿过整条正则链路，梯度为 0 但计算一步不少。

也就是说两条 run 的口径**完全相同**（都付了全额正则开销），`peak_memory_allocated_gb` 两边逐位相同（28.444942951202393）正是佐证，那 -2.8% 只是噪声。根因是 `08_train_opd.slurm` 当时没有传 `--measure_replay_when_lambda_zero`，`train.py` 的默认值 1 生效了（已修，现在该脚本暴露 `MEASURE_REPLAY` 开关）。

2026-08-08 补做了真 vanilla（Job 2832597，`17_bench_opd_vanilla.slurm`），除该开关外与 2649408 逐项同配置：

| OPD run | 正则口径 | train wall (s) | ms/step 全程 | ms/step 稳态 | cuda:0 | cuda:1 | 合计 peak | C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **真 vanilla**（2832597） | 不加载 C，不计算 | **618.9** | **309.5** | **198.3** | 11.235 | 16.329 | 27.564 | 0 |
| λ=0（2649408） | 加载并计算，不进 loss | 638.5 | 319.2 | 208.9 | 12.116 | 16.329 | 28.445 | 0.875 |
| λ=1e-4（2649408） | 加载并计算，进 loss | 620.6 | 310.3 | 209.1 | 12.116 | 16.329 | 28.445 | 0.875 |

逐区间速度（真 vanilla 来自新增的 `win=` 字段，另两条从累计平均反推）：

| 步区间 | 真 vanilla | λ=0 | λ=1e-4 | 正则开销 |
|---|---:|---:|---:|---:|
| 1–500 | 561.8 | 569 | 537 | +7.2 |
| 501–1000 | 246.2 | 257 | 253 | +10.8 |
| 1001–1500 | 231.5 | 242 | 242 | +10.5 |
| 1501–2000 | **198.3** | **208.9** | 209.1 | **+10.6** |

**对比对象必须是 λ=0 而不是 λ=1e-4**，两个理由：

- λ=0 与真 vanilla 的训练轨迹**逐位相同**（下面一条），所以差值纯粹是正则开销，不掺入「参数不同导致 rollout 长度不同」的混淆；λ=1e-4 的轨迹是真的不一样。
- λ=0 与真 vanilla 都是各自作业的**第一个** run，冷启动位置对称。λ=1e-4 是第二个 run、白捡了热环境，它的 1–500 区间是 537 ms，比真 vanilla 的 561.8 还快——用它当对比对象会把冷启动差算成正则的收益，这正是原先那 -2.8% 的来源。而 λ=0 与 λ=1e-4 的稳态几乎重合（208.9 vs 209.1，差 0.1%），说明两者计算量确实相同。

**顺带实证了「λ=0 加载 C 不影响训练结果」，而且是逐位相同而非近似**：真 vanilla 与 λ=0 的 `train_task_loss` 都是 1.226359612776665、`val_loss` 都是 0.9462042945623398，四个打印步的 loss 也逐位一致（0.325594 / 2.077461 / 0.313816 / 0.718801）。连 `temperature=1.0` 的采样 rollout 都完全复现，说明正则确实不消耗随机数。因此两次训练产出的 LoRA adapter 是同一份权重，**真 vanilla 的 retention 直接沿用 `cs_opd_lam0_seed1` 的评测结果**，无需重跑评测（该 run 用 `--save 0`，本就没存 adapter）。

**为什么 OPD 的相对开销（+5.3%）低于 SFT b8（+10.2%）**：正则的绝对开销与 batch、与 paradigm 都无关（见下节四次测量），而 OPD 的单步基数是 198 ms 而非 88 ms，被 rollout 和 teacher 打分摊薄了。显存侧同理：C 那 0.875 GiB 相对 OPD 的瓶颈（teacher 卡 16.3 GiB）不构成压力。

补充一句跨范式的参考（口径不同，只看吞吐）：OPD 按 token 吞吐是 1472 vs SFT b8 的约 10200，慢约 7 倍；总 wall time 只有 619 s 是因为只训了 8000 个样本（SFT 一个 epoch 是 168715 个），不是 OPD 更快。

### 正则绝对开销：四次独立测量都落在 9–11 ms/step

| 场景 | 无正则 | 有正则 | 绝对开销 | 相对开销 |
|---|---:|---:|---:|---:|
| SFT b8（21090 步稳态，2649412） | 88.0 ms | 97 ms | 9.0 ms | +10.2% |
| SFT b8（profile 分解，2649412） | 93.0 ms | 103.2 ms | 10.2 ms | +11.0% |
| SFT b32（稳态区间，2815257） | 366.9 ms | 375.4 ms | 8.5 ms | +2.3% |
| OPD b4（稳态区间，2832597 vs 2649408） | 198.3 ms | 208.9 ms | 10.6 ms | +5.3% |

batch 跨 4 倍、paradigm 跨 SFT/OPD，绝对开销都是 9–11 ms。原因在代码上是显然的：正则算的是 `tr(ΔW C ΔWᵀ)`，只读 LoRA 的 A/B 权重和 C 矩阵，batch 里有几条样本、序列多长都不进这个式子。profile 数据给出了去向：**前向 4.2 ms + 反向 5.5 ms**。两者都与 task_loss 的 kernel 在同一个 CUDA stream 上串行执行，即使 GPU 有空闲 SM 也不会被「吸收」，所以是加性的——这也解释了为什么 b8 测出的开销没有比 b32 更小。

### batch 饱和：b8 已吃满吞吐

Job 2831163（独占单卡，纯 vanilla，两档各吃 6400 样本）。详见 `2026-08-08_batch_size饱和测试.md`。

| batch | steps | 最快 train (s) | vs b8 | 单步倍数 | ms/step | tokens/s | peak alloc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 800 | 71.39 | 基准 | 基准 | 89.2 | **10181** | 21.29 GiB |
| 16 | 400 | 72.49 | **1.015** | 2.03x | 181.2 | 10026 | 39.28 GiB |

相同计算量下两档几乎同时完成，且 b8 的 `tokens/s` 反而高 1.55%（大 batch 的动态 padding 浪费更多）。所以**加大 micro-batch 换不到速度，只换显存**，老师担心的「batch 小没吃满、batch×2 时间不翻倍」在这个配置下不成立。

两个连带结论：

- **b8 的稳态基线确认是 89.2 ms**，与 2649412 全量 run 反推的 88.0 ms 吻合，上面整张单步时间表的分母站得住。
- **replay 的混合方案应选 `4+4` + accumulation ×2**（总时间 ×2.00、峰值显存维持 21.29 GiB），而不是 `8+8`（×2.03、39.28 GiB）。`4+4` 的成本与饱和度无关，`8+8` 的成本完全取决于饱和度。

## 结论

1. **OPD retention 几乎无损**：OPD λ=1e-4 在 IFEval strict prompt 上与 Base 持平（66.54%），Multi-IF avg strict prompt 仅比 Base 低 0.96 个百分点（48.51% vs 49.47%），优于此前最好的 OneReplay λ=1e-2（47.21%）。λ=1e-4 一致地略好于 λ=0，说明协方差正则在 OPD 之上仍有小幅增益。
2. **Vanilla replay 结果异常，全面差于 Vanilla SFT**：三个 ratio 的 IFEval strict prompt（39.7%–47.0%）都明显低于不加 replay 的 Vanilla SFT（55.08%），且 ratio 越大 retention 越差，方向与预期相反。已排除模板问题——`onereplay/data/replay.py` 与 Commonsense 共用 `build_sft_tokenize_fn`，同样应用 chat template 并对 prompt 做 -100 掩码。剩余的候选解释：(a) FLAN 的监督信号（`targets`）是极短答案，混入越多越强化「短输出」倾向，与 IFEval/Multi-IF 的长文本+格式约束直接冲突，这与 ratio 单调恶化的趋势吻合；(b) 55.08% 的 Vanilla 基线来自重构前的旧作业，本轮 `bench_vanilla` 只跑 1 epoch 且 `SAVE=0`，没有同代码版本的基线可比。**下结论前需要用当前代码重跑一个 r=0 的 3-epoch 基线。**
3. **正则的时间开销是一个与 batch、与 paradigm 都无关的常数：9–11 ms/step。** 此前记的「可忽略」是测量假象——基准 run 后段被外部因素拖慢约 208 s，恰好抵消了正则多花的 187 s，才得出 -1.0%。四次独立测量（SFT b8 稳态 9.0 ms、SFT b8 profile 10.2 ms、SFT b32 8.5 ms、OPD b4 10.6 ms）互相印证。**所以相对百分比完全由分母决定，报数时必须标注 batch**：b8 +10.2%、b32 +2.3%、OPD b4 +5.3%。显存代价同样是常数：C 常驻 0.875 GiB，peak +0.881 GiB，在 b8/b16/b32 三档逐位一致，与 λ 取值也无关。作为对比，传统 replay r=0.10 单步 +20.2%，且随 ratio 单调上升。
4. **这 9–11 ms 几乎全是开销而非算力，但优化没有原先想的那么容易。** profile 分解显示前向 4.2 ms、反向 5.5 ms，而正则的理论 FLOP 只需约 0.2 ms（11 GFLOP），也就是 98% 花在 kernel launch 与 CPU 侧遍历上：`lora_covariance_regularizer` 每步重新遍历 `named_modules()`，对每个 LoRA 层调 `lookup_covariance` 做字符串 `endswith` 匹配（56 × 56 ≈ 3000 次比较），再 launch 56 组小 kernel。缓存 (module, C) 配对能消掉前向那部分，但**反向那 5.5 ms 是 autograd 逐层求梯度的 launch 开销，缓存消不掉**，要压下去得把 56 层的小 GEMM 合并成 batched 操作。**在此之前，对外不应宣称「零开销」。**
5. **OPD 的代价在于 teacher**：需要第二张卡放 8B teacher（+16.3 GiB），按 token 吞吐比 SFT 慢约 7 倍；但本轮只训了 2000 步（8000 样本）就达到上述 retention，总 wall time 反而最短（约 620 s）。正则在 OPD 里只占 +5.3%，比 SFT b8 的 +10.2% 低一半，因为单步基数被 rollout 摊薄了。
6. **micro-batch=8 已经吃满 GPU 吞吐**（Job 2831163）：相同样本量下 b8 与 b16 的总时间比是 1.015，b8 的 `tokens/s` 反而高 1.55%。因此 replay 的混合方案定为 **`4+4` + accumulation ×2**（总时间 ×2.00、峰值显存 21.29 GiB），而不是 `8+8`（×2.03、39.28 GiB）。同时这确认了整张单步时间表的分母 88–89 ms 是可靠的。详见 `2026-08-08_batch_size饱和测试.md`。
7. **待办**：(a) OPD 的新任务能力需用生成式指标评测（当前 CE=10.6 不能说明学没学会）；(b) replay 需用自蒸馏目标重跑——已排除模板问题（与 Commonsense 共用 `build_sft_tokenize_fn`），gold target 那三条的崩溃来自 FLAN 极短答案抢走梯度；同时需要一个当前代码版本的 r=0 3-epoch 基线；(c) 老师要求的 `replay_ratio = 0.5` 需要先解决两个前置问题：口径（老师的「0.5」指 batch 内一半是 replay，等于代码 `replay_ratio` 的 **1.0**）和语料量（`build_replay_dataset` 不做重复采样，pool 只有 20k，请求超出时会静默退化到 actual≈0.119）；(d) OPD 目前只有 seed=1、2000 步，如结果要写进报告建议补 seed 和训练长度消融。

原始数据：`onereplay/slurm/outputs/` 下的 `or_opd-2649408.out`、`or_replay-2649411_{0,1,2}.out`、`or_bench-2649412.out`，以及 2026-08-08 补做的 `or_opdvan-2832597.out`（OPD 真 vanilla）与 `or_sat-2831163.out`（batch 饱和）；集群上的成本表 `/hpcwork/xsz96350/Chen_logs/onereplay/results/metrics/cost/cost_table.md`。

修订记录：2026-08-08 修正了正则开销的口径（补 OPD 真 vanilla 基线，明确相对百分比依赖 batch），补入 batch 饱和结论。此前版本中「OPD 内部对比无法回答开销问题」与「缓存后可压到 1–2%」两处结论已被新数据取代。
