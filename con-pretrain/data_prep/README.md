# CPT 数据准备

三个数据集的下载 → 划分 → tokenize → 检查。产物是 litdata chunk，`litgpt pretrain` 的
`LitData` data module 可以直接吃。

所有命令都在 `con-pretrain/` 目录下以模块方式跑：

```bash
cd con-pretrain
python -m data_prep.download --dataset biomed --dry-run
```

## 产物布局

```
con-pretrain/data/
  raw/                          # snapshot_download 落地的原始 parquet
    fineweb_edu_pool/           #   sample/10BT，14 片 28.5 GB
    biomed/                     #   data/commercial-*，26 片 49 GB
    finemath/                   #   finemath-4plus，64 片 18.4 GB
  chunks/                       # litdata 产物，pretrain 直接读
    fineweb_edu/
      seg00/ seg01/ seg02/      #   replay 段，同时也是采 C 的来源
      probe/                    #   干净的 held-out，取自 2025 的 dump
      pool_url_index.npy        #   10BT 的 URL 指纹，checks overlap 用
      replay_plan.json          #   每条臂读哪些段、各多少 token
    biomed/  train/ val/ test/
    finemath/ train/ val/ test/
```

每个 split 目录下有一份 `kres_manifest.json`，记录来源分片、划分盐值与区间、
`block_size` / `chunk_blocks`、以及写入了多少文章和 token。几周后拿到一堆 chunk
目录，靠它分清哪个对应哪条臂。

## 三条铁律

这三条在 `common.py` 里实现，下游脚本不要各自造第二套。

**1. 划分在 tokenize 之前、以文章为最小单位。**
litdata 把文档拉平成一条 token 流后按固定 block 切块，完全不管 chunk 边界是否跨文章。
一旦划分发生在 tokenize 之后，同一篇文章的前半段会进训练集、后半段进 val，而且看不出来。
实现是 `Router`：对文章主键做 blake2b 哈希后按 ppm 区间路由。用 blake2b 而不是内置
`hash()`，因为后者受 `PYTHONHASHSEED` 随机化影响，26 个 worker 会各划一套。

主键分别是：Biomed 用 `article_id`，FineWeb-Edu 用 `id`，FineMath 没有 id 列、用 `url`。

**2. tokenize 一律 `bos=False, eos=True`。**
与基座的 GPT-NeoX 约定一致。这个 tokenizer 的 `bos_id == eos_id == 0`
（`<|endoftext|>`），eos 同时充当文档分隔符。

**3. 写 litdata chunk 必须传 `item_loader=TokensLoader(block_size)`。**
不传的话 chunk 元数据里没有 block 布局，读回时 `StreamingDataset` 直接抛
`unsupported operand type(s) for //: NoneType and int`。litgpt 自己的
`openwebtext.py` 是不传的，那条路在当前 litdata 版本下已经不能用。

## 执行顺序

### 第 0 步：环境

```bash
wsl -e /home/czq/ENTER/bin/python test_code_CPT/setup_kres_env.py
```

集群上要装 CUDA 版 torch，但 `litdata` / `pyarrow` / `huggingface_hub` 是一样的。
注意 `torchvision` 必须和 torch 同一个 index（见 setup 脚本里的注释）。

### 第 1 步：下载（登录节点，有网）

计算节点是离线的（`HF_HUB_OFFLINE=1` 等），下载和 prepare 都必须在登录节点做完。
先 `--dry-run` 看清单和体积。

```bash
python -m data_prep.download --dataset fineweb_edu_pool   # 28.5 GB
python -m data_prep.download --dataset fineweb_edu_probe  #  3.1 GB
python -m data_prep.download --dataset biomed             # 49 GB
python -m data_prep.download --dataset finemath           # 18.4 GB
```

共约 99 GB。probe 默认取 `CC-MAIN-2025-18` 和 `CC-MAIN-2025-26` **各 1 片**（单片约
1.4 GB ≈ 4.9 亿 token，而 probe 只要 16M，1 片就够了；取两个 dump 各 1 片是因为分片
大致按抓取顺序排，跨爬取月份取样比在同一次爬取里多取一片更能代表旧分布）。

**不能取 `CC-MAIN-2024-18` 之前的 dump**，脚本对此有硬检查。依据是查证过的：
open-sci-ref 的论文、LAION 博客、GitHub 三处都把参考数据集写成
「FineWeb-Edu-1.4T (v1.0.0)」，模型名里的 `-1.4t-` 就是他们给 v1.0.0 的版本标签
（1.4T 是 1.3T 的 GPT-2 计数换成 GPT-NeoX-20B 的数）；而 HF 上 `v1.0.0` 分支实测是
95 个 dump、最晚 `CC-MAIN-2024-10`，`2024-18` 是 `v1.2.0` 才加进来的。实际用 2025 的
dump 又比这个下限隔了一年，就算基座用的其实是个稍新的快照也还在安全侧。

### 第 2 步：先量 token 数，再定 ppm

val/test 的 ppm 该给多少，取决于数据集的真实 token 总量，而这个量必须实测：

```bash
python -m data_prep.checks token-count --target-val-tokens 5e7
```

它会读 parquet footer 拿到精确行数，抽样 tokenize 算出每行平均 token 数，外推总量并
直接给出「想让 val 拿到 5000 万 token，`--val-ppm` 该设多少」。

对 FineMath 它还会额外报 GPT-NeoX / Llama 的 token 数之比——官方标的 9.6B 是 Llama
计数，换成 GPT-NeoX 是 9B 还是 12B 方向都不确定，而**这个数决定新域预算定 4B 还是 8B**。

### 第 3 步：FineWeb-Edu 保护侧

```bash
python -m data_prep.checks pool-tokens          # 打出可粘贴的 --pool-tokens
python -m data_prep.prepare_fineweb_edu --stage url-index
python -m data_prep.prepare_fineweb_edu --stage tokenize --role seg00 --pool-tokens <上一步>
python -m data_prep.prepare_fineweb_edu --stage tokenize --role seg01 --pool-tokens <同>
python -m data_prep.prepare_fineweb_edu --stage tokenize --role seg02 --pool-tokens <同>
python -m data_prep.prepare_fineweb_edu --stage tokenize --role probe
python -m data_prep.prepare_fineweb_edu --stage plan  --pool-tokens <同>
```

`--pool-tokens` 三次分段调用和 `plan` 都必须传同一个值，`plan` 会把各段的 ppm 区间与
本次换算结果逐一核对，不一致直接报错（边界一动，段之间就会重叠或漏文档）。

probe 不需要 `--pool-tokens`：它换了来源，抽样 ppm 由 `--probe-tokens`（默认 16M）和
probe 分片自己的 `token_count` 列自动换算，不用再手工量一个数。这里没沿用池子那套
GPT-2→GPT-NeoX 的比值换算，因为 probe 的 token 数不是实验变量，差几个百分点不影响任何
结论；池子那边必须换算，段的 token 数就是 replay 1B/4B/8B 本身。

`--probe-tokens` 是 **URL 排除之前**的量。排除发生在抽样之后，所以最终会少掉与 10BT
重合的那一部分，跑完会把目标和实际并排打出来；剩不到一半会警告（说明 dump 选得太旧）。

`url-index` 必须在 probe 之前跑：probe 靠它排除 10BT 的 URL。它同时也给 `checks
overlap` 检测新域（FineMath）与 FineWeb-Edu 的 URL 重合。

`pool-tokens` 单独做一个子命令而不是复用 `token-count`，是因为估偏的代价太高：三段
tokenize 各要过一遍 28.5GB，估偏几个百分点就得整个重跑。`token-count` 用的是「头部抽样
外推每行均值」，而分片内部大致按抓取顺序排、文档长度又是长尾分布，很容易偏几个百分点。
`pool-tokens` 改成只抽样一个比值：`token_count` 列是每行精确的 GPT-2 计数，只读这一列
不解压正文、全量求和几乎免费，长度的全部变化都被精确捕获；再抽几千篇算 Σneox/Σgpt2，
两个都是 50k 量级的英文 BPE，比值很稳。合成数据上实测偏差 −0.5%。它同时会检查池子装得
下最大的 replay 目标并报余量，装不下直接非零退出。

**replay 和 C 同源，这是设计要求。** 本方法要论证「用旧数据估出来的 C 可以替代重放
这些旧数据」，要让这个对比是同类相比，两条臂就必须面对同一批旧数据。所以 10BT 被切成
若干**累积嵌套**的段，每段既是某条 replay 臂重放的数据、也是那条臂的 C 的来源：

| 段 | 增量 | 累积 | 哪条臂读它 |
|---|---|---|---|
| `seg00` | 1B | 1B | replay 1B / 4B / 8B |
| `seg01` | 3B | 4B | replay 4B / 8B |
| `seg02` | 4B | 8B | replay 8B |
| 未覆盖 | 约 2B | — | 缓冲，池子实测 token 数与估计有出入时的余量 |

各臂的 replay 量由 `--replay-tokens`（默认 `1e9,4e9,8e9`，累积量）直接指定，段是相邻
两个目标之差。划分只能按文档主键哈希做，所以还要 `--pool-tokens`——池子用**本项目
tokenizer** 实测的总 token 数，跑 `checks token-count` 拿，别用数据集名义上的 10B（那是
别家 tokenizer 数的）。ppm 边界由这两个数换算得出。

落到的实际 token 不会正好是 1B / 4B / 8B：ppm 切的是文档份额，而每篇文档长短不同，两者
只在期望意义上相等。`plan` 会把目标和实测并排打出来，差超过 2% 会提示重算。**训练和论文
都用实测那一列**——段要跑满一个 epoch，replay 与 C 才是逐文档相同的；为了凑整去截断，被
截掉的那部分就只进了 C、没被重放。

「同一批」由**目录相同**机械保证，而不是靠「两边流式读到的第 1B 个 token 恰好一样」——
后者要求 shuffle seed、batch size、worker 数、混合 dataloader 的交错方式全都一致，任何
一个变了就悄悄不成立。三次 tokenize 调用的 `--replay-tokens` 和 `--pool-tokens` 必须
完全一致，改一个数所有段的边界就都变了；`plan` 阶段会拿各段 manifest 里记的区间对一遍，
对不上直接报错。

`plan` 阶段产出 `replay_plan.json`，这是 replay 侧和 C 侧唯一的真相来源：每条臂的
`replay_dirs` 与 `cov_dirs` 逐字相同，还带各段实测 token 数和该传给
`--train.max_tokens` 的值。混合 dataloader 和 `kres.mix_covariances` 都从它取目录，
不要各自去拼 `seg%02d`——两边各拼一次就有各拼错一次的机会。

C 侧对每个段各采一次，再按 token 计数加权合并：

```bash
python -m kres.collect_cov --chunks .../fineweb_edu/seg00 --output cov/seg00.pt
python -m kres.mix_covariances --from-plan .../replay_plan.json --arm 2 \
    --cov-dir cov --output cov/arm_4b.pt
```

加权合并与「直接在这几段的并集上采一次」严格等价（存的是 `C_k = Σxxᵀ/n_k` 和 `n_k`，
`Σ(C_k·n_k)/Σn_k` 正好还原成并集的 `Σxxᵀ/Σn`），所以 3 条嵌套的臂只需 3 次采集而不是
1+2+3=6 次。等权平均只在各段 token 数完全相等时才等价，而哈希分段各段会差几个百分点，
所以不提供等权这个选项。

`probe` 换来源，并按 URL 减掉与 10BT 重合的部分，所以必须先跑 `url-index`。

### 第 4 步：Biomed-Enriched（第一个新域）

```bash
CAP="--train-tokens 8e9 --corpus-tokens <token-count 实测的>"
python -m data_prep.prepare_biomed --split train --val-ppm <上一步算的> --test-ppm <同> $CAP
python -m data_prep.prepare_biomed --split val   --val-ppm <同> --test-ppm <同> $CAP
python -m data_prep.prepare_biomed --split test  --val-ppm <同> --test-ppm <同> $CAP
```

三次调用的参数必须完全一致，否则区间边界一动，划分就变了。

`--train-tokens` 不给的话，train 会吃掉 val/test 之外的全部——对 Biomed 是约 28B token、
约 56 GB chunk，而新域预算只有 8B，多出来的 20B 训练时压根读不到。给了就只切需要的那些
文章，剩下的不路由、直接丢弃。

`--train-margin` 默认 1.25，即按目标的 1.25 倍去切。留余量是必须的：`--corpus-tokens`
是抽样外推的，而且它没扣掉 en 过滤丢掉的文章，切少了就不够预算——**而不够不会报错**，
litgpt 的 `CycleIterator` 读完会静默重新开始，「8B 新数据」就变成「把 7B 里的一部分喂
两遍」。所以 `--split train` 跑完会拿实际写入量和 `--train-tokens` 对一次，不够就非零
退出并给出该把 margin 调到多少。val/test 的区间在开头，给 train 设上限不会挪动它们，
已有测试盯着这条。

只用 commercial split（noncommercial 的 text 列是空的，要靠几百 GB 的 PMC OA XML
dump 回填，已放弃）。段落按 `article_id` 分组、按 `_pN` 的整数后缀排序、`\n\n` 拼接。
`language == "en"` 过滤在文章级做：一篇文章里 en 段落占比低于 `--en-threshold`
（默认 0.9）就整篇丢弃，留下来的再剔掉个别非 en 段落。不用 `educational_score` 之类的
curation 标注——那是 Biomed-Enriched 那篇论文自己的贡献，用了就分不清增益来自我们的
正则还是来自它的数据筛选。

跨分片的文章直接丢弃（每个 worker 丢掉自己遇到的第一个和最后一个 run），26 个分片
最多丢 26 篇，对 2400 万篇是 1e-6 量级。脚本会统计这个数，明显超过 `2 × 分片数`
就以非零码退出——那说明段落在文件里不是按文章连续排的，重组结果不可信。

### 第 5 步：FineMath-4+（第二个新域，可推迟）

```bash
python -m data_prep.prepare_finemath --split train
python -m data_prep.prepare_finemath --split val
python -m data_prep.prepare_finemath --split test
```

正式用之前必须先跑重合检查（FineMath 和 FineWeb-Edu 同源于 CommonCrawl）：

```bash
python -m data_prep.checks overlap
```

它算全量 URL 重合（精确）和抽样 50-gram 重合（抓同一内容换了 URL 的情况）。两个数字
都要看，URL 不重合不代表内容不重合。

### 第 6 步：验收

```bash
python -m data_prep.checks tokenizer   # 批量编码路径与 litgpt 逐条路径逐 token 一致
python -m data_prep.checks router      # 划分无重叠、比例对、跨进程一致
python -m data_prep.checks chunks      # 产物能被 pretrain 的读取路径读回
python -m data_prep.checks leakage --dataset biomed   # val/test 与 train 无 50-gram 重叠
```

`leakage` 是唯一一个不依赖「划分代码写对了」这个前提的检查：它只看产物，在 token 级
比 50-gram。无论泄漏是因为划分列选错、哈希不稳定、还是同一篇文章有两个不同主键，
都能抓到。阈值默认 0.5%，不设 0——自然语言里的套话（版权声明、期刊模板句）本来就
会在不同文章之间重复。

## 给 pretrain 用

```bash
litgpt pretrain \
  --data LitData \
  --data.data_path con-pretrain/data/chunks/biomed \
  --data.split_names "[train,val]" \
  --train.tie_embeddings=true \
  ...
```

`--train.tie_embeddings=true` 不能漏。`tie_embeddings` 不是 `Config` 字段而是
`TrainArgs` 的（`args.py:34`），在 `pretrain.py:205` 才生效。转换脚本会把 `wte`
复制给 `lm_head` 所以初值是对的，但不传这个参数，两个矩阵会在 CPT 中各自漂移，
且不报任何错。

**尚未实现**：replay 那几条臂需要把新域和若干个 replay 段混在同一个 dataloader 里，而
`LitData` 只接一个目录。这属于核心代码的活，不在数据准备范围内；本目录保证的是两边的
chunk 分开产出、可以被混合，以及 `replay_plan.json` 里已经写好了每条臂要接哪些目录。

那个 dataloader 有两件事必须做对，否则新域预算会静默走偏：

- **`--train.max_tokens` 要用 `replay_plan.json` 里的 `train_max_tokens`**，它等于
  新域预算 + 该臂的 replay token。直接传 8B 的话，8B 里有一部分被 replay 占掉，新域
  实际只吃到 8B 减去 replay 的量，几条臂的新域预算就不一样了，对比不成立。
- **replay 段要恰好跑满一个 epoch**。litgpt 的 `CycleIterator` 数据读完会静默重新开始，
  于是「replay 1B」可能变成「把 0.5B 重放两遍」，而日志上看不出区别。旧数据在数据流里
  该占的采样比例是 plan 里的 `replay_share_of_stream`（注意它不是「replay 相对新域的
  倍数」：replay 8B 配 8B 新域时这个值是 0.5 而不是 1.0）。

## 实测到的 litdata 约束

来自 `test_code_CPT/probe_litdata.py`（litdata 0.2.75）：

- `optimize` 的 `fn` 必须是模块级函数或它的 `functools.partial`。litdata 把
  multiprocessing 的 start_method 设成 spawn，闭包 pickle 不了。
- 写出时必须传 `item_loader`，见上面的铁律 3。
- `StreamingDataset[i]` 返回的是 numpy 数组而不是 tensor，dtype 与写入时一致。
  所以 token 用 `uint16` 存（`padded_vocab_size=50304 < 65536`），8B token 从
  int64 的 64GB 降到 16GB，`pretrain.py` 里的 `.long()` 能正确提升回 int64。
- 每个 chunk 末尾不足一个 block 的 token 会被丢弃，损耗量级是 `1 / chunk_blocks`。
  默认 `chunk_blocks=8192`（chunk ≈ 33.6M token ≈ 67MB），损耗约 0.01%。调到个位数
  就会掉百分之几，`checks chunks` 会拦。

## 端到端自测

不需要下载真数据：

```bash
wsl -e /home/czq/ENTER/bin/python test_code_CPT/run_kres.py pipeline.log \
    /mnt/e/LLMs_learning/k-res/test_code_CPT/check_data_prep_pipeline.py
```

它造一批合成 parquet（段落级、段落号有空洞且行序打乱、文章跨分片、混多语种），
给每篇文章埋唯一标记，跑完整流水线后把产物解码回文本、抽出标记，直接验证三个 split
的文章集合互不相交、每篇落在 Router 预测的 split、全 fr 文章被过滤、段落被正确重组，
以及 probe 的 URL 排除真的生效、抽样 ppm 与 token_count 换算出的一致。
