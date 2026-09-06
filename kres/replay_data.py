"""replay 臂的混合 dataloader：新域数据 + 若干 replay 段，按确定性图案交错。

litgpt 自带的 `LitData` 只接一个目录，而 replay 臂要同时读新域和 1~3 个 replay 段。
读哪些目录来自 `replay_plan.json`，不在这里重新拼路径——两边各拼一次就有各拼错一次的
机会。

**为什么不用 litdata 自带的加权采样。** `CombinedStreamingDataset(weights=...)` 每取一个
样本掷一次骰子决定来源，跑完全程后 replay 侧被消耗的总量对，但具体喂了哪些文档不受控：
有的抽到两次，有的一次都没抽到。而正则臂的 C 是在整个段上精确统计的，所以那样一来
「重放的数据就是采 C 的数据」只在总量意义上成立，逐文档看不成立——而整个对比的前提正是
两条臂面对同一批旧数据。这里改成确定性分配，每个段恰好跑满一个 epoch。

**为什么连 `iterate_over_all=True` 也不用。** 那个模式的覆盖本身是对的（实测两个目录
各跑满一遍），但它在 `num_workers>0` 下**存档续跑会静默丢样本**：48 条的合成数据，
第 3 个 batch 处存档再续，回来只剩 46 条，不重复、就是少了 2 条，没有任何报错。单个
`StreamingDataset` 在同样条件下完好无损。6 小时 walltime 意味着要续 4~5 次，所以这里
一个段一个 loader，全部走单源路径。

**配额由磁盘决定，不由 plan 的 token 数决定。** 每个段该喂多少个 micro-batch 直接取
`len(loader)`，也就是那个目录里真实有多少条序列。plan 里的 token 数是给人看和给 C 加权
用的，拿它去除以 batch 大小会有取整误差，而误差的方向恰好是「少喂一点」——那部分文档
进了 C 却没被重放，正是要避免的情形。

**交错的粒度是 micro-batch 而不是 micro-batch 内部。** 两者对累积后的梯度**完全等价**：
每个 micro-batch 的 loss 都除以 gradient_accumulation_iters，batch 内部又对 token 取
平均，所以每条序列的权重一律是 1/(grad_accum × micro_batch)，与它和谁同批无关。按
micro-batch 交错能直接复用每个流自己的 worker 和续跑状态，代价只是单个 micro-batch 里
不是新旧混装。

**耗尽一律抛 RuntimeError，绝不抛 StopIteration。** litgpt 把 train_dataloader 包在
`CycleIterator` 里，它吃掉 StopIteration 然后从头再来一遍，于是「replay 1B」会变成
「把 0.6B 重放两遍」，日志上完全看不出区别。这是这份代码要挡的头号静默失效。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer

NO_REPLAY = "no_replay"
"""不重放的臂（baseline 与协方差正则臂）也走这个数据模块。

它们本可以直接用 litgpt 的 `LitData`，但那样新域的迭代顺序由另一套代码决定，
「C 臂与 baseline 除了惩罚项以外完全一致」这句话就少了一层保证。同一个模块、同一个
seed，新域数据流才是逐 batch 相同的。
"""

NEW = "new"


def allocate(counts: Sequence[int]) -> list[int]:
    """把 N 个 micro-batch 的名额按 counts 分给各个流，返回长度 sum(counts) 的流下标序列。

    每一步把名额给「当前欠得最多」的流：最大化 `(i+1)·n_s − c_s·N`。整数运算，没有浮点
    累积误差（跑到第八千个 micro-batch 时浮点足以让计数差一两个）。

    两条性质，测试里都验了：任意前缀 i 处每个流的已发数与理想值 `i·n_s/N` 差不到 1
    （所以每个段都均匀铺在全程上，而不是集中在某一段时间）；跑满 N 步后各流恰好等于
    `n_s`（所以每个段恰好一个 epoch，不多不少）。
    """
    total = sum(counts)
    served = [0] * len(counts)
    order = []
    for i in range(total):
        # 平局固定按下标取，保证同一份配额永远产出同一个序列（续跑要靠这个复现）
        best = max(range(len(counts)), key=lambda s: ((i + 1) * counts[s] - served[s] * total, -s))
        served[best] += 1
        order.append(best)
    return order


def load_arm(plan_path: Path, arm: str) -> dict:
    """从 replay_plan.json 里取一条臂。找不到就把可选项列出来。"""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    names = [a["name"] for a in plan["arms"]]
    if arm == NO_REPLAY:
        # 不重放臂在 plan 里没有对应条目（plan 只描述 replay 侧），这里造一条零重放的
        base = plan["arms"][0]
        return {
            "name": NO_REPLAY,
            "segments": [],
            "replay_dirs": [],
            "cov_dirs": [],
            "replay_tokens": 0,
            "new_tokens": base["new_tokens"],
            "train_max_tokens": base["new_tokens"],
        }
    for a in plan["arms"]:
        if a["name"] == arm:
            # 同源是这套实验的地基，读的时候再断言一次：写 plan 时已经断言过，但 plan 是
            # 个 json，中途被谁改过一笔完全可能，而改坏了不会有任何症状
            if a["replay_dirs"] != a["cov_dirs"]:
                raise ValueError(
                    f"臂 {arm} 的 replay_dirs 与 cov_dirs 不一致，重放的数据不是采 C 的数据：\n"
                    f"  replay={a['replay_dirs']}\n  cov   ={a['cov_dirs']}"
                )
            return a
    raise ValueError(f"{plan_path} 里没有臂 {arm!r}，可选：{names + [NO_REPLAY]}")


class _Placeholder(IterableDataset):
    """占位数据集。父类 `DataLoader` 的取数机制整个不走（`__iter__` 被覆盖了），它只为
    两件事存在：`setup_dataloaders` 要求参数是 `DataLoader` 实例，而且它靠
    `isinstance(dl.dataset, IterableDataset)` 判断能不能跳过分布式采样器包装——包装了就
    会重新实例化这个类，直接炸。
    """

    def __iter__(self):
        return iter(())


class MixedReplayLoader(DataLoader):
    """按确定性图案在新域与各 replay 段之间取 micro-batch。"""

    def __init__(
        self,
        loaders: dict[str, DataLoader],
        batches: dict[str, int],
        *,
        arm: str,
        tokens_per_batch: int,
    ) -> None:
        super().__init__(_Placeholder(), batch_size=None, num_workers=0)
        self.arm = arm
        self.tokens_per_batch = tokens_per_batch
        self.names = list(loaders)
        self._loaders = loaders
        self.batches = batches
        self._order = allocate([batches[n] for n in self.names])
        self._cursor = 0
        self._consumed = {n: 0 for n in self.names}
        self._fresh = True

    # -- 供 pretrain.py 交叉核对与收尾汇报 --------------------------------------

    @property
    def replay_batches(self) -> int:
        return sum(v for k, v in self.batches.items() if k != NEW)

    @property
    def remaining(self) -> int:
        """还剩几个 micro-batch 名额。续跑后不等于 `len(self)`。"""
        return len(self._order) - self._cursor

    @property
    def new_tokens_consumed(self) -> int:
        """到目前为止喂进去的**新域** token 数。

        跨臂画曲线的横轴只能用这个，不能用 step：各臂的 step 里掺的 replay 比例不同，
        同样是第 200 步，vanilla 已经吃了 419M 新域 token，replay 8B 只吃了约 210M。
        按 step 对齐等于拿「学了两倍新域」的点去比「忘得更少」，两条曲线不可比。
        """
        return self._consumed.get(NEW, 0) * self.tokens_per_batch

    def consumption_summary(self) -> str:
        t = self.tokens_per_batch
        want = " ".join(f"{n}={self.batches[n] * t / 1e9:.3f}B" for n in self.names)
        got = " ".join(f"{n}={self._consumed[n] * t / 1e9:.3f}B" for n in self.names)
        lines = [f"数据流（臂 {self.arm}）计划 {want}", f"                实际 {got}"]
        short = [n for n in self.names if self._consumed[n] < self.batches[n]]
        if short:
            lines.append(
                f"  ⚠ {', '.join(short)} 没跑满。没被重放的那部分文档仍然进了 C，"
                f"「replay 与 C 同源」对它们就不成立了。多半是 --train.max_tokens 小于 plan "
                f"里的 train_max_tokens，或者训练被 max_steps / max_time 提前截断了"
            )
        return "\n".join(lines)

    # -- 取数 ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self):
        # 中途重新 iter() 会让各子流从头开一个新 epoch，而本对象的游标还停在原处——数据被
        # 重发，名额却照扣。生产路径上 CycleIterator 只在开头 iter 一次（我们从不抛
        # StopIteration，所以它不会重开），能走到这里说明调用方式变了，直接拦住。
        #
        # 这个检查必须放在**非**生成器的 __iter__ 里：写成生成器的话，iter() 只是造一个
        # 生成器对象，函数体要等第一次 next() 才执行，护栏就形同虚设
        if not self._fresh:
            raise RuntimeError(
                f"臂 {self.arm} 的 dataloader 在跑到第 {self._cursor} 个 micro-batch 时被重新 "
                f"iter()。各子流会从头开一个新 epoch 而游标不回退，等于把前面的数据又喂一遍。"
            )
        self._fresh = False
        return self._generate()

    def _generate(self):
        iters = {n: iter(dl) for n, dl in self._loaders.items()}
        while self._cursor < len(self._order):
            name = self.names[self._order[self._cursor]]
            try:
                batch = next(iters[name])
            except StopIteration:
                done = self._consumed[name]
                raise RuntimeError(
                    f"流 {name} 在第 {self._cursor} 个 micro-batch 就取空了"
                    f"（已消耗 {done}/{self.batches[name]} 个 batch）。\n"
                    f"这里必须硬失败而不能重头再来：litgpt 的 CycleIterator 会吃掉 StopIteration "
                    f"并静默重开，于是「replay 恰好一个 epoch」变成「不足的那部分重放两遍」，"
                    f"日志上看不出区别。\n"
                    f"多半是这个目录的 chunk 没生成全，或者续跑时载入了别的臂的状态。"
                ) from None
            self._cursor += 1
            self._consumed[name] += 1
            yield batch

        # 名额跑完还在要数据，说明 --train.max_tokens 比 plan 大。让它落到上面 while 的
        # 出口会抛 StopIteration，又被 CycleIterator 静默重开，所以这里显式挡住
        raise RuntimeError(
            f"臂 {self.arm} 的 {len(self._order)} 个 micro-batch 名额已用完，但训练还在继续。"
            f"--train.max_tokens 应当等于 plan 里的 train_max_tokens。"
        )

    # -- 续跑 ------------------------------------------------------------------
    # pretrain.py 把 train_dataloader 放进 fabric.load 的 state 里（`_unwrap_objects` 会
    # 把 _FabricDataLoader 拆回本对象，所以下面两个方法确实会被调到），而且**没有**快进
    # 逻辑（initial_iter 只用来算 ETA）。也就是说续跑能不能接上全看这两个方法：漏了
    # _cursor，交错图案会从头开始，各段的消耗量就全错了。

    def state_dict(self) -> dict:
        return {
            "arm": self.arm,
            "cursor": self._cursor,
            "consumed": dict(self._consumed),
            "streams": {n: dl.state_dict() for n, dl in self._loaders.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("arm") != self.arm:
            raise ValueError(
                f"检查点存的是臂 {state.get('arm')!r}，当前是 {self.arm!r}。"
                f"接着跑会按错误的配额分配名额，各段消耗量全错。"
            )
        if set(state["streams"]) != set(self._loaders):
            raise ValueError(
                f"检查点的数据流与当前不一致：{sorted(state['streams'])} vs {sorted(self._loaders)}"
            )
        self._cursor = int(state["cursor"])
        self._consumed = dict(state["consumed"])
        self._fresh = True
        for n, dl in self._loaders.items():
            dl.load_state_dict(state["streams"][n])


@dataclass
class ReplayMixedData(DataModule):
    """新域 + replay 段的混合数据源，目录与配额取自 replay_plan.json 与磁盘上的实际序列数。

    用法（臂名见 plan 里 arms[].name，不重放的臂传 `no_replay`）：

        --data kres.replay_data.ReplayMixedData
        --data.plan_path  .../chunks/fineweb_edu/replay_plan.json
        --data.arm        replay_8.00B
        --data.new_data_path  .../chunks/biomed/train
        --data.val_data_path  .../chunks/biomed/val
        --data.probe_data_path .../chunks/fineweb_edu/probe
        --data.test_data_path .../chunks/biomed/test
    """

    plan_path: Path = Path("replay_plan.json")
    """replay_plan.json 的路径，由 prepare_fineweb_edu --stage plan 产出。"""
    arm: str = NO_REPLAY
    """臂名。plan 里 arms[].name 之一，或 `no_replay`。"""
    new_data_path: Path = Path("data/")
    """新域训练集的 chunk 目录（比如 chunks/biomed/train）。"""
    val_data_path: Path | None = None
    """新域 held-out（chunks/biomed/val）。塑性侧的主验证集，训练中周期性评。"""
    probe_data_path: Path | None = None
    """旧域 held-out（chunks/fineweb_edu/probe）。遗忘量就是它相对基座涨了多少，训练中周期性评。"""
    test_data_path: Path | None = None
    """新域 test（chunks/biomed/test）。**只在收尾评一次**，见 `final_eval_dataloaders`。"""
    seed: int = 42
    num_workers: int = 2
    """**每个流**的 worker 数。总进程数是 (1 + 段数) × 这个值，8B 臂就是 4 倍。"""

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def __post_init__(self) -> None:
        super().__init__()
        self._arm = load_arm(self.plan_path, self.arm)

    def connect(
        self, tokenizer: Tokenizer | None = None, batch_size: int = 1, max_seq_length: int | None = None
    ) -> None:
        self.batch_size = batch_size
        # 与 LitData 一致：多取一个 token 当 next-token 目标
        self.seq_length = max_seq_length + 1

    @property
    def expected_max_tokens(self) -> int:
        """该传给 `--train.max_tokens` 的值 = 新域预算 + 该臂的 replay 量。"""
        return int(self._arm["train_max_tokens"])

    def _loader(self, path: str, seed: int, shuffle: bool, num_workers: int | None = None) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        return StreamingDataLoader(
            StreamingDataset(
                input_dir=str(path),
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=shuffle,
                seed=seed,
            ),
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers if num_workers is None else num_workers,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        dirs = {NEW: str(self.new_data_path)}
        for name, d in zip(self._arm["segments"], self._arm["replay_dirs"]):
            dirs[name] = d
        missing = [d for d in dirs.values() if not Path(d, "index.json").exists()]
        if missing:
            raise SystemExit(f"这些目录下没有 litdata 的 index.json，chunk 没生成或路径不对：{missing}")

        # 每个流一个不同的 seed，否则各段的洗牌置换相同——它们是同一个池子的哈希分片，
        # 长度相近，同置换会让「第 i 篇」在各段之间产生本不该有的对应关系
        loaders = {n: self._loader(d, self.seed + i, shuffle=True) for i, (n, d) in enumerate(dirs.items())}

        tokens_per_batch = self.batch_size * (self.seq_length - 1)
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        # 与 pretrain.py 的 max_iters 同口径：它也是先按 world_size 分摊再除以每 iter 的 token
        total = self.expected_max_tokens // world // tokens_per_batch

        # replay 各段的名额直接取磁盘上真实有多少个整 batch，不用 plan 里的 token 数除一下：
        # 除法的取整误差方向是「少喂一点」，那部分文档进了 C 却没被重放
        batches = {n: len(dl) for n, dl in loaders.items() if n != NEW}
        batches[NEW] = total - sum(batches.values())
        self._check(batches, loaders, total, tokens_per_batch)

        return MixedReplayLoader(
            {NEW: loaders[NEW], **{n: loaders[n] for n in dirs if n != NEW}},
            batches,
            arm=self._arm["name"],
            tokens_per_batch=tokens_per_batch,
        )

    def _check(self, batches: dict[str, int], loaders: dict[str, DataLoader], total: int, tpb: int) -> None:
        """开跑前就把配额对不上的情形拦下来，别等到第 4 小时才发现。"""
        if batches[NEW] <= 0:
            raise SystemExit(
                f"新域名额算成 {batches[NEW]}：replay 段的实际序列数已经占满了全部 {total} 个 "
                f"micro-batch。plan 里这条臂的 train_max_tokens 与磁盘上的 chunk 对不上。"
            )
        if len(loaders[NEW]) < batches[NEW]:
            raise SystemExit(
                f"新域只有 {len(loaders[NEW])} 个 micro-batch，需要 {batches[NEW]} 个。"
                f"prepare_biomed 的 --train-tokens 给小了，或者 --train-margin 不够"
                f"（不够不会自己暴露：跑到一半 CycleIterator 会静默重开，新域被喂两遍）。"
            )
        want = int(self._arm["replay_tokens"])
        got = sum(v for k, v in batches.items() if k != NEW) * tpb
        if want and abs(got - want) > 0.02 * want:
            raise SystemExit(
                f"磁盘上 replay 段共 {got / 1e9:.3f}B token，plan 里写的是 {want / 1e9:.3f}B，"
                f"差了 {abs(got - want) / want:.1%}。chunk 没生成全，或者 plan 与数据不是同一批。"
            )
        print(
            f"混合数据流（臂 {self._arm['name']}）：共 {total} 个 micro-batch，"
            + "，".join(f"{n} {batches[n]}" for n in batches)
        )

    # -- 评估集 --------------------------------------------------------------------
    # 三个 held-out 各约 16M token，分工不同：
    #   val   新域 biomed/val        塑性：新域学进去了多少
    #   probe 旧域 fineweb_edu/probe 稳定性：遗忘量。2025 的 CC dump 且按 URL 排除过
    #                                10BT，对基座是干净的
    #   test  新域 biomed/test       只在收尾评一次
    #
    # test 为什么不跟着一起周期性评：λ 要扫三个点，而扫描是看着曲线选的。选的时候看过
    # 的集合就不再是 held-out 了。val + probe 参与选 λ，test 只在最后报一次，主表里那个
    # 数才是干净的。多评一个集合的算力成本可以忽略，被污染的 test 却没法补救。

    def _eval_loader(self, path: str) -> DataLoader:
        """评估用的 loader：不洗牌、**不开 worker**。

        `num_workers=0` 是有意的，两个原因：
        - litdata 按 chunk 把数据分给 worker，而一个 chunk 是 8192 个 block ≈ 33.6M
          token，三个评估集各约 16M，也就是一两个 chunk。worker 一多就有 worker 空手，
          那条流提前取空，评估悄悄少算一批数据而不报错。
        - 评估 loader 活得和训练一样久，而 8B replay 臂已经有 4 条训练流。in-flight 的
          共享内存张量数是 流数 × num_workers × prefetch_factor，603854 正是被 ENFILE
          （errno 23，全系统 file table 满）打死的。评估只有 100 个 batch、GPU 前向占
          绝大部分时间，取数从来不是瓶颈，多开 worker 是纯亏。

        不洗牌 + 固定 seed，所以每次评的是同一批 batch。这是曲线可比的前提：换一批数据
        重评，step 之间的差里就掺进了数据难度的差。
        """
        if not Path(path, "index.json").exists():
            raise SystemExit(f"{path} 下没有 litdata 的 index.json，chunk 没生成或路径不对")
        return self._loader(path, self.seed, shuffle=False, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        """主验证集（新域 held-out）。上游 litgpt 只认这一个，保留它的契约。"""
        if self.val_data_path is None:
            raise SystemExit(
                "没给 --data.val_data_path（新域 held-out，一般指向 chunks/biomed/val）"
            )
        return self._eval_loader(str(self.val_data_path))

    def eval_dataloaders(self) -> dict[str, DataLoader]:
        """训练中每 `--eval.interval` 步评一次的集合。第一个是主验证集。"""
        loaders = {"val": self.val_dataloader()}
        if self.probe_data_path is not None:
            loaders["probe"] = self._eval_loader(str(self.probe_data_path))
        return loaders

    def final_eval_dataloaders(self) -> dict[str, DataLoader]:
        """只在收尾评一次的集合。见上面那段关于 test 为什么不参与调参的说明。"""
        if self.test_data_path is None:
            return {}
        return {"test": self._eval_loader(str(self.test_data_path))}
