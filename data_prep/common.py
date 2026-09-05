"""CPT 数据准备的共享部件。

三条铁律全部在这里落实，下游脚本不要自己造第二套：

1. **划分必须在 tokenize 之前、以文章为最小单位。**
   litdata 把文档拉平成一条 token 流后按固定 block 切块，完全不管 chunk 边界是否
   跨文章。所以一旦划分发生在 tokenize 之后，同一篇文章的前半段会进训练集、后半段
   进 val，评测就被污染了而且看不出来。这里的实现是 `Router`：对文章主键做确定性
   哈希后按 ppm 区间路由，无状态、可并行、跨机器可复现。

2. **tokenize 一律 bos=False, eos=True。**
   与基座的 GPT-NeoX 约定一致。注意这个 tokenizer 的 bos_id == eos_id == 0
   (`<|endoftext|>`)，eos 同时充当文档分隔符。`encode_batch` 逐字节复刻了
   `litgpt.Tokenizer.encode` 的两个边界分支，一致性由 checks.py 的 tokenizer
   对齐检查保证。

3. **写 litdata chunk 必须传 item_loader=TokensLoader(block_size)。**
   实测（test_code_CPT/probe_litdata.py）：不传的话 chunk 元数据里没有 block 布局，
   读回时 StreamingDataset 直接抛 `unsupported operand type(s) for //: NoneType and int`。
   litgpt 自己的 openwebtext.py 是不传的，那条路在当前 litdata 版本下已经不能用。

另有两个实测到的约束：

- litdata 把 multiprocessing 的 start_method 设成 spawn，传给 `optimize` 的 `fn`
  必须是模块级函数（用 functools.partial 绑参数）。闭包会 pickle 失败。
- 每个 chunk 末尾不足一个 block 的 token 会被丢弃。chunk_size = block_size ×
  chunk_blocks，默认 8192 倍，配 ~1000 token 的文档时丢弃率约 0.01%，可忽略；
  但 chunk_blocks 调小到个位数就会掉百分之几。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SPLIT_DENOM = 1_000_000
"""划分粒度：百万分之一（ppm）。区间都用 ppm 表示，便于把 val 定到 0.1% 这种量级。"""

TOKEN_DTYPE = np.uint16
"""token 存盘 dtype。padded_vocab_size=50304 < 65536，uint16 装得下；
8B token 用 int64 存是 64GB，uint16 是 16GB。实测 litdata 会原样保留这个 dtype，
pretrain.py 里的 .long() 能正确提升回 int64。"""

MAX_TOKEN_ID_EXCLUSIVE = 50304
"""模型的 padded_vocab_size。超过这个值说明 tokenizer 和模型对不上，必须硬失败——
lit.vocab_size 返回的 50254 是错的上界（实际 token id 会到 50276），别拿它来卡。"""

DEFAULT_BLOCK_SIZE = 4097
"""seq_len 4096 + 1（pretrain.py 需要多一个 token 做 target 位移）。"""

DEFAULT_CHUNK_BLOCKS = 8192
"""每个 chunk 装多少个 block。4097 × 8192 ≈ 33.6M token ≈ 67MB/chunk（uint16）。"""

SALT = {
    "fineweb_edu": "kres-cpt/fineweb-edu/v1",
    "biomed": "kres-cpt/biomed-enriched-commercial/v1",
    "finemath": "kres-cpt/finemath-4plus/v1",
}
"""每个数据集的划分盐值。**改动它等于重新洗牌所有划分**，只有在明确要重做划分时才改，
并且要同步改版本号后缀，让旧产物和新产物不会被混用。"""

DOC_KEY = {
    "fineweb_edu": "id",  # <urn:uuid:...>，逐文档唯一
    "biomed": "article_id",  # PMC11276146，一篇文章的所有段落共用
    "finemath": "url",  # 没有 id 列，url 就是文档主键
}
"""各数据集的文档/文章主键列。划分只允许基于这一列。"""


# ---------------------------------------------------------------------------
# 确定性划分
# ---------------------------------------------------------------------------


def key_bucket(key: str, salt: str) -> int:
    """把文章主键映射到 [0, SPLIT_DENOM) 的确定性桶号。

    用 blake2b 而不是内置 `hash()`：后者受 PYTHONHASHSEED 随机化影响，换进程换机器
    结果就不一样，而划分必须在 26 个并行 worker 和多次重跑之间完全一致。
    """
    digest = hashlib.blake2b(f"{salt}\x00{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SPLIT_DENOM


@dataclass(frozen=True)
class Router:
    """把文章主键路由到某个 split。区间用 ppm 表示，半开区间 [lo, hi)。

    区间之间不允许重叠——这是「同一篇文章不能同时出现在两个 split」的机械保证。
    未被任何区间覆盖的 key 返回 None（表示丢弃），用于只取子集的场景（比如 probe
    只需要几千万 token，不需要整个 dump）。
    """

    salt: str
    ranges: dict[str, tuple[int, int]]

    def __post_init__(self) -> None:
        if not self.ranges:
            raise ValueError("Router.ranges 不能为空")
        spans = sorted((lo, hi, name) for name, (lo, hi) in self.ranges.items())
        for lo, hi, name in spans:
            if not (0 <= lo < hi <= SPLIT_DENOM):
                raise ValueError(f"split {name!r} 的区间 [{lo}, {hi}) 越界（上限 {SPLIT_DENOM}）")
        for (lo1, hi1, n1), (lo2, hi2, n2) in zip(spans, spans[1:]):
            if hi1 > lo2:
                raise ValueError(f"split {n1!r} [{lo1},{hi1}) 与 {n2!r} [{lo2},{hi2}) 重叠")

    def route(self, key: str) -> str | None:
        b = key_bucket(key, self.salt)
        for name, (lo, hi) in self.ranges.items():
            if lo <= b < hi:
                return name
        return None

    @property
    def names(self) -> list[str]:
        return list(self.ranges)

    def describe(self) -> dict[str, Any]:
        return {
            "salt": self.salt,
            "denom": SPLIT_DENOM,
            "ranges": {k: list(v) for k, v in self.ranges.items()},
            "expected_fraction": {k: (hi - lo) / SPLIT_DENOM for k, (lo, hi) in self.ranges.items()},
        }


def three_way_router(
    dataset: str, val_ppm: int, test_ppm: int, train_ppm: int | None = None
) -> Router:
    """新域数据集的 train/val/test 三分。

    val/test 用 ppm 而不是百分比，因为这两个集合按 token 预算定（各几千万 token 就
    够算 PPL），而 Biomed 有 300 亿量级的 token，用百分比会写成 0.001% 这种容易点错
    小数点的数。具体该给多少 ppm 取决于实测总 token 数，先跑 checks.py 的
    `token-count` 再定。

    `train_ppm=None` 时 train 吃掉 val/test 之外的全部，这对 Biomed 是约 28B token，
    而新域预算只有 8B——多出来的 20B 训练时压根读不到，白花几小时和几十 GB。给了
    `train_ppm` 就只切出那么多，剩下的不路由、直接丢弃。

    val/test 放在区间开头、train 紧接其后，所以给 train 设上限不会动 val/test 的边界，
    三次调用之间只有 train 那一段的长度会变。
    """
    lo = val_ppm + test_ppm
    hi = SPLIT_DENOM if train_ppm is None else lo + train_ppm
    if hi > SPLIT_DENOM:
        raise ValueError(
            f"val({val_ppm}) + test({test_ppm}) + train({train_ppm}) = {hi} 超过 {SPLIT_DENOM}"
        )
    return Router(
        salt=SALT[dataset],
        ranges={
            "val": (0, val_ppm),
            "test": (val_ppm, lo),
            "train": (lo, hi),
        },
    )


def segment_name(index: int) -> str:
    """段目录名。补零是为了 sorted() 的字典序与数值序一致（seg10 不会排在 seg2 前面）。"""
    return f"seg{index:02d}"


def segment_router(segment_ppms: Sequence[int]) -> Router:
    """FineWeb-Edu 保护侧：把 sample-10BT 池子切成若干**累积嵌套**的段。

    每一段既是某条 replay 臂重放的数据，也是那条臂的 C 的来源——**同一批文档，同时
    服务两个用途**。这是整个对比的前提：本方法要论证的是「用旧数据估出来的 C 可以替代
    重放这些旧数据」，两条臂必须面对同一批旧数据，否则就凭空多了一个「看的不是同一批
    数据」的差异。

    段是增量而不是累积量，臂通过读若干个段来组成自己的池子。默认三段对应主表的
    replay {1B, 4B, 8B}：

        seg00 = 1B          → replay 1B 臂读 [seg00]
        seg01 = 3B          → replay 4B 臂读 [seg00, seg01]
        seg02 = 4B          → replay 8B 臂读 [seg00, seg01, seg02]

    ppm 一般不直接给，用 segment_ppms_from_tokens 从「各臂要重放多少 token」换算。

    C 侧对应地把各段的 C 按 token 计数加权合并（见 kres/mix_covariances.py）。加权合并
    与「直接在合并后的数据上采一次」逐比特等价，因为每段存的是 C 和 n，而
    Σ(C_k·n_k)/Σn_k 正好还原成总的 Σxxᵀ/Σn。用简单平均只有在各段 token 数完全相等时
    才等价，而按主键哈希分段时各段会差几个百分点，所以合并一律用加权。

    probe 不在这里：它换来源（2025 的 CC dump，基座没见过），用 `subsample_router`。

    剩余未被任何段覆盖的部分直接丢弃，留作缓冲：池子的真实 token 数要跑
    `checks pool-tokens` 才知道，段的 ppm 是按估计值定的，留一截余量才不至于因为估偏
    了导致最后一段不足。

    **注意**：这里的历史版本曾把「采 C 用」和「replay 用」切成互不重叠的两半，理由是
    「同一批样本会让正则保护的正好是重放过的」。那个理由不成立——协方差正则那条臂的
    replay 比例是 0，它压根不重放任何东西，不存在这个混淆。只有在「同时 replay 又加
    正则」的混合臂上才需要担心，而主表里没有这样的臂。
    """
    if not segment_ppms:
        raise ValueError("segment_ppms 不能为空")
    if any(p <= 0 for p in segment_ppms):
        raise ValueError(f"每段的 ppm 必须为正，收到 {list(segment_ppms)}")
    total = sum(segment_ppms)
    if total > SPLIT_DENOM:
        raise ValueError(
            f"各段 ppm 之和 {total} 超过 {SPLIT_DENOM}（即超过池子的 100%）"
        )

    ranges: dict[str, tuple[int, int]] = {}
    lo = 0
    for i, ppm in enumerate(segment_ppms):
        ranges[segment_name(i)] = (lo, lo + ppm)
        lo += ppm
    return Router(salt=SALT["fineweb_edu"], ranges=ranges)


def segment_ppms_from_tokens(cumulative_tokens: Sequence[int], pool_tokens: int) -> list[int]:
    """把「各条臂要重放多少 token」换算成分段的 ppm。

    实验定义的是固定 token 数（replay 1B / 4B / 8B），而划分只能按文档主键的哈希做，
    所以要先把 token 目标折成池子的比例。落到的实际 token 数不会正好等于目标：ppm 切的
    是文档份额，而每篇文档的长度不同，两者只在期望意义上相等。千万级文档下偏差在 1%
    以内，`plan` 阶段会把目标和实测并排打出来，超过 2% 会警告。

    `pool_tokens` 必须是**用本项目 tokenizer 实测**的池子总 token 数（`checks
    token-count`），不能用数据集名义上的 10B——那是别家 tokenizer 数的，差几个百分点
    就会让最后一段不够。
    """
    if pool_tokens <= 0:
        raise ValueError(f"pool_tokens 必须为正，收到 {pool_tokens}")
    if not cumulative_tokens:
        raise ValueError("cumulative_tokens 不能为空")
    if any(b <= a for a, b in zip(cumulative_tokens, cumulative_tokens[1:])):
        raise ValueError(f"各臂的 replay token 必须严格递增（段是增量），收到 {list(cumulative_tokens)}")
    if cumulative_tokens[0] <= 0:
        raise ValueError(f"replay token 必须为正，收到 {list(cumulative_tokens)}")
    if cumulative_tokens[-1] > pool_tokens:
        raise ValueError(
            f"最大的 replay 量 {cumulative_tokens[-1]:,} 超过池子实测的 {pool_tokens:,} token"
        )

    bounds = [round(t / pool_tokens * SPLIT_DENOM) for t in cumulative_tokens]
    ppms = [bounds[0]] + [b - a for a, b in zip(bounds, bounds[1:])]
    if any(p <= 0 for p in ppms):
        raise ValueError(
            f"换算出的 ppm 有非正值 {ppms}：相邻两个 replay 目标离得太近，"
            f"1 ppm 在这个池子里是 {pool_tokens // SPLIT_DENOM:,} token"
        )
    return ppms


def ppm_from_tokens(tokens: int, source_tokens: int, what: str) -> int:
    """把一个 token 目标折成来源的 ppm。给 probe 这种单段用，段列表用
    `segment_ppms_from_tokens`。
    """
    if source_tokens <= 0:
        raise ValueError(f"source_tokens 必须为正，收到 {source_tokens}")
    if tokens <= 0:
        raise ValueError(f"{what} 必须为正，收到 {tokens}")
    if tokens > source_tokens:
        raise ValueError(
            f"{what}={tokens:,} 超过来源的 {source_tokens:,} token，多下几个分片"
        )
    ppm = round(tokens / source_tokens * SPLIT_DENOM)
    if ppm <= 0:
        raise ValueError(
            f"{what}={tokens:,} 在这个来源里不到 1 ppm"
            f"（1 ppm = {source_tokens // SPLIT_DENOM:,} token），调大一点"
        )
    return ppm


def subsample_router(dataset: str, name: str, ppm: int) -> Router:
    """从一个来源里按 ppm 抽一个子集，其余丢弃。用于 held-out probe。

    抽样按主键哈希而不是按流式位置截断：CC-MAIN 的 parquet 分片大致按抓取顺序排列，
    取前 N 行会偏向特定站点/时间段，而哈希抽样是无偏的。
    """
    return Router(salt=SALT[dataset], ranges={name: (0, ppm)})


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


def load_tokenizer(tokenizer_dir: str | Path):
    """加载 litgpt 的 Tokenizer，并把这个项目依赖的前提断言掉。"""
    from litgpt.tokenizer import Tokenizer

    tok = Tokenizer(Path(tokenizer_dir))
    if tok.backend != "huggingface":
        raise RuntimeError(f"期望 tokenizers 后端（读 tokenizer.json），实际是 {tok.backend!r}")
    if tok.eos_id is None:
        raise RuntimeError("tokenizer 没有 eos_id，无法逐文档追加文档分隔符")
    if tok.use_bos:
        raise RuntimeError(f"{tokenizer_dir} 的 use_bos 是 True，与基座的 bos=False 约定冲突")
    return tok


def encode_batch(tok, texts: Sequence[str]) -> list[np.ndarray]:
    """批量编码，语义与 `tok.encode(text, bos=False, eos=True)` 逐 token 一致。

    走 `processor.encode_batch` 是为了速度：8B token 规模下逐条调 Python 层的 encode
    会成为整个流水线的瓶颈。下面两个分支是从 litgpt/tokenizer.py L131-144 抄来的，
    包括那个反直觉的行为——文本本身以 eos 结尾时 litgpt 会把它删掉而不是保留。
    checks.py 的 `tokenizer` 检查会拿真实文本比对这两条路径。
    """
    bos_id, eos_id = tok.bos_id, tok.eos_id
    out: list[np.ndarray] = []
    for enc in tok.processor.encode_batch(list(texts)):
        ids = enc.ids
        if ids and bos_id is not None and ids[0] == bos_id:
            ids = ids[1:]
        if not ids or ids[-1] != eos_id:
            ids = [*ids, eos_id]
        else:
            ids = ids[:-1]
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size and int(arr.max()) >= MAX_TOKEN_ID_EXCLUSIVE:
            raise ValueError(
                f"出现 token id {int(arr.max())} >= padded_vocab_size {MAX_TOKEN_ID_EXCLUSIVE}，"
                "tokenizer 和模型对不上"
            )
        out.append(arr.astype(TOKEN_DTYPE))
    return out


# ---------------------------------------------------------------------------
# parquet 读取
# ---------------------------------------------------------------------------


def iter_parquet_batches(
    path: str | Path, columns: Sequence[str], batch_size: int = 2048
) -> Iterator[dict[str, list]]:
    """逐批读 parquet，产出 {列名: 值列表}。

    按批而不是按行：批量喂给 encode_batch 才有速度，而且 pyarrow 的 iter_batches 不会
    把整个 1.9GB 分片读进内存。
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    missing = set(columns) - set(pf.schema_arrow.names)
    if missing:
        raise KeyError(f"{path} 缺少列 {sorted(missing)}；实际列：{pf.schema_arrow.names}")
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns)):
        yield {name: batch.column(name).to_pylist() for name in columns}


# ---------------------------------------------------------------------------
# litdata 写出
# ---------------------------------------------------------------------------


def prepare_output_dir(output_dir: str | Path, overwrite: bool) -> Path:
    """litdata 往非空目录里写会产生半新半旧的 chunk，且 index.json 只反映新的那批。
    宁可在这里硬失败。"""
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} 已存在且非空。确认要重做的话加 --overwrite（会整目录删除）")
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_token_chunks(
    fn,
    inputs: list,
    output_dir: str | Path,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    chunk_blocks: int = DEFAULT_CHUNK_BLOCKS,
    num_workers: int | None = None,
    fast_dev_run: bool = False,
    overwrite: bool = False,
) -> Path:
    """把 token 流写成 litdata chunk，产物可直接给 `LitData` data module。

    `fn` 必须是模块级函数或它的 functools.partial（litdata 用 spawn，闭包 pickle 不了），
    对每个 input 产出若干 numpy uint16 数组。
    """
    from litdata import optimize
    from litdata.streaming import TokensLoader

    out = prepare_output_dir(output_dir, overwrite)
    if num_workers is None:
        num_workers = max(1, min(len(inputs), (os.cpu_count() or 2) - 1))

    optimize(
        fn=fn,
        inputs=inputs,
        output_dir=str(out),
        num_workers=num_workers,
        chunk_size=block_size * chunk_blocks,
        item_loader=TokensLoader(block_size=block_size),
        fast_dev_run=fast_dev_run,
    )
    return out


# ---------------------------------------------------------------------------
# 统计与 manifest
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    """worker 侧的计数器。

    litdata 不回收 `fn` 的返回值，所以每个 worker 把自己的计数写成一个 json 边车文件，
    主进程再 `merge_stats` 汇总。没有这些数字就没法填 manifest，也没法判断 val_ppm
    该调到多少。
    """

    counters: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def dump(self, stats_dir: str | Path, tag: str) -> None:
        d = Path(stats_dir)
        d.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
        (d / f"{safe}.json").write_text(json.dumps(self.counters, indent=2), encoding="utf-8")


def merge_stats(stats_dir: str | Path) -> dict[str, int]:
    total: dict[str, int] = {}
    for p in sorted(Path(stats_dir).glob("*.json")):
        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
            total[k] = total.get(k, 0) + int(v)
    return total


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """每份产物旁边写一个 manifest。

    里面必须能回答「这批 chunk 是谁、用什么划分、什么盐值、多少 token 生成的」——
    否则几周后拿到一堆 chunk 目录根本分不清哪个对应哪条臂。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "token_dtype": np.dtype(TOKEN_DTYPE).name,
        **payload,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_tokens(n: float) -> str:
    """token 数按量级选单位。

    固定用 B 打印会把小规模的跑（抽样子集、合成数据自测）显示成 `0.00B`，分不清
    是真的没数据还是精度掉了。
    """
    n = float(n)
    for scale, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return f"{n:.0f}"


def url_hash(url: str) -> np.uint64:
    """URL 的 64 位指纹，用于 probe 与 replay 池的去重。

    存 8 字节整数而不是原串：10BT 池子有千万级文档，整数数组配 searchsorted 做成员
    查询是 80MB 的内存和对数时间，存原串要几 GB。64 位空间下千万级 key 的碰撞概率
    在 1e-5 量级，对「排除掉污染样本」这个用途足够。
    """
    return np.frombuffer(hashlib.blake2b(url.encode("utf-8"), digest_size=8).digest(), dtype=">u8")[0].astype(
        np.uint64
    )


def url_hashes(urls: Sequence[str]) -> np.ndarray:
    return np.fromiter((url_hash(u) for u in urls), dtype=np.uint64, count=len(urls))
