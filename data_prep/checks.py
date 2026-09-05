"""数据准备的检查套件。五个子命令，每个都针对一种「不报错但结论是错的」失效。

    python -m data_prep.checks tokenizer      # 批量编码与 litgpt.Tokenizer 逐 token 一致？
    python -m data_prep.checks router         # 划分是不是一个真正的划分（无重叠、比例对、跨进程一致）？
    python -m data_prep.checks token-count    # 各数据集在 GPT-NeoX 下的真实 token 数是多少？
    python -m data_prep.checks chunks         # 产出的 chunk 能被 pretrain 的读取路径正确读回？
    python -m data_prep.checks leakage        # val/test 与 train 之间真的没有 50-gram 重叠？
    python -m data_prep.checks overlap        # FineMath 与 FineWeb-Edu 的 URL / 50-gram 重合有多少？

约定：每个检查失败就以非零码退出，可以直接串进 shell 的 && 链里。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from data_prep.common import (
    DEFAULT_BLOCK_SIZE,
    MAX_TOKEN_ID_EXCLUSIVE,
    SALT,
    SPLIT_DENOM,
    TOKEN_DTYPE,
    encode_batch,
    fmt_int,
    fmt_tokens,
    iter_parquet_batches,
    key_bucket,
    load_tokenizer,
    three_way_router,
    url_hashes,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = ROOT / "model" / "open-sci-ref-v0.02-0.4b-fineweb-edu-1.4t-300B-4096"
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_CHUNKS = ROOT / "data" / "chunks"

# 各数据集的正文列与主键列，token-count / overlap 都靠这张表定位
SOURCE = {
    "fineweb_edu_pool": ("fineweb_edu_pool", "text", "id"),
    "fineweb_edu_probe": ("fineweb_edu_probe", "text", "id"),
    "biomed": ("biomed", "text", "article_id"),
    "finemath": ("finemath", "text", "url"),
}


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# 1) tokenizer：批量路径与 litgpt 的逐条路径是否逐 token 一致
# ---------------------------------------------------------------------------


def check_tokenizer(args) -> int:
    """common.encode_batch 走的是 processor.encode_batch，litgpt 走的是逐条 encode。

    两者的差异会静默地改变整个语料的 token 边界——多一个少一个 eos 都会让 loss 对不上，
    但不会报错。所以拿真实文本逐 token 比对，而不是相信「我抄对了那两个分支」。
    """
    tok = load_tokenizer(args.tokenizer_dir)
    print(f"tokenizer: {args.tokenizer_dir}")
    print(f"  backend={tok.backend} use_bos={tok.use_bos} bos_id={tok.bos_id} eos_id={tok.eos_id}")
    print(f"  vocab_size={tok.vocab_size}（注意这是错的上界，实际 token id 会超过它）")

    failures = 0
    if tok.bos_id != tok.eos_id:
        bad(f"预期 bos_id == eos_id == 0（<|endoftext|>），实际 bos={tok.bos_id} eos={tok.eos_id}")
        failures += 1
    else:
        ok(f"bos_id == eos_id == {tok.eos_id}")

    texts = _sample_texts(args, n=args.sample)
    # 掺进几个边界样本：空串、纯空白、本身以 eos 文本结尾的串
    texts += ["", "   ", "hello world", "ends with eos<|endoftext|>"]
    print(f"\n拿 {len(texts)} 条文本比对批量路径 vs litgpt 逐条路径：")

    mismatch = 0
    for i, t in enumerate(texts):
        mine = encode_batch(tok, [t])[0]
        theirs = tok.encode(t, bos=False, eos=True).numpy()
        if mine.shape != theirs.shape or not np.array_equal(mine.astype(np.int64), theirs.astype(np.int64)):
            mismatch += 1
            if mismatch <= 3:
                bad(f"第 {i} 条不一致：batch={mine[:12].tolist()}... litgpt={theirs[:12].tolist()}...")
    if mismatch:
        bad(f"{mismatch}/{len(texts)} 条不一致")
        failures += 1
    else:
        ok(f"{len(texts)} 条全部逐 token 一致")

    # 每篇文档必须正好以一个 eos 收尾，这是 chunk 里唯一的文档边界信息
    arrs = encode_batch(tok, [t for t in texts if t.strip()])
    if all(a.size and int(a[-1]) == tok.eos_id for a in arrs):
        ok("每篇非空文档都以 eos 结尾")
    else:
        bad("有文档没有以 eos 结尾")
        failures += 1

    max_id = max((int(a.max()) for a in arrs if a.size), default=0)
    if max_id < MAX_TOKEN_ID_EXCLUSIVE:
        ok(f"最大 token id {max_id} < padded_vocab_size {MAX_TOKEN_ID_EXCLUSIVE}，uint16 装得下")
    else:
        bad(f"最大 token id {max_id} 越界")
        failures += 1

    return 1 if failures else 0


def _sample_texts(args, n: int) -> list[str]:
    """从已下载的任一数据集里取一些真实文本；没下载就退化成内置样本。"""
    for name, (sub, text_col, _) in SOURCE.items():
        shards = sorted((Path(args.raw_root) / sub).rglob("*.parquet"))
        if not shards:
            continue
        print(f"  取样来源：{name} / {shards[0].name}")
        out: list[str] = []
        for batch in iter_parquet_batches(shards[0], (text_col,), batch_size=min(n, 512)):
            out.extend(t for t in batch[text_col] if t)
            if len(out) >= n:
                break
        return out[:n]
    print("  （raw 下没有 parquet，退化为内置样本）")
    return [
        "The mitochondrion is a double-membrane-bound organelle.",
        "Let $f(x) = x^2 + 3x - 4$. Solve $f(x) = 0$.\n\nFactoring gives $(x+4)(x-1) = 0$.",
        "第一段中文。\n\n第二段中文，用来看 tokenizer 对非拉丁文的处理。",
    ] * (n // 3 + 1)


# ---------------------------------------------------------------------------
# 2) router：划分是不是一个真正的划分
# ---------------------------------------------------------------------------


def check_router(args) -> int:
    """三件事：每个 key 只落一个 split、经验比例与 ppm 相符、跨进程结果一致。

    第三件最容易被忽略：如果用了内置 `hash()`，PYTHONHASHSEED 的随机化会让 26 个
    worker 各自算出不同的划分，同一篇文章在不同 worker 里进不同 split，而且每次
    重跑都不一样。这个检查会真的 spawn 一个子进程去比。
    """
    failures = 0
    router = three_way_router("biomed", args.val_ppm, args.test_ppm)
    print(f"router: {json.dumps(router.describe(), ensure_ascii=False)}")

    keys = [f"PMC{i:08d}" for i in range(args.sample)]
    routed = [router.route(k) for k in keys]

    if all(r is not None for r in routed):
        ok(f"{fmt_int(len(keys))} 个 key 全部被覆盖（三分区间无空隙）")
    else:
        bad(f"{sum(r is None for r in routed)} 个 key 没落进任何 split")
        failures += 1

    print("\n经验比例 vs 期望比例：")
    for name, (lo, hi) in router.ranges.items():
        want = (hi - lo) / SPLIT_DENOM
        got = routed.count(name) / len(routed)
        # 样本量 n 下比例的标准差约 sqrt(p/n)，放 5 倍容差
        tol = 5 * (max(want, 1 / len(routed)) / len(routed)) ** 0.5
        flag = "PASS" if abs(got - want) <= tol else "FAIL"
        print(f"  [{flag}] {name:<6} 期望 {want:.5%}  实际 {got:.5%}  (±{tol:.5%})")
        if flag == "FAIL":
            failures += 1

    # 同一篇文章的所有段落必须路由到同一个 split——这是 Biomed 段落级数据的核心要求
    art = "PMC11276146"
    para_splits = {router.route(art) for _ in range(100)}
    if len(para_splits) == 1:
        ok(f"同一 article_id 的重复路由稳定落在 {para_splits.pop()!r}")
    else:
        bad(f"同一 article_id 路由不稳定：{para_splits}")
        failures += 1

    # 跨进程一致性：blake2b 不受 PYTHONHASHSEED 影响，内置 hash() 会
    probe = (
        "import sys; sys.path.insert(0, %r);"
        "from data_prep.common import key_bucket, SALT;"
        "print(','.join(str(key_bucket('PMC%%08d' %% i, SALT['biomed'])) for i in range(20)))"
    ) % str(ROOT)
    mine = ",".join(str(key_bucket(f"PMC{i:08d}", SALT["biomed"])) for i in range(20))
    theirs = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
    )
    if theirs.returncode != 0:
        bad(f"子进程一致性检查跑不起来：{theirs.stderr.strip()[:200]}")
        failures += 1
    elif theirs.stdout.strip() == mine:
        ok("换 PYTHONHASHSEED 的子进程算出同样的桶号（划分跨 worker / 跨机器可复现）")
    else:
        bad("子进程算出的桶号不同——划分依赖了随机化的哈希，26 个 worker 会各划一套")
        failures += 1

    return 1 if failures else 0


# ---------------------------------------------------------------------------
# 3) token-count：GPT-NeoX 下的真实 token 数
# ---------------------------------------------------------------------------


def check_token_count(args) -> int:
    """抽样实测每个数据集的 token 总量。

    FineMath 那一列 `token_count` 是 Llama token 数，换 GPT-NeoX（50k 词表）之后是
    9B 还是 12B 方向都不确定，而这个数直接决定新域预算定 4B 还是 8B。Biomed 侧则要
    这个数来反推 val/test 的 ppm 该给多少。
    """
    import pyarrow.parquet as pq

    tok = load_tokenizer(args.tokenizer_dir)
    names = [args.dataset] if args.dataset else list(SOURCE)
    failures = 0

    for name in names:
        sub, text_col, key_col = SOURCE[name]
        shards = sorted((Path(args.raw_root) / sub).rglob("*.parquet"))
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        if not shards:
            print(f"  跳过：{Path(args.raw_root) / sub} 下没有 parquet")
            continue

        # 总行数从 parquet footer 读，是精确值且不用解压任何数据
        total_rows = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
        print(f"  分片 {len(shards)} 个，总行数 {fmt_int(total_rows)}（读自 parquet footer，精确）")

        # 抽样：每个分片取头部若干行。分片内部大致按抓取顺序排，头部不是无偏样本，
        # 但对「平均每行多少 token」这个量级估计够用，且成本极低。
        sample_texts: list[str] = []
        llama_sum = 0
        per_shard = max(1, args.sample // len(shards))
        cols = (text_col, "token_count") if name == "finemath" else (text_col,)
        for s in shards[: args.max_shards]:
            got = 0
            for batch in iter_parquet_batches(s, cols, batch_size=min(per_shard, 512)):
                sample_texts.extend(batch[text_col])
                if "token_count" in batch:
                    llama_sum += int(sum(batch["token_count"]))
                got += len(batch[text_col])
                if got >= per_shard:
                    break

        sample_texts = [t for t in sample_texts if t]
        if not sample_texts:
            print("  跳过：抽样到的正文全为空（noncommercial split 就是这样，text 列是空的）")
            failures += 1
            continue

        arrs = encode_batch(tok, sample_texts)
        toks = np.array([a.size for a in arrs], dtype=np.int64)
        mean, sd = float(toks.mean()), float(toks.std())
        est = mean * total_rows
        # 均值的标准误 → 外推总量的 95% 区间
        se = sd / len(toks) ** 0.5 * total_rows

        print(f"  抽样 {fmt_int(len(toks))} 行：每行 {mean:.1f} ± {sd:.1f} token（中位数 {np.median(toks):.0f}）")
        print(f"  外推总量：{fmt_tokens(est)} token  (95% CI ±{fmt_tokens(1.96 * se)})")

        if name == "finemath" and llama_sum:
            ratio = toks.sum() / llama_sum
            print(f"  GPT-NeoX / Llama token 比：{ratio:.3f}")
            print(f"  → 官方标称 9.6B（Llama 计数）对应 GPT-NeoX 约 {9.6 * ratio:.2f}B")

        if name == "biomed":
            # Biomed 的「行」是段落，要换算成文章数才能定 val/test 的 ppm
            print(f"  提示：这里的行是段落。ppm 是按 article_id 划的，")
            print(f"        想让 val 拿到 {fmt_tokens(args.target_val_tokens)} token，")
            print(f"        --val-ppm 应设为约 {int(args.target_val_tokens / est * SPLIT_DENOM)}")
        else:
            print(f"  提示：想让 val 拿到 {fmt_tokens(args.target_val_tokens)} token，")
            print(f"        --val-ppm 应设为约 {int(args.target_val_tokens / est * SPLIT_DENOM)}")

    return 1 if failures else 0


# ---------------------------------------------------------------------------
# 3b) pool-tokens：FineWeb-Edu 池子的总 token 数，给 --pool-tokens 用
# ---------------------------------------------------------------------------


def check_pool_tokens(args) -> int:
    """算 sample-10BT 在本项目 tokenizer 下的总 token 数。

    这个数是 `prepare_fineweb_edu --pool-tokens` 的输入，段的 ppm 边界全由它换算。
    估偏的代价很高：三段 tokenize 各要过一遍 28.5GB，估偏 5% 就得整个重跑，所以这里
    不用 `token-count` 那套「头部抽样外推每行均值」的办法——分片内部大致按抓取顺序
    排，头部不是无偏样本，而文档长度是长尾分布，均值外推很容易偏几个百分点。

    改成只抽样一个比值：

    1. `token_count` 列是**每行精确的** token 数（FineWeb-Edu 官方用 GPT-2 数的），
       只读这一列不解压正文，全量求和几乎免费，长度的全部变化都被精确捕获。
    2. 抽样一批文档，用本项目 tokenizer 编一遍，算 Σneox / Σgpt2 这个比值。两个都是
       50k 量级的英文 BPE，比值非常稳，抽几千篇的相对标准误就在千分之几。

    于是 pool_tokens = 精确的 GPT-2 总量 × 比值，唯一的不确定性来自那个稳定的比值。

    口径必须和 tokenize 实际写进去的一致，否则换算出的边界就是错的：
    - 只算 `language == "en"` 的行（prepare_fineweb_edu 会丢掉其余的）；
    - 编码走 encode_batch，即 bos=False / eos=True，每篇多一个 eos，这一个 token
      被比值自然吸收；
    - 不扣 chunk 末尾的丢弃，因为 plan 读的 `tokens_written` 也是写 chunk 之前记的。
    """
    tok = load_tokenizer(args.tokenizer_dir)
    sub, text_col, _ = SOURCE["fineweb_edu_pool"]
    shards = sorted((Path(args.raw_root) / sub).rglob("*.parquet"))
    if not shards:
        print(f"  {Path(args.raw_root) / sub} 下没有 parquet，先跑 download.py")
        return 1

    # 第一遍：只读 token_count 与 language，全量精确求和。
    gpt2_sum = 0
    en_rows = 0
    total_rows = 0
    for s in shards:
        for batch in iter_parquet_batches(s, ("token_count", "language"), batch_size=16384):
            lang = np.asarray(batch["language"])
            cnt = np.asarray(batch["token_count"], dtype=np.float64)
            keep = (lang == "en") & np.isfinite(cnt)
            total_rows += lang.size
            en_rows += int(keep.sum())
            gpt2_sum += int(cnt[keep].sum())
        print(f"  {s.name}: 累计 en 行 {fmt_int(en_rows)} / {fmt_int(total_rows)}")

    if not en_rows:
        print("  没有 language == 'en' 的行，语言列的取值和预期不符")
        return 1

    # 第二遍：抽样算比值。跨分片摊开取，不只摸头几个分片。
    per_shard = max(1, args.sample // len(shards))
    texts: list[str] = []
    gpt2_sample: list[float] = []
    for s in shards:
        got = 0
        for batch in iter_parquet_batches(s, (text_col, "token_count", "language"), batch_size=min(per_shard, 512)):
            for t, c, lang in zip(batch[text_col], batch["token_count"], batch["language"]):
                if lang == "en" and t and c:
                    texts.append(t)
                    gpt2_sample.append(float(c))
                    got += 1
            if got >= per_shard:
                break

    if len(texts) < 100:
        print(f"  只抽到 {len(texts)} 篇，太少，调大 --sample")
        return 1

    neox = np.array([a.size for a in encode_batch(tok, texts)], dtype=np.float64)
    gpt2 = np.array(gpt2_sample, dtype=np.float64)

    # 比值估计量用「和之比」而不是「逐篇比值的均值」：要放大的是总量，前者才是无偏的。
    ratio = float(neox.sum() / gpt2.sum())
    m = neox.size
    resid = neox - ratio * gpt2
    se_ratio = float(np.sqrt((resid**2).sum() / (m * (m - 1))) / gpt2.mean())
    est = gpt2_sum * ratio
    rel = 1.96 * se_ratio / ratio

    print(f"\n  分片 {len(shards)} 个，总行 {fmt_int(total_rows)}，其中 en {fmt_int(en_rows)}"
          f"（{en_rows / total_rows:.1%}）")
    print(f"  token_count 精确求和（en 行，GPT-2 计数）：{fmt_int(gpt2_sum)}  ({fmt_tokens(gpt2_sum)})")
    print(f"  抽样 {fmt_int(m)} 篇算比值：本项目 tokenizer / GPT-2 = {ratio:.4f}"
          f"  (95% CI ±{1.96 * se_ratio:.4f})")
    print(f"\n  池子总量 ≈ {fmt_int(int(est))}  ({fmt_tokens(est)})，95% CI ±{rel:.2%}")
    print(f"\n  直接粘给下一步：--pool-tokens {est:.6g}")

    # 段的 ppm 分辨率是 1e-6，最大的 replay 目标又要能装下，两头都得留出余量。
    if est < args.max_replay_tokens:
        print(f"\n  ⚠ 池子只有 {fmt_tokens(est)}，装不下最大的 replay 目标 "
              f"{fmt_tokens(args.max_replay_tokens)}。要么降 --replay-tokens，要么多下几个分片")
        return 1
    headroom = est / args.max_replay_tokens - 1
    print(f"  余量：最大 replay 目标 {fmt_tokens(args.max_replay_tokens)} 之外还剩 {headroom:.0%}"
          f"（比值的 CI 是 ±{rel:.2%}，余量远大于它才安全）")
    return 0


# ---------------------------------------------------------------------------
# 4) chunks：产物能否被 pretrain 的读取路径正确读回
# ---------------------------------------------------------------------------


def iter_blocks(chunk_dir: Path, block_size: int, limit: int | None = None):
    """按 pretrain 的读取路径（StreamingDataset + TokensLoader）逐 block 读回。"""
    from litdata.streaming import StreamingDataset, TokensLoader

    ds = StreamingDataset(input_dir=str(chunk_dir), item_loader=TokensLoader(block_size=block_size), shuffle=False)
    n = len(ds) if limit is None else min(len(ds), limit)
    for i in range(n):
        yield np.asarray(ds[i]).reshape(-1)


def check_chunks(args) -> int:
    """把产物当成 pretrain 会怎么读就怎么读一遍，并与 manifest 对账。"""
    tok = load_tokenizer(args.tokenizer_dir)
    dirs = sorted(p.parent for p in Path(args.chunks_root).rglob("index.json"))
    if not dirs:
        print(f"{args.chunks_root} 下没有 litdata 产物（找不到 index.json）")
        return 1

    failures = 0
    for d in dirs:
        print(f"\n{'=' * 70}\n{d.relative_to(args.chunks_root)}\n{'=' * 70}")
        index = json.loads((d / "index.json").read_text(encoding="utf-8"))
        n_chunks = len(index.get("chunks", []))

        try:
            blocks = list(iter_blocks(d, args.block_size, limit=args.sample))
        except Exception as e:
            bad(f"读不回来：{type(e).__name__}: {e}")
            failures += 1
            continue
        if not blocks:
            bad("读回 0 个 block")
            failures += 1
            continue

        from litdata.streaming import StreamingDataset, TokensLoader

        ds = StreamingDataset(input_dir=str(d), item_loader=TokensLoader(block_size=args.block_size), shuffle=False)
        total_blocks = len(ds)
        print(f"  chunk 文件 {n_chunks} 个，block {fmt_int(total_blocks)} 个，"
              f"合计 {fmt_tokens(total_blocks * args.block_size)} token")

        b = blocks[0]
        if b.size == args.block_size:
            ok(f"block 长度 {b.size} == block_size（= seq_len 4096 + 1）")
        else:
            bad(f"block 长度 {b.size} != {args.block_size}")
            failures += 1

        if b.dtype == np.dtype(TOKEN_DTYPE):
            ok(f"dtype {b.dtype}（uint16 存盘，8B token 省 48GB）")
        else:
            bad(f"dtype {b.dtype} != {np.dtype(TOKEN_DTYPE)}；存储会膨胀数倍")
            failures += 1

        stacked = np.concatenate(blocks).astype(np.int64)
        if int(stacked.max()) < MAX_TOKEN_ID_EXCLUSIVE:
            ok(f"最大 token id {int(stacked.max())} < padded_vocab_size {MAX_TOKEN_ID_EXCLUSIVE}")
        else:
            bad(f"最大 token id {int(stacked.max())} 越界，embedding 会越界访问")
            failures += 1

        # eos 密度 → 平均文档长度。数量级不对（比如一个 block 里几百个 eos）
        # 说明正文被截断或者拼接方式错了
        n_eos = int((stacked == tok.eos_id).sum())
        if n_eos:
            ok(f"eos 出现 {fmt_int(n_eos)} 次 → 平均文档 {stacked.size / n_eos:.0f} token")
        else:
            bad("抽样里没有 eos——文档分隔符丢了")
            failures += 1

        # pretrain.py 的输入/标签切法，确认位移对得上
        import torch

        td = torch.from_numpy(np.stack(blocks[: min(4, len(blocks))]).astype(np.int64))
        input_ids = td[:, 0 : args.block_size - 1]
        targets = td[:, 1 : args.block_size]
        if torch.equal(input_ids[:, 1:], targets[:, :-1]):
            ok(f"pretrain 的 input/target 位移自洽，形状 {tuple(input_ids.shape)}")
        else:
            bad("input/target 位移不自洽")
            failures += 1

        if args.show_text:
            text = tok.decode(torch.from_numpy(blocks[0][:256].astype(np.int64)))
            print(f"  首个 block 解码前 256 token：\n    {text[:400]!r}")

        man = d / "kres_manifest.json"
        if man.exists():
            m = json.loads(man.read_text(encoding="utf-8"))
            written = m.get("stats", {}).get("tokens_written")
            if written:
                kept = total_blocks * args.block_size
                loss = 1 - kept / written
                # 每个 chunk 末尾不足一个 block 的 token 会被丢弃
                flag = "PASS" if loss < args.max_discard else "FAIL"
                print(f"  [{flag}] manifest 记录写入 {fmt_int(written)} token，"
                      f"读回 {fmt_int(kept)}，chunk 边界丢弃 {loss:.3%}")
                if flag == "FAIL":
                    print(f"         丢弃率超过 {args.max_discard:.1%}，把 --chunk-blocks 调大")
                    failures += 1
            print(f"  划分：{json.dumps(m.get('router', {}), ensure_ascii=False)}")
        else:
            print("  （没有 kres_manifest.json，无法对账）")

    return 1 if failures else 0


# ---------------------------------------------------------------------------
# 5) leakage：val/test 与 train 的 n-gram 重叠
# ---------------------------------------------------------------------------


def ngram_hashes(tokens: np.ndarray, n: int, stride: int) -> set[bytes]:
    """token 级 n-gram 指纹。

    直接在 token id 上做而不是解码回文本：省掉解码开销，而且比较的粒度正好是模型
    真正见到的东西。
    """
    out = set()
    for i in range(0, max(0, tokens.size - n + 1), stride):
        out.add(hashlib.blake2b(tokens[i : i + n].tobytes(), digest_size=8).digest())
    return out


def check_leakage(args) -> int:
    """端到端的划分检查：不看代码，只看产物里 val/test 和 train 有没有共同的 50-gram。

    这是唯一一个不依赖「我的划分逻辑写对了」这个前提的检查。无论泄漏是因为划分列选错、
    哈希不稳定、还是同一篇文章在数据集里有两个不同的主键，它都能抓到。
    """
    root = Path(args.chunks_root) / args.dataset
    train_dir = root / "train"
    if not (train_dir / "index.json").exists():
        print(f"缺少 {train_dir}，先跑 prepare_*.py --split train")
        return 1

    print(f"建 train 的 {args.n}-gram 指纹（抽 {fmt_int(args.train_blocks)} 个 block，stride={args.stride}）")
    train_ngrams: set[bytes] = set()
    for i, b in enumerate(iter_blocks(train_dir, args.block_size, limit=args.train_blocks)):
        train_ngrams |= ngram_hashes(b, args.n, args.stride)
        if (i + 1) % 2000 == 0:
            print(f"  {fmt_int(i + 1)} block → {fmt_int(len(train_ngrams))} 个指纹")
    print(f"  合计 {fmt_int(len(train_ngrams))} 个指纹")

    failures = 0
    for split in ("val", "test"):
        d = root / split
        if not (d / "index.json").exists():
            print(f"\n{split}: 不存在，跳过")
            continue
        hit = tot = 0
        for b in iter_blocks(d, args.block_size, limit=args.eval_blocks):
            grams = ngram_hashes(b, args.n, args.stride)
            tot += len(grams)
            hit += len(grams & train_ngrams)
        rate = hit / tot if tot else 0.0
        # 阈值不设 0：自然语言里的套话（版权声明、期刊模板句）本来就会在不同文章间重复
        flag = "PASS" if rate <= args.max_rate else "FAIL"
        print(f"\n  [{flag}] {split}: {fmt_int(hit)}/{fmt_int(tot)} 个 {args.n}-gram 与 train 重合 = {rate:.4%}"
              f"（阈值 {args.max_rate:.2%}）")
        if flag == "FAIL":
            print("         超阈值说明有文章被劈开分进了两个 split，划分不可用")
            failures += 1

    return 1 if failures else 0


# ---------------------------------------------------------------------------
# 6) overlap：FineMath 与 FineWeb-Edu 的重合
# ---------------------------------------------------------------------------


def check_overlap(args) -> int:
    """FineMath 和 FineWeb-Edu 同源于 CommonCrawl，重合是结构性风险。

    URL 重合是精确的（两边都有 url 列，全量算）；50-gram 重合抓的是同一内容换了 URL
    的情况，只能抽样。两个数字都要看：URL 不重合不代表内容不重合。
    """
    fwe = sorted((Path(args.raw_root) / "fineweb_edu_pool").rglob("*.parquet"))
    fm = sorted((Path(args.raw_root) / "finemath").rglob("*.parquet"))
    if not fwe or not fm:
        print("需要 fineweb_edu_pool 和 finemath 两边的原始 parquet 都已下载")
        return 1

    print("=" * 70)
    print("URL 重合（全量，精确）")
    print("=" * 70)

    def all_url_hashes(shards, tag):
        parts, n = [], 0
        for s in shards:
            for batch in iter_parquet_batches(s, ("url",), batch_size=8192):
                parts.append(url_hashes(batch["url"]))
                n += len(batch["url"])
            print(f"  {tag} {s.name}: 累计 {fmt_int(n)}")
        return np.unique(np.concatenate(parts)), n

    a, na = all_url_hashes(fwe, "fineweb-edu")
    b, nb = all_url_hashes(fm, "finemath")
    inter = np.intersect1d(a, b, assume_unique=True)
    print(f"\n  fineweb-edu: {fmt_int(na)} 行 → {fmt_int(len(a))} 个唯一 URL")
    print(f"  finemath   : {fmt_int(nb)} 行 → {fmt_int(len(b))} 个唯一 URL")
    print(f"  交集       : {fmt_int(len(inter))}")
    print(f"  占 finemath 的 {len(inter) / max(1, len(b)):.3%}")

    print()
    print("=" * 70)
    print(f"{args.n}-gram 重合（抽样）")
    print("=" * 70)
    tok = load_tokenizer(args.tokenizer_dir)

    def sample_ngrams(shards, tag, n_docs):
        grams, got = set(), 0
        for s in shards:
            for batch in iter_parquet_batches(s, ("text",), batch_size=512):
                for arr in encode_batch(tok, [t for t in batch["text"] if t]):
                    grams |= ngram_hashes(arr, args.n, args.stride)
                got += len(batch["text"])
                if got >= n_docs:
                    break
            if got >= n_docs:
                break
        print(f"  {tag}: {fmt_int(got)} 篇 → {fmt_int(len(grams))} 个指纹")
        return grams

    ga = sample_ngrams(fwe, "fineweb-edu", args.docs)
    gb = sample_ngrams(fm, "finemath", args.docs)
    inter_g = ga & gb
    print(f"\n  交集 {fmt_int(len(inter_g))}，占 finemath 抽样的 {len(inter_g) / max(1, len(gb)):.4%}")
    print("\n  这两个数字用来决定 FineMath 能不能直接当第二个新域；Biomed 来自 PMC 全文，")
    print("  污染风险低一个量级，所以先做 Biomed，这项检查可以推迟但不能跳过。")
    return 0


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--chunks-root", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tokenizer", help="批量编码与 litgpt 逐条编码是否一致")
    p.add_argument("--sample", type=int, default=512)
    p.set_defaults(fn=check_tokenizer)

    p = sub.add_parser("router", help="划分是否为真划分、比例是否对、跨进程是否一致")
    p.add_argument("--sample", type=int, default=2_000_000)
    p.add_argument("--val-ppm", type=int, default=1_500)
    p.add_argument("--test-ppm", type=int, default=1_500)
    p.set_defaults(fn=check_router)

    p = sub.add_parser("token-count", help="抽样实测 GPT-NeoX 下的 token 总量")
    p.add_argument("--dataset", choices=list(SOURCE))
    p.add_argument("--sample", type=int, default=4096, help="总抽样行数")
    p.add_argument("--max-shards", type=int, default=4, help="最多摸几个分片")
    p.add_argument("--target-val-tokens", type=float, default=5e7, help="想让 val 拿到多少 token")
    p.set_defaults(fn=check_token_count)

    p = sub.add_parser("pool-tokens", help="FineWeb-Edu 池子的总 token 数，给 --pool-tokens 用")
    p.add_argument("--sample", type=int, default=8192, help="算 tokenizer 比值的抽样篇数")
    p.add_argument(
        "--max-replay-tokens",
        type=float,
        default=8e9,
        help="最大的 replay 目标（--replay-tokens 的末项），用来检查池子装得下且有余量",
    )
    p.set_defaults(fn=check_pool_tokens)

    p = sub.add_parser("chunks", help="产物能否被 pretrain 的读取路径正确读回")
    p.add_argument("--sample", type=int, default=256, help="抽几个 block")
    p.add_argument("--show-text", action="store_true", help="解码首个 block 看一眼")
    p.add_argument(
        "--max-discard",
        type=float,
        default=0.01,
        help="chunk 末尾丢弃率的上限。丢弃量级是 1/chunk_blocks，"
             "默认 chunk_blocks=8192 时约 0.01%%；小 chunk 的自测要相应放宽",
    )
    p.set_defaults(fn=check_chunks)

    p = sub.add_parser("leakage", help="val/test 与 train 的 n-gram 重叠")
    p.add_argument("--dataset", default="biomed")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--train-blocks", type=int, default=20000)
    p.add_argument("--eval-blocks", type=int, default=2000)
    p.add_argument("--max-rate", type=float, default=0.005)
    p.set_defaults(fn=check_leakage)

    p = sub.add_parser("overlap", help="FineMath 与 FineWeb-Edu 的 URL / n-gram 重合")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--docs", type=int, default=20000)
    p.set_defaults(fn=check_overlap)

    args = ap.parse_args(argv)
    print(f"### {args.cmd}\n")
    rc = args.fn(args)
    print(f"\n### {args.cmd}: {'FAILED' if rc else 'OK'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
