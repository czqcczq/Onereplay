"""Biomed-Enriched（新域）→ 文章级 train/val/test 的 litdata chunk。

只用 commercial split。noncommercial 的 text 列是空的（许可限制），要靠本地几百 GB
的 PMC OA XML dump 用 biomed-enriched 包回填，已决定放弃。

数据集是**段落级**的（98.6M 段落 / 约 2400 万篇文章，49GB parquet / 127GB 原文），
每行一个段落，`id` 形如 `PMC11276146_p3`，`article_id` 是 `PMC11276146`。段落做过
≥64 token 的过滤，所以 `_pN` 的 N 有空洞。用之前必须先按 article_id 重组成文章。

三个决定：

1. **划分按 article_id 哈希，不按流式位置。**
   这一条同时解决了并行难题：路由只取决于 article_id，所以 26 个 worker 各干各的，
   同一篇文章的所有段落必然被算到同一个 split，不需要任何跨进程协调。反过来说，
   即使段落在文件里不连续，也不可能造成 split 泄漏——最坏情况只是一篇文章被切成
   几段各自加 eos，是数据质量问题而不是评测污染问题。

2. **跨分片的边界文章直接丢弃。**
   每个 worker 丢掉自己遇到的第一个和最后一个 article 的 run。26 个分片最多丢 26 篇，
   对 2400 万篇是 1e-6 量级。这样换来完全的分片并行，比为了这 26 篇去做两阶段
   外部排序（要把 127GB 重写一遍）划算得多。丢弃数会计入统计，明显超过 26 就说明
   段落不连续，需要回头查。

3. **不用它的 curation 标注。**
   educational_score / clinical 上采样是 Biomed-Enriched 那篇论文自己的贡献，用了
   就成了混淆变量——分不清增益来自我们的正则还是来自它的数据筛选。
   `language == "en"` 的过滤是必须的（数据集含 en/fr/es/zh/de/it/pt/ko/ru，混进去
   会把领域漂移和语言漂移搅在一起）。

用法：
    python -m data_prep.prepare_biomed --split train
    python -m data_prep.prepare_biomed --split val
    python -m data_prep.prepare_biomed --split test

val/test 的 ppm 默认值是按「各约 5000 万 token」估的，但 Biomed 的实际总 token 数
要先跑 `checks.py token-count` 才知道，拿到数字后回来调 --val-ppm / --test-ppm。
"""

from __future__ import annotations

import argparse
import json
import re
from functools import partial
from pathlib import Path

from data_prep.common import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CHUNK_BLOCKS,
    DOC_KEY,
    SPLIT_DENOM,
    Router,
    Stats,
    encode_batch,
    fmt_int,
    fmt_tokens,
    iter_parquet_batches,
    load_tokenizer,
    merge_stats,
    three_way_router,
    write_manifest,
    write_token_chunks,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "data" / "raw" / "biomed"
DEFAULT_OUT = ROOT / "data" / "chunks" / "biomed"
DEFAULT_TOKENIZER = ROOT / "model" / "open-sci-ref-v0.02-0.4b-fineweb-edu-1.4t-300B-4096"

COLUMNS = ("id", "article_id", "text", "language")
_PARA_SUFFIX = re.compile(r"_p(\d+)$")

_TOK = None


def _tokenizer(tokenizer_dir: str):
    global _TOK
    if _TOK is None:
        _TOK = load_tokenizer(tokenizer_dir)
    return _TOK


def paragraph_index(row_id: str) -> int:
    """从 `PMC11276146_p3` 里取出 3，用于把段落排回原文顺序。

    parquet 里的行序看起来已经是段落序，但依赖它是没根据的（前一个段落被 ≥64 token
    过滤掉之后 N 会跳号，行序和 N 是否一致没有任何保证），所以显式按 N 排。
    """
    m = _PARA_SUFFIX.search(row_id)
    if m is None:
        raise ValueError(f"id {row_id!r} 不符合 `<article_id>_p<N>` 的形式")
    return int(m.group(1))


def assemble(run: list[tuple[int, str, str]], en_threshold: float, st: Stats) -> str | None:
    """把一篇文章的段落 run 拼成正文。run 里是 (段落号, 文本, 语言)。

    先在文章级判语言再在段落级过滤：`language` 是逐段落的分类器输出，短段落容易误判，
    要求 100% 是 en 会白扔掉不少好文章；反过来只做段落级过滤又会把多语种文章拼成
    半英半法的怪东西。所以先看这篇文章里 en 段落的占比，过阈值才留，留下来之后再把
    个别非 en 段落去掉。
    """
    en = sum(1 for _, _, lang in run if lang == "en")
    if en / len(run) < en_threshold:
        st.add("articles_dropped_language")
        return None

    paras = [(idx, text) for idx, text, lang in run if lang == "en" and text]
    st.add("paragraphs_dropped_non_en", len(run) - len(paras))
    if not paras:
        st.add("articles_dropped_empty")
        return None

    paras.sort(key=lambda p: p[0])
    return "\n\n".join(text.strip() for _, text in paras)


def encode_articles(tok, texts: list[str], st: Stats):
    """把攒好的一批文章编码成 token 数组。

    攒批是必要的：逐篇调 tokenizer，Python 层的调用开销会成为整个流水线的瓶颈。
    """
    for arr in encode_batch(tok, texts):
        st.add("articles_written")
        st.add("tokens_written", int(arr.size))
        yield arr
    texts.clear()


def tokenize_shard(
    shard: str,
    *,
    split: str,
    router: Router,
    tokenizer_dir: str,
    stats_dir: str,
    en_threshold: float,
    batch_size: int,
    encode_batch_size: int,
    check_contiguity: bool,
):
    """litdata 的 worker 入口：读一个 parquet 分片，重组文章，产出属于 `split` 的 token。

    必须是模块级函数（litdata 用 spawn，闭包 pickle 不了）。
    """
    tok = _tokenizer(tokenizer_dir)
    st = Stats()

    run_key: str | None = None  # 当前正在累积的 article_id
    run: list[tuple[int, str, str]] = []  # (段落号, 文本, 语言)
    is_first_run = True
    closed: set[str] = set()  # 已收尾的 article_id，用于段落连续性断言
    pending: list[str] = []  # 待编码的文章正文

    for batch in iter_parquet_batches(shard, COLUMNS, batch_size=batch_size):
        for row_id, art, text, lang in zip(batch["id"], batch["article_id"], batch["text"], batch["language"]):
            if art != run_key:
                if run:
                    if is_first_run:
                        # 分片的第一个 run 可能是上一个分片某篇文章的后半段，丢弃
                        st.add("articles_dropped_shard_boundary")
                        is_first_run = False
                    else:
                        st.add("articles_seen")
                        if router.route(run_key) == split:
                            body = assemble(run, en_threshold, st)
                            if body is not None:
                                pending.append(body)
                    if check_contiguity:
                        closed.add(run_key)
                        if art in closed:
                            raise RuntimeError(
                                f"{Path(shard).name}: article_id {art!r} 的段落在文件里不连续。"
                                "重组会把它切成几段，需要改成按 article_id 分桶的两阶段处理。"
                            )
                if len(pending) >= encode_batch_size:
                    yield from encode_articles(tok, pending, st)
                run_key, run = art, []
            run.append((paragraph_index(row_id), text, lang))

    # 分片的最后一个 run 同样丢弃：它可能是下一个分片某篇文章的前半段。
    if run:
        st.add("articles_dropped_shard_boundary")
    if pending:
        yield from encode_articles(tok, pending, st)

    st.dump(stats_dir, f"{split}-{Path(shard).stem}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--val-ppm", type=int, default=1_500, help="val 占比（ppm）")
    ap.add_argument("--test-ppm", type=int, default=1_500, help="test 占比（ppm）")
    ap.add_argument(
        "--train-tokens",
        type=float,
        help="train 要多少 token（新域预算，比如 8e9）。不给就把 val/test 之外的全要——"
             "对 Biomed 是约 28B，而预算只有 8B，多出来的训练时读不到，白花时间和磁盘。"
             "给了要同时给 --corpus-tokens。**三次调用都传同一个值**",
    )
    ap.add_argument(
        "--corpus-tokens",
        type=float,
        help="Biomed commercial 的实测总 token 数，跑 `checks token-count --dataset biomed` 拿。"
             "用来把 --train-tokens 折成 ppm",
    )
    ap.add_argument(
        "--train-margin",
        type=float,
        default=1.25,
        help="按 --train-tokens 的这个倍数去切，默认 1.25。留余量是因为 corpus-tokens 是"
             "抽样外推的，而且它没扣掉 en 过滤丢掉的文章，切少了就会不够 8B——"
             "而不够会被 CycleIterator 静默地变成「把少的那部分重放两遍」",
    )
    ap.add_argument(
        "--en-threshold",
        type=float,
        default=0.9,
        help="一篇文章里 en 段落占比的下限，低于它整篇丢弃",
    )
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    ap.add_argument("--chunk-blocks", type=int, default=DEFAULT_CHUNK_BLOCKS)
    ap.add_argument("--batch-size", type=int, default=2048, help="parquet 读批大小（行）")
    ap.add_argument("--encode-batch", type=int, default=256, help="攒多少篇文章调一次 tokenizer")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-contiguity-check", action="store_true", help="关掉段落连续性断言（省内存）")
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    shards = sorted(str(p) for p in args.raw_dir.rglob("commercial-*.parquet"))
    if not shards:
        raise SystemExit(f"{args.raw_dir} 下没有 commercial-*.parquet，先跑 download.py --dataset biomed")

    if (args.train_tokens is None) != (args.corpus_tokens is None):
        raise SystemExit("--train-tokens 和 --corpus-tokens 要么都给，要么都不给")
    train_ppm = None
    if args.train_tokens is not None:
        if args.corpus_tokens <= 0:
            raise SystemExit(f"--corpus-tokens 必须为正，收到 {args.corpus_tokens}")
        want = args.train_tokens * args.train_margin
        if want > args.corpus_tokens:
            raise SystemExit(
                f"--train-tokens {args.train_tokens:.3g} × 余量 {args.train_margin} "
                f"= {want:.3g} 超过语料实测的 {args.corpus_tokens:.3g}，直接全量切（不传这两个参数）即可"
            )
        # 向上取整：宁可多切一点，少了就不够预算。
        train_ppm = -(-int(want * SPLIT_DENOM) // int(args.corpus_tokens))

    router = three_way_router("biomed", args.val_ppm, args.test_ppm, train_ppm)
    out_dir = args.out_root / args.split
    stats_dir = args.out_root / "_stats" / args.split
    if stats_dir.exists():
        for old in stats_dir.glob("*.json"):
            old.unlink()

    print(f"split={args.split}  分片 {len(shards)} 个  →  {out_dir}")
    print(f"划分：{json.dumps(router.describe(), ensure_ascii=False)}")

    write_token_chunks(
        partial(
            tokenize_shard,
            split=args.split,
            router=router,
            tokenizer_dir=str(args.tokenizer_dir),
            stats_dir=str(stats_dir),
            en_threshold=args.en_threshold,
            batch_size=args.batch_size,
            encode_batch_size=args.encode_batch,
            check_contiguity=not args.no_contiguity_check,
        ),
        shards,
        out_dir,
        block_size=args.block_size,
        chunk_blocks=args.chunk_blocks,
        num_workers=args.workers,
        fast_dev_run=args.fast_dev_run,
        overwrite=args.overwrite,
    )

    stats = merge_stats(stats_dir)
    write_manifest(
        out_dir / "kres_manifest.json",
        {
            "dataset": "almanach/Biomed-Enriched",
            "hf_split": "commercial",
            "split": args.split,
            "source_dir": str(args.raw_dir),
            "shards": [Path(s).name for s in shards],
            "doc_key": DOC_KEY["biomed"],
            "router": router.describe(),
            "train_tokens_target": args.train_tokens,
            "corpus_tokens": args.corpus_tokens,
            "train_margin": args.train_margin if args.train_tokens is not None else None,
            "en_threshold": args.en_threshold,
            "curation_columns_used": [],
            "block_size": args.block_size,
            "chunk_blocks": args.chunk_blocks,
            "stats": stats,
        },
    )

    print("\n统计：")
    for k in sorted(stats):
        print(f"  {k:<34} {fmt_int(stats[k])}")

    dropped = stats.get("articles_dropped_shard_boundary", 0)
    if dropped > 2 * len(shards):
        print(
            f"\n!! 边界丢弃数 {dropped} 超过 2×分片数 {2 * len(shards)}，"
            "说明段落在文件里不是按文章连续排列的，重组结果不可信。"
        )
        return 1

    # 切少了必须在这里拦住。corpus-tokens 是抽样外推的，而且没扣掉 en 过滤丢掉的文章，
    # 所以有可能不够预算。而不够不会报错：litgpt 的 CycleIterator 读完会静默重新开始，
    # 「8B 新数据」就变成「把 7B 里的一部分喂两遍」，日志上看不出区别。
    if args.split == "train" and args.train_tokens is not None:
        got = stats.get("tokens_written", 0)
        if got < args.train_tokens:
            need = args.train_margin * args.train_tokens / max(got, 1)
            print(
                f"\n!! train 只切出 {fmt_tokens(got)} token，不够 --train-tokens "
                f"{fmt_tokens(args.train_tokens)}。\n"
                f"   --corpus-tokens 估高了（它没扣 en 过滤丢掉的文章）。"
                f"把 --train-margin 调到约 {need:.2f} 后重跑 train。"
            )
            return 1
        print(
            f"\ntrain 切出 {fmt_tokens(got)} token，覆盖预算 "
            f"{fmt_tokens(args.train_tokens)} 的 {got / args.train_tokens:.2f} 倍"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
