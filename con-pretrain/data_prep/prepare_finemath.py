"""FineMath-4+（第二个新域）→ 文档级 train/val/test 的 litdata chunk。

结构最简单的一个：逐行一个独立文档，不需要重组。注意两点：

1. **没有 id 列，主键是 url。** 划分按 url 哈希做。同一个 url 在数据集里可能出现多次
   （`snapshot_type` 有 longest/latest 之类的取法），按 url 划分正好保证这些副本落进
   同一个 split，不会互相污染。

2. **token_count 列不能信。** 它的定义是 Llama token 数，换成 GPT-NeoX（50k 词表）
   之后是 9B 还是 12B 方向都不确定。这个数直接决定新域预算定 4B 还是 8B，必须自己
   抽样实测：先跑 `checks.py token-count --dataset finemath`。

另外 FineMath 和 FineWeb-Edu 同源于 CommonCrawl，重合是结构性风险，正式用之前要跑
`checks.py overlap` 测 URL 和 50-gram 重叠。因为当前先做 Biomed（来自 PMC 全文，
污染风险低一个量级），这两项可以推迟，但不能跳过。

用法：
    python -m data_prep.prepare_finemath --split train
    python -m data_prep.prepare_finemath --split val
    python -m data_prep.prepare_finemath --split test
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from data_prep.common import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CHUNK_BLOCKS,
    DOC_KEY,
    Router,
    Stats,
    encode_batch,
    fmt_int,
    iter_parquet_batches,
    load_tokenizer,
    merge_stats,
    three_way_router,
    write_manifest,
    write_token_chunks,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "data" / "raw" / "finemath"
DEFAULT_OUT = ROOT / "data" / "chunks" / "finemath"
DEFAULT_TOKENIZER = ROOT / "model" / "open-sci-ref-v0.02-0.4b-fineweb-edu-1.4t-300B-4096"

COLUMNS = ("url", "text", "language", "token_count")

_TOK = None


def _tokenizer(tokenizer_dir: str):
    global _TOK
    if _TOK is None:
        _TOK = load_tokenizer(tokenizer_dir)
    return _TOK


def tokenize_shard(
    shard: str,
    *,
    split: str,
    router: Router,
    tokenizer_dir: str,
    stats_dir: str,
    batch_size: int,
):
    """litdata 的 worker 入口。必须是模块级函数（litdata 用 spawn，闭包 pickle 不了）。"""
    tok = _tokenizer(tokenizer_dir)
    st = Stats()
    key_col = DOC_KEY["finemath"]

    for batch in iter_parquet_batches(shard, COLUMNS, batch_size=batch_size):
        n = len(batch[key_col])
        st.add("rows_seen", n)
        # token_count 只用来对照实测值，不参与任何筛选
        st.add("llama_token_count_sum", int(sum(batch["token_count"])))

        keep = [i for i in range(n) if batch["language"][i] == "en" and router.route(batch[key_col][i]) == split]
        if not keep:
            continue

        for arr in encode_batch(tok, [batch["text"][i] for i in keep]):
            st.add("docs_written")
            st.add("tokens_written", int(arr.size))
            yield arr

    st.dump(stats_dir, f"{split}-{Path(shard).stem}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--val-ppm", type=int, default=5_000, help="val 占比（ppm，默认 0.5%%）")
    ap.add_argument("--test-ppm", type=int, default=5_000, help="test 占比（ppm，默认 0.5%%）")
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    ap.add_argument("--chunk-blocks", type=int, default=DEFAULT_CHUNK_BLOCKS)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    shards = sorted(str(p) for p in args.raw_dir.rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"{args.raw_dir} 下没有 parquet，先跑 download.py --dataset finemath")

    router = three_way_router("finemath", args.val_ppm, args.test_ppm)
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
            batch_size=args.batch_size,
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
            "dataset": "HuggingFaceTB/finemath",
            "hf_config": "finemath-4plus",
            "split": args.split,
            "source_dir": str(args.raw_dir),
            "shards": [Path(s).name for s in shards],
            "doc_key": DOC_KEY["finemath"],
            "router": router.describe(),
            "block_size": args.block_size,
            "chunk_blocks": args.chunk_blocks,
            "stats": stats,
        },
    )

    print("\n统计：")
    for k in sorted(stats):
        print(f"  {k:<24} {fmt_int(stats[k])}")

    llama = stats.get("llama_token_count_sum", 0)
    ours = stats.get("tokens_written", 0)
    if llama and ours:
        print(f"\nGPT-NeoX / Llama token 数之比：{ours / llama:.3f}（这个 split 内）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
