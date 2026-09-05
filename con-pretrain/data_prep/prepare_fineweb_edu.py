"""FineWeb-Edu 保护侧：若干个 replay 段（同时也是 C 的来源）+ 干净的 held-out probe。

**replay 和 C 同源，这是设计要求而不是巧合。** 本方法要论证的是「用旧数据估出来的 C
可以替代重放这些旧数据」。要让这个对比是同类相比，两条臂就必须面对同一批旧数据。所以
sample-10BT 被切成若干**累积嵌套**的段，每一段既是某条 replay 臂重放的数据、也是那条
臂的 C 的来源：

    seg00 ≈ 1B → replay 1B 臂读 [seg00]，          C = C(seg00)
    seg01 ≈ 3B → replay 4B 臂读 [seg00, seg01]，   C = 加权合并这两段
    seg02 ≈ 4B → replay 8B 臂读 [seg00..02]，      C = 加权合并这三段

"同一批"由**目录相同**机械保证，而不是靠"两边流式读到的第 1B 个 token 恰好一样"——后者
要求 shuffle seed、batch size、worker 数、以及混合 dataloader 的交错方式全都一致，任何
一个变了就悄悄不成立。

`probe` 必须换来源。sample-10BT 整个取自 FineWeb-Edu v1.0.0 的池子，而基座就是在
v1.0.0 上训的，拿它当 held-out 大约 23% 是污染的。干净的 probe 取 CC-MAIN-2024-18 及
之后的 dump（v1.0.0 之后才加入，基座证明没见过），再按 URL 减掉与 10BT 池子重合的部分。

四个 stage，按顺序跑：

    # 1) 建 10BT 池子的 URL 指纹索引（probe 排除用）
    python -m data_prep.prepare_fineweb_edu --stage url-index

    # 2) 各个 replay 段（一段一次调用，--replay-tokens 和 --pool-tokens 必须每次都一样）
    python -m data_prep.prepare_fineweb_edu --stage tokenize --role seg00 --pool-tokens 1.03e10
    python -m data_prep.prepare_fineweb_edu --stage tokenize --role seg01 --pool-tokens 1.03e10
    python -m data_prep.prepare_fineweb_edu --stage tokenize --role seg02 --pool-tokens 1.03e10

    # 3) 干净 probe（自动排除 10BT 的 URL）
    python -m data_prep.prepare_fineweb_edu --stage tokenize --role probe

    # 4) 汇总成 replay_plan.json：每条臂该读哪些目录、实测各多少 token
    python -m data_prep.prepare_fineweb_edu --stage plan --pool-tokens 1.03e10

`--replay-tokens` 默认 `1e9,4e9,8e9`，即主表的 replay {1B, 4B, 8B}，段是相邻两个目标之
差。`--pool-tokens` 是池子用**本项目 tokenizer** 实测的总 token 数（跑 `checks
pool-tokens` 拿，它会直接打出可粘贴的值；别用名义上的 10B，那是别家 tokenizer 数的），
段的 ppm 边界由这两个数换算得出。落到的实际 token 不会正好是 1B/4B/8B——ppm 切的是文档
份额而每篇文档长短不同——`plan` 会把目标和实测并排打出来，差超过 2% 会提示重算。

产物给 pretrain 的用法：replay 侧读 replay_plan.json 里对应臂的目录列表（混合
dataloader 要接多个目录）；probe 侧
    --data LitData --data.data_path .../chunks/fineweb_edu --data.split_names "[seg00,probe]"
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from data_prep.common import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CHUNK_BLOCKS,
    DOC_KEY,
    Router,
    Stats,
    encode_batch,
    fmt_int,
    fmt_tokens,
    iter_parquet_batches,
    load_tokenizer,
    merge_stats,
    segment_name,
    segment_ppms_from_tokens,
    segment_router,
    subsample_router,
    url_hashes,
    write_manifest,
    write_token_chunks,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "data" / "chunks" / "fineweb_edu"
DEFAULT_TOKENIZER = ROOT / "model" / "open-sci-ref-v0.02-0.4b-fineweb-edu-1.4t-300B-4096"

COLUMNS = ("text", "id", "url", "language", "token_count")

# worker 侧的惰性单例。litdata 用 spawn，tokenizer 和排除索引都不能跨进程传，
# 只能在每个 worker 里各建一次。
_TOK = None
_EXCLUDE: np.ndarray | None = None


def _tokenizer(tokenizer_dir: str):
    global _TOK
    if _TOK is None:
        _TOK = load_tokenizer(tokenizer_dir)
    return _TOK


def _exclude_index(path: str | None) -> np.ndarray | None:
    global _EXCLUDE
    if path is None:
        return None
    if _EXCLUDE is None:
        _EXCLUDE = np.load(path)
        if _EXCLUDE.dtype != np.uint64:
            raise TypeError(f"排除索引 dtype 应为 uint64，实际 {_EXCLUDE.dtype}")
    return _EXCLUDE


def _is_excluded(index: np.ndarray, urls: list[str]) -> np.ndarray:
    """在排序好的指纹数组里做批量成员查询。"""
    if index.size == 0:
        return np.zeros(len(urls), dtype=bool)
    h = url_hashes(urls)
    pos = np.clip(np.searchsorted(index, h), 0, index.size - 1)
    return index[pos] == h


def tokenize_shard(
    shard: str,
    *,
    role: str,
    router: Router,
    tokenizer_dir: str,
    stats_dir: str,
    exclude_path: str | None,
    batch_size: int,
):
    """litdata 的 worker 入口：读一个 parquet 分片，产出属于 `role` 的文档 token。

    必须是模块级函数（litdata 用 spawn，闭包 pickle 不了）。
    """
    tok = _tokenizer(tokenizer_dir)
    index = _exclude_index(exclude_path)
    st = Stats()
    key_col = DOC_KEY["fineweb_edu"]

    for batch in iter_parquet_batches(shard, COLUMNS, batch_size=batch_size):
        n = len(batch[key_col])
        st.add("rows_seen", n)

        en = [i for i in range(n) if batch["language"][i] == "en"]
        st.add("rows_dropped_language", n - len(en))
        keep = [i for i in en if router.route(batch[key_col][i]) == role]
        st.add("rows_other_split", len(en) - len(keep))
        if not keep:
            continue

        if index is not None:
            flags = _is_excluded(index, [batch["url"][i] for i in keep])
            dropped = int(flags.sum())
            if dropped:
                st.add("rows_url_excluded", dropped)
                keep = [i for i, bad in zip(keep, flags) if not bad]
            if not keep:
                continue

        texts = [batch["text"][i] for i in keep]
        for arr in encode_batch(tok, texts):
            st.add("docs_written")
            st.add("tokens_written", int(arr.size))
            yield arr

    st.dump(stats_dir, f"{role}-{Path(shard).stem}")


def build_url_index(raw_root: Path, out_path: Path, batch_size: int) -> None:
    """把 sample-10BT 全部文档的 URL 指纹存成一个排序好的 uint64 数组。

    存指纹不存原串：池子有千万级文档，uint64 数组是几十 MB，配 searchsorted 做成员
    查询是对数时间；存原串要几 GB 内存，每个 worker 还要各存一份。
    """
    shards = sorted((raw_root / "fineweb_edu_pool").rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"{raw_root / 'fineweb_edu_pool'} 下没有 parquet，先跑 download.py")

    parts: list[np.ndarray] = []
    total = 0
    for s in shards:
        for batch in iter_parquet_batches(s, ("url",), batch_size=batch_size):
            parts.append(url_hashes(batch["url"]))
            total += len(batch["url"])
        print(f"  {s.name}: 累计 {fmt_int(total)} 个 URL")

    index = np.unique(np.concatenate(parts))  # unique 顺带排序
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, index)
    print(f"URL 指纹索引写入 {out_path}")
    print(f"  文档 {fmt_int(total)} 个 → 去重后指纹 {fmt_int(len(index))} 个（{index.nbytes / 1e6:.1f} MB）")


def write_replay_plan(
    out_root: Path, segment_ppms: list[int], replay_tokens: list[int], new_tokens: int
) -> None:
    """把各段的实测 token 数汇总成 replay_plan.json：每条臂读哪些目录、跑多少 token。

    这个文件是 replay 侧和 C 侧唯一的真相来源。混合 dataloader 和 collect_cov 都从它取
    目录列表，而不是各自去拼 `seg%02d`——两边各拼一次，就有各拼错一次的机会。
    """
    segments = []
    for i, ppm in enumerate(segment_ppms):
        name = segment_name(i)
        manifest = out_root / name / "kres_manifest.json"
        if not manifest.exists():
            raise SystemExit(
                f"缺少 {manifest}，先跑 --stage tokenize --role {name}"
                f"（--replay-tokens 和 --pool-tokens 要和现在这次完全一致）"
            )
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        # 段的边界是「每条臂读的到底是哪批文档」的全部定义，manifest 里没记或记的和
        # 这次要求的不一样，都不能照算——那会产出一份指向错误目录的 plan。
        got = meta.get("router", {}).get("ranges", {}).get(name)
        want = [sum(segment_ppms[:i]), sum(segment_ppms[: i + 1])]
        if got is None:
            raise SystemExit(f"{manifest} 里没记 {name} 的 ppm 区间，不是 segment_router 产出的，重跑那一段")
        if list(got) != want:
            raise SystemExit(
                f"{name} 是用 ppm 区间 {got} 切的，本次换算出的是 {want}。边界不一致意味着"
                f"段之间会重叠或漏文档。检查各次调用的 --replay-tokens / --pool-tokens 是否一致，"
                f"然后重跑那一段"
            )
        segments.append(
            {
                "name": name,
                "dir": str(out_root / name),
                "ppm": ppm,
                "tokens": int(meta["stats"].get("tokens_written", 0)),
                "docs": int(meta["stats"].get("docs_written", 0)),
            }
        )

    arms = []
    for k in range(1, len(segments) + 1):
        used = segments[:k]
        dirs = [s["dir"] for s in used]
        actual = sum(s["tokens"] for s in used)
        target = replay_tokens[k - 1]
        arms.append(
            {
                "name": f"replay_{fmt_tokens(target)}",
                "segments": [s["name"] for s in used],
                # 这两个字段必须逐字相同：replay 和 C 同源就是靠它。
                "replay_dirs": dirs,
                "cov_dirs": dirs,
                # target 是实验设计里那个整数（1B / 4B / 8B），actual 是这些段真正装了多少。
                # 训练和论文都用 actual：段跑满一个 epoch 才能保证 replay 与 C 逐文档相同，
                # 为了凑整去截断 actual 会让被截掉的那部分只进 C、不被重放。
                "replay_tokens_target": target,
                "replay_tokens": actual,
                "target_deviation": actual / target - 1,
                # C 侧按 token 计数加权合并，权重就是各段的 token 数。
                "cov_merge_weights": [s["tokens"] for s in used],
                "new_tokens": new_tokens,
                # 新域 token 要恰好是 new_tokens，所以训练总预算得把 replay 的那份加回去。
                "train_max_tokens": new_tokens + actual,
                # 混合 dataloader 里旧数据该占的采样比例（不是「replay 相对新域的倍数」）。
                "replay_share_of_stream": actual / (new_tokens + actual),
            }
        )

    for arm in arms:
        if arm["replay_dirs"] != arm["cov_dirs"]:
            raise AssertionError(f"replay 与 C 的来源目录不一致：{arm}")

    plan_path = out_root / "replay_plan.json"
    write_manifest(
        plan_path,
        {
            "pool_root": str(out_root),
            "segment_ppms": segment_ppms,
            "replay_tokens_target": replay_tokens,
            "segments": segments,
            "arms": arms,
            "note": "每条臂的 replay_dirs 与 cov_dirs 逐字相同：重放的数据就是采 C 的数据",
        },
    )

    print(f"replay 计划写入 {plan_path}\n")
    print(f"  {'段':<8}{'ppm':>9}{'文档':>14}{'token':>16}")
    for s in segments:
        print(
            f"  {s['name']:<8}{s['ppm']:>9}{fmt_int(s['docs']):>14}"
            f"{fmt_int(s['tokens']):>16}  ({fmt_tokens(s['tokens'])})"
        )
    print(f"\n  新域预算固定 {fmt_tokens(new_tokens)} token，各臂：")
    print(f"  {'读的段':<22}{'目标':>8}{'实际':>10}{'偏差':>9}{'max_tokens':>13}{'占数据流':>10}")
    for arm in arms:
        print(
            f"  {'+'.join(arm['segments']):<22}"
            f"{fmt_tokens(arm['replay_tokens_target']):>8}"
            f"{fmt_tokens(arm['replay_tokens']):>10}"
            f"{arm['target_deviation']:>+8.1%}"
            f"{fmt_tokens(arm['train_max_tokens']):>13}"
            f"{arm['replay_share_of_stream']:>10.1%}"
        )

    worst = max(arms, key=lambda a: abs(a["target_deviation"]))
    if abs(worst["target_deviation"]) > 0.02:
        print(
            f"\n  ⚠ {'+'.join(worst['segments'])} 与目标差 {worst['target_deviation']:+.1%}，超过 2%。"
            f"\n    ppm 切的是文档份额而不是 token 份额，一两个百分点是正常的；差这么多说明"
            f"\n    --pool-tokens 与实际不符，按 segments 表里的实测值重算后重跑分段。"
        )

    print(
        "\n  训练和论文都用「实际」那一列：段要跑满一个 epoch，replay 与 C 才是逐文档相同的。"
        "\n  为了凑整去截断，被截掉的那部分就只进了 C、没被重放。"
        "\n  train_max_tokens 是要传给 --train.max_tokens 的值：它包含 replay 的那部分，这样"
        "\n  新域才恰好消耗 new_tokens。直接传 new_tokens 会让新域少喂 replay 那么多。"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["url-index", "tokenize", "plan"])
    ap.add_argument("--role", help="tokenize 阶段必填：seg00 / seg01 / ... 或 probe")
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument(
        "--replay-tokens",
        default="1e9,4e9,8e9",
        help="各条 replay 臂要重放多少 token（逗号分隔，累积量，必须严格递增）。"
             "默认对应主表的 replay {1B, 4B, 8B}。段是相邻两个目标之差",
    )
    ap.add_argument(
        "--pool-tokens",
        type=float,
        help="sample-10BT 用本项目 tokenizer 实测的总 token 数，跑 `checks pool-tokens` 拿。"
             "别用数据集名义上的 10B，那是别家 tokenizer 数的。"
             "**所有 tokenize 调用和 plan 都必须传同一个值**，改一个数所有段的边界就变了",
    )
    ap.add_argument(
        "--probe-ppm",
        type=int,
        default=100_000,
        help="从 probe dump 里抽多少（ppm，默认 10%%）。先跑 checks.py token-count 看实际 token 数再调",
    )
    ap.add_argument(
        "--new-tokens",
        type=int,
        default=8_000_000_000,
        help="plan 阶段用：新域的 token 预算，用来算各臂的 --train.max_tokens 和 replay 占比",
    )
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    ap.add_argument("--chunk-blocks", type=int, default=DEFAULT_CHUNK_BLOCKS)
    ap.add_argument("--batch-size", type=int, default=2048, help="parquet 读批大小 / tokenizer 批大小")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fast-dev-run", action="store_true", help="每个分片只处理少量数据，用于跑通流程")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    index_path = args.out_root / "pool_url_index.npy"

    if args.stage == "url-index":
        build_url_index(args.raw_root, index_path, args.batch_size)
        return 0

    if args.stage == "tokenize" and args.role is None:
        raise SystemExit("--stage tokenize 需要 --role：seg00 / seg01 / ... 或 probe")

    # probe 不参与分段，也就不需要池子的 token 数；其余情况都要，且不能给默认值——
    # 名义上的 10B 是别家 tokenizer 数的，拿它换算会让最后一段悄悄不够。
    needs_segments = args.stage == "plan" or args.role != "probe"
    segment_ppms: list[int] = []
    replay_tokens: list[int] = []
    if needs_segments:
        try:
            replay_tokens = [int(float(x)) for x in args.replay_tokens.split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"--replay-tokens 只能是逗号分隔的数字，收到 {args.replay_tokens!r}")
        if args.pool_tokens is None:
            raise SystemExit(
                "缺 --pool-tokens：段的边界按「replay 目标 token / 池子实测 token」换算。"
                "先跑 `python -m data_prep.checks pool-tokens`，它会直接打出可粘贴的值"
            )
        try:
            segment_ppms = segment_ppms_from_tokens(replay_tokens, int(args.pool_tokens))
        except ValueError as e:
            raise SystemExit(str(e))
    segment_roles = [segment_name(i) for i in range(len(segment_ppms))]

    if args.stage == "plan":
        write_replay_plan(args.out_root, segment_ppms, replay_tokens, args.new_tokens)
        return 0

    if args.role == "probe":
        src = args.raw_root / "fineweb_edu_probe"
        router = subsample_router("fineweb_edu", "probe", args.probe_ppm)
        if not index_path.exists():
            raise SystemExit(f"缺少 {index_path}，先跑 --stage url-index（probe 必须排除 10BT 池子的 URL）")
        exclude_path = str(index_path)
    elif args.role in segment_roles:
        src = args.raw_root / "fineweb_edu_pool"
        # 整套段一起构造再取一个，是为了让每次调用看到的边界完全一致：段 k 的起点是
        # 前 k 段 ppm 之和，只算自己那一段是算不出来的。
        router = segment_router(segment_ppms)
        exclude_path = None
    else:
        raise SystemExit(
            f"--role {args.role} 不在 {segment_roles + ['probe']} 里。"
            f"段数由 --replay-tokens（当前 {replay_tokens}）决定"
        )

    shards = sorted(str(p) for p in src.rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"{src} 下没有 parquet，先跑 download.py")

    out_dir = args.out_root / args.role
    stats_dir = args.out_root / "_stats" / args.role
    if stats_dir.exists():
        for old in stats_dir.glob("*.json"):
            old.unlink()

    print(f"role={args.role}  分片 {len(shards)} 个  →  {out_dir}")
    print(f"划分：{json.dumps(router.describe(), ensure_ascii=False)}")

    write_token_chunks(
        partial(
            tokenize_shard,
            role=args.role,
            router=router,
            tokenizer_dir=str(args.tokenizer_dir),
            stats_dir=str(stats_dir),
            exclude_path=exclude_path,
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
            "dataset": "fineweb-edu",
            "role": args.role,
            # 换算依据一起记下：plan 要拿它核对三次 tokenize 用的是不是同一套边界。
            "replay_tokens_target": replay_tokens or None,
            "pool_tokens": int(args.pool_tokens) if args.pool_tokens else None,
            "segment_ppms": segment_ppms or None,
            "source_dir": str(src),
            "shards": [Path(s).name for s in shards],
            "doc_key": DOC_KEY["fineweb_edu"],
            "router": router.describe(),
            "url_exclusion": exclude_path,
            "block_size": args.block_size,
            "chunk_blocks": args.chunk_blocks,
            "stats": stats,
        },
    )
    print("\n统计：")
    for k in sorted(stats):
        print(f"  {k:<20} {fmt_int(stats[k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
