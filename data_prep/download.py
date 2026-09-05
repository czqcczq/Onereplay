"""在登录节点下载三个数据集的原始 parquet。

计算节点是离线的（HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1 / TRANSFORMERS_OFFLINE=1），
所以下载和后面的 prepare 都必须在登录节点做完，进队列时只能读本地文件。

分片数量是显式参数而不是「下整个 config」，因为体积差别巨大：
  fineweb-edu sample/10BT   14 片   28.5 GB   —— 全要（replay 池 + 采 C 的来源）
  fineweb-edu CC-MAIN-2024-18  50 片  55.2 GB —— 只要 2~4 片，probe 用不了几千万 token
  finemath   finemath-4plus  64 片   18.4 GB  —— 全要
  Biomed-Enriched data/commercial-*  26 片  49.0 GB —— 全要

用法（先 --dry-run 看清单和体积，再去掉它真下）：
    python -m data_prep.download --dataset fineweb_edu_pool  --dry-run
    python -m data_prep.download --dataset fineweb_edu_probe --probe-dumps CC-MAIN-2024-18,CC-MAIN-2024-22 --shards-per-dump 2
    python -m data_prep.download --dataset biomed
    python -m data_prep.download --dataset finemath
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_RAW_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEMATH_REPO = "HuggingFaceTB/finemath"
BIOMED_REPO = "almanach/Biomed-Enriched"

# 基座 open-sci-ref-v0.02 用的 FineWeb-Edu v1.0.0 覆盖到 CC-MAIN-2024-10 为止，
# 所以干净的 held-out probe 只能取 2024-18 及之后的 dump。
FIRST_CLEAN_DUMP = "CC-MAIN-2024-18"
DEFAULT_PROBE_DUMPS = ("CC-MAIN-2024-18", "CC-MAIN-2024-22")


def list_shards(repo: str, prefix: str) -> list[tuple[str, int]]:
    """列出 repo 下某前缀的所有文件（路径, 字节数），按路径排序。

    不去猜文件命名（sample/ 下是 `000_00000.parquet`，data/ 下是
    `train-00000-of-000NN.parquet`，finemath 下又不一样），直接问 HF 要清单，
    然后把确切文件名交给 snapshot_download 的 allow_patterns。
    """
    from huggingface_hub import HfApi

    entries = [
        (e.path, e.size or 0)
        for e in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True, expand=False)
        if getattr(e, "size", None) is not None and e.path.startswith(prefix) and e.path.endswith(".parquet")
    ]
    if not entries:
        raise RuntimeError(f"{repo} 下 {prefix!r} 没有 parquet 文件")
    return sorted(entries)


def plan(dataset: str, args) -> tuple[str, list[tuple[str, int]]]:
    """返回 (repo, [(文件路径, 字节数)])。"""
    if dataset == "fineweb_edu_pool":
        return FINEWEB_REPO, list_shards(FINEWEB_REPO, "sample/10BT/")

    if dataset == "fineweb_edu_probe":
        dumps = [d.strip() for d in args.probe_dumps.split(",") if d.strip()]
        for d in dumps:
            if d < FIRST_CLEAN_DUMP:
                raise SystemExit(
                    f"dump {d} 早于 {FIRST_CLEAN_DUMP}，属于基座 FineWeb-Edu v1.0.0 的池子，不能当 held-out probe"
                )
        files: list[tuple[str, int]] = []
        for d in dumps:
            shards = list_shards(FINEWEB_REPO, f"data/{d}/")
            files.extend(shards[: args.shards_per_dump])
        return FINEWEB_REPO, files

    if dataset == "finemath":
        return FINEMATH_REPO, list_shards(FINEMATH_REPO, "finemath-4plus/")

    if dataset == "biomed":
        # 只要 commercial：noncommercial 的 text 列是空的（许可限制），要靠本地几百 GB
        # 的 PMC OA XML dump 回填，已决定放弃。
        return BIOMED_REPO, list_shards(BIOMED_REPO, "data/commercial-")

    raise SystemExit(f"未知 dataset: {dataset}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["fineweb_edu_pool", "fineweb_edu_probe", "finemath", "biomed"],
    )
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="原始 parquet 落地根目录")
    ap.add_argument("--probe-dumps", default=",".join(DEFAULT_PROBE_DUMPS), help="probe 用哪些 CC dump（逗号分隔）")
    ap.add_argument("--shards-per-dump", type=int, default=2, help="每个 probe dump 取几个分片")
    ap.add_argument("--workers", type=int, default=8, help="并发下载数")
    ap.add_argument("--dry-run", action="store_true", help="只打印清单和体积，不下载")
    args = ap.parse_args(argv)

    repo, files = plan(args.dataset, args)
    total = sum(sz for _, sz in files)
    dest = args.raw_root / args.dataset

    print(f"repo     : {repo}")
    print(f"目标目录 : {dest}")
    print(f"分片数   : {len(files)}")
    print(f"总体积   : {total / 1e9:.2f} GB")
    print("清单：")
    for p, sz in files:
        print(f"  {p:<52} {sz / 1e9:>7.2f} GB")

    if args.dry_run:
        print("\n--dry-run，未下载。")
        return 0

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[p for p, _ in files],
        local_dir=str(dest),
        max_workers=args.workers,
    )
    print(f"\n下载完成：{path}")

    # snapshot_download 保留 repo 内的目录结构，落地后确认文件数对得上，
    # 免得后面 prepare 静默处理了残缺的输入。
    got = sorted(str(q.relative_to(dest)).replace("\\", "/") for q in Path(dest).rglob("*.parquet"))
    want = sorted(p for p, _ in files)
    if got != want:
        print(f"!! 落地文件与清单不一致：期望 {len(want)} 个，实际 {len(got)} 个", file=sys.stderr)
        for miss in sorted(set(want) - set(got)):
            print(f"   缺失 {miss}", file=sys.stderr)
        return 1
    print(f"校验通过：{len(got)} 个 parquet 全部落地。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
