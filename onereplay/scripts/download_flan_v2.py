"""Pull a row sample out of SirNeural/flan_v2 without ever storing the raw dump.

Run this on a login node: the compute nodes set HF_HUB_OFFLINE=1 because this
site hangs on outbound network, so nothing can be downloaded from inside a job.

The four files this reads are the Flan 2021 sub-mixture in its four template
styles, which is the whole of what the IF line wants (no t0, no niv2, no cot, no
dialog):

    flan_zs_opt_train.jsonl.gz            10.9 GB gz
    flan_zs_noopt_train.jsonl.gz          10.7 GB gz
    flan_fs_opt_train_part{1,2,3}.jsonl.gz  83.3 GB gz, split at raw byte
                                          boundaries, so only the concatenation
                                          is a valid gzip stream
    flan_fs_noopt_train.jsonl.gz          31.4 GB gz

127 GiB compressed, ~340 GiB decompressed, against ~150 GB of free scratch. So
the raw files are never written: each one is streamed over HTTP, decompressed
incrementally in memory, and only the sampled rows reach the disk. Peak disk is
the output jsonl, a few tens of MB. Peak memory is one buffer of --chunk_mb.

Why a prefix is enough
----------------------
Each file is one deflate stream with no sync points, so there is no random
access: rows can only be read from the head. That is acceptable because the
official generator writes with seqio's `shuffle=True` (see run_example.py), so
the write order is already a random interleaving of tasks rather than blocked by
task -- but "should be interleaved" and "is interleaved" are different claims,
and this dump carries no task column to check it with (`task` is the constant
string "flan" on every row of every flan_* file, i.e. the sub-mixture name, not
the task name).

So rather than assume it, this reads a window of --read_mb decompressed bytes,
which is 5-10x the rows actually wanted, reservoir-samples the quota out of that
window, and prints a per-block breakdown of the window by template family. Five
blocks whose family shares agree with the window's is evidence the stream is
stationary; a drifting table is a reason to raise --read_mb or to distrust the
draw. The check costs one pass over a window that is already being read.

The same missing task column is why FLAN's own examples-proportional mixing with
a rate maximum (Wei et al. 2021) cannot be applied here the way
old_knowledge.allocate_task_quota applies it to the v1 dump. The family table is
a lexical stand-in for the mixture audit, not a substitute for the policy.

Usage
-----
    python -m onereplay.scripts.download_flan_v2 \\
        --out_dir /scratch/weiliu87/student/czq/Onereplay/datasets/flan_v2

    # watch it more closely / cut it short
    python -m onereplay.scripts.download_flan_v2 --out_dir ... \\
        --read_mb 200 --progress_mb 25

Defaults pull 60k/60k/15k/15k rows, which covers a 50k training pool plus a 20k
replay pool plus a held-out probe slice at the 40/40/10/10 remix weighting with
room to re-slice later without re-downloading. prepare_flan_v2.py does the
slicing.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import time
import zlib
from pathlib import Path
from typing import Any, Iterator

BASE_URL = "https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main"

# label -> (files in concatenation order, default row quota)
# The parts of flan_fs_opt are listed in order because only their concatenation
# is a valid gzip stream; a --read_mb small enough to stay inside part1 never
# reaches the others, which is the normal case.
REMIXES: dict[str, tuple[tuple[str, ...], int]] = {
    "zs_opt": (("flan_zs_opt_train.jsonl.gz",), 60000),
    "zs_noopt": (("flan_zs_noopt_train.jsonl.gz",), 60000),
    "fs_opt": (
        (
            "flan_fs_opt_train_part1.jsonl.gz",
            "flan_fs_opt_train_part2.jsonl.gz",
            "flan_fs_opt_train_part3.jsonl.gz",
        ),
        15000,
    ),
    "fs_noopt": (("flan_fs_noopt_train.jsonl.gz",), 15000),
}

# Lexical stand-in for the task column this dump does not have. First match
# wins, so the shares are a partition rather than per-family recall, and
# calibrating "translation" against a dump that does carry task names put its
# recall near 0.7 -- read every number as a lower bound. This exists to detect
# drift across blocks of one stream, where a consistent bias cancels out, and to
# give the manifest a mixture fingerprint. It is not a task distribution.
FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "translation",
        re.compile(
            r"translat|Übersetz|traduire|"
            r"\b(French|German|Russian|Czech|Finnish|Romanian|Turkish)\b"
            r".{0,40}\b(say|sentence|text)\b",
            re.I,
        ),
    ),
    (
        "summarization",
        re.compile(r"summar|\bshort summary\b|\bheadline\b|\btl;?dr\b|in one sentence", re.I),
    ),
    ("nli", re.compile(r"\bentail|hypothesis|premise|can we (conclude|infer)", re.I)),
    (
        "sentiment",
        re.compile(r"sentiment|positive or negative|\breview\b.{0,60}\b(rating|star)", re.I),
    ),
    (
        "qa_reading",
        re.compile(
            r"\bquestion:|answer the question|based on (the|this) (passage|article|paragraph)",
            re.I,
        ),
    ),
    ("multiple_choice", re.compile(r"^\s*Options:|\bOPT:|\n- |choose|which of", re.I | re.M)),
    ("math", re.compile(r"\bsolve\b|\bwhat is the value\b|=\s*-?\d|\bequation\b", re.I)),
)

REQUIRED_KEYS = ("inputs", "targets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-sample the Flan 2021 sub-mixture out of SirNeural/flan_v2."
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--remixes",
        type=str,
        default="zs_opt,zs_noopt,fs_opt,fs_noopt",
        help="Which of the four template styles to pull. The IF line wants all four.",
    )
    parser.add_argument(
        "--rows",
        type=str,
        default="",
        help="Override the per-remix quotas as LABEL=N pairs, e.g. "
        "'zs_opt=60000,fs_opt=15000'. Unlisted remixes keep their default.",
    )
    parser.add_argument(
        "--read_mb",
        type=int,
        default=400,
        help="Decompressed megabytes to read per remix before stopping. The "
        "reservoir draws the quota from this whole window, so a larger value "
        "buys a wider draw and a stronger stationarity check at the cost of "
        "network only. 400 MB is roughly 400k zs rows or 130k fs rows.",
    )
    parser.add_argument(
        "--chunk_mb",
        type=int,
        default=8,
        help="HTTP read size. Also the granularity at which the run can be "
        "interrupted, and the peak memory of the decompression buffer.",
    )
    parser.add_argument(
        "--progress_mb",
        type=int,
        default=50,
        help="Print a progress line every this many decompressed megabytes.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Blocks the read window is split into for the stationarity table.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--base_url",
        type=str,
        default=BASE_URL,
        help="Point at a mirror, or at a file:// directory of already-downloaded gz.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-request socket timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="Retries per file. The connection to huggingface.co drops often "
        "enough that a single-shot download is not worth attempting; a retry "
        "restarts that file's stream from the beginning, which is cheap "
        "because only --read_mb is ever read.",
    )
    parser.add_argument(
        "--force",
        type=int,
        default=0,
        help="1 re-pulls a remix whose output jsonl already exists.",
    )
    return parser.parse_args()


def parse_row_overrides(spec: str, defaults: dict[str, int]) -> dict[str, int]:
    quotas = dict(defaults)
    for piece in (part.strip() for part in spec.split(",")):
        if not piece:
            continue
        if "=" not in piece:
            raise SystemExit(f"--rows entry {piece!r} is not LABEL=N")
        label, value = piece.split("=", 1)
        label = label.strip()
        if label not in defaults:
            raise SystemExit(f"--rows names unknown remix {label!r}; known: {sorted(defaults)}")
        quotas[label] = int(value)
    return quotas


def classify(text: str) -> str:
    for name, pattern in FAMILY_PATTERNS:
        if pattern.search(text):
            return name
    return "other"


def stream_bytes(
    url: str, chunk_size: int, timeout: int
) -> Iterator[bytes]:
    """Yield raw bytes of one remote or local file."""

    if url.startswith("file://") or "://" not in url:
        path = Path(url[7:] if url.startswith("file://") else url)
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    return
                yield chunk
        return

    import requests

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk


def read_window(
    label: str,
    files: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream a decompressed window of one remix and reservoir-sample it.

    The gzip state carries across files so the parts of a split stream
    concatenate correctly. Returns the sampled rows plus the stats the manifest
    and the stationarity table need.
    """

    quota = args.quotas[label]
    limit_bytes = args.read_mb * 1024 * 1024
    chunk_size = args.chunk_mb * 1024 * 1024
    progress_bytes = max(args.progress_mb, 1) * 1024 * 1024

    rng = random.Random(f"{args.seed}:{label}")
    reservoir: list[dict[str, Any]] = []
    # Family counts per block of the read window. Accumulated over every row the
    # stream yields rather than over the survivors, so the drift check describes
    # the window itself and does not inherit the reservoir's own randomness.
    block_families: list[collections.Counter] = [
        collections.Counter() for _ in range(max(args.blocks, 1))
    ]

    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = b""
    seen = 0
    decompressed = 0
    # Bytes handed to the row loop, which advances per line. `decompressed`
    # advances per HTTP chunk instead, and a chunk is wider than a block at the
    # default settings, so using it for the block index would leave blocks empty
    # and make the drift table skip rows.
    consumed = 0
    compressed = 0
    malformed = 0
    empty = 0
    next_progress = progress_bytes
    started = time.time()
    stopped_early = False

    for name in files:
        url = f"{args.base_url.rstrip('/')}/{name}"
        print(f"  streaming {name}")
        for raw in stream_bytes(url, chunk_size, args.timeout):
            compressed += len(raw)
            try:
                text = decompressor.decompress(raw)
            except zlib.error as exc:
                raise SystemExit(
                    f"{label}: gzip stream broke after {decompressed / 1e6:.0f} MB ({exc}). "
                    "For flan_fs_opt this usually means a part was read out of order."
                )
            if not text:
                continue
            decompressed += len(text)
            pending += text
            lines = pending.split(b"\n")
            pending = lines.pop()

            for line in lines:
                consumed += len(line) + 1
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if any(not str(row.get(key) or "").strip() for key in REQUIRED_KEYS):
                    empty += 1
                    continue

                # Block the row belongs to, by position in the window. Counted
                # before the reservoir decision so the family table covers the
                # whole window rather than only the survivors.
                block = min(
                    int(consumed / max(limit_bytes, 1) * len(block_families)),
                    len(block_families) - 1,
                )
                block_families[block][classify(str(row["inputs"]))] += 1

                slim = {
                    "inputs": str(row["inputs"]),
                    "targets": str(row["targets"]),
                    "task": str(row.get("task") or ""),
                }
                if len(reservoir) < quota:
                    reservoir.append(slim)
                else:
                    # Standard reservoir sampling: every row of the window ends
                    # up equally likely to survive, so the draw does not favor
                    # the head of the stream the way a plain head-cut would.
                    victim = rng.randrange(seen + 1)
                    if victim < quota:
                        reservoir[victim] = slim
                seen += 1

            if decompressed >= next_progress:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"    {decompressed / 1e6:7.0f} MB jsonl "
                    f"({compressed / 1e6:6.0f} MB gz, {compressed / elapsed / 1e6:4.1f} MB/s)  "
                    f"{seen:>9,} rows seen  {len(reservoir):>6,} held"
                )
                next_progress += progress_bytes

            if decompressed >= limit_bytes:
                stopped_early = True
                break
        if stopped_early:
            break

    # Reservoir position is not a uniform permutation -- a slot never chosen for
    # replacement still holds the row that landed there during fill, so early
    # rows are over-represented at the front. prepare_flan_v2 cuts nested slices
    # off the front of this file, so the order has to be shuffled or those
    # slices would be biased toward the head of the stream.
    random.Random(f"{args.seed}:order:{label}").shuffle(reservoir)

    stats = {
        "files_read": list(files),
        "rows_seen": seen,
        "rows_kept": len(reservoir),
        "compressed_bytes_read": compressed,
        "decompressed_bytes_read": decompressed,
        "read_seconds": round(time.time() - started, 1),
        "stopped_at_read_mb_limit": stopped_early,
        "malformed_lines": malformed,
        "rows_dropped_empty": empty,
        "block_families": [dict(counter) for counter in block_families],
    }
    return reservoir, stats


def stationarity_table(stats: dict[str, Any]) -> dict[str, float]:
    """Print family shares per block and return the worst block-vs-window TVD.

    A stationary stream makes every block agree with the window, so the draw
    does not depend on how far into the file it reached. Drift here is the one
    failure mode a head-only read cannot recover from, so it is printed rather
    than merely stored.
    """

    blocks = [collections.Counter(block) for block in stats["block_families"]]
    totals = collections.Counter()
    for block in blocks:
        totals.update(block)
    total = sum(totals.values())
    if total == 0:
        print("    (no rows classified)")
        return {}

    names = [name for name, _ in sorted(totals.items(), key=lambda item: -item[1])]
    header = "    " + "block".ljust(7) + "".join(name[:11].rjust(13) for name in names)
    print(header)
    tvds = []
    for index, block in enumerate(blocks):
        size = sum(block.values())
        if size == 0:
            continue
        row = f"    {index:<7}"
        for name in names:
            row += f"{block[name] / size:>12.1%} "
        tvds.append(
            0.5 * sum(abs(block[name] / size - totals[name] / total) for name in names)
        )
        print(row)
    row = "    " + "WINDOW".ljust(7)
    for name in names:
        row += f"{totals[name] / total:>12.1%} "
    print(row)
    print(
        f"    worst TVD(block, window) = {max(tvds):.3f}"
        "   <- 越接近 0 说明流是平稳的、只读开头不引入偏置"
    )
    return {name: totals[name] / total for name in names}


def main() -> None:
    args = parse_args()
    labels = [piece.strip() for piece in args.remixes.split(",") if piece.strip()]
    unknown = [label for label in labels if label not in REMIXES]
    if unknown:
        raise SystemExit(f"unknown remix(es) {unknown}; known: {sorted(REMIXES)}")

    args.quotas = parse_row_overrides(
        args.rows, {label: REMIXES[label][1] for label in REMIXES}
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==== flan_v2 (Flan 2021 sub-mixture) 取样 ====")
    print(f"  源        : {args.base_url}")
    print(f"  输出      : {out_dir}")
    print(f"  每路读窗口: {args.read_mb} MB 解压后，从中蓄水池抽样")
    print(f"  配额      : {', '.join(f'{k}={args.quotas[k]}' for k in labels)}")
    print("  原始 gz 不落盘，只写抽到的行\n")

    manifest: dict[str, Any] = {
        "source": args.base_url,
        "note": "SirNeural/flan_v2 flan_* files only = the Flan 2021 sub-mixture in "
        "its four template styles (zs/fs x opt/noopt). No t0 / niv2 / cot / dialog. "
        "The dump's `task` column is the constant string 'flan', so it carries no "
        "task granularity and FLAN's capped-proportional task balancing cannot be "
        "applied to it; `families` below is a lexical approximation of the mixture "
        "kept only as a fingerprint and a drift check, not a task distribution. "
        "Each file is one deflate stream with no random access, so rows are read "
        "from the head of the stream and reservoir-sampled within a --read_mb window.",
        "read_mb": args.read_mb,
        "seed": args.seed,
        "remixes": {},
    }

    for label in labels:
        files, _ = REMIXES[label]
        out_path = out_dir / f"flan_v2_{label}.jsonl"
        if out_path.exists() and args.force != 1:
            existing = sum(1 for _ in out_path.open("r", encoding="utf-8"))
            print(f"==== {label} ==== 跳过：{out_path} 已有 {existing} 行（--force 1 重拉）\n")
            manifest["remixes"][label] = {
                "path": str(out_path),
                "rows": existing,
                "skipped_existing": True,
            }
            continue

        print(f"==== {label} ====")
        rows: list[dict[str, Any]] = []
        stats: dict[str, Any] = {}
        for attempt in range(1, args.retries + 1):
            try:
                rows, stats = read_window(label, files, args)
                break
            except SystemExit:
                raise
            except Exception as exc:
                print(f"  第 {attempt}/{args.retries} 次失败: {type(exc).__name__}: {exc}")
                if attempt == args.retries:
                    raise SystemExit(
                        f"{label}: {args.retries} 次都没读完。降低 --read_mb 或换 --base_url。"
                    )
                time.sleep(min(5 * attempt, 60))

        # Written in reservoir order, which is already a random permutation of
        # the window, so prepare_flan_v2 can cut nested slices off the front.
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        size_mb = out_path.stat().st_size / 1e6
        print(
            f"  读了 {stats['decompressed_bytes_read'] / 1e6:.0f} MB jsonl "
            f"/ {stats['compressed_bytes_read'] / 1e6:.0f} MB gz，"
            f"看过 {stats['rows_seen']:,} 行，抽出 {stats['rows_kept']:,} 行 "
            f"-> {out_path.name} ({size_mb:.1f} MB)"
        )
        if not stats["stopped_at_read_mb_limit"]:
            print(
                "  !! 整个文件读完了都没到 --read_mb，说明窗口比文件还大，"
                "抽样退化成全量随机（这对 flan_* 不该发生，检查 --base_url）"
            )
        if stats["rows_kept"] < args.quotas[label]:
            print(
                f"  !! 只抽到 {stats['rows_kept']} 行，少于配额 {args.quotas[label]}。"
                "调大 --read_mb。"
            )
        if stats["malformed_lines"] or stats["rows_dropped_empty"]:
            print(
                f"  丢弃: {stats['malformed_lines']} 行 JSON 解析失败, "
                f"{stats['rows_dropped_empty']} 行 inputs/targets 为空"
            )
        print("  平稳性检查（按模板家族，词法近似，只看跨块是否一致）:")
        families = stationarity_table(stats)

        entry = {key: value for key, value in stats.items() if key != "block_families"}
        entry["path"] = str(out_path)
        entry["quota"] = args.quotas[label]
        entry["families"] = families
        entry["output_bytes"] = out_path.stat().st_size
        manifest["remixes"][label] = entry
        print()

    manifest_path = out_dir / "flan_v2_download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"写出 {manifest_path}")
    print(
        "\n下一步（同样在登录节点，CPU 即可）:\n"
        "  python -m onereplay.scripts.prepare_flan_v2 \\\n"
        f"    --sample_dir {out_dir} \\\n"
        "    --out_dir <.../datasets/flan_v2_if_50k> \\\n"
        "    --tokenizer_path <.../models/Qwen3-1.7B-Base>"
    )


if __name__ == "__main__":
    main()
