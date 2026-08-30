"""Preview HumanEval-style rewrites on the local educational_instruct dump.

    python test_code/preview_opc_heval.py
    python test_code/preview_opc_heval.py --num 8 --seed 1
    python test_code/preview_opc_heval.py --save codedata_check/data/opc_heval_preview.json

Reads the save_to_disk directory you already downloaded. Walks the dataset in
a fixed-seed random order and prints the first N rows that convert; rows that
cannot be rewritten are counted but skipped, so what you see is the form that
would actually enter the pool as `style=heval`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onereplay.scripts.prepare_opc_ccode import build_heval_view  # noqa: E402


def load_local_opc(data_dir: str):
    """Load the save_to_disk dump without going through Features.from_dict.

    The dump was written by a newer `datasets` that serializes list columns as
    `_type: List`. Older installs only know `Sequence`, so load_from_disk
    raises ValueError before any row is read. The arrow table itself is fine.
    """

    import pyarrow as pa

    source = Path(data_dir)
    arrow_files = sorted(source.glob("*.arrow"))
    if not arrow_files:
        raise SystemExit(f"no .arrow files under {source}")

    def read_arrow(path: Path):
        with path.open("rb") as handle:
            try:
                return pa.ipc.open_file(handle).read_all()
            except (pa.ArrowInvalid, pa.lib.ArrowInvalid):
                handle.seek(0)
                return pa.ipc.open_stream(handle).read_all()

    table = read_arrow(arrow_files[0])
    for extra in arrow_files[1:]:
        table = pa.concat_tables([table, read_arrow(extra)])
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="codedata_check/data/opencoder_educational",
    )
    parser.add_argument("--num", type=int, default=6, help="How many converted examples to show.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_scan",
        type=int,
        default=400,
        help="Give up after this many random rows if --num conversions are not found.",
    )
    parser.add_argument("--save", type=str, default="", help="Optional JSON dump of the previews.")
    return parser.parse_args()


def shorten(text: object, limit: int = 1200) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[TRUNCATED]..."


def main() -> None:
    args = parse_args()
    table = load_local_opc(args.data_dir)
    print(f"loaded {table.num_rows} rows from {args.data_dir}")
    print(f"columns: {table.column_names}")

    rng = random.Random(args.seed)
    order = list(range(table.num_rows))
    rng.shuffle(order)

    previews: list[dict] = []
    scanned = 0
    fallback = 0
    for index in order:
        if scanned >= args.max_scan or len(previews) >= args.num:
            break
        scanned += 1
        row = {name: table[name][index].as_py() for name in table.column_names}
        built = build_heval_view(
            str(row.get("code") or ""),
            str(row.get("entry_point") or ""),
            str(row.get("instruction") or ""),
        )
        if built is None:
            fallback += 1
            continue
        stub, body = built
        previews.append(
            {
                "index": index,
                "seq_id": row.get("seq_id"),
                "entry_point": row.get("entry_point"),
                "instruction": row.get("instruction"),
                "code": row.get("code"),
                "heval_prompt": stub,
                "reference_body": body,
                "num_tests": len(row.get("testcase") or []),
            }
        )

    print(
        f"scanned {scanned} rows: converted {len(previews)}, "
        f"fallback {fallback} (would stay bare in the pool)"
    )

    for i, item in enumerate(previews, start=1):
        print("\n" + "=" * 88)
        print(f"PREVIEW {i}/{len(previews)}  index={item['index']}  entry_point={item['entry_point']}")
        print("=" * 88)
        print("\n[original instruction]")
        print(item["instruction"])
        print("\n[original code]")
        print(shorten(item["code"]))
        print("\n[heval prompt  ——  this is what the pool would store as `inputs`]")
        print(item["heval_prompt"] + "<<< model completes from here >>>")
        print("\n[reference body  ——  `targets`, overwritten later by self-distill]")
        print(item["reference_body"])

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved {len(previews)} previews -> {out}")


if __name__ == "__main__":
    main()
