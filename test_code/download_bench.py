"""Download the eleven evaluation benchmarks from HuggingFace, once, into place.

    python test_code/download_bench.py                  # 全部
    python test_code/download_bench.py --only medqa,fpb
    python test_code/download_bench.py --dest /scratch/.../datasets
    python test_code/download_bench.py --only fpb,fiqasa,tfns \\
        --fingpt_train datasets/raw/fingpt_sentiment_train.parquet   # 顺带查污染

Everything lands under --dest with exactly the paths 90_specialist_sft.pbs
expects, so after this finishes the PBS defaults just work:

    datasets/bench/medqa/test.jsonl
    datasets/bench/pubmedqa/test.jsonl
    datasets/bench/medmcqa/test.jsonl
    datasets/bench/careqa/test.jsonl
    datasets/bench/fpb/test.jsonl
    datasets/bench/fiqasa/test.jsonl
    datasets/bench/tfns/test.jsonl
    datasets/code/humaneval_test.parquet
    datasets/code/humanevalplus_test.parquet
    datasets/code/mbppplus_test.parquet
    datasets/code/mbpp_full/

The code four go to datasets/code/ rather than alongside the rest because
download_code_data.py has been putting them there since before this line
existed and other scripts still read that path. Anything already present is
skipped, so running this on the cluster fetches only what is genuinely missing.

Why every entry pins allow_patterns
-----------------------------------
snapshot_download takes the whole repo by default, and these repos are mostly
not the eval split:

    OctoMed/MedQA-5options       train is ~500MB (16 teacher responses per
                                 question), test is 0.7MB
    openlifescienceai/medmcqa    train is 182,822 rows / 132MB; the benchmark is
                                 the 4,183-row validation split
    qiaojin/PubMedQA             pqa_artificial is 211k generated rows; only the
                                 1000 expert-labeled ones are the benchmark

Unpinned, this script would pull well over a gigabyte of data that never gets
evaluated.

Two substitutions worth knowing about
-------------------------------------
* FPB comes from ChanceFocus/en-fpb, not TheFinAI/en-fpb. They are the same
  bytes (identical splits: 3100/776/970, identical dataset_size), but the
  TheFinAI copy is gated behind a click-through, which makes an unattended
  download on a login node fail with a 401.
* TFNS is the validation split, because the released test split is unlabeled.
  Same reason MedMCQA is evaluated on validation.

Behind the Great Firewall set a mirror before running:

    $env:HF_ENDPOINT = "https://hf-mirror.com"     # PowerShell
    export HF_ENDPOINT=https://hf-mirror.com       # bash
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DEST = "datasets"


@dataclass
class Spec:
    name: str
    # Path relative to --dest, subdirectory included.
    target: str
    # HuggingFace repo, or "" when the benchmark comes straight from a url.
    repo_id: str = ""
    # Pinned on purpose; see the module docstring.
    allow: list[str] = field(default_factory=list)
    # Glob, relative to the snapshot, selecting the file(s) to publish.
    source: str = ""
    # Direct download, used when no HuggingFace repo carries the actual data.
    url: str = ""
    # "parquet" converts to jsonl, "csv" likewise, "copy" keeps the bytes,
    # "auto" decides by extension.
    mode: str = "parquet"
    note: str = ""


SPECS: list[Spec] = [
    # --- medical ------------------------------------------------------------
    Spec(
        name="medqa",
        repo_id="OctoMed/MedQA-5options",
        allow=["data/test-*"],
        source="data/test-*.parquet",
        target="bench/medqa/test.jsonl",
        note="1273 USMLE questions, 5 options",
    ),
    Spec(
        name="pubmedqa",
        repo_id="qiaojin/PubMedQA",
        # pqa_labeled = PQA-L, the 1000 expert-annotated rows. pqa_artificial
        # (211k) and pqa_unlabeled are training material, not the benchmark.
        allow=["pqa_labeled/*"],
        source="pqa_labeled/*.parquet",
        target="bench/pubmedqa/test.jsonl",
        note="1000 expert-labeled yes/no/maybe",
    ),
    Spec(
        name="medmcqa",
        repo_id="openlifescienceai/medmcqa",
        # validation, not test: the released test split has no cop column.
        allow=["data/validation-*"],
        source="data/validation-*.parquet",
        target="bench/medmcqa/test.jsonl",
        note="4183 AIIMS/NEET-PG questions, 4 options, cop is 0-based",
    ),
    Spec(
        name="careqa",
        repo_id="HPAI-BSC/CareQA",
        # Closed-ended English only. The _open variant is free-response and the
        # _es one is Spanish; neither is graded by this line.
        allow=["CareQA_en.json"],
        source="CareQA_en.json",
        target="bench/careqa/test.jsonl",
        mode="auto",
        note="5621 Spanish FSE exam questions in English, cop is 1-based",
    ),
    # --- finance ------------------------------------------------------------
    Spec(
        name="fpb",
        repo_id="ChanceFocus/en-fpb",
        allow=["data/test-*"],
        source="data/test-*.parquet",
        target="bench/fpb/test.jsonl",
        note="970 sentences; non-gated mirror of TheFinAI/en-fpb",
    ),
    Spec(
        name="fiqasa",
        repo_id="TheFinAI/fiqa-sentiment-classification",
        allow=["data/test-*"],
        source="data/test-*.parquet",
        target="bench/fiqasa/test.jsonl",
        note="234 rows; class comes from the sign of `score`",
    ),
    Spec(
        name="tfns",
        repo_id="zeroshot/twitter-financial-news-sentiment",
        # csv, not parquet -- this repo ships two csv files and nothing else.
        allow=["sent_valid.csv"],
        source="sent_valid.csv",
        target="bench/tfns/test.jsonl",
        mode="csv",
        note="2388 tweets; label 0=Bearish 1=Bullish 2=Neutral",
    ),
    # --- code ---------------------------------------------------------------
    Spec(
        name="humaneval",
        repo_id="openai/openai_humaneval",
        allow=["openai_humaneval/test-*"],
        source="openai_humaneval/test-*.parquet",
        # Copied, not converted: HumanEvalMetric reads it back as parquet.
        target="code/humaneval_test.parquet",
        mode="copy",
        note="164 tasks, pass@1",
    ),
    # EvalPlus: same tasks, ~80x the test cases. humanevalplus has exactly the
    # columns the base metric reads, so it is the same file in a different
    # place. mbppplus is 378 rows (EvalPlus dropped the ambiguous ones) and is
    # graded on its own `test` column -- see eval/metrics/evalplus.py.
    Spec(
        name="humanevalplus",
        repo_id="evalplus/humanevalplus",
        allow=["data/test-*"],
        source="data/test-*.parquet",
        target="code/humanevalplus_test.parquet",
        mode="copy",
        note="164 tasks, extended tests",
    ),
    Spec(
        name="mbppplus",
        repo_id="evalplus/mbppplus",
        allow=["data/test-*"],
        source="data/test-*.parquet",
        target="code/mbppplus_test.parquet",
        mode="copy",
        note="378 tasks (not 500); graded on `test`, not `test_list`",
    ),
    Spec(
        name="mbpp",
        repo_id="google-research-datasets/mbpp",
        # The whole `full` config: MBPPMetric calls load_from_disk(...)[split],
        # so the split names have to survive, and 90_specialist_sft.pbs asks for
        # `test` (500 tasks). The default split in evaluate.py is validation,
        # which is 90 tasks -- too few to read a pass@1 difference off.
        allow=["full/*"],
        source="full/*.parquet",
        target="code/mbpp_full",
        mode="save_to_disk",
        note="974 tasks across splits; the line evaluates test (500)",
    ),
]

# Which bench files the contamination check reads, and the column holding the
# graded sentence in each.
FINANCE_BENCHES = {
    "fpb": ("text", "sentence"),
    "fiqasa": ("sentence", "text"),
    "tfns": ("text", "sentence"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download evaluation benchmarks.")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of: " + ", ".join(spec.name for spec in SPECS),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch and rebuild even when the target file already exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print what would be downloaded and exit.",
    )
    parser.add_argument(
        "--fingpt_train",
        type=str,
        default="",
        help="FinGPT sentiment-train parquet. Given, the three finance benches "
        "are checked against it for overlap -- that corpus is built from the "
        "train splits of these same three sets, so a leak is plausible enough "
        "to be worth one set comparison.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    """Convert numpy containers to plain json types, recursively.

    parquet gives back numpy arrays for list columns, and json.dumps(default=str)
    would stringify one into "['A: ...' 'B: ...']" -- valid json holding a numpy
    repr, so nothing raises and the options list silently becomes one string.
    """

    import numpy as np

    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def to_jsonl(rows: Any, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def parquet_to_jsonl(paths: list[Path], target: Path) -> int:
    import pandas as pd

    frames = [pd.read_parquet(path) for path in sorted(paths)]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return to_jsonl(frame.to_dict("records"), target)


def csv_to_jsonl(paths: list[Path], target: Path) -> int:
    import pandas as pd

    frames = [pd.read_csv(path) for path in sorted(paths)]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return to_jsonl(frame.to_dict("records"), target)


def jsonish_to_jsonl(path: Path, target: Path) -> int:
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        with target.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "test", "dev", "validation", "rows"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if isinstance(payload, dict):
        # Column- or index-oriented json (CareQA_en.json could be either), which
        # pandas can reorient and a hand-rolled reader would get wrong.
        import pandas as pd

        return to_jsonl(pd.read_json(path).to_dict("records"), target)
    if not isinstance(payload, list):
        raise ValueError(f"{path.name}: expected a list of records, got {type(payload).__name__}")
    return to_jsonl(payload, target)


def parquet_to_disk(paths: list[Path], target: Path) -> str:
    """Rebuild a DatasetDict on disk, keeping the split names.

    MBPPMetric calls ``load_from_disk(path)[split]``, so a bare parquet file
    will not do -- the split names have to survive the trip, and they only exist
    in the file names (``full/test-00000-of-00001.parquet``).
    """

    import pandas as pd
    from datasets import Dataset, DatasetDict

    by_split: dict[str, list[Path]] = {}
    for path in sorted(paths):
        by_split.setdefault(path.stem.split("-")[0], []).append(path)

    bundle = DatasetDict(
        {
            split: Dataset.from_pandas(
                pd.concat([pd.read_parquet(item) for item in files], ignore_index=True)
            )
            for split, files in sorted(by_split.items())
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    bundle.save_to_disk(str(target))
    return ", ".join(f"{name}={len(rows)}" for name, rows in sorted(bundle.items()))


def fetch_url(spec: Spec, dest: Path, target: Path) -> tuple[bool, str]:
    import urllib.request

    raw = dest / "_hf" / spec.name / Path(spec.url).name
    raw.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec.url, timeout=120) as response:
        raw.write_bytes(response.read())

    if spec.mode == "copy":
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw, target)
        return True, f"{raw.name} -> {spec.target} ({target.stat().st_size / 1e6:.1f} MB)"
    rows = jsonish_to_jsonl(raw, target)
    return True, f"{rows} 行 -> {spec.target}"


def fetch(spec: Spec, dest: Path, force: bool) -> tuple[bool, str]:
    target = dest / spec.target
    if target.exists() and not force:
        if target.is_dir():
            files = sum(1 for path in target.rglob("*") if path.is_file())
            return True, f"已存在，跳过 (目录，{files} 个文件) -- --force 可重下"
        size = target.stat().st_size / 1e6
        return True, f"已存在，跳过 ({size:.1f} MB) -- --force 可重下"

    if spec.url:
        return fetch_url(spec, dest, target)

    from huggingface_hub import snapshot_download

    # Raw snapshot kept next to the converted file: when a loader turns out to
    # disagree with the data, the untouched original is still on disk.
    snapshot_dir = dest / "_hf" / spec.name
    snapshot_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        allow_patterns=spec.allow,
        local_dir=str(snapshot_dir),
    )

    matches = sorted(path for path in snapshot_dir.glob(spec.source) if path.is_file())
    if not matches:
        available = sorted(
            str(path.relative_to(snapshot_dir))
            for path in snapshot_dir.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )
        return False, f"下到了但没找到 {spec.source!r}；snapshot 里有: {available[:8]}"

    if spec.mode == "copy":
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matches[0], target)
        size = target.stat().st_size / 1e6
        return True, f"{matches[0].name} -> {spec.target} ({size:.1f} MB)"

    if spec.mode == "save_to_disk":
        parquet = [path for path in matches if path.suffix.lower() == ".parquet"]
        if not parquet:
            return False, f"没有 parquet 匹配 {spec.source!r}，实际: {[p.name for p in matches]}"
        return True, f"{parquet_to_disk(parquet, target)} -> {spec.target}"

    if spec.mode == "csv":
        csv_files = [path for path in matches if path.suffix.lower() == ".csv"]
        if not csv_files:
            return False, f"没有 csv 匹配 {spec.source!r}，实际: {[p.name for p in matches]}"
        return True, f"{csv_to_jsonl(csv_files, target)} 行 -> {spec.target}"

    parquet = [path for path in matches if path.suffix.lower() == ".parquet"]
    if parquet or spec.mode == "parquet":
        if not parquet:
            return False, f"没有 parquet 匹配 {spec.source!r}，实际匹配到: {[p.name for p in matches]}"
        rows = parquet_to_jsonl(parquet, target)
    else:
        rows = jsonish_to_jsonl(matches[0], target)
    return True, f"{rows} 行 -> {spec.target}"


def normalize_sentence(text: str) -> str:
    """Punctuation- and case-insensitive key for the overlap check."""

    return re.sub(r"\W+", " ", str(text or "").lower()).strip()


def check_contamination(dest: Path, fingpt_path: str) -> None:
    """Report how many finance bench rows also appear in the training corpus.

    FinGPT sentiment-train is assembled from the *train* splits of these three
    sets, so the expected answer is "almost none". It is checked rather than
    assumed because an overlap here would turn the finance numbers into a
    memorization measurement, and the check costs one pass over 77k strings.
    """

    import pandas as pd

    path = Path(fingpt_path)
    if not path.exists():
        print(f"!! 跳过污染自检：找不到 {path}")
        return

    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_json(
        path, lines=path.suffix.lower() in (".jsonl", ".ndjson")
    )
    column = "input" if "input" in frame.columns else frame.columns[0]
    train_keys = {normalize_sentence(value) for value in frame[column].tolist()}
    train_keys.discard("")
    print(f"训练集 {path.name}: {len(frame)} 行，{len(train_keys)} 个唯一句子")

    for name, keys in FINANCE_BENCHES.items():
        bench_file = dest / "bench" / name / "test.jsonl"
        if not bench_file.exists():
            print(f"  {name:8} 还没下载，跳过")
            continue
        sentences = []
        with bench_file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for key in keys:
                    if str(record.get(key) or "").strip():
                        sentences.append(str(record[key]))
                        break
        hits = [text for text in sentences if normalize_sentence(text) in train_keys]
        share = len(hits) / max(len(sentences), 1) * 100
        print(f"  {name:8} {len(sentences)} 行中 {len(hits)} 行出现在训练集 ({share:.1f}%)")
        for sample in hits[:3]:
            print(f"      {sample[:96]}")


def main() -> None:
    args = parse_args()
    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    specs = [spec for spec in SPECS if not wanted or spec.name in wanted]
    unknown = wanted - {spec.name for spec in SPECS}
    if unknown:
        print(f"未知的 bench: {sorted(unknown)}", file=sys.stderr)
        raise SystemExit(2)

    dest = Path(args.dest)
    endpoint = os.environ.get("HF_ENDPOINT", "")
    print(f"目标目录 : {dest.resolve()}")
    print(f"HF_ENDPOINT: {endpoint or '(默认 huggingface.co)'}")
    if not endpoint:
        print("  国内网络建议先设 HF_ENDPOINT=https://hf-mirror.com")
    print()

    if args.list:
        for spec in specs:
            origin = spec.url if spec.url else f"{spec.repo_id}  {spec.allow}"
            print(f"{spec.name:14} -> {spec.target}")
            print(f"{'':14}    {origin}")
            if spec.note:
                print(f"{'':14}    {spec.note}")
        return

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("缺 huggingface_hub，先 pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(2)

    failures: list[str] = []
    for spec in specs:
        print(f"[{spec.name}] {spec.repo_id}")
        if spec.note:
            print(f"    {spec.note}")
        try:
            ok, message = fetch(spec, dest, args.force)
        except Exception as error:  # noqa: BLE001
            ok, message = False, f"{type(error).__name__}: {error}"
        print(f"    {'OK  ' if ok else 'FAIL'} {message}")
        if not ok:
            failures.append(spec.name)
        print()

    if args.fingpt_train:
        print("=" * 62)
        print("金融三个 bench 的去污染自检")
        check_contamination(dest, args.fingpt_train)
        print()

    print("=" * 62)
    if failures:
        print(f"失败 {len(failures)} 个: {', '.join(failures)}")
        print("网络问题的话设 HF_ENDPOINT=https://hf-mirror.com 后重跑，已下好的会跳过。")
        raise SystemExit(1)

    print(f"全部完成，共 {len(specs)} 个。接着用小 limit 验一遍判分器读得懂这些文件：")
    print()
    print("  python -m onereplay.scripts.evaluate --metrics medqa,pubmedqa,medmcqa,careqa \\")
    print("      --limit 20 --out_dir /tmp/bench_smoke ...")


if __name__ == "__main__":
    main()
