"""Download the nine evaluation benchmarks from HuggingFace, once, into place.

    python test_code/download_bench.py                  # 全部
    python test_code/download_bench.py --only medqa,finqa
    python test_code/download_bench.py --dest /scratch/.../datasets

Everything lands under --dest with exactly the paths 90_specialist_sft.pbs
expects, so after this finishes the PBS defaults just work:

    datasets/math/gsm8k_test.jsonl
    datasets/math/math500_test.jsonl
    datasets/math/minervamath_test.jsonl
    datasets/bench/medqa/test.jsonl
    datasets/bench/pubmedqa/test.jsonl
    datasets/bench/medxpertqa/test.jsonl
    datasets/bench/finqa/test.jsonl
    datasets/bench/convfinqa/dev_turn.json
    datasets/bench/tatqa/tatqa_dataset_dev.json

The math three go to datasets/math/<name>_test.jsonl rather than alongside the
rest because that is where this project has always kept them, and gsm8k and
math500 are already sitting there on the cluster. Anything already present is
skipped, so running this on the cluster fetches only what is genuinely missing.

Why every entry pins allow_patterns
-----------------------------------
snapshot_download takes the whole repo by default, and these repos are mostly
not the eval split:

    OctoMed/MedQA-5options   train is ~500MB (16 teacher responses per question),
                             test is 0.7MB
    TsinghuaC3I/MedXpertQA   images.zip is 517MB and belongs to the MM subset,
                             which is not evaluated here
    AdaptLLM/ConvFinQA       train_turn.json is 166MB, dev_turn.json is 21MB
    qiaojin/PubMedQA         pqa_artificial is 211k generated rows; only the
                             1000 expert-labeled ones are the benchmark

Unpinned, this script would pull well over a gigabyte of data that never gets
evaluated. Pinned, the whole set is roughly 45MB.

Behind the Great Firewall set a mirror before running:

    $env:HF_ENDPOINT = "https://hf-mirror.com"     # PowerShell
    export HF_ENDPOINT=https://hf-mirror.com       # bash
"""

from __future__ import annotations

import argparse
import json
import os
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
    # "parquet" converts to jsonl, "copy" keeps the bytes, "auto" decides by
    # extension.
    mode: str = "parquet"
    note: str = ""


SPECS: list[Spec] = [
    # --- math ---------------------------------------------------------------
    Spec(
        name="gsm8k",
        repo_id="openai/gsm8k",
        allow=["main/test-*"],
        source="main/test-*.parquet",
        target="math/gsm8k_test.jsonl",
        note="1319 grade-school word problems",
    ),
    Spec(
        name="math500",
        repo_id="HuggingFaceH4/MATH-500",
        allow=["test.jsonl", "*.parquet"],
        source="test*",
        target="math/math500_test.jsonl",
        mode="auto",
        note="500 problems from the MATH test split",
    ),
    Spec(
        name="minervamath",
        repo_id="math-ai/minervamath",
        allow=["*.parquet", "*.jsonl"],
        source="*test*",
        target="math/minervamath_test.jsonl",
        mode="auto",
        note="272 MIT OpenCourseWare problems",
    ),
    # --- medical ------------------------------------------------------------
    Spec(
        name="medqa",
        repo_id="OctoMed/MedQA-5options",
        # test only: train carries 16 teacher responses per question (~500MB)
        # and is what the Medical specialist is fine-tuned on.
        allow=["data/test-*"],
        source="data/test-*.parquet",
        target="bench/medqa/test.jsonl",
        note="in-domain: same corpus the Medical specialist trains on",
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
        name="medxpertqa",
        repo_id="TsinghuaC3I/MedXpertQA",
        # Text only. images.zip (517MB) serves the MM subset, which this line
        # does not evaluate.
        allow=["Text/test.jsonl"],
        source="Text/test.jsonl",
        target="bench/medxpertqa/test.jsonl",
        mode="copy",
        note="difficulty ruler only -- 10 options, chance is 10%",
    ),
    # --- finance ------------------------------------------------------------
    Spec(
        name="finqa",
        # Straight from the authors' repo, not HuggingFace. dreamerdeo/finqa is
        # only a loading script with no data files, and the mirrors that do
        # carry data (flare-finqa, finqa-updated) flatten the qa block and drop
        # exe_ans -- the executed numeric answer this metric grades against.
        url="https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
        target="bench/finqa/test.jsonl",
        note="1147 questions, original format with qa.exe_ans",
    ),
    Spec(
        name="convfinqa",
        repo_id="AdaptLLM/ConvFinQA",
        # The authors' own turn-level split. train_turn.json is 166MB.
        allow=["dev_turn.json"],
        source="dev_turn.json",
        target="bench/convfinqa/dev_turn.json",
        mode="copy",
        note="1490 turns over 421 conversations",
    ),
    Spec(
        name="tatqa",
        repo_id="next-tat/TAT-QA",
        # dev, not test: the official test split is blind (leaderboard only).
        allow=["tatqa_dataset_dev.json"],
        source="tatqa_dataset_dev.json",
        target="bench/tatqa/tatqa_dataset_dev.json",
        mode="copy",
        note="dev split; the official test split has no public answers",
    ),
    # --- code ---------------------------------------------------------------
    # These two land under datasets/code/ rather than datasets/bench/, because
    # download_code_data.py has been putting them there since well before this
    # line existed and other scripts still read that path. Kept here anyway so
    # one command fetches everything the specialist runs evaluate against.
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

    parquet = [path for path in matches if path.suffix.lower() == ".parquet"]
    if parquet or spec.mode == "parquet":
        if not parquet:
            return False, f"没有 parquet 匹配 {spec.source!r}，实际匹配到: {[p.name for p in matches]}"
        rows = parquet_to_jsonl(parquet, target)
    else:
        rows = jsonish_to_jsonl(matches[0], target)
    return True, f"{rows} 行 -> {spec.target}"


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
            print(f"{spec.name:12} -> {spec.target}")
            print(f"{'':12}    {origin}")
            if spec.note:
                print(f"{'':12}    {spec.note}")
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

    print("=" * 62)
    if failures:
        print(f"失败 {len(failures)} 个: {', '.join(failures)}")
        print("网络问题的话设 HF_ENDPOINT=https://hf-mirror.com 后重跑，已下好的会跳过。")
        raise SystemExit(1)

    print(f"全部完成，共 {len(specs)} 个。接着验一遍 loader 能不能读懂这些文件：")
    print()
    print("  python test_code/check_bench_loaders.py")


if __name__ == "__main__":
    main()
