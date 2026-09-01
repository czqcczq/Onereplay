"""Check that the gold-target ablation changes the answers and nothing else.

The suspicion under test in the experiment this supports is that the base
model's self-distilled answers on the protection corpus are often wrong, so C,
F and replay are all anchoring the model to bad answers. Swapping in the
corpus's own answers is the way to find out -- but only if the swap is clean.
The self-distilled file carries gold_targets on *every* row, including the
truncated ones whose self-distilled target is empty, so the naive swap silently
hands the gold arm ~13% more rows and the result would confound target quality
with pool size.

This asserts the swap is clean on both sides:

  cov/fisher  --target_column gold_targets --require_target_column targets
              keeps exactly the rows --target_column targets keeps
  replay      --replay_target_column gold_targets keeps exactly the rows the
              self-distilled path keeps, in the same order, with only `output`
              differing

and that dropping --require_target_column is in fact unsafe, so the flag is
load-bearing rather than decorative.

    python test_code/smoke_gold_target_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onereplay.data.old_knowledge import (  # noqa: E402
    filter_incomplete_rows,
    load_old_knowledge_dataset,
)
from onereplay.data.replay import (  # noqa: E402
    load_self_distilled_pool,
    replay_target_flavor,
    to_sft_schema,
)

# Row 2 and 5 hit the generation cap: empty self-distilled target, truncated
# flag set, gold still present. That asymmetry is the whole point of the test.
ROWS = [
    {
        "index": i,
        "inputs": f"question {i}",
        "targets": "" if i in (2, 5) else f"self-distilled answer {i}",
        "gold_targets": f"gold answer {i}",
        "truncated": i in (2, 5),
        "prompt_tokens": 10,
        "target_tokens": 0 if i in (2, 5) else 20,
    }
    for i in range(8)
]
EXPECTED_KEPT = {f"question {i}" for i in range(8) if i not in (2, 5)}


def cov_args(corpus: Path, cache: str, target_column: str, require_column: str):
    return argparse.Namespace(
        dataset_path="",
        dataset_name="",
        dataset_config="",
        dataset_split="train",
        data_files=str(corpus),
        cache_dir=cache,
        streaming=0,
        text_column="",
        input_column="inputs",
        target_column=target_column,
        require_target_column=require_column,
        require_target=1,
    )


def cov_rows(corpus: Path, cache: str, target_column: str, require_column: str):
    """The (prompt, answer) pairs collect_cov would forward through the model."""

    args = cov_args(corpus, cache, target_column, require_column)
    dataset = filter_incomplete_rows(load_old_knowledge_dataset(args), args)
    return [(row["inputs"], row[target_column]) for row in dataset]


def replay_rows(corpus: Path, cache: str, target_column: str):
    """The (instruction, output) pairs the replay loader would train on."""

    args = argparse.Namespace(
        replay_self_distill_file=str(corpus),
        replay_cache_dir=cache,
        replay_drop_truncated=1,
        replay_input_column="inputs",
        replay_target_column=target_column,
    )
    dataset = to_sft_schema(load_self_distilled_pool(args), args)
    return [(row["instruction"], row["output"]) for row in dataset]


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        cache = str(Path(tmp) / "cache")
        corpus = Path(tmp) / "selfdistill.jsonl"
        with corpus.open("w", encoding="utf-8") as file:
            for row in ROWS:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        # -- cov / fisher side ------------------------------------------------
        sd = cov_rows(corpus, cache, "targets", "")
        gold = cov_rows(corpus, cache, "gold_targets", "targets")
        naive = cov_rows(corpus, cache, "gold_targets", "")

        if [p for p, _ in sd] != [p for p, _ in gold]:
            failures.append(f"cov: 行集变了\n  自蒸馏 {[p for p, _ in sd]}\n  gold {[p for p, _ in gold]}")
        if {p for p, _ in sd} != EXPECTED_KEPT:
            failures.append(f"cov: 自蒸馏行集不是预期的 {EXPECTED_KEPT}，实际 {[p for p, _ in sd]}")
        if [a for _, a in gold] != [f"gold answer {i}" for i in range(8) if i not in (2, 5)]:
            failures.append(f"cov: gold 分支没有读到 gold 文本，实际 {[a for _, a in gold]}")
        if len(naive) != len(ROWS):
            failures.append(
                "cov: 少了 require_target_column 本应多留下截断行，说明这个测试没在测真正的风险"
            )
        else:
            print(
                f"[ok] cov: 不加 require_target_column 会从 {len(sd)} 行涨到 {len(naive)} 行，"
                "加上之后行集与自蒸馏完全一致"
            )

        # -- replay side ------------------------------------------------------
        sd_replay = replay_rows(corpus, cache, "targets")
        gold_replay = replay_rows(corpus, cache, "gold_targets")

        if [p for p, _ in sd_replay] != [p for p, _ in gold_replay]:
            failures.append("replay: 两种 target 下的行集/顺序不一致")
        if [a for _, a in sd_replay] == [a for _, a in gold_replay]:
            failures.append("replay: 换了 target_column 但答案没变，说明参数没生效")
        if {p for p, _ in sd_replay} != EXPECTED_KEPT:
            failures.append(f"replay: 行集不是预期的 {EXPECTED_KEPT}，实际 {[p for p, _ in sd_replay]}")
        if not failures:
            print(f"[ok] replay: {len(sd_replay)} 行同序，仅 output 从自蒸馏换成 gold")

        # -- cov and replay must agree ----------------------------------------
        if [p for p, _ in gold] != [p for p, _ in gold_replay]:
            failures.append("gold 分支下 cov 和 replay 的行集不一致，C 会覆盖 replay 没训的行")
        else:
            print("[ok] gold 分支下 cov 与 replay 的行集一致")

        # -- run log ----------------------------------------------------------
        # The multi-domain arms pass --replay_mix_files instead of the single-file
        # flag, so the label has to follow that too. It used to check only the
        # single-file flag, which labelled every mixed self-distilled arm
        # "gold-target" -- the exact opposite of what it trained on.
        mix_spec = f"if={corpus},code={corpus}"
        flavors = {
            "self-distilled": replay_target_flavor(
                argparse.Namespace(
                    replay_self_distill_file=str(corpus),
                    replay_mix_files="",
                    replay_target_column="targets",
                )
            ),
            "gold ablation": replay_target_flavor(
                argparse.Namespace(
                    replay_self_distill_file=str(corpus),
                    replay_mix_files="",
                    replay_target_column="gold_targets",
                )
            ),
            "raw gold pool": replay_target_flavor(
                argparse.Namespace(
                    replay_self_distill_file="",
                    replay_mix_files="",
                    replay_target_column="targets",
                )
            ),
            "mixed self-distilled": replay_target_flavor(
                argparse.Namespace(
                    replay_self_distill_file="",
                    replay_mix_files=mix_spec,
                    replay_target_column="targets",
                )
            ),
            "mixed gold ablation": replay_target_flavor(
                argparse.Namespace(
                    replay_self_distill_file="",
                    replay_mix_files=mix_spec,
                    replay_target_column="gold_targets",
                )
            ),
        }
        expected = {
            "self-distilled": "self-distilled",
            "mixed self-distilled": "self-distilled",
        }
        for key, want in expected.items():
            if flavors[key] != want:
                failures.append(f"flavor[{key}] 应为 {want!r}，实得 {flavors[key]!r}")
        for key in ("gold ablation", "mixed gold ablation"):
            if "gold" not in flavors[key]:
                failures.append(f"flavor[{key}] 里没有 gold 字样：{flavors[key]!r}")
        if not failures:
            for key, value in flavors.items():
                print(f"[ok] flavor {key:>21} -> {value}")

    if failures:
        print("\n失败:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\n全部通过：gold 消融只换答案，不动行集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
