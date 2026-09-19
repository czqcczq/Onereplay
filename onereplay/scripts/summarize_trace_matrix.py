"""Assemble TRACE's forgetting matrix from the per-stage summary.json files.

97_trace_continual.pbs evaluates, after stage i, every task it has seen so far,
which fills the upper triangle of an 8x8 matrix:

    rows    = training stage (which tasks the model has been through)
    columns = task being scored
    R[i][j] = score on task j after training through stage i, defined for j <= i

Three readings come out of it, and they answer different questions:

  diagonal  R[j][j]   what task j looked like the moment it finished training.
                      This is the reference for forgetting -- not the base model,
                      which never learned the task at all.
  last row  R[n][j]   what survived to the end.
  BWT       mean over j < n of (R[n][j] - R[j][j]). TRACE's backward transfer:
                      negative means forgetting, and its magnitude is the headline
                      number for "does sequential SFT on this benchmark forget".

The base row is printed too but excluded from BWT. It is the answer to "how much
of this task could the model already do", which is what tells you whether a high
diagonal means the task was learned or was easy all along -- a distinction BWT
cannot make.

Each task has its own primary metric (accuracy, ROUGE-L, fuzzy similarity, SARI),
so the columns are not on a comparable scale and the matrix is not meant to be
read across a row. The metric classes record which field is primary under
``primary_metric``, and that is what is read here rather than a guess by name.

    python -m onereplay.scripts.summarize_trace_matrix \
        --eval_root results_trace_eval_Llama-3.2-3B-Instruct --config fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.scripts.prepare_trace import TASK_SPECS  # noqa: E402

# Fallback order when a summary predates the primary_metric field. Deliberately
# ordered from most to least specific so a generation metric does not get read as
# an accuracy it happens to also carry.
SCORE_KEY_ORDER = ("sari", "similarity", "accuracy", "rouge_l", "pass_at_1")

# Primary field per general benchmark, same convention 94/95 use.
GENERAL_SCORE_KEYS = ("accuracy", "pass_at_1", "strict_prompt_accuracy")
GENERAL_SCORE_OVERRIDE = {"ifbench": "loose_prompt_accuracy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the TRACE forgetting matrix.")
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--config", type=str, default="fast")
    parser.add_argument("--base_run", type=str, default="base")
    parser.add_argument("--num_stages", type=int, default=len(TASK_SPECS))
    parser.add_argument(
        "--general_metrics",
        type=str,
        default="ifeval,gsm8k,math500,humaneval,mbpp,ifbench",
    )
    parser.add_argument("--json_out", type=str, default="")
    return parser.parse_args()


def read_summary(eval_root: Path, metric: str, run: str) -> dict[str, Any] | None:
    path = eval_root / metric / run / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def primary_score(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    key = payload.get("primary_metric")
    if key and key in payload:
        value = payload[key]
        return float(value) if isinstance(value, (int, float)) else None
    for candidate in SCORE_KEY_ORDER:
        if candidate in payload and isinstance(payload[candidate], (int, float)):
            return float(payload[candidate])
    return None


def general_score(payload: dict[str, Any] | None, metric: str) -> float | None:
    if not payload:
        return None
    override = GENERAL_SCORE_OVERRIDE.get(metric)
    if override:
        value = payload.get(override)
        return float(value) if isinstance(value, (int, float)) else None
    for candidate in GENERAL_SCORE_KEYS:
        if candidate in payload and isinstance(payload[candidate], (int, float)):
            return float(payload[candidate])
    return None


def cell(value: float | None, width: int = 9, spec: str = ".4f") -> str:
    return f"{'-' if value is None else format(value, spec):>{width}}"


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    specs = list(TASK_SPECS)[: args.num_stages]
    stages = list(range(1, args.num_stages + 1))

    # matrix[stage][task_key] and a parallel copy of the strict-accuracy column,
    # which is what distinguishes real forgetting from a format change.
    matrix: dict[int, dict[str, float | None]] = {}
    strict: dict[int, dict[str, float | None]] = {}
    base_row: dict[str, float | None] = {}
    base_strict: dict[str, float | None] = {}

    for spec in specs:
        metric = f"trace_{spec.key}"
        payload = read_summary(eval_root, metric, args.base_run)
        base_row[spec.key] = primary_score(payload)
        base_strict[spec.key] = (payload or {}).get("accuracy_strict")

    for stage in stages:
        run = f"trace_{args.config}_stage{stage}"
        matrix[stage] = {}
        strict[stage] = {}
        for spec in specs:
            payload = read_summary(eval_root, f"trace_{spec.key}", run)
            matrix[stage][spec.key] = primary_score(payload)
            strict[stage][spec.key] = (payload or {}).get("accuracy_strict")

    header_width = 12
    print(f"eval_root = {eval_root}")
    print(f"config    = {args.config}")
    print()
    print("主指标（每列一个任务，每列的指标不同，不要横向比）")
    print(
        f"{'stage':<{header_width}}"
        + "".join(f"{spec.key[:9]:>10}" for spec in specs)
    )
    print("-" * (header_width + 10 * len(specs)))
    print(f"{'base':<{header_width}}" + "".join(cell(base_row[s.key], 10) for s in specs))
    for stage in stages:
        label = f"after {stage}"
        print(
            f"{label:<{header_width}}"
            + "".join(cell(matrix[stage][spec.key], 10) for spec in specs)
        )

    print()
    print("每个任务的指标：", ", ".join(f"{s.key}={s.directory}" for s in specs))

    # ---- learned vs final ----
    print()
    print("学会了多少 / 最后剩多少")
    print(
        f"{'task':<14}{'base':>10}{'learned':>10}{'final':>10}"
        f"{'forget(pp)':>12}{'retained':>10}"
    )
    print("-" * 66)
    forgetting: list[float] = []
    per_task: list[dict[str, Any]] = []
    last_stage = stages[-1] if stages else 0
    for index, spec in enumerate(specs, start=1):
        learned = matrix.get(index, {}).get(spec.key)
        final = matrix.get(last_stage, {}).get(spec.key)
        base_value = base_row.get(spec.key)
        drop = (final - learned) * 100 if (learned is not None and final is not None) else None
        # Retained is relative to what the task looked like when it finished
        # training, not to the base model: a task the base could already do would
        # otherwise look "retained" while having been overwritten.
        retained = (final / learned) if (learned and final is not None) else None
        if drop is not None and index != last_stage:
            forgetting.append(drop)
        print(
            f"{spec.key:<14}"
            + cell(base_value, 10)
            + cell(learned, 10)
            + cell(final, 10)
            + cell(drop, 12, "+.2f")
            + cell(retained, 10)
        )
        per_task.append(
            {
                "task": spec.directory,
                "key": spec.key,
                "stage": index,
                "base": base_value,
                "learned": learned,
                "final": final,
                "forget_pp": drop,
                "retained": retained,
            }
        )

    backward_transfer = sum(forgetting) / len(forgetting) if forgetting else None
    print()
    if backward_transfer is None:
        print("BWT = -（缺格子：前面的阶段还没评，或 run 名不匹配）")
    else:
        print(
            f"BWT = {backward_transfer:+.2f} pp"
            f"（前 {len(forgetting)} 个任务 final - learned 的均值；负 = 遗忘）"
        )

    # ---- strict vs lenient, the format-collapse check ----
    strict_rows = [
        (spec.key, strict.get(index, {}).get(spec.key), strict.get(last_stage, {}).get(spec.key))
        for index, spec in enumerate(specs, start=1)
        if strict.get(index, {}).get(spec.key) is not None
    ]
    if strict_rows:
        print()
        print("严格 EM 的同一张表（只有这一列掉 = 不再输出裸标签，不是真忘了）")
        print(f"{'task':<14}{'learned':>10}{'final':>10}{'forget(pp)':>12}")
        print("-" * 46)
        for key, learned, final in strict_rows:
            drop = (final - learned) * 100 if (learned is not None and final is not None) else None
            print(f"{key:<14}" + cell(learned, 10) + cell(final, 10) + cell(drop, 12, "+.2f"))

    # ---- general ability ----
    general = [name.strip() for name in args.general_metrics.split(",") if name.strip()]
    general_rows: list[dict[str, Any]] = []
    if general:
        final_run = f"trace_{args.config}_stage{last_stage}"
        print()
        print(f"通用能力：base vs {final_run}（方向：别掉。掉得多 = 这套训练够狠）")
        print(f"{'metric':<14}{'base':>10}{'final':>10}{'delta(pp)':>12}{'relative':>10}")
        print("-" * 56)
        for metric in general:
            base_value = general_score(read_summary(eval_root, metric, args.base_run), metric)
            final_value = general_score(read_summary(eval_root, metric, final_run), metric)
            delta = (
                (final_value - base_value) * 100
                if (base_value is not None and final_value is not None)
                else None
            )
            relative = (final_value / base_value) if (base_value and final_value is not None) else None
            print(
                f"{metric:<14}"
                + cell(base_value, 10)
                + cell(final_value, 10)
                + cell(delta, 12, "+.2f")
                + cell(relative, 10)
            )
            general_rows.append(
                {
                    "metric": metric,
                    "base": base_value,
                    "final": final_value,
                    "delta_pp": delta,
                    "relative": relative,
                }
            )
        missing = [row["metric"] for row in general_rows if row["final"] is None]
        if missing:
            print()
            print(f"缺 final 的指标：{', '.join(missing)} —— 先跑 MODE=final_eval")

    if args.json_out:
        payload = {
            "eval_root": str(eval_root),
            "config": args.config,
            "num_stages": args.num_stages,
            "task_order": [spec.directory for spec in specs],
            "base": base_row,
            "base_strict": base_strict,
            "matrix": {str(stage): matrix[stage] for stage in stages},
            "matrix_strict": {str(stage): strict[stage] for stage in stages},
            "per_task": per_task,
            "backward_transfer_pp": backward_transfer,
            "general": general_rows,
        }
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"matrix -> {target}")


if __name__ == "__main__":
    main()
