"""Direct-safety generation metric (NSPO-style protocol).

The generation half only: render each harmful request through the chat
template and greedy-decode one response per prompt, under settings that are
provably identical across every model (Base and the four adapters). Judging is
deliberately separate -- AdvBench/HarmBench go to a GPT-4 judge off-cluster and
SORRY-Bench to a local fine-tuned Mistral -- so this file never assigns a
harm label or an ASR. It just writes responses.jsonl.

Batching is not optional here. The batched decoder this metric introduced now
lives in onereplay.eval.generation and every metric uses it.

Unlike the other metrics this one does NOT append to a shared
{metric}_summary.csv: two generation jobs run in parallel (see
pbs/52_direct_safety_gen.pbs) and a shared append would interleave and corrupt
the file. Cross-arm aggregation is analyze_direct_safety.py's job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from onereplay.eval.generation import batched_generate


def load_prompts(path: str, limit: int = 0) -> list[dict[str, Any]]:
    """Read the normalized {bench, id, prompt, meta} rows."""

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


class DirectSafetyMetric:
    name = "direct_safety"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        prompts_path = cfg.get("safety_prompts", "")
        if not prompts_path:
            raise ValueError(
                "direct_safety needs --safety_prompts pointing at the JSONL from "
                "prepare_direct_safety_data.py."
            )
        max_new_tokens = int(cfg.get("safety_max_new_tokens", cfg.get("max_new_tokens", 512)))
        batch_size = int(cfg.get("safety_batch_size", 128))
        limit = int(cfg.get("limit", 0))
        run_name = cfg.get("run_name", "base")

        response_path = output_dir / "responses.jsonl"
        # Whole-file resume: a finished responses.jsonl with the right row count
        # is trusted, so re-submitting the job skips arms that already ran.
        rows = load_prompts(prompts_path, limit)
        if response_path.exists():
            done = sum(1 for _ in response_path.open(encoding="utf-8"))
            if done == len(rows):
                print(f"direct_safety: {response_path} already has {done} rows, skipping")
                return self._summarize(rows, run_name, cfg, output_dir, skipped=True)

        responses = batched_generate(
            model,
            tokenizer,
            [row["prompt"] for row in rows],
            device,
            max_new_tokens,
            batch_size,
            log_label=self.name,
        )

        with response_path.open("w", encoding="utf-8") as file:
            for row, response in zip(rows, responses):
                file.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "bench": row["bench"],
                            "prompt": row["prompt"],
                            "response": response,
                            "meta": row.get("meta", {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return self._summarize(rows, run_name, cfg, output_dir, skipped=False)

    def _summarize(
        self,
        rows: list[dict[str, Any]],
        run_name: str,
        cfg: dict[str, Any],
        output_dir: Path,
        skipped: bool,
    ) -> dict[str, Any]:
        per_bench: dict[str, int] = {}
        for row in rows:
            per_bench[row["bench"]] = per_bench.get(row["bench"], 0) + 1
        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "num_prompts": len(rows),
            "per_bench": per_bench,
            "max_new_tokens": int(cfg.get("safety_max_new_tokens", cfg.get("max_new_tokens", 512))),
            "batch_size": int(cfg.get("safety_batch_size", 128)),
            "skipped_existing": skipped,
            "output_dir": str(output_dir),
            "note": "generation only; harm labels come from the per-bench judges",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
