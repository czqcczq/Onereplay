"""Direct-safety generation metric (NSPO-style protocol).

The generation half only: render each harmful request through the chat
template and greedy-decode one response per prompt, under settings that are
provably identical across every model (Base and the four adapters). Judging is
deliberately separate -- AdvBench/HarmBench go to a GPT-4 judge off-cluster and
SORRY-Bench to a local fine-tuned Mistral -- so this file never assigns a
harm label or an ASR. It just writes responses.jsonl.

Batching is not optional here. The stock generate_response in
onereplay.eval.generation decodes one prompt at a time, which turns 1280
prompts into hours; the batched path below finishes in minutes on an H200.
Prompts are length-sorted into buckets so a batch is not dragged out by one
long member, and each row keeps its original index so responses.jsonl comes
out in prompt order regardless of the bucket it was decoded in.

Unlike the other metrics this one does NOT append to a shared
{metric}_summary.csv: two generation jobs run in parallel (see
pbs/52_direct_safety_gen.pbs) and a shared append would interleave and corrupt
the file. Cross-arm aggregation is analyze_direct_safety.py's job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from onereplay.eval.generation import render_user_prompt


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


def batched_generate(
    model,
    tokenizer,
    prompts: list[str],
    device,
    max_new_tokens: int,
    batch_size: int,
    log_every: int = 5,
) -> list[str]:
    """Greedy-decode one response per prompt, batched and order-preserving.

    Returns responses aligned to the input order. Internally the prompts are
    rendered through the chat template, sorted by token length so each batch is
    roughly uniform, decoded, and scattered back to their original positions.
    """

    rendered = [render_user_prompt(tokenizer, prompt) for prompt in prompts]
    lengths = [
        len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in rendered
    ]
    order = sorted(range(len(rendered)), key=lambda index: lengths[index])

    responses: list[str] = [""] * len(rendered)
    num_batches = (len(order) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(order), batch_size), start=1):
        chunk = order[start : start + batch_size]
        encoded = tokenizer(
            [rendered[index] for index in chunk],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[:, encoded["input_ids"].shape[1] :]
        for index, ids in zip(chunk, generated):
            responses[index] = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if log_every > 0 and batch_number % log_every == 0:
            print(f"direct_safety generated batch {batch_number}/{num_batches}", flush=True)
    return responses


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
