"""Decide whether SV / FOBAR prompts are usable for C_math self-distillation.

SV and FOBAR restate the problem, hand the model the answer, and ask for an
unknown variable. The worry is that self-distillation on them yields a short
back-substitution rather than a real reasoning trace: the hard part is already
given away. If that happens, the activations collected there represent template
following, not math reasoning, and the "trajectory diversity" rationale for
including them collapses -- while they still consume a fifth of the pool.

This pilot answers that empirically and cheaply. It replicates the exact
self-distillation conditions of generate_replay_targets.py (same chat template,
greedy decode, per-batch budget capped by --max_len, and the same "no EOS means
truncated, drop the row" rule) over a small paired sample: the SAME questions
appear under every view, so difficulty is controlled and any length gap is
attributable to the view itself.

Verdict thresholds compare each view's median answer length to canonical's:
  >= 0.70   substantive trajectories, the view is usable
  0.50-0.70 shallow, shrink its share
  <  0.50   mostly echo/back-substitution, keep it out of the main pool

Run it on an interactive GPU (salloc); a few hundred rows per view take minutes.

Paths carry no cluster-specific defaults: pass --model_dir / --model_name, or
export MODEL_DIR / MODEL_NAME, so the same command works on another cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from onereplay.core.modeling import load_causal_lm_and_tokenizer, set_seed  # noqa: E402
from onereplay.data.chat import apply_prompt_template  # noqa: E402

VIEW_ORDER = ("canonical", "rephrased", "sv", "fobar")


def parse_args() -> argparse.Namespace:
    """Parse the per-view prompt files and generation settings."""

    parser = argparse.ArgumentParser(description="Pilot: compare self-distillation across views.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    # No cluster-specific default: pass --model_dir explicitly or export MODEL_DIR
    # so the same command line works unchanged on another cluster.
    parser.add_argument("--model_dir", type=str, default=os.environ.get("MODEL_DIR", ""))
    parser.add_argument(
        "--model_name", type=str, default=os.environ.get("MODEL_NAME", "Qwen3-1.7B")
    )
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument(
        "--view_dir",
        type=str,
        required=True,
        help="Directory holding <prefix>_<view>.jsonl from prepare_metamath_cmath.py.",
    )
    parser.add_argument("--prefix", type=str, default="metamath_cmath_pilot")
    parser.add_argument("--views", type=str, default=",".join(VIEW_ORDER))
    parser.add_argument("--limit", type=int, default=200, help="Rows per view.")
    # Mirror the real self-distillation budget so the pilot is faithful.
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=1280)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--min_new_tokens", type=int, default=16)
    parser.add_argument("--length_margin", type=int, default=8)
    parser.add_argument("--short_tokens", type=int, default=32)
    parser.add_argument("--num_examples", type=int, default=2, help="Generations to print per view.")
    parser.add_argument("--example_chars", type=int, default=600)
    parser.add_argument("--out_path", type=str, default="")
    return parser.parse_args()


def load_view(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read up to `limit` prompt records from one view file."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records


def generate_view(
    model,
    tokenizer,
    device,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Greedy-decode every record under the real self-distillation budget."""

    prompts = [apply_prompt_template(tokenizer, record["inputs"]) for record in records]
    prompt_lens = [
        len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]
    ]
    ceiling = args.max_len - args.min_new_tokens - args.length_margin
    eos_token_id = tokenizer.eos_token_id

    results: list[dict[str, Any]] = [{} for _ in records]
    order = sorted(
        (i for i in range(len(records)) if prompt_lens[i] <= ceiling),
        key=lambda i: prompt_lens[i],
    )
    for i in range(len(records)):
        if prompt_lens[i] > ceiling:
            results[i] = {
                "prompt_tokens": prompt_lens[i],
                "target_tokens": 0,
                "truncated": True,
                "over_long_prompt": True,
                "text": "",
            }

    for start in range(0, len(order), args.batch_size):
        chunk = order[start : start + args.batch_size]
        encoded = tokenizer(
            [prompts[i] for i in chunk],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        budget = min(
            args.max_new_tokens,
            args.max_len - max(prompt_lens[i] for i in chunk) - args.length_margin,
        )
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=budget,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
        generated = output_ids[:, encoded["input_ids"].shape[1] :]
        for i, ids in zip(chunk, generated):
            token_ids = ids.tolist()
            stopped = eos_token_id in token_ids
            answer_tokens = token_ids.index(eos_token_id) if stopped else len(token_ids)
            results[i] = {
                "prompt_tokens": prompt_lens[i],
                "target_tokens": answer_tokens,
                "truncated": not stopped,
                "over_long_prompt": False,
                "text": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
            }
        done = min(start + args.batch_size, len(order))
        print(f"  generated {done}/{len(order)}", flush=True)
    return results


def summarize(results: list[dict[str, Any]], short_tokens: int) -> dict[str, Any]:
    """Length distribution, drop rate, and answer-presence for one view."""

    import numpy as np

    n = max(len(results), 1)
    lengths = np.array([r["target_tokens"] for r in results], dtype=float)
    kept = [r for r in results if not r["truncated"]]
    kept_lengths = np.array([r["target_tokens"] for r in kept], dtype=float)
    boxed = sum("\\boxed" in r["text"] for r in kept)
    return {
        "n": len(results),
        "n_usable": len(kept),
        "frac_truncated": float(sum(r["truncated"] for r in results) / n),
        "median_tokens": float(np.median(kept_lengths)) if len(kept_lengths) else 0.0,
        "mean_tokens": float(kept_lengths.mean()) if len(kept_lengths) else 0.0,
        "p10_tokens": float(np.percentile(kept_lengths, 10)) if len(kept_lengths) else 0.0,
        "frac_short": float(np.mean(kept_lengths < short_tokens)) if len(kept_lengths) else 0.0,
        "boxed_rate": boxed / max(len(kept), 1),
        "all_median_tokens": float(np.median(lengths)) if len(lengths) else 0.0,
    }


def verdict(ratio: float) -> str:
    """Turn a median-length ratio against canonical into a recommendation."""

    if ratio >= 0.70:
        return "usable (轨迹实质)"
    if ratio >= 0.50:
        return "shallow (建议压缩份额)"
    return "echo/back-substitution (建议移出主池)"


def main() -> None:
    """Run the paired pilot and print a per-view comparison plus a verdict."""

    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    view_dir = Path(args.view_dir)
    view_names = [name.strip() for name in args.views.split(",") if name.strip()]

    available: dict[str, list[dict[str, Any]]] = {}
    for view in view_names:
        path = view_dir / f"{args.prefix}_{view}.jsonl"
        if not path.exists():
            print(f"skip {view}: missing {path}")
            continue
        available[view] = load_view(path, args.limit)

    if "canonical" not in available:
        raise SystemExit("canonical view is required as the length baseline")

    if not args.model_dir:
        raise SystemExit(
            "--model_dir is empty: pass it explicitly or `export MODEL_DIR=/path/to/checkpoints`."
        )
    model_path = Path(args.model_dir) / args.model_name
    if not model_path.is_dir():
        raise SystemExit(
            f"model not found: {model_path}\n"
            "check --model_dir / --model_name (or $MODEL_DIR / $MODEL_NAME) for this cluster."
        )

    print(f"loading base model and tokenizer from {model_path}")
    model, tokenizer = load_causal_lm_and_tokenizer(args.model_dir, args.model_name, args.use_bf16)
    model.to(device)
    model.eval()

    report: dict[str, Any] = {"settings": vars(args), "views": {}, "examples": {}}
    generations: dict[str, list[dict[str, Any]]] = {}
    for view, records in available.items():
        print(f"==== view {view}: {len(records)} rows ====", flush=True)
        results = generate_view(model, tokenizer, device, records, args)
        generations[view] = results
        report["views"][view] = summarize(results, args.short_tokens)
        report["examples"][view] = [
            {
                "prompt": records[i]["inputs"][: args.example_chars],
                "generation": results[i]["text"][: args.example_chars],
                "target_tokens": results[i]["target_tokens"],
            }
            for i in range(min(args.num_examples, len(records)))
        ]

    base_median = report["views"]["canonical"]["median_tokens"] or 1.0
    print("\n=================== per-view summary ===================")
    header = f"{'view':<11}{'n':>5}{'usable':>8}{'median':>8}{'p10':>6}{'short':>8}{'trunc':>8}{'boxed':>8}{'ratio':>7}"
    print(header)
    for view in view_names:
        if view not in report["views"]:
            continue
        stats = report["views"][view]
        ratio = stats["median_tokens"] / base_median
        stats["median_ratio_vs_canonical"] = ratio
        print(
            f"{view:<11}{stats['n']:>5}{stats['n_usable']:>8}"
            f"{stats['median_tokens']:>8.0f}{stats['p10_tokens']:>6.0f}"
            f"{stats['frac_short']:>8.3f}{stats['frac_truncated']:>8.3f}"
            f"{stats['boxed_rate']:>8.3f}{ratio:>7.2f}"
        )

    print("\n=================== verdict ===================")
    for view in ("rephrased", "sv", "fobar"):
        if view in report["views"]:
            ratio = report["views"][view]["median_ratio_vs_canonical"]
            decision = verdict(ratio)
            report["views"][view]["verdict"] = decision
            print(f"{view:<11} median ratio {ratio:.2f} -> {decision}")

    print("\n=================== sample generations ===================")
    for view in view_names:
        for example in report["examples"].get(view, []):
            print(f"\n--- [{view}] target_tokens={example['target_tokens']}")
            print(f"PROMPT: {example['prompt']}")
            print(f"GEN   : {example['generation']}")

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
