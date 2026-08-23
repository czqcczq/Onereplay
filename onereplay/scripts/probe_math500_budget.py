"""Measure the censored tail of MATH-500 generations to pick MATH_MAX_NEW_TOKENS.

A finished eval only tells you *that* responses hit the cap, not how long they
wanted to be: everything piles up on the cap and the tail is censored, so the
percentiles cannot say whether 2560 is enough or 4096 still is not.

This re-runs only the truncated items under a much larger cap and reports where
they actually terminate. Items that hit the probe cap too are non-terminating
loops -- no realistic budget saves those, so the recommended budget is derived
from the ones that do finish.

Prompt and decoding come from the metric itself, so what is measured here is
exactly what a real evaluation would produce.

  python -m onereplay.scripts.probe_math500_budget \
    --model_dir .../models --model_name Qwen3-1.7B \
    --responses .../results/math500/base/responses.jsonl \
    --cap 1792 --probe_max_new_tokens 4096 --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.eval.generation import generate_response  # noqa: E402
from onereplay.eval.metrics.math500 import build_prompt, extract_answer, is_equiv  # noqa: E402
from onereplay.eval.runner import load_eval_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the MATH-500 generation budget.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--responses", type=str, required=True, help="A finished responses.jsonl.")
    parser.add_argument("--cap", type=int, default=1792, help="Cap the responses were made with.")
    parser.add_argument("--probe_max_new_tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="Probe at most N items (0 = all).")
    parser.add_argument("--out_path", type=str, default="", help="Optional jsonl dump of probes.")
    return parser.parse_args()


def pct(values: list[int], q: float) -> int:
    """Percentile of an already-sorted list."""

    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def main() -> None:
    import torch

    args = parse_args()
    response_path = Path(args.responses)
    rows = [json.loads(line) for line in response_path.open(encoding="utf-8") if line.strip()]
    print(f"loaded {len(rows)} rows from {response_path}")

    model, tokenizer = load_eval_model(
        args.model_dir, args.model_name, use_bf16=args.use_bf16, adapter_path=args.adapter_path
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def token_len(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    truncated = [row for row in rows if token_len(row.get("response", "")) >= args.cap - 8]
    print(f"truncated at cap={args.cap}: {len(truncated)}/{len(rows)} ({len(truncated)/max(len(rows),1):.1%})")
    if args.limit > 0:
        truncated = truncated[: args.limit]
    if not truncated:
        raise SystemExit("没有被截断的样本，当前预算已经够用")

    probe_cap = args.probe_max_new_tokens
    print(f"probing {len(truncated)} items at max_new_tokens={probe_cap}\n")

    finished: list[int] = []
    still_capped = 0
    now_answered = 0
    now_correct = 0
    dump = []
    started = time.time()

    for idx, row in enumerate(truncated, start=1):
        response = generate_response(
            model, tokenizer, build_prompt(row["question"]), device, probe_cap
        )
        length = token_len(response)
        capped = length >= probe_cap - 8
        prediction = extract_answer(response)
        correct = bool(prediction) and is_equiv(prediction, row.get("gold", ""))
        still_capped += capped
        now_answered += prediction is not None
        now_correct += correct
        if not capped:
            finished.append(length)
        dump.append(
            {
                "question": row["question"],
                "gold": row.get("gold", ""),
                "probe_tokens": length,
                "hit_probe_cap": capped,
                "prediction": prediction,
                "correct": correct,
                "response": response,
            }
        )
        flag = "CAPPED" if capped else f"{length:>5}"
        print(f"  [{idx}/{len(truncated)}] {flag}  answered={prediction is not None}  correct={correct}")

    elapsed = time.time() - started
    print(f"\n==== probe summary ({elapsed/60:.1f} min) ====")
    print(f"  probed          : {len(truncated)} (all were truncated at {args.cap})")
    print(f"  仍顶到 {probe_cap}  : {still_capped} ({still_capped/len(truncated):.1%})  <- 不收敛，加预算也救不回")
    print(f"  正常终止        : {len(finished)} ({len(finished)/len(truncated):.1%})")
    print(f"  放开后能给出答案: {now_answered}  其中判对: {now_correct}")

    if finished:
        finished.sort()
        print(
            f"  终止长度        : P50={pct(finished,50)} P90={pct(finished,90)} "
            f"P95={pct(finished,95)} P99={pct(finished,99)} max={finished[-1]}"
        )
        suggested = int(pct(finished, 95) * 1.1 / 128 + 1) * 128
        print(
            f"\n建议 MATH_MAX_NEW_TOKENS >= {suggested}"
            f"（终止样本 P95 × 1.1，向上取整到 128 的倍数）"
        )
        print(
            "若 '仍顶到' 的比例很高，说明剩下的是不收敛循环，"
            "把预算定在这个建议值即可，再往上加只会烧卡时。"
        )
    else:
        print("\n所有探针样本都顶到了探针上限：这些是不收敛生成，加预算无用。")
        print("维持现有预算，把这部分记为已知的判分损耗。")

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as file:
            for record in dump:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n逐条探针结果: {out_path}")


if __name__ == "__main__":
    main()
