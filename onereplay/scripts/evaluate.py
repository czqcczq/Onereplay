"""Stage 3 CLI: evaluate one model on several metrics with a single model load.

    python -m onereplay.scripts.evaluate --metrics ifeval,multiif,commonsense ...

Available metrics: ifeval, ifbench, followbench, multiif, commonsense, gsm8k,
aime, math500, amc, humaneval, mbpp, direct_safety. Each metric writes
<out_dir>/<metric>/<run_name>/summary.json plus an appended row in
<out_dir>/<metric>_summary.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.eval.runner import run_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OneReplay evaluation metrics.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--metrics",
        type=str,
        default="commonsense,ifeval,multiif",
        help="Comma-separated metric names.",
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=768)

    # Decoding batch size. eval_batch_size is the fallback for every generative
    # metric; the per-family overrides exist because the token budgets differ by
    # an order of magnitude (math decodes 4096, code 512), so the KV cache a
    # given batch needs does too. 0 means "fall back to eval_batch_size".
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--math_batch_size", type=int, default=0)
    parser.add_argument("--code_batch_size", type=int, default=0)
    parser.add_argument("--ifeval_batch_size", type=int, default=0)
    parser.add_argument("--ifbench_batch_size", type=int, default=0)
    parser.add_argument("--followbench_batch_size", type=int, default=0)

    # commonsense loss
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--map_cache_dir", type=str, default="")

    # ifeval / ifbench / multiif
    parser.add_argument("--ifeval_input", type=str, default="")
    # Empty ifbench_input means "use IFBench_test.jsonl from the installed
    # ifbench wheel". Its constraints need a bigger budget than IFEval's.
    parser.add_argument("--ifbench_input", type=str, default="")
    parser.add_argument("--ifbench_limit", type=int, default=0)
    parser.add_argument("--ifbench_max_new_tokens", type=int, default=2048)
    # followbench: generation + the 350 rule-scored rows. The other 470 need a
    # judge and leave via judge_inputs.jsonl; see the metric's docstring.
    # Empty followbench_data means the vendored third_party/followbench/data.
    parser.add_argument("--followbench_data", type=str, default="")
    parser.add_argument("--followbench_categories", type=str, default="")
    parser.add_argument("--followbench_max_new_tokens", type=int, default=2048)
    # Subsamples whole example_id ladders per category, not rows: CSL and the
    # rule-side denominators both need a complete level-1..5 group.
    parser.add_argument("--followbench_group_limit", type=int, default=0)
    parser.add_argument("--multiif_input", type=str, default="")
    parser.add_argument("--multiif_language", type=str, default="English")
    parser.add_argument("--multiif_limit", type=int, default=0)
    parser.add_argument("--multiif_max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--math_max_new_tokens", type=int, default=1024)
    parser.add_argument("--code_max_new_tokens", type=int, default=512)

    # direct safety (generation half; judges run separately, off-cluster)
    parser.add_argument("--safety_prompts", type=str, default="")
    parser.add_argument("--safety_max_new_tokens", type=int, default=512)
    parser.add_argument("--safety_batch_size", type=int, default=128)

    # math / code probes
    parser.add_argument("--gsm8k_data_path", type=str, default="")
    parser.add_argument("--math500_data_path", type=str, default="")
    parser.add_argument("--amc_data_path", type=str, default="")
    parser.add_argument("--aime_data_path", type=str, default="")
    parser.add_argument("--question_field", type=str, default="")
    parser.add_argument("--answer_field", type=str, default="")
    parser.add_argument("--humaneval_data_file", type=str, default="")
    parser.add_argument("--mbpp_dataset_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="google-research-datasets/mbpp")
    parser.add_argument("--dataset_config", type=str, default="full")
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]

    metric_cfg = {key: value for key, value in vars(args).items() if value != ""}
    for consumed in ("metrics", "out_dir", "adapter_path", "run_name", "model_dir", "model_name"):
        metric_cfg.pop(consumed, None)

    run_evaluation(
        model_dir=args.model_dir,
        model_name=args.model_name,
        metric_names=metric_names,
        out_dir=args.out_dir,
        adapter_path=args.adapter_path,
        run_name=args.run_name,
        use_bf16=args.use_bf16,
        seed=args.seed,
        gpu=args.gpu,
        metric_cfg=metric_cfg,
    )


if __name__ == "__main__":
    main()
