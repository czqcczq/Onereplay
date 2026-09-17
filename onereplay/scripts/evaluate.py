"""Stage 3 CLI: evaluate one model on several metrics with a single model load.

    python -m onereplay.scripts.evaluate --metrics ifeval,multiif,commonsense ...

Available metrics: ifeval, ifbench, followbench, multiif, commonsense,
commonsense_qa, gsm8k, aime, math500, amc, minervamath, humaneval, mbpp,
dialogsum, direct_safety. Each metric writes
<out_dir>/<metric>/<run_name>/summary.json plus an appended row in
<out_dir>/<metric>_summary.csv.

commonsense and commonsense_qa are not the same measurement: the first is
held-out SFT loss on the Commonsense170k pool, the second is accuracy on the
eight LLM-Adapters test sets. Part 1 concludes from the second.

dialogsum is the one metric here that scores a *new* task rather than retention,
so its numbers are expected to rise above base, not hold. Reading a lambda sweep
needs both directions: retention alone is trivially won by not learning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.core.chat_policy import configure_system_prompt  # noqa: E402
from onereplay.eval.generation import configure_decoding, describe_decoding  # noqa: E402
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
    # Decoding policy, shared by every generative metric in the run. Every
    # default here reproduces the historical plain-greedy behavior, so results
    # stay comparable with runs made before these flags existed.
    #
    # Greedy is deterministic, so a model that walks into a repeating state
    # repeats it forever: on MATH500 that showed up as >50% of rows burning the
    # full token budget on "216 is 6**3, but 216 is 6**3, but ...". The first
    # two knobs break that loop. no_repeat_ngram is the targeted one -- a
    # 40-token verbatim repeat is degeneration rather than real math -- and it
    # keeps decoding deterministic, so arms stay comparable at a single seed.
    parser.add_argument("--decode_no_repeat_ngram", type=int, default=0)
    parser.add_argument("--decode_repetition_penalty", type=float, default=0.0)
    parser.add_argument("--decode_do_sample", type=int, default=0)
    parser.add_argument("--decode_temperature", type=float, default=0.0)
    parser.add_argument("--decode_top_p", type=float, default=0.0)
    # Off by default only to protect comparability: a chat turn ends
    # '<|im_end|><|endoftext|>' but eos_token on the base tokenizer is just
    # '<|endoftext|>', so 1 is the more correct setting. Turning it on shifts
    # every generative score, so all arms have to be re-decoded together.
    parser.add_argument("--decode_stop_on_im_end", type=int, default=0)
    # Qwen2.5-Math's template injects "put your final answer within \boxed{}"
    # when no system turn is given. That is the recipe on the math line and a
    # contaminant on the IF and commonsense lines, so those pass a neutral
    # system message here. It must match the value train.py was given, or the
    # model is evaluated under a prompt it was never trained on. Empty keeps
    # the template default, which is what every pre-Qwen2.5-Math run used.
    parser.add_argument("--system_prompt", type=str, default="")

    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--map_cache_dir", type=str, default="")

    # commonsense_qa accuracy: LLM-Adapters' dataset/<task>/test.json
    parser.add_argument("--cs_eval_dir", type=str, default="")
    parser.add_argument(
        "--cs_eval_tasks",
        type=str,
        default="",
        help="Comma-separated subset of boolq,piqa,social_i_qa,hellaswag,"
        "winogrande,ARC-Easy,ARC-Challenge,openbookqa. Empty runs all eight.",
    )
    parser.add_argument(
        "--cs_scoring",
        type=str,
        choices=["logprob", "gen", "both"],
        default="both",
        help="logprob ranks 'the correct answer is <label>' over the row's own "
        "candidate set and is the reportable number; gen decodes and regex-"
        "matches, which is only meaningful alongside its parse_rate.",
    )
    parser.add_argument("--cs_limit", type=int, default=0)
    parser.add_argument("--cs_max_new_tokens", type=int, default=32)
    parser.add_argument("--cs_batch_size", type=int, default=0)
    # Smaller than the decoding batch on purpose: the scoring forward pass
    # materializes a [batch, seq, 151936] logit tensor and reuses no KV cache.
    parser.add_argument("--cs_score_batch_size", type=int, default=8)

    # dialogsum: the only new-task metric here, so its scores are meant to rise
    # relative to base rather than hold. Input is the held-out JSONL from
    # prepare_dialogsum.py, one row per dialogue with every reference.
    parser.add_argument("--dialogsum_input", type=str, default="")
    parser.add_argument("--dialogsum_limit", type=int, default=0)
    parser.add_argument("--dialogsum_max_new_tokens", type=int, default=256)
    parser.add_argument("--dialogsum_batch_size", type=int, default=0)
    # BERTScore needs a second model resident on the card and an offline node
    # cannot fetch it, so it is opt-in and the path is mandatory once it is on.
    # Layers must be passed because bert-score looks its default up by hub name.
    parser.add_argument("--dialogsum_bertscore", type=int, default=0)
    parser.add_argument("--dialogsum_bertscore_model", type=str, default="")
    parser.add_argument("--dialogsum_bertscore_layers", type=int, default=17)
    parser.add_argument("--dialogsum_bertscore_batch_size", type=int, default=64)

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
    parser.add_argument("--minervamath_data_path", type=str, default="")
    parser.add_argument("--aime_data_path", type=str, default="")
    # Decode k responses per question and report their mean accuracy
    # (average@k). Only amc and minervamath honor it: MATH-500 and GSM8K are
    # reported as greedy@1 everywhere, so they ignore it by construction.
    # Meaningless without --decode_do_sample 1, since k greedy passes are k
    # copies of the same response.
    parser.add_argument("--math_num_samples", type=int, default=1)
    # Minerva golds are measured quantities given to a stated precision, so its
    # scorer compares numbers with this relative tolerance and only falls back
    # to string equality for symbolic answers. See the metric's comments.
    parser.add_argument("--minerva_rel_tol", type=float, default=0.01)
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

    configure_system_prompt(args.system_prompt)
    configure_decoding(
        stop_on_im_end=bool(args.decode_stop_on_im_end),
        no_repeat_ngram_size=args.decode_no_repeat_ngram,
        repetition_penalty=args.decode_repetition_penalty,
        do_sample=bool(args.decode_do_sample),
        temperature=args.decode_temperature,
        top_p=args.decode_top_p,
    )
    print(f"[evaluate] decoding: {describe_decoding()}", flush=True)

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
