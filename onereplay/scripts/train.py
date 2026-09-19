"""Stage 2 CLI: train on Commonsense170k with OneReplay regularization.

Two paradigms share the same regularizer, data pipeline, and logging:

  --paradigm sft   cross-entropy on assistant tokens (default)
  --paradigm opd   on-policy distillation against a frozen teacher

Two adaptation modes share the same penalty tr(DeltaW C DeltaW^T):

  default            LoRA adapter, DeltaW = scale * B A
  --full_finetune 1  every parameter trains, DeltaW = W - W0

Usage: python -m onereplay.scripts.train [args]
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from onereplay.core.modeling import (  # noqa: E402
    build_lora_model,
    load_causal_lm_and_tokenizer,
    print_trainable_parameters,
    set_seed,
    snapshot_reference_weights,
)
from onereplay.core.chat_policy import configure_system_prompt  # noqa: E402
from onereplay.core.regularizer import EWCRegularizer, ReplayRegularizer  # noqa: E402
from onereplay.data.chat import build_loader, build_opd_loader  # noqa: E402
from onereplay.data.commonsense import load_and_prepare_dataset  # noqa: E402
from onereplay.data.batch_mix import (  # noqa: E402
    build_batch_mixed_loader,
    build_step_mixed_loader,
)
from onereplay.data.old_val import build_old_val_loader, build_prior_val_loaders  # noqa: E402
from onereplay.data.probe import build_probe_loaders  # noqa: E402
from onereplay.data.replay import (  # noqa: E402
    build_replay_pools,
    mix_replay_into_train,
    parse_replay_mix,
    replay_max_len,
)
from onereplay.trainers.opd import OPDTrainer  # noqa: E402
from onereplay.trainers.sft import SFTTrainer  # noqa: E402

REG_DTYPES = {"fp32": torch.float32, "fp64": torch.float64, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    """Parse model, dataset, LoRA, OneReplay, paradigm, and saving settings."""

    parser = argparse.ArgumentParser(description="Commonsense170k OneReplay training (LoRA or full)")
    parser.add_argument("--paradigm", type=str, choices=["sft", "opd"], default="sft")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    # Qwen2.5-Math's template injects "put your final answer within \boxed{}"
    # when no system turn is given, so every FLAN and Commonsense row would be
    # trained under a math instruction. Those lines pass a neutral system
    # message; the math line leaves this empty and keeps the injected one,
    # which is the recipe it is reproducing. evaluate.py and the C/F collectors
    # must be given the same value, or the model is measured and regularized
    # under a prompt it was never trained on.
    parser.add_argument("--system_prompt", type=str, default="")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/weiliu1/huggingface/datasets/commonsense_170k",
    )
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument(
        "--map_cache_dir",
        type=str,
        default="",
        help="Writable directory for HuggingFace map cache files.",
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=0,
        help=(
            "Batch size for the validation loader; 0 reuses --batch_size. "
            "val_loss averages over batches, not tokens, so changing the batch "
            "size shifts it slightly. Pin this to the baseline's batch size when "
            "--batch_size differs (batch-level replay counts new-task rows only)."
        ),
    )
    parser.add_argument("--accumulation_size", type=int, default=64)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["constant", "cosine", "linear", "constant_with_warmup"],
        default="constant",
        help=(
            "LR schedule over optimizer updates. constant is the default so "
            "every run recorded before this flag existed reproduces exactly; "
            "the recipes that specify a schedule (NuminaMath full-parameter SFT "
            "at lr 5e-5) need cosine with --warmup_ratio 0.1. The horizon is "
            "--epochs worth of updates, or --max_steps worth when that caps the "
            "epoch, so an lr_sweep arm walks a complete schedule over its own "
            "truncated budget instead of stopping partway down a full one."
        ),
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of total updates spent warming up. Ignored by "
            "--lr_scheduler constant. Full-parameter fine-tuning at 5e-5 needs "
            "this: the first updates land on a model whose loss is far from its "
            "own optimum, and an unwarmed step of that size is where the loss "
            "spike comes from."
        ),
    )
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument(
        "--eval_before_train",
        type=int,
        default=0,
        help=(
            "1 scores the validation split once on the untouched starting model "
            "and logs it as record_type=baseline, epoch 0. It is the only "
            "reference that makes the epoch-1 val_loss readable: without it a "
            "number like 0.62 says nothing about whether the run learned "
            "anything. Same loader, same batch size, no regularizer, so the two "
            "are directly subtractable. Costs one pass over --max_val_samples "
            "rows. Default 0 keeps every metrics file written before this flag "
            "existed byte-identical."
        ),
    )
    # The domain the model arrived already knowing, scored on the same schedule
    # as the new task's validation split so one record answers both "did it
    # learn this" and "did it keep that". Off by default: only a continual run
    # has a second domain. See data/old_val.py for why the rows are cut from
    # behind the old-knowledge subset rather than taken from that domain's own
    # validation split.
    parser.add_argument(
        "--old_val_jsonl",
        type=str,
        default="",
        help=(
            "The old domain's corpus, in the {question, response} schema the "
            "prepare_* scripts write. Rows behind the old-knowledge subset are "
            "scored every epoch as old_val_loss, next to the new task's "
            "val_loss. Empty leaves the metrics records as they were."
        ),
    )
    parser.add_argument(
        "--old_val_pool_size",
        type=int,
        default=0,
        help=(
            "Rows the old-knowledge subset takes off the shuffled corpus, i.e. "
            "collect_cov's --max_samples and the replay arm's "
            "--replay_pool_size. The scored slice starts right after them, so "
            "this is what keeps it clear of the rows replay trains on."
        ),
    )
    parser.add_argument(
        "--old_val_sample_seed",
        type=int,
        default=-1,
        help="Shuffle seed of that subset: collect_cov's --sample_seed.",
    )
    parser.add_argument(
        "--old_val_rows",
        type=int,
        default=-1,
        help="Rows to score; negative inherits --max_val_samples.",
    )
    # The {question, response} pair the prepare_* scripts write is not what a
    # corpus reached through a replay pool is called: FLAN is {inputs, targets}.
    # Both flags also cover --prior_val_jsonl, so a chain whose domains disagree
    # on their schema has to be normalized before it gets here.
    parser.add_argument(
        "--old_val_input_column",
        type=str,
        default="question",
        help=(
            "Prompt column of --old_val_jsonl. Must be the column the replay arm "
            "reads from the same corpus (--replay_input_column), or the two arms "
            "are not looking at the same old knowledge."
        ),
    )
    parser.add_argument(
        "--old_val_target_column",
        type=str,
        default="response",
        help="Answer column of --old_val_jsonl; the replay arm's --replay_target_column.",
    )
    parser.add_argument(
        "--old_val_max_len",
        type=int,
        default=0,
        help=(
            "Token budget for the held-out rows only; 0 reuses --max_len. Set it "
            "to the domain's --replay_max_len: truncation keeps the last tokens, "
            "so an over-long row scored at the new task's budget would be a "
            "different row than the one replay trained on."
        ),
    )
    parser.add_argument(
        "--old_val_presliced",
        type=int,
        default=0,
        help=(
            "1 means --old_val_jsonl (and every --prior_val_jsonl path) already "
            "holds only held-out rows, so they are read as-is instead of being "
            "cut from behind the old-knowledge subset. Needed when the pool cut "
            "is not 'shuffle(seed) then a prefix': a corpus sampled with "
            "rng.sample or a stratified quota has no offset to point at, and "
            "load_self_distilled_pool ignores --replay_pool_size, so the split "
            "has to exist on disk for the replay arm to really skip those rows."
        ),
    )
    # A chain of three or more stages leaves domains that are neither the new
    # task nor the checkpoint's immediate predecessor. At medical -> law,
    # finance was learned two stages back and val_loss/old_val_loss say nothing
    # about it.
    parser.add_argument(
        "--prior_val_jsonl",
        type=str,
        default="",
        help=(
            "Comma-separated label=path for domains learned before the one this "
            "stage starts from. Each is cut the same way --old_val_jsonl is, so "
            "it compares with what an earlier stage recorded for that domain, "
            "and each lands in its own column old_val_loss_<label>. Empty "
            "leaves the metrics records as they were."
        ),
    )

    parser.add_argument(
        "--profile",
        type=int,
        default=0,
        help=(
            "1 adds per-phase timers (prepare_batch / task_loss / replay_reg / "
            "backward / optimizer) so the regularizer's own cost is visible. "
            "Needs cuda synchronize per phase, which slows training by a few "
            "percent, so use it for a short measurement run rather than the "
            "runs whose wall time you report. Peak memory and epoch wall time "
            "are recorded either way."
        ),
    )

    parser.add_argument(
        "--full_finetune",
        type=int,
        default=0,
        help=(
            "1 trains every parameter instead of wrapping the model in a LoRA "
            "adapter. The OneReplay penalty is unchanged; DeltaW switches from "
            "scale * B A to W - W0 against a frozen snapshot. In this mode "
            "--target_modules no longer selects what trains (everything does) "
            "but still decides which layers the penalty covers, so it must "
            "match the --target_modules that collect_cov used."
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    parser.add_argument(
        "--osft",
        type=int,
        default=0,
        help=(
            "1 runs the OSFT baseline (arXiv:2504.07097) instead of a DeltaW penalty. "
            "Every targeted 2D weight is replaced by its SVD, the top singular "
            "directions are frozen as old knowledge, and both gradients and "
            "parameters are projected into the orthogonal complement at each step. "
            "The decomposition and the projections are executed out of the authors' "
            "repository under baseline/mini_trainer; see onereplay/core/osft.py. "
            "Requires --full_finetune 1 and is mutually exclusive with the penalty "
            "arms, since it constrains DeltaW structurally rather than by a loss term."
        ),
    )
    parser.add_argument(
        "--osft_unfreeze_rank_ratio",
        type=float,
        default=-1.0,
        help=(
            "Fraction of each matrix's singular directions that train. This is "
            "upstream's user-facing knob and runs in the protection direction you "
            "would expect: 0.0 freezes every targeted matrix, 1.0 degenerates to "
            "plain full fine-tuning, and their README's example is 0.25. It is the "
            "method's only hyperparameter, so it plays the role --replay_lambda "
            "plays for OneReplay and needs its own sweep. Required with --osft 1."
        ),
    )
    parser.add_argument(
        "--osft_target_patterns",
        type=str,
        default="",
        help=(
            "Comma-separated layer-name patterns to decompose, e.g. "
            "'self_attn.q_proj,mlp.down_proj'. Empty lets upstream resolve them from "
            "the checkpoint path against its per-architecture table, which is what "
            "their own runs do. Worth setting explicitly when the model path contains "
            "a substring that collides with another architecture's key -- their "
            "lookup scans the path and 'opt' is tested before 'qwen'."
        ),
    )
    parser.add_argument(
        "--osft_upcast_dtype",
        type=str,
        choices=["fp32", "fp64"],
        default="fp32",
        help=(
            "Precision the SVD and the weight reconstruction run in. fp32 is "
            "upstream's default; the factors themselves are stored in the training "
            "dtype."
        ),
    )

    parser.add_argument(
        "--regularizer",
        type=str,
        choices=["onereplay", "ewc"],
        default="onereplay",
        help=(
            "Which weighting matrix multiplies DeltaW in the penalty. onereplay uses "
            "C = E[x x^T] from --cov_path and gives tr(DeltaW C DeltaW^T); ewc uses the "
            "diagonal empirical Fisher from --fisher_path and gives "
            "sum_ij F_ij DeltaW_ij^2. Both are estimated once on the same old-knowledge "
            "rows and held fixed, so the two runs differ only in the weighting. "
            "--replay_lambda is the coefficient either way, but the two scales are not "
            "comparable and need separate sweeps."
        ),
    )
    parser.add_argument(
        "--cov_path", type=str, default="mycode/onereplay/results/cov_flan_chat_10k_qv.pt"
    )
    parser.add_argument(
        "--fisher_path",
        type=str,
        default="",
        help=(
            "Diagonal empirical Fisher from scripts/collect_fisher.py. Required by "
            "--regularizer ewc. Must have been estimated on the same pool fingerprint "
            "as --cov_path or the two baselines are not comparable."
        ),
    )
    parser.add_argument("--replay_lambda", type=float, default=0.0)
    parser.add_argument("--normalize_replay_by_layers", type=int, default=1)
    parser.add_argument("--measure_replay_when_lambda_zero", type=int, default=1)
    parser.add_argument(
        "--reg_once_per_update",
        type=int,
        default=1,
        help=(
            "1 evaluates the penalty once per optimizer step instead of once per "
            "micro-batch. The penalty depends only on weights, which are frozen "
            "across an accumulation window, so both give the same accumulated "
            "gradient while 1 costs 1/accumulation_steps as much. 0 restores the "
            "old per-micro-batch behavior; only useful for reproducing the timing "
            "of runs made before this flag existed, or as the control arm of the "
            "equivalence check."
        ),
    )
    parser.add_argument(
        "--reg_impl",
        type=str,
        choices=["autograd", "analytic"],
        default="autograd",
        help=(
            "How dR/dW is obtained on the full fine-tuning path. autograd builds "
            "sum((DeltaW C) * DeltaW) into the loss and lets backward derive the "
            "gradient, which costs a second matmul of the same size and holds every "
            "layer's fp32 DeltaW in the graph until backward returns. analytic uses "
            "dR/dDeltaW = 2 DeltaW C, which the forward pass already computed, and "
            "writes it into .grad before optimizer.step(); same penalty, half the "
            "arithmetic, and per-layer temporaries instead of all-layers. No effect on "
            "the LoRA path, which is already rank x rank and costs ~1.6%% of a step."
        ),
    )
    parser.add_argument(
        "--reg_allow_tf32",
        type=int,
        default=1,
        help=(
            "1 lets the penalty's matmul use TF32 tensor cores (11 mantissa bits) "
            "instead of the FP32 pipeline (24 bits). C stays fp32 in memory; only the "
            "tensor core's inputs are rounded. On H100 that is ~7x throughput for a "
            "~5e-4 relative perturbation of C, an order of magnitude below the 5e-3 "
            "penalty-ratio gap between two sampling strategies of the same corpus. "
            "The flag is scoped to the penalty and restored afterwards, so the rest of "
            "the process is unaffected. Only read by --reg_impl analytic."
        ),
    )
    parser.add_argument(
        "--reg_compute_dtype",
        type=str,
        choices=["fp32", "fp64", "bf16"],
        default="fp32",
        help=(
            "Precision the penalty's matmul runs in. fp64 is the ground-truth arm of "
            "the precision check: it is ~15x slower than fp32 but exact enough to say "
            "which of the other arms is closer to the true gradient, which comparing "
            "them against each other cannot. Only read by --reg_impl analytic."
        ),
    )
    parser.add_argument(
        "--identity_cov",
        type=int,
        default=0,
        help=(
            "1 replaces each loaded C with an identity matrix, turning the "
            "penalty into plain L2 on DeltaW (||DeltaW||_F^2). Ablation control "
            "to isolate the effect of the old-knowledge covariance structure. "
            "cov_path is still required, only for layer keys and shapes."
        ),
    )

    # Vanilla replay baseline. Instead of penalizing DeltaW through C, keep the
    # old-knowledge corpus and re-train on a fraction of it. Point these at the
    # same dump and sample_seed that collect_cov used so both routes consume
    # the same old knowledge. Use with --replay_lambda 0.
    parser.add_argument(
        "--replay_ratio",
        type=float,
        default=0.0,
        help=(
            "Replay samples appended per new-task sample. 0 disables replay. "
            "0.05 adds 5%% more rows drawn from the old-knowledge corpus, so an "
            "epoch costs 1.05x a vanilla epoch."
        ),
    )
    parser.add_argument(
        "--replay_per_batch",
        type=int,
        default=0,
        help=(
            "Replay rows carved out of every micro-batch, i.e. strict batch-level "
            "mixing. 0 disables it and leaves --replay_ratio's data-level mixing "
            "in charge. --batch_size stays the total micro-batch size, so each "
            "step holds batch_size - replay_per_batch new-task rows; keeping "
            "batch_size at the baseline's value makes per-step time and peak "
            "memory directly comparable. Raise --accumulation_size to hold the "
            "new-task rows per update near the baseline (at batch_size=8, "
            "replay_per_batch=4 needs accumulation_size=128 for 64). The replay "
            "pool is cycled, so any share is reachable without enlarging it "
            "past the rows that produced C."
        ),
    )
    parser.add_argument(
        "--replay_steps_per_update",
        type=int,
        default=0,
        help=(
            "Replay micro-batches per optimizer update, i.e. step-level mixing. "
            "0 disables it. The window holds --accumulation_size / --batch_size "
            "micro-batches; this many of them are drawn entirely from the replay "
            "pool and the rest entirely from the new task, so the accumulated "
            "gradient is exactly (1 - w) * L_new + w * L_replay with w = R / K. "
            "Two things this buys over --replay_per_batch, and both matter only "
            "when the ratio is the experiment's independent variable: the ratio "
            "stops being quantized by --batch_size (at batch_size 4 the "
            "batch-level scheme can only do 25/50/75%%), and the loss weight "
            "stops depending on answer length, because each micro-batch averages "
            "over its own supervised tokens instead of sharing one average with "
            "the other corpus. Set --accumulation_size to "
            "(new_steps + R) * batch_size so the trainer's window boundary "
            "coincides with the loader's; new_steps * batch_size is then the "
            "new-task rows per update, which every arm must hold equal."
        ),
    )
    parser.add_argument("--replay_dataset_path", type=str, default="")
    parser.add_argument(
        "--replay_data_files",
        type=str,
        default="",
        help="Glob for the old-knowledge json/jsonl files, e.g. /path/flan/train/*.jsonl",
    )
    parser.add_argument("--replay_split", type=str, default="train")
    parser.add_argument("--replay_cache_dir", type=str, default="")
    parser.add_argument("--replay_input_column", type=str, default="inputs")
    parser.add_argument("--replay_target_column", type=str, default="targets")
    parser.add_argument(
        "--replay_pool_size",
        type=int,
        default=20000,
        help=(
            "Rows kept from the shuffled corpus before sampling. Match "
            "collect_cov's --max_samples so replay subsets stay nested inside "
            "the pool that produced C. 0 uses the whole corpus."
        ),
    )
    parser.add_argument(
        "--replay_sample_seed",
        type=int,
        default=1,
        help="Must match collect_cov's --sample_seed for a nested replay subset.",
    )
    parser.add_argument(
        "--replay_self_distill_file",
        type=str,
        default="",
        help=(
            "JSONL from scripts/generate_replay_targets.py. When set, replay "
            "trains on the base model's own answers instead of the corpus's "
            "gold targets, so the penalty-free baseline anchors to W0's "
            "behavior the same way the OneReplay regularizer does. The pool "
            "settings above are ignored: that file already carries the "
            "shuffled, cut and index-stamped pool."
        ),
    )
    parser.add_argument(
        "--replay_drop_truncated",
        type=int,
        default=1,
        help=(
            "1 drops self-distilled rows whose answer hit the generation "
            "budget. Those answers have no stop token, and training on them "
            "teaches the model not to end its turn."
        ),
    )
    parser.add_argument(
        "--replay_max_len",
        type=int,
        default=0,
        help=(
            "Token budget for replay rows only; 0 reuses --max_len. The new "
            "task must stay at --max_len on every arm or the comparison gains "
            "a second variable, but old-knowledge corpora have their own "
            "lengths. Truncation keeps the last max_len tokens, so an over-long "
            "row keeps its answer and loses its question, which turns a "
            "question-answer rehearsal into bare continuation. Set this to match "
            "the max_len the domain's C was collected at, otherwise replay sees "
            "strictly less old knowledge than the penalty encodes. Measure the "
            "right value rather than inheriting it: it is a property of the "
            "corpus *and* of the answer column, and MetaMath's self-distilled "
            "chains reach 2008 tokens where the very same rows' gold_targets "
            "have a P99 of 766. It also sets the memory ceiling, because the "
            "collator pads each micro-batch to its longest row, so a long tail "
            "of a few rows decides the peak."
        ),
    )
    parser.add_argument(
        "--replay_system_prompt",
        type=str,
        default="",
        help=(
            "System turn for replay rows only; empty inherits --system_prompt. "
            "Continual training needs it: the new task and the replay corpus "
            "were serialized under different system turns during their own SFT "
            "stages (code asks for a Python code block, math and medical ask "
            "for \\boxed{}), and chat_policy holds one turn per process. "
            "Without this the replay rows arrive under the new task's turn and "
            "stop being byte-identical to the rows the model actually learned, "
            "which costs this arm the one property that makes it a rehearsal. "
            "It also shifts every replay row's length, so set --replay_max_len "
            "from the same measurement."
        ),
    )
    parser.add_argument(
        "--replay_mix_files",
        type=str,
        default="",
        help=(
            "Comma-separated LABEL=PATH self-distilled corpora to draw replay "
            "rows from, e.g. 'if=/path/flan_selfdistill.jsonl,"
            "math=/path/metamath_selfdistill.jsonl'. This is the replay-side "
            "counterpart of C_mix: the OneReplay arm protects two domains "
            "through w_if * C_if + w_math * C_math, and this arm draws replay "
            "rows from both corpora instead. Batch-level mixing only "
            "(--replay_per_batch)."
        ),
    )
    parser.add_argument(
        "--replay_mix_weights",
        type=str,
        default="",
        help=(
            "Row shares matching --replay_mix_files, normalized, e.g. "
            "'0.5,0.5' or '0.8,0.2'. Rows are not the unit the loss averages "
            "over: with FLAN at 90 supervised tokens per row and MetaMath at "
            "357, equal rows put ~79%% of the replay loss on math, and 0.8/0.2 "
            "rows is what equalizes supervised tokens. The startup log prints "
            "both shares; compare them against scripts/stat_replay_pools.py."
        ),
    )

    # Retention probes. Score fixed old-knowledge sets every N micro-batches so
    # the shape of forgetting is visible during training rather than only at
    # the three epoch boundaries. Off by default: the probes cost forward
    # passes and the production runs should not pay for them.
    parser.add_argument(
        "--probe_every_updates",
        type=int,
        default=0,
        help=(
            "Optimizer updates between probe evaluations. 0 disables all "
            "probes. Counted in updates rather than micro-batches on purpose: "
            "batch-level replay halves the new-task rows per micro-batch and "
            "doubles accumulation_size to compensate, so a replay run and a "
            "vanilla run take the same 7908 updates over the same 168k rows "
            "while the replay run takes twice the micro-batches. Probing on "
            "updates puts every run's curve on the same x axis, and lands "
            "every probe on weights that are not mid-window."
        ),
    )
    parser.add_argument(
        "--probe_heldout_file",
        type=str,
        default="",
        help=(
            "Self-distilled JSONL for FLAN rows the replay pool does not "
            "contain, produced by generate_replay_targets.py with "
            "--pool_offset set past the training pool. This is the curve that "
            "answers whether the old ability survives, as opposed to whether "
            "the replay pool has been memorized."
        ),
    )
    parser.add_argument(
        "--probe_inpool_file",
        type=str,
        default="",
        help=(
            "Self-distilled JSONL to draw the in-pool probe from; empty reuses "
            "--replay_self_distill_file. Pass it explicitly on runs that do no "
            "replay, so vanilla and OneReplay get the same two FLAN curves and "
            "the comparison with the replay run is like for like."
        ),
    )
    parser.add_argument(
        "--probe_heldout_size",
        type=int,
        default=1000,
        help="Rows taken from the held-out file, evenly strided; 0 uses all of them.",
    )
    parser.add_argument(
        "--probe_inpool_size",
        type=int,
        default=1000,
        help=(
            "Rows taken from the replay pool, evenly strided; 0 uses all ~17k, "
            "which makes every probe an order of magnitude slower. Keep it "
            "equal to --probe_heldout_size: the two curves are meant to be "
            "subtracted, and unequal sample sizes would put different noise on each."
        ),
    )
    parser.add_argument(
        "--probe_old_val_size",
        type=int,
        default=1000,
        help=(
            "Rows of --old_val_jsonl's held-out slice to probe, evenly strided; "
            "0 skips that curve, and it is skipped anyway when --old_val_jsonl "
            "is unset. This is the protected-domain curve for a line with no "
            "self-distilled corpus to point --probe_heldout_file at: same rows "
            "and same rendering as the epoch-level old_val_loss, just scored on "
            "the probe schedule and token-weighted instead of batch-averaged."
        ),
    )
    parser.add_argument(
        "--probe_cs_val_size",
        type=int,
        default=1000,
        help="Rows of the new task's validation split to probe; 0 skips that curve.",
    )
    parser.add_argument(
        "--probe_batch_size",
        type=int,
        default=0,
        help="Probe loader batch size; 0 falls back to --eval_batch_size then --batch_size.",
    )

    # OPD-only settings. The teacher must share the student's tokenizer/vocab.
    parser.add_argument("--teacher_model_name", type=str, default="Qwen3-8B")
    parser.add_argument("--teacher_model_dir", type=str, default="")
    parser.add_argument("--teacher_gpu", type=int, default=-1)
    parser.add_argument("--opd_max_new_tokens", type=int, default=128)
    parser.add_argument("--opd_temperature", type=float, default=1.0)
    parser.add_argument("--opd_kl_temperature", type=float, default=1.0)

    parser.add_argument("--save", type=int, default=1)
    parser.add_argument(
        "--save_path", type=str, default="mycode/onereplay/results/adapters/commonsense_lora"
    )
    parser.add_argument("--metrics_path", type=str, default="")
    return parser.parse_args()


def build_regularizer(args: argparse.Namespace, device) -> ReplayRegularizer | EWCRegularizer | None:
    """Load the penalty matrices once unless nothing needs them.

    Both regularizers expose the same __call__(model) -> (tensor, stats), so the
    choice made here is the only place in the training path that knows which
    baseline is running.
    """

    if args.regularizer == "ewc" and not args.fisher_path:
        raise ValueError("--regularizer ewc requires --fisher_path")
    penalty_path = args.fisher_path if args.regularizer == "ewc" else args.cov_path

    should_load = args.replay_lambda > 0 or (
        args.measure_replay_when_lambda_zero == 1 and Path(penalty_path).exists()
    )
    if not should_load:
        return None

    if args.regularizer == "ewc":
        if args.identity_cov == 1:
            raise ValueError(
                "--identity_cov replaces C with an identity matrix; it is the covariance "
                "path's L2 control and has no meaning for a Fisher. Use "
                "--regularizer onereplay --identity_cov 1 for that ablation."
            )
        regularizer = EWCRegularizer.from_path(
            penalty_path,
            device=device,
            normalize_by_layers=bool(args.normalize_replay_by_layers),
        )
        print(
            f"loaded {len(regularizer.fishers)} Fisher matrices from {penalty_path} "
            f"({regularizer.memory_bytes() / 1024**3:.3f} GiB resident)"
        )
    else:
        regularizer = ReplayRegularizer.from_path(
            penalty_path,
            device=device,
            identity=args.identity_cov == 1,
            normalize_by_layers=bool(args.normalize_replay_by_layers),
            reg_impl=args.reg_impl,
            allow_tf32=bool(args.reg_allow_tf32),
            compute_dtype=REG_DTYPES[args.reg_compute_dtype],
        )
        if args.identity_cov == 1:
            print(
                "identity_cov=1: using identity C (L2 on DeltaW) for "
                f"{len(regularizer.covariances)} layers"
            )
        print(
            f"loaded {len(regularizer.covariances)} covariance matrices from {penalty_path} "
            f"({regularizer.memory_bytes() / 1024**3:.3f} GiB resident)"
        )

    if args.profile == 1 and args.replay_lambda == 0:
        print(
            "warning: replay_lambda=0 but the penalty matrices are still loaded and "
            "evaluated every step (measure_replay_when_lambda_zero=1), so this run pays "
            "the regularizer's cost. Pass --measure_replay_when_lambda_zero 0 to time a "
            "true vanilla baseline."
        )
    return regularizer


def build_osft_factory(args: argparse.Namespace):
    """A model_factory that swaps in the OSFT-decomposed class.

    Returned as a closure rather than plumbed through the loader's signature so
    that core/modeling.py stays unaware of OSFT, and so the OSFT arm still picks
    up the loader's config munging, dtype and pad-token handling unchanged.
    """

    from onereplay.core.osft import build_osft_model, parse_target_patterns

    patterns = parse_target_patterns(args.osft_target_patterns)
    upcast_dtype = {"fp32": torch.float32, "fp64": torch.float64}[args.osft_upcast_dtype]

    def factory(model_path, *, config, torch_dtype, tokenizer):
        return build_osft_model(
            model_path,
            unfreeze_rank_ratio=args.osft_unfreeze_rank_ratio,
            target_patterns=patterns,
            torch_dtype=torch_dtype,
            config=config,
            upcast_dtype=upcast_dtype,
            tokenizer=tokenizer,
        )

    return factory


def validate_osft_args(args: argparse.Namespace) -> None:
    """Refuse the combinations where an OSFT run would not be an OSFT run.

    OSFT constrains DeltaW structurally: it deletes each targeted weight and
    replaces it with SVD factors whose frozen half is never updated. That makes
    it incompatible with the two things this script otherwise does.

    LoRA, because PEFT wraps a base layer whose `weight` no longer exists.

    The DeltaW penalties, because snapshot_reference_weights reads
    module.weight off every covered Linear to freeze W0, which now raises -- and
    because stacking a second constraint on top would produce a run that is
    neither baseline. Combining them may be worth measuring one day, but it
    cannot be the arm labelled OSFT in a comparison table.
    """

    if args.osft != 1:
        if args.osft_unfreeze_rank_ratio >= 0:
            raise ValueError(
                "--osft_unfreeze_rank_ratio was set but --osft is 0, so nothing would read "
                "it and the run would silently be a vanilla one. Pass --osft 1."
            )
        return

    if args.osft_unfreeze_rank_ratio < 0:
        raise ValueError(
            "--osft 1 requires --osft_unfreeze_rank_ratio. It is the method's only "
            "hyperparameter (fraction of singular directions that train, 0.25 in "
            "upstream's README) and has no defensible default, the same way "
            "--replay_lambda has none."
        )
    if args.full_finetune != 1:
        raise ValueError(
            "--osft 1 requires --full_finetune 1. OSFT replaces each targeted weight "
            "with SVD factors and removes the original parameter, so there is no "
            "base_layer.weight for a LoRA adapter to wrap."
        )
    if args.replay_lambda > 0:
        raise ValueError(
            "--osft 1 with --replay_lambda > 0 would apply the subspace constraint and a "
            "DeltaW penalty at once, which is neither baseline. Run them as separate arms."
        )
    if args.replay_ratio > 0 or args.replay_per_batch > 0 or args.replay_steps_per_update > 0:
        raise ValueError(
            "--osft 1 with replay mixing enabled would be an OSFT+replay combination, not "
            "the OSFT baseline. Run them as separate arms."
        )
    if args.paradigm != "sft":
        raise ValueError("--osft 1 is only defined for --paradigm sft")


def main() -> None:
    args = parse_args()
    print("the file is " + str(Path(__file__).resolve()))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for attr, value in sorted(vars(args).items()):
        print(f"\t{attr.upper()}={value}")
    set_seed(args.seed)
    configure_system_prompt(args.system_prompt)
    validate_osft_args(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
        model_factory=build_osft_factory(args) if args.osft == 1 else None,
    )
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if args.full_finetune == 1:
        model.to(device)
        print_trainable_parameters(model)
    else:
        model = build_lora_model(
            model,
            target_modules,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model.to(device)
        model.print_trainable_parameters()

    osft_record: dict[str, Any] = {}
    if args.osft == 1:
        from onereplay.core.osft import describe_osft

        osft_record = describe_osft(model, args.osft_unfreeze_rank_ratio)

    # validate_osft_args already refused a penalty on this arm; skipping the load
    # entirely also keeps C off the device, since measure_replay_when_lambda_zero
    # would otherwise pull it in just to report a number no longer being applied.
    regularizer = None if args.osft == 1 else build_regularizer(args, device)
    if args.full_finetune == 1 and regularizer is not None:
        # The snapshot has to be taken before the first optimizer step, and
        # after .to(device) so DeltaW never crosses devices mid-training.
        regularizer.set_reference_weights(
            snapshot_reference_weights(model, regularizer.layer_keys())
        )
        print(
            f"full finetune: froze W0 for {len(regularizer.reference_weights)} layers "
            f"({regularizer.reference_memory_bytes() / 1024**3:.3f} GiB resident)"
        )

    print("loading and tokenizing Commonsense170k")
    train_dataset, valid_dataset = load_and_prepare_dataset(args, tokenizer)
    if args.reg_impl == "analytic":
        if args.reg_once_per_update != 1:
            raise ValueError(
                "--reg_impl analytic always injects one penalty gradient per optimizer "
                "step, so --reg_once_per_update 0 cannot be honored. Use "
                "--reg_impl autograd for the per-micro-batch control arm."
            )
        if args.full_finetune != 1:
            print(
                "warning: --reg_impl analytic only changes the full fine-tuning path. "
                "This is a LoRA run, so the rank x rank autograd shortcut is used and "
                "the flag has no effect beyond being recorded in the metrics."
            )
    schemes = [
        name
        for name, enabled in (
            ("--replay_ratio (data-level)", args.replay_ratio > 0),
            ("--replay_per_batch (batch-level)", args.replay_per_batch > 0),
            ("--replay_steps_per_update (step-level)", args.replay_steps_per_update > 0),
        )
        if enabled
    ]
    if len(schemes) > 1:
        raise ValueError(
            f"{' and '.join(schemes)} are different mixing schemes; set exactly one."
        )
    replay_mix = parse_replay_mix(args)
    if replay_mix:
        if args.replay_self_distill_file:
            raise ValueError(
                "--replay_mix_files already names every corpus; passing "
                "--replay_self_distill_file too leaves it ambiguous which one the single-pool "
                "path would use. Put the IF corpus in --replay_mix_files as well."
            )
        if args.replay_per_batch <= 0 and args.replay_steps_per_update <= 0:
            raise ValueError(
                "--replay_mix_files needs --replay_per_batch or --replay_steps_per_update: "
                "the domain ratio is spent per draw by the scheduler. Data-level mixing "
                "(--replay_ratio) would only reach the ratio in expectation, and cannot reach "
                "0.5 at all with a 17k pool."
            )
    if (args.replay_per_batch > 0 or args.replay_steps_per_update > 0) and args.paradigm == "opd":
        # OPD replaces the batch's targets with a student rollout scored by the
        # teacher. What that should mean for a replay row is undefined, so
        # refuse rather than silently distilling the replay corpus too.
        raise ValueError(
            "--replay_per_batch / --replay_steps_per_update are not defined for --paradigm opd"
        )

    if args.replay_ratio > 0:
        # Validation stays pure new-task so val_loss remains comparable with
        # the vanilla and OneReplay runs.
        train_dataset = mix_replay_into_train(args, tokenizer, train_dataset)
    loader_factory = build_opd_loader if args.paradigm == "opd" else build_loader
    if args.replay_per_batch > 0:
        new_per_batch = args.batch_size - args.replay_per_batch
        if new_per_batch <= 0:
            raise ValueError(
                f"--replay_per_batch {args.replay_per_batch} leaves {new_per_batch} new-task "
                f"rows in a --batch_size {args.batch_size} micro-batch; replay_per_batch must "
                "be smaller than batch_size"
            )
        replay_pools = build_replay_pools(args, tokenizer)
        train_loader = build_batch_mixed_loader(
            train_dataset,
            None,
            tokenizer,
            new_per_batch=new_per_batch,
            replay_per_batch=args.replay_per_batch,
            seed=args.seed,
            replay_pools=replay_pools,
        )
        print(train_loader.describe(), flush=True)
        print(
            f"accumulation: {max(args.accumulation_size // args.batch_size, 1)} steps x "
            f"{new_per_batch} new-task rows = "
            f"{max(args.accumulation_size // args.batch_size, 1) * new_per_batch} new-task rows "
            f"per update",
            flush=True,
        )
    elif args.replay_steps_per_update > 0:
        # The loader's window has to be the trainer's window, or the optimizer
        # would step in the middle of a new/replay pattern and every update
        # would see a different mix. accumulation_steps is the only thing the
        # trainer counts, so derive the new-task steps from it rather than
        # taking a second flag that could disagree with it.
        window_steps = max(args.accumulation_size // args.batch_size, 1)
        new_steps = window_steps - args.replay_steps_per_update
        if new_steps <= 0:
            raise ValueError(
                f"--replay_steps_per_update {args.replay_steps_per_update} leaves {new_steps} "
                f"new-task micro-batches in a window of {window_steps} "
                f"(--accumulation_size {args.accumulation_size} / --batch_size "
                f"{args.batch_size}). Raise --accumulation_size to "
                f"(new_steps + {args.replay_steps_per_update}) * {args.batch_size}."
            )
        if args.accumulation_size % args.batch_size != 0:
            raise ValueError(
                f"--accumulation_size {args.accumulation_size} is not a multiple of "
                f"--batch_size {args.batch_size}, so the window the loader builds and the "
                "window the trainer closes would drift apart after the first update."
            )
        replay_pools = build_replay_pools(args, tokenizer)
        train_loader = build_step_mixed_loader(
            train_dataset,
            None,
            tokenizer,
            batch_size=args.batch_size,
            new_steps_per_window=new_steps,
            replay_steps_per_window=args.replay_steps_per_update,
            seed=args.seed,
            replay_pools=replay_pools,
        )
        print(train_loader.describe(), flush=True)
        print(
            f"accumulation: {window_steps} steps/window x {args.batch_size} rows = "
            f"{args.accumulation_size} rows per update, of which "
            f"{train_loader.new_per_update} are new-task rows "
            f"({new_steps} steps) and {train_loader.replay_per_update} are replay "
            f"({args.replay_steps_per_update} steps)",
            flush=True,
        )
    else:
        train_loader = loader_factory(
            train_dataset, tokenizer, batch_size=args.batch_size, train=True
        )
    valid_loader = loader_factory(
        valid_dataset,
        tokenizer,
        batch_size=args.eval_batch_size or args.batch_size,
        train=False,
    )
    if (args.old_val_jsonl or args.prior_val_jsonl) and args.paradigm == "opd":
        # The retention number is the cross-entropy of the old domain's own
        # targets. OPD's prepare_batch throws those away and substitutes a
        # student rollout scored by the teacher, so what came back would not be
        # the quantity this claims to report.
        raise ValueError("--old_val_jsonl is not defined for --paradigm opd")
    old_val_loader = build_old_val_loader(args, tokenizer)
    prior_val_loaders = build_prior_val_loaders(args, tokenizer)

    # The trainer counts micro-batches, so convert once here. Going through
    # accumulation_steps is what makes the interval mean the same amount of
    # new-task data on every run.
    accumulation_steps = max(args.accumulation_size // args.batch_size, 1)
    args.probe_every = args.probe_every_updates * accumulation_steps
    if args.probe_every > 0:
        if args.paradigm == "opd":
            # The probes score cross-entropy against fixed targets. OPD's
            # prepare_batch throws those targets away and replaces them with a
            # student rollout the teacher scores, so the number it produced
            # would not be the retention loss the curves claim to show.
            raise ValueError("--probe_every_updates is not defined for --paradigm opd")
        print(
            f"probing every {args.probe_every_updates} updates "
            f"= {args.probe_every} micro-batches (accum_steps={accumulation_steps})"
        )
    probe_loaders = build_probe_loaders(args, tokenizer, valid_dataset)

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
    )
    if args.osft == 1:
        from onereplay.core.osft import wrap_optimizer

        # Before the scheduler is built: an LRScheduler patches optimizer.step
        # with its own call counter, and it has to sit outside the projections so
        # that one step() is still one scheduler tick.
        optimizer = wrap_optimizer(optimizer, model)

    # The schedule advances once per optimizer update, not once per micro-batch,
    # so its horizon has to be counted in the same unit the trainer steps in.
    # train_one_epoch also steps on the final partial window (step ==
    # total_steps), hence the ceiling rather than a floor: undercounting here
    # would drive the cosine past its endpoint and hand the last few updates a
    # negative or clipped LR.
    scheduler = None
    steps_per_epoch = len(train_loader)
    if args.max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    updates_per_epoch = -(-steps_per_epoch // accumulation_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = max(0, round(total_updates * args.warmup_ratio))
    if args.lr_scheduler != "constant":
        from transformers import get_scheduler

        scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_updates,
            num_training_steps=total_updates,
        )
        print(
            f"lr schedule: {args.lr_scheduler} over {total_updates} updates "
            f"({updates_per_epoch}/epoch x {args.epochs}), "
            f"warmup {warmup_updates} updates (ratio={args.warmup_ratio}), "
            f"peak lr={args.lr}",
            flush=True,
        )
    elif args.warmup_ratio > 0:
        raise ValueError(
            "--warmup_ratio > 0 has no effect with --lr_scheduler constant; "
            "pass --lr_scheduler cosine (or constant_with_warmup) as well."
        )

    # What the mixing scheme actually did, resolved once. new_per_update is the
    # quantity every arm of a comparison has to hold equal -- it is the new-task
    # data behind one optimizer step -- and each scheme reaches it differently,
    # so deriving it from the flags at read time is how a table ends up
    # comparing runs that are not comparable.
    if args.replay_steps_per_update > 0:
        new_per_update = train_loader.new_per_update
        replay_per_update = train_loader.replay_per_update
        replay_ratio_r = train_loader.replay_ratio_r
        # Exact only here. Under batch-level mixing the two corpora share one
        # token average inside the micro-batch, so their gradient shares follow
        # answer length rather than row counts; batch_mix's describe() prints
        # the measured token share for that case.
        replay_loss_weight = train_loader.replay_loss_weight
    elif args.replay_per_batch > 0:
        new_per_update = accumulation_steps * (args.batch_size - args.replay_per_batch)
        replay_per_update = accumulation_steps * args.replay_per_batch
        replay_ratio_r = args.replay_per_batch / (args.batch_size - args.replay_per_batch)
        replay_loss_weight = None
    else:
        new_per_update = args.accumulation_size
        replay_per_update = 0
        replay_ratio_r = args.replay_ratio
        replay_loss_weight = None

    common = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
        "regularizer": regularizer,
        "replay_lambda": args.replay_lambda,
        "batch_size": args.batch_size,
        "accumulation_size": args.accumulation_size,
        "log_every": args.log_every,
        "metrics_path": args.metrics_path,
        "max_steps": args.max_steps,
        "profile": args.profile,
        "reg_once_per_update": args.reg_once_per_update,
        "probe_loaders": probe_loaders,
        "probe_every": args.probe_every,
    }

    if args.paradigm == "opd":
        teacher_dir = args.teacher_model_dir or args.model_dir
        teacher_device = (
            device
            if args.teacher_gpu < 0
            else torch.device(f"cuda:{args.teacher_gpu}" if torch.cuda.is_available() else "cpu")
        )
        print(f"loading teacher {args.teacher_model_name} onto {teacher_device}")
        teacher_model, _ = load_causal_lm_and_tokenizer(
            teacher_dir, args.teacher_model_name, args.use_bf16
        )
        teacher_model.to(teacher_device)
        trainer = OPDTrainer(
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            max_new_tokens=args.opd_max_new_tokens,
            temperature=args.opd_temperature,
            kl_temperature=args.opd_kl_temperature,
            **common,
        )
    else:
        trainer = SFTTrainer(**common)

    trainer.train(
        train_loader,
        epochs=args.epochs,
        val_loader=valid_loader,
        eval_before_train=args.eval_before_train,
        extra_record={
            "paradigm": args.paradigm,
            "seed": args.seed,
            "full_finetune": args.full_finetune,
            "regularizer": args.regularizer,
            # Which matrix file the penalty actually came from. "regularizer: ewc"
            # alone stops being enough to identify a run once several Fisher
            # estimates coexist (one per old-knowledge domain, plus their mixes),
            # and the same holds for C.
            "penalty_path": args.fisher_path if args.regularizer == "ewc" else args.cov_path,
            "identity_cov": args.identity_cov,
            # Timing is only comparable across runs that share this setting.
            "reg_once_per_update": args.reg_once_per_update,
            # And so is any claim about the penalty's cost or its numerics.
            "reg_impl": args.reg_impl,
            "reg_allow_tf32": args.reg_allow_tf32,
            "reg_compute_dtype": args.reg_compute_dtype,
            "batch_size": args.batch_size,
            "accumulation_size": args.accumulation_size,
            # val_loss averages over batches rather than tokens, so it only
            # means the same thing across runs that used the same value here.
            "eval_batch_size": args.eval_batch_size or args.batch_size,
            "replay_ratio": args.replay_ratio,
            "replay_per_batch": args.replay_per_batch,
            "replay_steps_per_update": args.replay_steps_per_update,
            # train_samples counts replay rows too, so record the split needed
            # to recover the new-task volume from the cost metrics. These two
            # describe the batch-level split only; step-level micro-batches are
            # homogeneous and leave them at 0, like vanilla.
            "new_per_batch": (
                args.batch_size - args.replay_per_batch if args.replay_per_batch > 0 else 0
            ),
            "replay_row_share": (
                args.replay_per_batch / args.batch_size if args.replay_per_batch > 0 else 0.0
            ),
            "new_per_update": new_per_update,
            "replay_per_update": replay_per_update,
            # N_replay / N_new, which is also the extra cost this arm pays: the
            # run takes (1 + r) times the micro-batches of a vanilla one.
            "replay_ratio_r": replay_ratio_r,
            # Replay's exact share of the accumulated gradient, or null on the
            # schemes where row share and loss weight come apart.
            "replay_loss_weight": replay_loss_weight,
            "replay_self_distill": int(
                bool(args.replay_self_distill_file) or bool(replay_mix)
            ),
            # Replay rows can be tokenized at a different budget than the new
            # task, so the effective one has to be recorded rather than inferred
            # from max_len.
            "replay_max_len": replay_max_len(args) if args.replay_per_batch > 0 else 0,
            # Whether the rehearsal rows arrived in the form the model was
            # taught them in. Empty means they inherited the new task's turn,
            # which is right for a single-corpus run and wrong for a continual
            # one -- and the two are indistinguishable after the fact without
            # this field.
            "replay_system_prompt": args.replay_system_prompt,
            "replay_mix_files": args.replay_mix_files,
            "replay_mix_weights": (
                ",".join(f"{weight:.4f}" for _, _, weight in replay_mix) if replay_mix else ""
            ),
            "replay_mix_labels": ",".join(label for label, _, _ in replay_mix),
            "probe_every_updates": args.probe_every_updates,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "eval_before_train": args.eval_before_train,
            # Which rows old_val_loss is measured on, and how they were
            # rendered. Without these the column is a number with no provenance,
            # and the slice moves if any of them is changed between arms.
            "old_val_jsonl": args.old_val_jsonl,
            "old_val_pool_size": args.old_val_pool_size,
            "old_val_sample_seed": args.old_val_sample_seed,
            "old_val_max_len": args.old_val_max_len,
            "old_val_target_column": args.old_val_target_column,
            "prior_val_jsonl": args.prior_val_jsonl,
            # A loss curve only means something next to the schedule that
            # produced it, and "lr: 5e-5" alone no longer identifies a run now
            # that the same peak can be constant or cosine-decayed.
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "warmup_updates": warmup_updates if args.lr_scheduler != "constant" else 0,
            "total_updates": total_updates,
            "osft": args.osft,
            # Realized ranks and the upstream commit they came out of. Absent on
            # every non-OSFT run, which keeps their records byte-identical to the
            # ones already in results_log.
            **osft_record,
        },
        save_path=args.save_path if args.save == 1 else "",
        tokenizer=tokenizer,
        old_val_loader=old_val_loader,
        prior_val_loaders=prior_val_loaders,
    )


if __name__ == "__main__":
    main()
