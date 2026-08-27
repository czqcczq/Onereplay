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
from onereplay.core.regularizer import EWCRegularizer, ReplayRegularizer  # noqa: E402
from onereplay.data.chat import build_loader, build_opd_loader  # noqa: E402
from onereplay.data.commonsense import load_and_prepare_dataset  # noqa: E402
from onereplay.data.batch_mix import build_batch_mixed_loader  # noqa: E402
from onereplay.data.probe import build_probe_loaders  # noqa: E402
from onereplay.data.replay import build_replay_dataset, mix_replay_into_train  # noqa: E402
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
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=0)
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
            "the LoRA path, which is already rank x rank and costs ~1.6% of a step."
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
        "--probe_cs_val_size",
        type=int,
        default=1000,
        help="Rows of the Commonsense validation split to probe; 0 skips that curve.",
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


def main() -> None:
    args = parse_args()
    print("the file is " + str(Path(__file__).resolve()))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for attr, value in sorted(vars(args).items()):
        print(f"\t{attr.upper()}={value}")
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir,
        args.model_name,
        args.use_bf16,
        args,
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

    regularizer = build_regularizer(args, device)
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
    if args.replay_per_batch > 0 and args.replay_ratio > 0:
        raise ValueError(
            "--replay_per_batch (batch-level) and --replay_ratio (data-level) are two "
            "different mixing schemes; set exactly one."
        )
    if args.replay_per_batch > 0 and args.paradigm == "opd":
        # OPD replaces the batch's targets with a student rollout scored by the
        # teacher. What that should mean for a replay row is undefined, so
        # refuse rather than silently distilling the replay corpus too.
        raise ValueError("--replay_per_batch is not defined for --paradigm opd")

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
        replay_dataset = build_replay_dataset(args, tokenizer)
        train_loader = build_batch_mixed_loader(
            train_dataset,
            replay_dataset,
            tokenizer,
            new_per_batch=new_per_batch,
            replay_per_batch=args.replay_per_batch,
            seed=args.seed,
        )
        print(train_loader.describe(), flush=True)
        print(
            f"accumulation: {max(args.accumulation_size // args.batch_size, 1)} steps x "
            f"{new_per_batch} new-task rows = "
            f"{max(args.accumulation_size // args.batch_size, 1) * new_per_batch} new-task rows "
            f"per update",
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

    common = {
        "model": model,
        "optimizer": optimizer,
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
        extra_record={
            "paradigm": args.paradigm,
            "seed": args.seed,
            "full_finetune": args.full_finetune,
            "regularizer": args.regularizer,
            "identity_cov": args.identity_cov,
            # Timing is only comparable across runs that share this setting.
            "reg_once_per_update": args.reg_once_per_update,
            # And so is any claim about the penalty's cost or its numerics.
            "reg_impl": args.reg_impl,
            "reg_allow_tf32": args.reg_allow_tf32,
            "reg_compute_dtype": args.reg_compute_dtype,
            "batch_size": args.batch_size,
            "accumulation_size": args.accumulation_size,
            "replay_ratio": args.replay_ratio,
            "replay_per_batch": args.replay_per_batch,
            # train_samples counts replay rows too, so record the split needed
            # to recover the new-task volume from the cost metrics.
            "new_per_batch": (
                args.batch_size - args.replay_per_batch if args.replay_per_batch > 0 else 0
            ),
            "replay_row_share": (
                args.replay_per_batch / args.batch_size if args.replay_per_batch > 0 else 0.0
            ),
            "new_per_update": (
                max(args.accumulation_size // args.batch_size, 1)
                * (args.batch_size - args.replay_per_batch)
                if args.replay_per_batch > 0
                else args.accumulation_size
            ),
            "replay_self_distill": int(bool(args.replay_self_distill_file)),
            "probe_every_updates": args.probe_every_updates,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
        },
        save_path=args.save_path if args.save == 1 else "",
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
