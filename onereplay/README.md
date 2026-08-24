# OneReplay

OneReplay preserves old-knowledge behavior while a LoRA adapter learns a new task:

```text
y = W x + scale * B A x
L = L_task + lambda * E_old ||scale * B A x||^2
  = L_task + lambda * tr(DeltaW C DeltaW^T)
```

where `C = E_old[x x^T]` is estimated once from old-knowledge data such as
FLAN Collection.

For relative-error preservation, collect `C` with normalized hidden states:

```text
x' = x / ||W x||
C = E_old[x' x'^T]
```

Then the same training regularizer becomes approximately:

```text
E_old ||scale * B A x||^2 / ||W x||^2
```

## Layout

```text
onereplay/
├── core/         regularizer.py  covariance.py  fisher.py  modeling.py
│                                                             framework-agnostic kernel
├── data/         chat.py  commonsense.py  old_knowledge.py  replay.py  probe.py
├── trainers/     base.py  sft.py  opd.py                      shared loop + two paradigms
├── eval/         runner.py  generation.py  metrics/           one model load, many metrics
├── scripts/      collect_cov.py  collect_fisher.py  generate_replay_targets.py
│                 train.py  evaluate.py  check_old_knowledge_pool.py
│                 compare_cov_scale.py                        the only CLI entry points
├── legacy/       archived single-purpose scripts, not on the active path
├── slurm/        01..27 cluster jobs
├── pbs/          01..33 the same jobs for the PBS + Singularity cluster
└── third_party/  vendored IFEval checkers, Multi-IF data
```

Run entry points as modules from the repository root so `onereplay` and
`process_dataset` are both importable:

```bash
python -m onereplay.scripts.train ...
```

## Stage 1: collect C

```bash
python -m onereplay.scripts.collect_cov \
  --model_dir /home/weiliu1/huggingface/models/ \
  --model_name Qwen3-1.7B \
  --dataset_name Muennighoff/flan \
  --cache_dir /home/weiliu1/huggingface/datasets/cache \
  --streaming 1 \
  --dataset_split train \
  --use_chat_template 1 \
  --include_target_in_chat 1 \
  --max_samples 20000 \
  --target_modules q_proj,v_proj \
  --output_path results/cov/flan_qwen3_qv_cov.pt
```

To collect the relative-error covariance, add:

```bash
  --cov_normalization base_output_norm \
  --cov_norm_eps 1e-6
```

This changes the collected matrix from `E[x x^T]` to
`E[x x^T / max(||W x||, eps)^2]` for each target Linear module.

If FLAN is saved locally or converted to jsonl, use `--dataset_path` or
`--data_files` instead of `--dataset_name`.

The HuggingFace version checked for this implementation has columns:

```text
inputs, targets, task
```

The collection script formats each row with the model chat template by default:

```text
user: inputs
assistant: targets
```

This is important because the hidden states used to estimate `C` should match
the chat-formatted distribution used during LoRA fine-tuning. For an ablation
with raw text, set `--use_chat_template 0`.

## Stage 1b: collect F (EWC baseline only)

EWC is the regularization-family baseline OneReplay is compared against. It
replaces `C` with the diagonal empirical Fisher of the frozen base weights and
keeps everything else identical:

```text
L = L_task + lambda * sum_ij F_ij (DeltaW_ij)^2
F_i = (1/N) sum_n (d L_n / d W_i)^2
L_n = -sum_{t in assistant} log p(y_t | x, y_<t)
```

`L_n` sums over the answer's tokens rather than averaging them, because the
score behind the Fisher is `grad log p(y|x) = sum_t grad log p(y_t|...)`; a
`1/T_n` factor would give a length-normalized variant instead. Only assistant
tokens are supervised, so `F` measures response sensitivity, which is what the
IFEval and Multi-IF retention metrics probe.

```bash
python -m onereplay.scripts.collect_fisher \
  --model_dir /home/weiliu1/huggingface/models/ \
  --model_name Qwen3-1.7B \
  --use_bf16 0 \
  --data_files /path/to/flan/train/*.jsonl \
  --use_chat_template 1 \
  --include_target_in_chat 1 \
  --max_samples 20000 --sample_seed 1 \
  --target_modules q_proj,v_proj \
  --output_path results/fisher/flan_qwen3_qv_fisher.pt
```

Every dataset flag has to match Stage 1 exactly, or the two baselines consumed
different old knowledge and the comparison measures that instead of the
weighting. Both scripts print `pool rows=... fingerprint=...` for the rows they
selected; `scripts/check_old_knowledge_pool.py` prints the same value in seconds
without loading a model, and `slurm/22_collect_fisher.slurm` compares the two
before it starts.

`--use_bf16 0` and `batch_size=1` are not tuning knobs. A bf16 gradient is
already rounded before it can be squared, and squaring a batch-summed gradient
computes `(sum_n g_n)^2` rather than `sum_n g_n^2`, which is a different matrix
and fails silently.

The run reports two numbers that belong in the paper next to `F`: the fitted
exponent `a` in `||g_n||^2 ~ T_n^a`, and the effective sample size
`(sum w)^2 / sum w^2`. Sequence-sum scores grow with answer length somewhere
between `a=1` (uncorrelated per-token gradients) and `a=2` (perfectly aligned
ones), and which one this corpus lands on decides whether `N=20000` is enough.

## Stage 2: train

Supervised fine-tuning:

```bash
python -m onereplay.scripts.train --paradigm sft \
  --dataset_path /path/to/commonsense_170k_hf \
  --cov_path results/cov/flan_qwen3_qv_cov.pt \
  --replay_lambda 1e-4 \
  --save_path results/adapters/cs_onereplay_lam1e-4_seed1
```

The EWC baseline swaps the weighting matrix and changes nothing else:

```bash
python -m onereplay.scripts.train --paradigm sft \
  --dataset_path /path/to/commonsense_170k_hf \
  --regularizer ewc --fisher_path results/fisher/flan_qwen3_qv_fisher.pt \
  --replay_lambda 1e5 \
  --save_path results/adapters/cs_ewc_lam1e5_seed1
```

`--replay_lambda` is the coefficient for either regularizer, but the two scales
are unrelated: `F` is built from squared gradients and lands five to six orders
of magnitude below `C`, so the OneReplay grid transfers to nothing. Sweep EWC
separately with `slurm/23_sweep_ewc_lambda.slurm`, and compare the two methods
at equal `val_loss` rather than at some chosen lambda each. That protocol
matters here specifically: EWC-LoRA (arXiv 2602.17559) reports that a
precomputed fixed Fisher has the lowest plasticity of the variants it tried, so
a retention gain that comes from failing to learn the new task is the failure
mode to rule out.

On-policy distillation against a frozen same-family teacher:

```bash
python -m onereplay.scripts.train --paradigm opd \
  --teacher_model_name Qwen3-8B --teacher_gpu 1 \
  --dataset_path /path/to/commonsense_170k_hf \
  --cov_path results/cov/flan_qwen3_qv_cov.pt \
  --replay_lambda 1e-4
```

The teacher must share the student's tokenizer and vocabulary, otherwise the
per-token KL is meaningless. Always run the `--replay_lambda 0` OPD baseline
too: a strong instruct teacher may preserve instruction following on its own,
and without that baseline the teacher effect cannot be separated from the
regularizer effect.

Watch the printed values:

```text
task_loss
replay_reg
lambda_reg = replay_lambda * replay_reg
```

A practical first setting is to choose `replay_lambda` so `lambda_reg` is about
1% to 10% of `task_loss` near the start of training.

### Measuring training cost

Every run records its own wall time and GPU memory into the `metrics_path`
jsonl and prints a short block after each epoch:

```text
---- cost ----
train wall time      : 1832.4 s
eval  wall time      : 41.2 s
throughput           : 9.21 samples/s
                       3987 tokens/s
per-step time        : 108.5 ms
peak GPU memory      : 21.44 GiB allocated / 23.10 GiB reserved
  of which C matrices: 0.875 GiB
```

Add `--profile 1` to also split the step into phases, which is how you get the
regularizer's own cost rather than inferring it from a difference:

```text
  backward          : 612.3 s  (33.4% of train)
  optimizer         :  48.1 s  (2.6% of train)
  prepare_batch     :   9.7 s  (0.5% of train)
  replay_reg        :  74.6 s  (4.1% of train)
  task_loss         : 1081.2 s  (59.0% of train)
```

Phase timers call `torch.cuda.synchronize()`, which costs a few percent of
throughput, so use `--profile 1` with a small `--max_steps` for a dedicated
measurement run rather than on the runs whose wall time you report. Peak memory
and epoch wall time are always recorded and cost nothing.

For a vanilla-LoRA cost baseline, also pass `--measure_replay_when_lambda_zero 0`.
Otherwise `C` stays loaded and is evaluated every step even at `--replay_lambda 0`,
so the baseline would pay the regularizer's cost too.

On the cluster the training jobs expose this as environment variables:

```bash
PROFILE=1 MAX_STEPS=200 SAVE=0 MEASURE_REPLAY=0 sbatch onereplay/slurm/02_train_vanilla.slurm
PROFILE=1 MAX_STEPS=200 SAVE=0 sbatch onereplay/slurm/03_train_onereplay.slurm
```

Two controls matter for the claim:

- `--replay_lambda 0` — vanilla LoRA, shows the forgetting the method must fix.
- `--identity_cov 1` — replaces every `C` with the identity, turning the penalty
  into plain L2 on `DeltaW`. This separates "the covariance directions help"
  from "any shrinkage helps".

### The replay baseline: gold vs self-distilled targets

`--replay_ratio` appends old-knowledge rows to the training set instead of
penalizing `DeltaW`. Which target those rows carry decides what the baseline
actually measures.

With FLAN's gold targets (the default) the rows are bare one-line answers. For
an already instruction-tuned base model that is a new task, not a rehearsal:
it starts near loss 3.0 while the new task sits near 0.05, so a nominal 10% of
rows takes over roughly 80% of the loss and drags the model toward FLAN's
answer-key style. Measured on Qwen3-1.7B full fine-tuning, IFEval fell from
0.634 to 0.486, answers degenerated into repetition in 42% of prompts, and
many replies contained only the requested format marker with no content.
OneReplay, by contrast, only ever sees FLAN *inputs* and its own `W0`, so the
two routes are not consuming the same information.

`generate_replay_targets.py` closes that gap by replacing every target with
the base model's own greedy answer to the same prompt:

```bash
python -m onereplay.scripts.generate_replay_targets \
  --model_name Qwen3-1.7B \
  --data_files /path/to/flan/train/*.jsonl \
  --pool_size 20000 --sample_seed 1 \
  --output_path results/replay/flan_selfdistill_20000_seed1.jsonl

python -m onereplay.scripts.train --paradigm sft \
  --replay_ratio 0.10 \
  --replay_self_distill_file results/replay/flan_selfdistill_20000_seed1.jsonl
```

Replay then starts near zero loss and only anchors the model to `W0`'s
behavior, which is what the penalty approximates through `C`. The pool order,
shuffle seed and schema filtering are identical on both paths, so the two
flavors train on the same rows and the only variable is the target.

Answers that hit the generation budget carry `truncated: true` and are dropped
by `--replay_drop_truncated 1`; training on a cut-off answer would teach the
model not to end its turn.

### Retention probes during training

The replay pool holds ~17k usable rows and a 50% run consumes ~169k replay rows
per epoch, so every row is trained on about 29 times over three epochs. At that
repetition, "IFEval only fell 1.1 points" is equally consistent with keeping the
old ability and with memorizing those 17k rows, and no end-of-training metric
can tell the two apart.

`--probe_every_updates` scores fixed sets on the side, mid-training:

```bash
# A FLAN slice the training pool does not contain, from the same shuffle.
python -m onereplay.scripts.generate_replay_targets \
  --data_files /path/to/flan/train/*.jsonl \
  --pool_offset 20000 --pool_size 1500 --sample_seed 1 \
  --output_path results/replay/flan_selfdistill_heldout_off20000_1500_seed1.jsonl

python -m onereplay.scripts.train --paradigm sft \
  --probe_every_updates 64 \
  --probe_heldout_file results/replay/flan_selfdistill_heldout_off20000_1500_seed1.jsonl \
  --probe_inpool_file results/replay/flan_selfdistill_20000_seed1.jsonl
```

Three curves land in the metrics JSONL as `record_type: probe` rows:
`flan_heldout` (rows replay never trains on, so old-knowledge generalization),
`flan_inpool` (rows it does, so memorization), and `cs_val`. Reading heldout
alone is ambiguous — a rise could be overfitting to the pool or the new task
dragging everything — and the gap between the two FLAN curves separates those.
Run the same probes on a vanilla arm for the control: neither slice is in its
training data, so its two curves must coincide, which is what proves a gap on
the replay arm comes from replay rather than from the slices differing.

The interval counts optimizer updates, not micro-batches, because batch-level
replay halves the new-task rows per micro-batch and doubles `accumulation_size`
to compensate. Every arm therefore probes at the same points on an x axis of
new-task data consumed. Probe time is subtracted from `train_sec`,
`sec_per_step` and the `win=` field so a probed run still reports the same
steady-state ms/step; `elapsed_sec` is wall clock and still includes it.

`slurm/27_probe_heldout_curve.slurm` runs the whole thing (held-out generation,
the three arms, an overlap check that the held-out slice really is disjoint, and
a shape verdict printed to the log).

## Stage 3: evaluate

One model load, several metrics:

```bash
python -m onereplay.scripts.evaluate \
  --model_name Qwen3-1.7B \
  --adapter_path results/adapters/YOUR_ADAPTER --run_name YOUR_RUN \
  --metrics commonsense,ifeval,multiif \
  --out_dir results \
  --dataset_path /path/to/commonsense_170k_hf
```

Available metrics:

- `commonsense` — new-task fitting (held-out Commonsense170k loss).
- `ifeval` — instruction following, 541 prompts.
- `multiif` — instruction following, multi-turn cumulative, English subset.
- `gsm8k`, `aime` — math retention probes.
- `humaneval`, `mbpp` — code retention probes.

Results land in `<out_dir>/<metric>/<run_name>/summary.json` with one appended
row per run in `<out_dir>/<metric>_summary.csv`.

## Refactor parity check

`scripts/verify_refactor.py` runs a few steps through both the archived
training loop and the new `SFTTrainer` with the same seed and fails if
`train_task_loss` or `replay_reg` differ:

```bash
python -m onereplay.scripts.verify_refactor \
  --dataset_path /path/to/commonsense_170k_hf \
  --cov_path results/cov/flan_qwen3_qv_cov.pt \
  --max_train_samples 64
```
