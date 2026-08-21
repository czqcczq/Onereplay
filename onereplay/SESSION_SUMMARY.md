# OneReplay Session Summary

This file summarizes the current discussion and implementation state so a new
Codex session can continue quickly.

## Research Idea

We start from LoRA fine-tuning:

```text
y = W x + scale * B A x
DeltaW = scale * B A
```

The concern is that the LoRA update `DeltaW` may change model behavior on old
knowledge. For an old-knowledge hidden state `x`, we want:

```text
DeltaW x ≈ 0
```

For many old hidden states `X_old = [x_1, ..., x_n]`, the regularizer is:

```text
||DeltaW X_old||_F^2
```

If:

```text
C = E_x[x x^T]
```

then the average old-knowledge perturbation is:

```text
E_x ||DeltaW x||_2^2 = tr(DeltaW C DeltaW^T)
```

So the proposed training objective is:

```text
L = L_task + lambda * tr(DeltaW C DeltaW^T)
```

This is better than constraining `DeltaW` to be orthogonal to `W`, because the
real goal is to make the LoRA update small on old hidden-state directions.

## Implementation Location

All new code is under:

```text
/home/weiliu1/mypaper/2026/微调/mycode/onereplay
```

Files:

```text
mycode/onereplay/__init__.py
mycode/onereplay/lora_cov_utils.py
mycode/onereplay/collect_flan_cov.py
mycode/onereplay/train_onereplay_lora.py
mycode/onereplay/README.md
mycode/onereplay/SESSION_SUMMARY.md
```

The original vanilla LoRA script was not modified:

```text
mycode/standard_ft/vanilla_lora.py
```

## Environment

Use this Python environment:

```bash
/home/weiliu1/mypaper/2026/微调/.venv/bin/python
```

From the workspace root:

```bash
cd /home/weiliu1/mypaper/2026/微调
```

Validated commands:

```bash
.venv/bin/python -m py_compile mycode/onereplay/*.py
```

This passed.

## Stage 1: Collect Old-Knowledge C

Script:

```text
mycode/onereplay/collect_flan_cov.py
```

Purpose:

1. Load the frozen base model.
2. Load old-knowledge data, currently FLAN Collection.
3. Register forward pre-hooks on LoRA target linear layers such as `q_proj` and
   `v_proj`.
4. For each target layer, collect its input hidden states `x`.
5. Accumulate `X^T X` in streaming form.
6. Save:

```text
C_l = E[x x^T]
```

for each target layer.

Important correction made later:

The script originally joined FLAN `inputs + targets` as plain text. This was
changed because Qwen fine-tuning data is chat-formatted. Now the default is to
format each FLAN row as chat messages:

```text
user: inputs
assistant: targets
```

Then the script calls:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

Relevant args:

```text
--use_chat_template 1
--include_target_in_chat 1
--system_prompt ""
```

For ablation with raw text:

```text
--use_chat_template 0
```

FLAN dataset checked:

```text
Muennighoff/flan
```

Its fields are:

```text
inputs, targets, task
```

HuggingFace cache directory:

```text
/home/weiliu1/huggingface/datasets/cache
```

Example command:

```bash
.venv/bin/python mycode/onereplay/collect_flan_cov.py \
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
  --output_path mycode/onereplay/flan_qwen3_qv_cov.pt
```

The script uses streaming by default so it does not need to download all FLAN
before collection.

## Stage 2: Train OneReplay LoRA

Script:

```text
mycode/onereplay/train_onereplay_lora.py
```

Purpose:

1. Load the base model and tokenizer.
2. Add PEFT LoRA to target modules.
3. Load precomputed covariance matrices from Stage 1.
4. Train on the existing BoolQ jsonl dataset.
5. Add the OneReplay regularizer:

```text
loss = task_loss + replay_lambda * replay_reg
```

where:

```text
replay_reg = mean_l tr(DeltaW_l C_l DeltaW_l^T)
```

Efficient low-rank computation:

```text
tr((scale * B A) C (scale * B A)^T)
= scale^2 * tr((B^T B)(A C A^T))
```

This avoids constructing the full `DeltaW`.

Example command:

```bash
.venv/bin/python mycode/onereplay/train_onereplay_lora.py \
  --cov_path mycode/onereplay/flan_qwen3_qv_cov.pt \
  --replay_lambda 1e-4
```

Useful printed values:

```text
task_loss
replay_reg
lambda_reg = replay_lambda * replay_reg
```

Suggested first tuning rule:

Choose `replay_lambda` so that `lambda_reg` is about 1% to 10% of `task_loss`
near the start of training.

## Current Validation

Completed:

1. Syntax check passed:

```bash
.venv/bin/python -m py_compile mycode/onereplay/*.py
```

2. Import check passed for utility functions.

3. Mathematical check passed:

The implemented low-rank regularizer matched explicit computation of:

```text
tr(DeltaW C DeltaW^T)
```

with absolute difference:

```text
0.0
```

4. `Muennighoff/flan` was checked through HuggingFace streaming. One sample had
keys:

```text
inputs, targets, task
```

Not yet done:

1. Full Stage 1 covariance collection with Qwen3-1.7B.
2. Full Stage 2 LoRA training run.
3. Hyperparameter sweep for `replay_lambda`.

## Next Good Steps

1. Run a very small Stage 1 smoke test, for example:

```bash
.venv/bin/python mycode/onereplay/collect_flan_cov.py \
  --max_samples 4 \
  --batch_size 1 \
  --output_path /tmp/onereplay_cov_smoke.pt
```

2. If that works, run a larger C collection with `--max_samples 20000`.

3. Run Stage 2 with a short training setting first, for example:

```bash
.venv/bin/python mycode/onereplay/train_onereplay_lora.py \
  --cov_path mycode/onereplay/flan_qwen3_qv_cov.pt \
  --epochs 1 \
  --replay_lambda 1e-4
```

4. Watch `lambda_reg` relative to `task_loss`, then adjust `replay_lambda`.

