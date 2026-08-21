#!/usr/bin/zsh
#
#SBATCH --job-name=or_bscale
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# Batch scaling 开销扫描：找 GPU 饱和点，并在每个 micro-batch 下对比
# vanilla LoRA vs OneReplay 的时间 / 显存。
#
# 老师的点：batch 太小时 GPU 没吃满，batch*2 的时间不一定翻倍，测出的
# 「replay 只贵 X%」会被低估。必须先扫到 tokens/s 涨不动的点，再在那里报开销。
#
# 口径：
#   - 有效 batch（每次更新的样本数）固定为 ACCUM=64，只改 micro-batch。
#     batch=8  -> accum_steps=8；batch=16 -> 4；batch=32 -> 2；batch=64 -> 1。
#   - 短跑 MAX_STEPS（默认 400），PROFILE=0：干净 wall time / 吞吐 / 峰值显存。
#   - 同一 job 串行、同一张卡，数字才可比。
#   - SAVE=0，只要开销数字。
#
# 用法:
#   sbatch onereplay/slurm/15_bench_batch_scale.slurm
# 只扫 8/16/32（更快）:
#   BATCHES="8 16 32" sbatch onereplay/slurm/15_bench_batch_scale.slurm
# 顺带在每个 batch 下做 PROFILE=1 短跑（看正则占比随 batch 下降）:
#   RUN_PROFILE=1 sbatch onereplay/slurm/15_bench_batch_scale.slurm
#
# 跑完汇总:
#   python -m onereplay.scripts.summarize_batch_scale \
#     --metrics_dir <RESULTS_ROOT>/metrics/cost/batch_scale
# =============================================================================

GCC_MODULE="${GCC_MODULE:-GCC}"
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.6.3}"
CONDA_ENV="${CONDA_ENV:-opd}"

module purge
module load "${GCC_MODULE}"
module load "${CUDA_MODULE}"
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
hash -r

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export USER="${USER:-$(whoami)}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Python          : $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- 项目路径 ----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
COST_DIR="${COST_DIR:-${RESULTS_ROOT}/metrics/cost/batch_scale}"

# ---- 超参 ----
SEED="${SEED:-1}"
LR="${LR:-1e-4}"
# 有效 batch 固定；micro-batch 变化时 accum_steps = ACCUM / BATCH 自动变。
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-200}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
LAMBDA="${LAMBDA:-1e-2}"

# ---- 扫描网格 ----
# zsh：空格分隔字符串拆成数组
BATCHES=(${=BATCHES:-8 16 32 64})
MAX_STEPS="${MAX_STEPS:-400}"
RUN_PROFILE="${RUN_PROFILE:-0}"
PROF_STEPS="${PROF_STEPS:-200}"
LOG_EVERY="${LOG_EVERY:-100}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || { echo "缺协方差文件: ${COV_PATH}"; exit 1; }
mkdir -p "${COST_DIR}"

echo "BATCHES=${BATCHES[*]} ACCUM=${ACCUM} MAX_STEPS=${MAX_STEPS} LAMBDA=${LAMBDA}"

# ---------------------------------------------------------------------------
# $1 run 名  $2 batch  $3 profile  $4 max_steps  $5.. 追加 train 参数
# ---------------------------------------------------------------------------
bench_run() {
  local run="$1"; shift
  local batch="$1"; shift
  local profile="$1"; shift
  local max_steps="$1"; shift

  # batch 不能超过 ACCUM，否则有效 batch 会悄悄变大
  if (( batch > ACCUM )); then
    echo "错误: BATCH=${batch} > ACCUM=${ACCUM}，有效 batch 会变成 ${batch}；请同步调大 ACCUM"
    exit 1
  fi

  echo "==== bench: ${run} (batch=${batch} accum=${ACCUM} profile=${profile} max_steps=${max_steps}) ===="
  mkdir -p "${RESULTS_ROOT}/map_cache/${run}"
  python -m onereplay.scripts.train --paradigm sft \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --epochs 1 --lr "${LR}" --batch_size "${batch}" --accumulation_size "${ACCUM}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TARGET_MODULES}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/${run}" \
    --log_every "${LOG_EVERY}" \
    --profile "${profile}" --max_steps "${max_steps}" \
    --save 0 \
    --metrics_path "${COST_DIR}/${run}.jsonl" \
    "$@"
}

VANILLA_FLAGS=(--cov_path "${COV_PATH}" --replay_lambda 0 --measure_replay_when_lambda_zero 0)
ONEREPLAY_FLAGS=(--cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0)

for batch in "${BATCHES[@]}"; do
  echo "======== micro-batch = ${batch} ========"
  bench_run "bscale_vanilla_b${batch}_seed${SEED}" "${batch}" 0 "${MAX_STEPS}" "${VANILLA_FLAGS[@]}"
  bench_run "bscale_onereplay_b${batch}_lam${LAMBDA}_seed${SEED}" "${batch}" 0 "${MAX_STEPS}" "${ONEREPLAY_FLAGS[@]}"

  if [[ "${RUN_PROFILE}" == "1" ]]; then
    bench_run "bscale_vanilla_b${batch}_prof_seed${SEED}" "${batch}" 1 "${PROF_STEPS}" "${VANILLA_FLAGS[@]}"
    bench_run "bscale_onereplay_b${batch}_lam${LAMBDA}_prof_seed${SEED}" "${batch}" 1 "${PROF_STEPS}" "${ONEREPLAY_FLAGS[@]}"
  fi
done

echo "==== summary ===="
python -m onereplay.scripts.summarize_batch_scale \
  --metrics_dir "${COST_DIR}" \
  --out "${COST_DIR}/batch_scale_table.md"

echo "Job finished at: $(date)"
