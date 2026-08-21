#!/usr/bin/zsh
#
#SBATCH --job-name=or_bench
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# 开销测量：vanilla LoRA vs OneReplay，全量数据各跑 1 个 epoch
#
# 为什么放在同一个 job 里串行跑：两条线共享同一个节点、同一块 GPU，wall time
# 才可比。分成两个 job 提交可能落到不同型号的卡上，数字没有意义。
#
# 表里另外两行不需要这个脚本：
#   - base   不训练，训练开销一栏是 "—"；推理开销与其他行相同（都是 LoRA）
#   - replay 由 09_sweep_replay_ratio.slurm 训练时顺带记录（train_sec、显存峰值
#            是免费采集的，无需 PROFILE）
#
# 两轮测量，因为它们互相冲突：
#   Round A  PROFILE=0 全量 1 epoch  -> 干净的 wall time / 吞吐 / 显存峰值
#   Round B  PROFILE=1 前若干步      -> 分阶段耗时（task_loss / replay_reg / backward）
# PROFILE=1 每个阶段都要 cuda synchronize，会拖慢几个百分点，所以分阶段占比只能
# 从 Round B 读，对外报的 wall time 必须来自 Round A。
#
# 用法:
#   sbatch onereplay/slurm/10_bench_cost.slurm
# 只跑分阶段短跑（几分钟出结果，先验证脚本通不通）:
#   RUN_FULL=0 sbatch onereplay/slurm/10_bench_cost.slurm
# 顺带测一次性协方差采集成本:
#   BENCH_COV=1 sbatch onereplay/slurm/10_bench_cost.slurm
#
# 跑完汇总成表:
#   python -m onereplay.scripts.summarize_cost \
#     --metrics_dir <RESULTS_ROOT>/metrics/cost --baseline bench_vanilla_seed1
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

# ---- 项目路径（按需覆盖）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
FLAN_DIR="${FLAN_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/flan}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"
COST_DIR="${COST_DIR:-${RESULTS_ROOT}/metrics/cost}"

# ---- 训练超参（必须与 02 / 06 / 09 逐字相同，否则 wall time 不可比）----
SEED="${SEED:-1}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

# ---- 被测的 λ（默认取 retention 最好的 1e-2）----
LAMBDA="${LAMBDA:-1e-2}"

# ---- 两轮开关 ----
RUN_FULL="${RUN_FULL:-1}"          # Round A：全量 1 epoch
RUN_PROFILE="${RUN_PROFILE:-1}"    # Round B：分阶段短跑
PROF_STEPS="${PROF_STEPS:-300}"    # Round B 的步数
BENCH_COV="${BENCH_COV:-0}"        # 一次性协方差采集成本
FLAN_SAMPLES="${FLAN_SAMPLES:-20000}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || { echo "缺协方差文件(先跑 01_collect_cov): ${COV_PATH}"; exit 1; }
mkdir -p "${COST_DIR}" "${DATASET_CACHE}"

# ---------------------------------------------------------------------------
# 一次训练测量。save=0，因为 adapter 之前的正式训练已经产出，这里只要开销数字。
#   $1 run 名  $2 profile  $3 max_steps  $4.. 追加给 train 的参数
# ---------------------------------------------------------------------------
bench_run() {
  local run="$1"; shift
  local profile="$1"; shift
  local max_steps="$1"; shift

  echo "==== bench: ${run} (profile=${profile} max_steps=${max_steps}) ===="
  mkdir -p "${RESULTS_ROOT}/map_cache/${run}"
  python -m onereplay.scripts.train --paradigm sft \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --epochs 1 --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TARGET_MODULES}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/${run}" \
    --profile "${profile}" --max_steps "${max_steps}" \
    --save 0 \
    --metrics_path "${COST_DIR}/${run}.jsonl" \
    "$@"
}

# vanilla 必须连 C 都不加载，否则每步仍在算正则，测出来的不是 vanilla。
VANILLA_FLAGS=(--cov_path "${COV_PATH}" --replay_lambda 0 --measure_replay_when_lambda_zero 0)
ONEREPLAY_FLAGS=(--cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0)

# ---------------------------------------------------------------------------
# Round A：全量 1 epoch，PROFILE=0，取 wall time / 吞吐 / 显存峰值
# ---------------------------------------------------------------------------
if [[ "${RUN_FULL}" == "1" ]]; then
  bench_run "bench_vanilla_seed${SEED}" 0 0 "${VANILLA_FLAGS[@]}"
  bench_run "bench_onereplay_lam${LAMBDA}_seed${SEED}" 0 0 "${ONEREPLAY_FLAGS[@]}"
fi

# ---------------------------------------------------------------------------
# Round B：短跑 + PROFILE=1，取分阶段耗时占比
# ---------------------------------------------------------------------------
if [[ "${RUN_PROFILE}" == "1" ]]; then
  bench_run "bench_vanilla_prof_seed${SEED}" 1 "${PROF_STEPS}" "${VANILLA_FLAGS[@]}"
  bench_run "bench_onereplay_lam${LAMBDA}_prof_seed${SEED}" 1 "${PROF_STEPS}" "${ONEREPLAY_FLAGS[@]}"
fi

# ---------------------------------------------------------------------------
# 可选：OneReplay 的一次性成本。所有 λ 共用同一个 C，这笔钱只付一次。
# ---------------------------------------------------------------------------
if [[ "${BENCH_COV}" == "1" ]]; then
  echo "==== bench: covariance collection (one-time cost) ===="
  COV_BENCH_PATH="${COST_DIR}/cov_bench_${FLAN_SAMPLES}.pt"
  COV_START=$(date +%s)
  python -m onereplay.scripts.collect_cov \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    --data_files "${FLAN_DIR}/train/*.jsonl" --dataset_split train \
    --cache_dir "${DATASET_CACHE}" \
    --use_chat_template 1 --include_target_in_chat 1 \
    --max_samples "${FLAN_SAMPLES}" \
    --sample_shuffle 1 --sample_seed "${SAMPLE_SEED}" \
    --batch_size 2 --max_len "${MAXLEN}" \
    --target_modules "${TARGET_MODULES}" \
    --output_path "${COV_BENCH_PATH}"
  echo "covariance collection wall time: $(( $(date +%s) - COV_START )) s for ${FLAN_SAMPLES} samples"
  ls -l "${COV_BENCH_PATH}"
  rm -f "${COV_BENCH_PATH}"
fi

# ---------------------------------------------------------------------------
# 汇总成表。--extra_metrics_dir 把 09 的 replay 运行一起拉进来（如果已经跑完）。
# ---------------------------------------------------------------------------
echo "==== summary ===="
python -m onereplay.scripts.summarize_cost \
  --metrics_dir "${COST_DIR}" \
  --extra_metrics_dir "${RESULTS_ROOT}/metrics" \
  --baseline "bench_vanilla_seed${SEED}" \
  --out "${COST_DIR}/cost_table.md"

echo "Job finished at: $(date)"
