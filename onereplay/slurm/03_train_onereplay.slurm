#!/usr/bin/zsh
#
#SBATCH --job-name=or_main
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# Stage 2b: OneReplay 主实验 (replay_lambda>0)
# 验证阶段固定一个 lambda：直接提交本脚本 (默认 LAMBDA=1e-4)
# 后续扫参：for L in 1e-5 1e-4 1e-3 1e-2; do
#            sbatch --export=ALL,LAMBDA=$L onereplay/slurm/03_train_onereplay.sbatch
#          done
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

# ---- 项目路径（按需覆盖）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"

# ---- 训练超参（与 vanilla 完全一致，唯一变量是 LAMBDA）----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
LAMBDA="${LAMBDA:-1e-4}"
# IDENTITY_COV=1 用单位阵 C 做 L2 对照 (C=I 消融)；此时用不同的 run 标签避免覆盖真实 C 的结果。
IDENTITY_COV="${IDENTITY_COV:-0}"
# 开销测量：PROFILE=1 打开分阶段计时（多花几个百分点），MAX_STEPS 限制步数做短跑
#   PROFILE=1 MAX_STEPS=200 SAVE=0 sbatch onereplay/slurm/03_train_onereplay.slurm
#   与 02 的同参数短跑对比，即可得到正则的时间与显存增量
PROFILE="${PROFILE:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
SAVE="${SAVE:-1}"
if [[ "${IDENTITY_COV}" == "1" ]]; then TAG="identcov"; else TAG="onereplay"; fi
RUN="cs_${TAG}_lam${LAMBDA}_seed${SEED}"

echo "replay_lambda=${LAMBDA} identity_cov=${IDENTITY_COV} run=${RUN}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || { echo "缺协方差文件(先跑 01_collect_cov): ${COV_PATH}"; exit 1; }
mkdir -p "${RESULTS_ROOT}/adapters" "${RESULTS_ROOT}/metrics" \
  "${RESULTS_ROOT}/map_cache/${RUN}"

python -m onereplay.scripts.train --paradigm sft \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --profile "${PROFILE}" --max_steps "${MAX_STEPS}" \
  --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov "${IDENTITY_COV}" \
  --save "${SAVE}" --save_path "${RESULTS_ROOT}/adapters/${RUN}" \
  --metrics_path "${RESULTS_ROOT}/metrics/${RUN}.jsonl"

echo "Job finished at: $(date)"
