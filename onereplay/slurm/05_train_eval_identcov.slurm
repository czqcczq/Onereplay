#!/usr/bin/zsh
#
#SBATCH --job-name=or_identcov
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# C=I 消融对照：一个自包含任务，先训练 identity_cov 版，再只评测该 adapter。
# 正则退化为 tr(DeltaW DeltaW^T) = ||DeltaW||_F^2（纯 L2 收缩），用于证明
# OneReplay 的收益来自协方差结构，而非任意收缩 DeltaW。
# 其余 base / vanilla / onereplay 的结果已有，本脚本只处理 identcov。
# 注意：训练超参必须与你已有的 vanilla / onereplay adapter 完全一致才公平。
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

# ---- 项目路径 ----
REPO_ROOT=/home/xsz96350/K-RES
CODE=onereplay
MODEL_DIR=/hpcwork/xsz96350/Chen_logs/checkpoints/
MODEL_NAME=Qwen3-1.7B
CS_DATASET=/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf
RESULTS_ROOT=/hpcwork/xsz96350/Chen_logs/onereplay/results
COV_PATH=${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt

# ---- 训练超参（务必与 vanilla / onereplay 一致）----
SEED=1
EPOCHS=3
LR=1e-4
BATCH=8
ACCUM=64
MAXLEN=512
VAL_FRAC=0.01
MAXVAL=1000
TARGET_MODULES=q_proj,v_proj
LAMBDA=1e-4          # 与真实 C 的主实验用同一个 lambda；如需匹配强度可另调

RUN=cs_identcov_lam${LAMBDA}_seed${SEED}
IFEVAL_INPUT=${CODE}/third_party/instruction_following_eval/data/input_data.jsonl
# Multi-IF（英文、多轮）：MULTIIF_LIMIT=0 全量，调试可设 20；先下载 facebook/Multi-IF
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || { echo "缺协方差文件(先跑 01_collect_cov): ${COV_PATH}"; exit 1; }
mkdir -p "${RESULTS_ROOT}/adapters" "${RESULTS_ROOT}/metrics" \
  "${RESULTS_ROOT}/map_cache/${RUN}"

# ---------------------------------------------------------------------------
# Step 1: 训练 C=I 对照 (identity_cov=1)
# ---------------------------------------------------------------------------
echo "==== train: ${RUN} (identity_cov=1, lambda=${LAMBDA}) ===="
python -m onereplay.scripts.train --paradigm sft \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 1 \
  --save 1 --save_path "${RESULTS_ROOT}/adapters/${RUN}" \
  --metrics_path "${RESULTS_ROOT}/metrics/${RUN}.jsonl"

# ---------------------------------------------------------------------------
# Step 2: 评测该 identcov adapter，三个指标共用一次模型加载
# ---------------------------------------------------------------------------
METRICS="commonsense,ifeval"
if [[ -f "${MULTIIF_INPUT}" ]]; then
  METRICS="${METRICS},multiif"
else
  echo "跳过 Multi-IF：未找到 ${MULTIIF_INPUT}（先在有网节点 git clone facebook/Multi-IF）"
fi

echo "==== eval: ${RUN} (${METRICS}) ===="
python -m onereplay.scripts.evaluate \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
  --adapter_path "${RESULTS_ROOT}/adapters/${RUN}" --run_name "${RUN}" \
  --metrics "${METRICS}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --ifeval_input "${IFEVAL_INPUT}" \
  --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
  --multiif_limit "${MULTIIF_LIMIT}"

echo "Job finished at: $(date)"
