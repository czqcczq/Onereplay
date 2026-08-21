#!/usr/bin/zsh
#
#SBATCH --job-name=or_sweep
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.err
#SBATCH --array=1-6 

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-local} task: ${SLURM_ARRAY_TASK_ID:-0}"

set -euo pipefail
set -x

# =============================================================================
# Stage 2c: OneReplay λ 扫描（训练 + IFEval 评测）
#
# 固定单 seed，遍历一组 replay_lambda，为后续画“新任务 vs retention”权衡曲线
# 产出多个 adapter。每个 λ：先训练保存 adapter，再跑 IFEval。
#   - 新任务轴：验证 loss 由训练脚本写入 metrics/*.jsonl（无需另跑 commonsense 评测）
#   - retention 轴：IFEval + Multi-IF English（均在本脚本内；Multi-IF 缺数据则跳过）
#
# 用法（一次提交整组 λ）:
#   sbatch onereplay/slurm/06_sweep_lambda_train.slurm
#
# 修改 λ 网格：编辑下面的 LAMBDAS 数组，并把 --array 的上界改成 (元素个数 - 1)。
# 例如 5 个 λ  -> #SBATCH --array=0-4
#
# 只想限制并发（例如最多同时 4 个）：
#   sbatch --array=0-6%4 onereplay/slurm/06_sweep_lambda_train.slurm
# =============================================================================

# ---- λ 网格（如需 vanilla 锚点，可自行加入 0；vanilla 已单独训练过则不必重复）----
LAMBDAS=(2e-2 3e-2 5e-2 1e-1)

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

# ---- 训练超参（务必与 vanilla / onereplay / identcov 完全一致，唯一变量是 LAMBDA）----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

# ---- 从 array 任务号选出本次的 λ（注意 zsh 数组是 1-based，array 任务号是 0-based）----
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LAMBDA="${LAMBDAS[$((TASK_ID + 1))]}"
if [[ -z "${LAMBDA}" ]]; then
  echo "错误: array 任务号 ${TASK_ID} 在 LAMBDAS(共 ${#LAMBDAS[@]} 个) 中没有对应的 λ；请让 --array 上界 = 元素个数-1"
  exit 1
fi

RUN="cs_onereplay_lam${LAMBDA}_seed${SEED}"
IFEVAL_INPUT="${CODE}/third_party/instruction_following_eval/data/input_data.jsonl"
# Multi-IF（英文、多轮）：MULTIIF_LIMIT=0 全量，调试可设 20；先下载 facebook/Multi-IF
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"
echo "sweep task ${TASK_ID}: replay_lambda=${LAMBDA} run=${RUN}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || { echo "缺协方差文件(先跑 01_collect_cov): ${COV_PATH}"; exit 1; }
mkdir -p "${RESULTS_ROOT}/adapters" "${RESULTS_ROOT}/metrics" \
  "${RESULTS_ROOT}/map_cache/${RUN}"

# ---------------------------------------------------------------------------
# Step 1: 训练并保存 adapter
# 新任务验证 loss 由训练脚本每个 epoch 写入 metrics/*.jsonl，无需另跑 commonsense 评测。
# ---------------------------------------------------------------------------
echo "==== train: ${RUN} (lambda=${LAMBDA}) ===="
python -m onereplay.scripts.train --paradigm sft \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0 \
  --save 1 --save_path "${RESULTS_ROOT}/adapters/${RUN}" \
  --metrics_path "${RESULTS_ROOT}/metrics/${RUN}.jsonl"

# ---------------------------------------------------------------------------
# Step 2: retention 评测 IFEval + Multi-IF English（一次模型加载跑完）
# ---------------------------------------------------------------------------
METRICS="ifeval"
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
  --ifeval_input "${IFEVAL_INPUT}" \
  --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
  --multiif_limit "${MULTIIF_LIMIT}"

echo "Job finished at: $(date)"
