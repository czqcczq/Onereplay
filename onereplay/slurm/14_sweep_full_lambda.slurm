#!/usr/bin/zsh
#
#SBATCH --job-name=or_fullsweep
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.err
#SBATCH --array=0-3

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-local} task: ${SLURM_ARRAY_TASK_ID:-0}"

set -euo pipefail
set -x

# =============================================================================
# 全量微调下的 OneReplay λ 扫描（训练 + IFEval/Multi-IF）
#
# 为什么要重扫：11 号 job 用的 λ=1e-4 在全量下惩罚项只有 1.15e-4，占 task loss
# (0.104) 的 0.11%，三个 epoch 的 train loss 与 vanilla 一致到小数点后四位，
# IFEval 0.6377 vs 0.6340 完全在噪声内——那一轮的正则实际上没有生效。
#
# 网格按 calibrate 的换算取：λ=1e-3 对应 λ·reg ≈ 1% 的 task loss，1e-2 对应
# 约 10%，3e-2 用来找到开始伤新任务的拐点。
#
# 成本提醒：全量下正则每步 561ms vs vanilla 96ms，一个 3-epoch 训练约 10 小时，
# 加评测约 12.5 小时，四个 λ 就是四个 GPU-day。想先便宜地定位区间，用
# EPOCHS=1 跑一轮（约 4.5 小时/λ），看 val_loss 和 IFEval 的走向再决定哪个 λ
# 值得跑满 3 个 epoch。
#
# 用法:
#   sbatch onereplay/slurm/14_sweep_full_lambda.slurm
#   sbatch --array=0-3%2 onereplay/slurm/14_sweep_full_lambda.slurm   # 限并发 2
#   EPOCHS=1 sbatch onereplay/slurm/14_sweep_full_lambda.slurm        # 便宜的定位轮
#   KEEP_CKPT=0 sbatch onereplay/slurm/14_sweep_full_lambda.slurm     # 评完删 checkpoint
#
# 改网格：编辑 LAMBDAS，并把 --array 上界改成 (元素个数 - 1)。
# =============================================================================

LAMBDAS=(1e-3 3e-3 1e-2 3e-2)

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

# ---- 项目路径（与 11 号脚本保持一致）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_full.pt}"
METRICS_DIR="${METRICS_DIR:-${RESULTS_ROOT}/metrics/full}"
CKPT_DIR="${CKPT_DIR:-${RESULTS_ROOT}/checkpoints/full}"

# ---- 超参：唯一的变量是 LAMBDA，其余逐字照抄 11 号脚本 ----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head}"
KEEP_CKPT="${KEEP_CKPT:-1}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LAMBDA="${LAMBDAS[$((TASK_ID + 1))]}"    # zsh 数组是 1-based，array 任务号是 0-based
if [[ -z "${LAMBDA}" ]]; then
  echo "错误: array 任务号 ${TASK_ID} 在 LAMBDAS(共 ${#LAMBDAS[@]} 个) 中没有对应的 λ"
  exit 1
fi

RUN="full_onereplay_lam${LAMBDA}_seed${SEED}"
if [[ "${EPOCHS}" != "3" ]]; then
  # 便宜的定位轮不能和正式 3-epoch 结果写进同一个名字，否则表会串。
  RUN="${RUN}_ep${EPOCHS}"
fi

IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺基座模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || {
  echo "缺全量协方差文件: ${COV_PATH}"
  echo "先跑: COV_SCOPE=full sbatch onereplay/slurm/01_collect_cov.slurm"
  exit 1
}
mkdir -p "${METRICS_DIR}" "${CKPT_DIR}" "${RESULTS_ROOT}/map_cache/${RUN}"

echo "sweep task ${TASK_ID}: lambda=${LAMBDA} epochs=${EPOCHS} run=${RUN}"

echo "==== train: ${RUN} (lambda=${LAMBDA}) ===="
python -m onereplay.scripts.train --paradigm sft --full_finetune 1 \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0 \
  --save 1 --save_path "${CKPT_DIR}/${RUN}" \
  --metrics_path "${METRICS_DIR}/${RUN}.jsonl"

METRICS="ifeval"
if [[ -f "${MULTIIF_INPUT}" ]]; then
  METRICS="${METRICS},multiif"
else
  echo "跳过 Multi-IF：未找到 ${MULTIIF_INPUT}"
fi

echo "==== eval: ${RUN} (${METRICS}) ===="
python -m onereplay.scripts.evaluate \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
  --adapter_path "${CKPT_DIR}/${RUN}" --run_name "${RUN}" \
  --metrics "${METRICS}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/eval_full_seed${SEED}" \
  --ifeval_input "${IFEVAL_INPUT}" \
  --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
  --multiif_limit "${MULTIIF_LIMIT}"

# 每个全量 checkpoint 约 3.4 GiB，四个 λ 就是 13.6 GiB。指标已经落盘，
# 不打算再复用权重时可以让 job 自己清掉。
if [[ "${KEEP_CKPT}" != "1" ]]; then
  echo "KEEP_CKPT=0：删除 ${CKPT_DIR}/${RUN}"
  rm -rf "${CKPT_DIR:?}/${RUN:?}"
fi

echo "Job finished at: $(date)"
