#!/usr/bin/zsh
#
#SBATCH --job-name=or_full_replay_eval
#SBATCH --partition=c23g
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
# 补测脚本：只评测 11_train_full_baselines.slurm 训出来的 full_replay_r0.10_seed1。
#
# 背景：11 号 job（2690387）24h 墙时间用完时，三件套已全部训完、base / vanilla /
# onereplay 也都评完，full_replay 的 IFEval 也已写盘，只有最后的 Multi-IF 被截断。
# 所以本脚本默认 METRICS=multiif，只补那一项；要重跑 IFEval 就传 METRICS=ifeval,multiif。
#
# checkpoint（全量微调，整个目录就是模型，不是 LoRA adapter）：
#   ${RESULTS_ROOT}/checkpoints/full/full_replay_r0.10_seed1
#
# 用法:
#   sbatch onereplay/slurm/12_eval_full_replay.slurm
#   METRICS=ifeval,multiif sbatch onereplay/slurm/12_eval_full_replay.slurm
#   MULTIIF_LIMIT=20 sbatch onereplay/slurm/12_eval_full_replay.slurm   # 小样本调试
#   RUN_COST=1 sbatch onereplay/slurm/12_eval_full_replay.slurm         # 顺带补开销表
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

# ---- 项目路径（与 11_train_full_baselines.slurm 保持一致）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
METRICS_DIR="${METRICS_DIR:-${RESULTS_ROOT}/metrics/full}"
CKPT_DIR="${CKPT_DIR:-${RESULTS_ROOT}/checkpoints/full}"

# ---- 目标 run ----
SEED="${SEED:-1}"
REPLAY_RATIO="${REPLAY_RATIO:-0.10}"
RUN_NAME="${RUN_NAME:-full_replay_r${REPLAY_RATIO}_seed${SEED}}"
CKPT_PATH="${CKPT_PATH:-${CKPT_DIR}/${RUN_NAME}}"

# ---- 评测设置 ----
METRICS="${METRICS:-multiif}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
MAXLEN="${MAXLEN:-512}"
IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"
RUN_COST="${RUN_COST:-0}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺基座模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CKPT_PATH}" ]] || {
  echo "缺 checkpoint: ${CKPT_PATH}"
  echo "现有 full checkpoint:"
  ls -1 "${CKPT_DIR}" 2>/dev/null || echo "  (${CKPT_DIR} 不存在)"
  exit 1
}
if [[ "${METRICS}" == *multiif* ]]; then
  [[ -f "${MULTIIF_INPUT}" ]] || { echo "缺 Multi-IF 数据: ${MULTIIF_INPUT}"; exit 1; }
fi
if [[ "${METRICS}" == *ifeval* ]]; then
  [[ -f "${IFEVAL_INPUT}" ]] || { echo "缺 IFEval 数据: ${IFEVAL_INPUT}"; exit 1; }
fi

echo "==== eval: ${RUN_NAME} (${METRICS}) ===="
echo "checkpoint: ${CKPT_PATH}"

python -m onereplay.scripts.evaluate \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
  --adapter_path "${CKPT_PATH}" --run_name "${RUN_NAME}" \
  --metrics "${METRICS}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/eval_full_seed${SEED}" \
  --ifeval_input "${IFEVAL_INPUT}" \
  --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
  --multiif_limit "${MULTIIF_LIMIT}"

# 11 号 job 被截断在评测阶段，开销表也没生成；需要时用 RUN_COST=1 一起补上
if [[ "${RUN_COST}" == "1" ]]; then
  echo "==== cost summary (epoch 1, from training metrics) ===="
  python -m onereplay.scripts.summarize_cost \
    --metrics_dir "${METRICS_DIR}" \
    --baseline "full_vanilla_seed${SEED}" \
    --epoch 1 \
    --out "${METRICS_DIR}/cost_table_full.md"
fi

echo "Job finished at: $(date)"
