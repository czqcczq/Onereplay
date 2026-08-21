#!/usr/bin/zsh
#
#SBATCH --job-name=or_safety_judge
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# 问题 1 - 安全性评测（只重判分，不重生成）
#   28_safety_end2end.slurm 已把 responses.jsonl 落盘；本作业只用 WildGuard 重跑
#   判分。适合修了 judge 逻辑（如长 prompt 截断、max_new_tokens）后重算标签，
#   省掉约 7h 的重复生成。判分脚本会覆盖 JUDGE_OUT 下同名 run 的结果。
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

REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"

JUDGE_MODEL_DIR="${JUDGE_MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-wildguard}"
JUDGE_BATCH="${JUDGE_BATCH:-16}"

# 默认判所有 run 的 safety 生成结果
RESPONSES="${RESPONSES:-${RESULTS_ROOT}/safety/*/responses.jsonl}"
JUDGE_OUT="${JUDGE_OUT:-${RESULTS_ROOT}/safety_judged}"

cd "${REPO_ROOT}"
[[ -d "${JUDGE_MODEL_DIR}/${JUDGE_MODEL_NAME}" ]] || {
  echo "缺 judge 模型: ${JUDGE_MODEL_DIR}/${JUDGE_MODEL_NAME}"; exit 1;
}

python -m onereplay.scripts.judge_safety \
  --responses "${RESPONSES}" \
  --judge_model_dir "${JUDGE_MODEL_DIR}" --judge_model_name "${JUDGE_MODEL_NAME}" \
  --out_dir "${JUDGE_OUT}" \
  --batch_size "${JUDGE_BATCH}"

echo "Job finished at: $(date)"
