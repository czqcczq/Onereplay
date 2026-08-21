#!/usr/bin/zsh
#
#SBATCH --job-name=or_eval
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# Stage 3: 一次性评测 base / vanilla / onereplay 三条线
#   指标: 新任务拟合 (commonsense held-out loss) + 指令遵循 (IFEval + Multi-IF English)
# LAMBDA / SEED 必须与 02、03 训练时一致
# 三个指标现在共用一次模型加载（scripts/evaluate.py），Multi-IF 缺数据时自动从指标列表移除
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

# ---- 与训练保持一致 ----
SEED="${SEED:-1}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
MAXLEN="${MAXLEN:-512}"
LAMBDA="${LAMBDA:-1e-4}"

IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
# Multi-IF（第二个指令遵循基准）：默认只测英文子集；MULTIIF_LIMIT=0 表示全量，
# 调试时可设为 20 先快速验证多轮生成/判分。先在有网节点下载 facebook/Multi-IF。
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"

cd "${REPO_ROOT}"
mkdir -p "${RESULTS_ROOT}/map_cache/eval_seed${SEED}"

METRICS="commonsense,ifeval"
if [[ -f "${MULTIIF_INPUT}" ]]; then
  METRICS="${METRICS},multiif"
else
  echo "跳过 Multi-IF：未找到 ${MULTIIF_INPUT}（先在有网节点 git clone facebook/Multi-IF）"
fi

run_eval () {
  local name=$1
  local adapter=$2
  local -a adapter_arg=()
  if [[ -n "${adapter}" ]]; then adapter_arg=(--adapter_path "${adapter}"); fi

  echo "==== eval: ${name} (${METRICS}) ===="
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    "${adapter_arg[@]}" --run_name "${name}" \
    --metrics "${METRICS}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/eval_seed${SEED}" \
    --ifeval_input "${IFEVAL_INPUT}" \
    --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
    --multiif_limit "${MULTIIF_LIMIT}"
}

run_eval "base" ""
run_eval "vanilla_seed${SEED}" "${RESULTS_ROOT}/adapters/cs_vanilla_seed${SEED}"
run_eval "onereplay_lam${LAMBDA}_seed${SEED}" "${RESULTS_ROOT}/adapters/cs_onereplay_lam${LAMBDA}_seed${SEED}"

echo "Job finished at: $(date)"
