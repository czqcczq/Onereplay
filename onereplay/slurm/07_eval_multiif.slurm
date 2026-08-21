#!/usr/bin/zsh
#
#SBATCH --job-name=or_multiif
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
# 一次性脚本：只对已训练好的模型补测 Multi-IF（英文、多轮）。
# 用于给之前没测 Multi-IF 的 adapter（vanilla / identcov / onereplay 各 λ）
# 以及 base 补上这个 retention 指标。新任务 loss / IFEval 已由其它脚本产出。
#
# 前置：先在有网的登录节点下载数据，例如：
#   cd /home/xsz96350/K-RES/onereplay/third_party
#   git clone https://huggingface.co/datasets/facebook/Multi-IF
#
# 用法：sbatch onereplay/slurm/07_eval_multiif.slurm
# 想改测哪些模型：编辑下面的 ADAPTERS 数组（用 adapters/ 下的目录名）。
# 调试可先设 MULTIIF_LIMIT=20 跑小样本。
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
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"

# ---- Multi-IF 设置 ----
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"     # 0=全量；调试可设 20

# ---- 要补测的 adapter（adapters/ 下的目录名），base 单独处理 ----
ADAPTERS=(
  cs_vanilla_seed1
  cs_identcov_lam1e-4_seed1
  cs_onereplay_lam1e-4_seed1
  cs_onereplay_lam1e-5_seed1
)

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -f "${MULTIIF_INPUT}" ]] || {
  echo "缺 Multi-IF 数据: ${MULTIIF_INPUT}"
  echo "请先在有网节点: cd ${CODE}/third_party && git clone https://huggingface.co/datasets/facebook/Multi-IF"
  exit 1
}
mkdir -p "${RESULTS_ROOT}/multiif"

run_multiif () {
  local name=$1
  local adapter=$2
  local -a adapter_arg=()
  if [[ -n "${adapter}" ]]; then adapter_arg=(--adapter_path "${adapter}"); fi

  echo "==== Multi-IF: ${name} ===="
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    "${adapter_arg[@]}" --run_name "${name}" \
    --metrics multiif --out_dir "${RESULTS_ROOT}" \
    --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
    --multiif_limit "${MULTIIF_LIMIT}"
}

# base 参考（retention 基线，务必测）
run_multiif "base" ""

# 各 adapter：用目录名作为 run_name，输出目录与 adapter 同名，便于对照
for adapter_name in "${ADAPTERS[@]}"; do
  adapter_path="${RESULTS_ROOT}/adapters/${adapter_name}"
  if [[ -d "${adapter_path}" ]]; then
    run_multiif "${adapter_name}" "${adapter_path}"
  else
    echo "跳过 ${adapter_name}：未找到 ${adapter_path}"
  fi
done

echo "Job finished at: $(date)"
