#!/usr/bin/zsh
#
#SBATCH --job-name=or_cov
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
# Stage 1: 从本地 FLAN(chat 模板) 采集旧知识协方差 C = E[x x^T]
# 计算节点无外网；FLAN 已 save_to_disk 到 FLAN_DIR，运行时按 seed 随机抽 max_samples
#
# COV_SCOPE 决定覆盖哪些层，其余参数（采样种子、样本数、chat 模板）两种 scope 完全
# 相同，这样 LoRA 与全量实验消费的是同一批旧知识：
#
#   qv    q_proj,v_proj —— 配合 LoRA 训练，只有这两类层有非零 ΔW
#   full  八类线性投影层 —— 配合全量微调，所有线性层都在动，覆盖必须跟上
#
# 全量下 embed_tokens 不会被选中（它是 nn.Embedding 不是 nn.Linear），tied 权重
# 因此只通过 lm_head 被约束一次，无需额外去重。
#
# 用法:
#   sbatch onereplay/slurm/01_collect_cov.slurm                    # 默认 qv
#   COV_SCOPE=full sbatch onereplay/slurm/01_collect_cov.slurm     # 全量八类层
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
# 离线：禁止任何 HuggingFace 联网调用
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Python          : $(which python)"
python --version

# ---- 项目路径（按需覆盖）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"        # 含 onereplay/ 与 process_dataset/ 的目录
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
FLAN_DIR="${FLAN_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/flan}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"
MAXLEN="${MAXLEN:-512}"

# ---- 覆盖范围预设。显式给 TARGET_MODULES / COV_PATH 可覆盖预设值 ----
COV_SCOPE="${COV_SCOPE:-full}"
case "${COV_SCOPE}" in
  qv)
    TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
    COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
    ;;
  full)
    TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head}"
    COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_full.pt}"
    ;;
  *)
    echo "未知 COV_SCOPE=${COV_SCOPE}，可选 qv / full"; exit 1
    ;;
esac
echo "COV_SCOPE=${COV_SCOPE} TARGET_MODULES=${TARGET_MODULES}"

FLAN_SAMPLES="${FLAN_SAMPLES:-20000}"
SAMPLE_SHUFFLE="${SAMPLE_SHUFFLE:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${FLAN_DIR}/train" ]] || { echo "缺 FLAN JSONL 目录: ${FLAN_DIR}/train"; exit 1; }
mkdir -p "$(dirname "${COV_PATH}")" "${DATASET_CACHE}"

python -m onereplay.scripts.collect_cov \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
  --data_files "${FLAN_DIR}/train/*.jsonl" --dataset_split train \
  --cache_dir "${DATASET_CACHE}" \
  --use_chat_template 1 --include_target_in_chat 1 \
  --max_samples "${FLAN_SAMPLES}" \
  --sample_shuffle "${SAMPLE_SHUFFLE}" --sample_seed "${SAMPLE_SEED}" \
  --batch_size 2 --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --output_path "${COV_PATH}"

echo "Job finished at: $(date)"
