#!/usr/bin/zsh
#
#SBATCH --job-name=or_regab
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# 等价性 A/B：--reg_once_per_update 1 是否改变了训练
#
# 背景
# ----
# 正则 tr(ΔW C ΔWᵀ) 只读权重，而一个梯度累积窗口内权重是冻结的，所以旧实现在
# 窗口内的 8 个 micro-batch 上算出的是同一个数，7/8 是白算。修复后每次参数更新
# 只算一次、且不参与 /accumulation_steps 的除法，累积梯度不变：
#
#   旧  Σ_i ∇(L_i/A + λR/A) = mean_i ∇L_i + λ∇R
#   新  ∇(L_1/A + λR) + Σ_{i>1} ∇(L_i/A) = 同一个值
#
# 算法层面这件事已经由 scripts/verify_reg_once_per_update.py 在 float64 上直接
# 比对累积 .grad 证明了（最大差 1.1e-16，正则调用 24→3 次）。
#
# 本脚本回答的是另一个问题：bf16 + 真实模型 + 真实数据下，那点舍入差异会不会在
# 上千步里累积成轨迹分岔。如果不会，已有的 IFEval / Multi-IF / val loss 就能直接
# 沿用，不必重跑整个 λ 扫描（那是几十个 GPU 小时）。
#
# 为什么是三条而不是两条
# --------------------
#   A1  旧行为
#   A2  旧行为，与 A1 逐字相同 —— 用来量出 run-to-run 的噪声底
#   B   新行为
#
# 没有 A2，看到 1e-6 的差异无法判断它来自这次修改还是来自 GPU 本身的非确定性
# （cuBLAS/cuDNN 的 atomic 累加顺序默认不保证位级可复现）。判据是
# |B−A1| 与 |A2−A1| 同量级。顺带一提，OPD 那两条 run 的 train_task_loss 曾逐位
# 相同（1.226359612776665），所以这里 A2−A1 很可能就是 0。
#
# 几个参数上的讲究
# --------------
#   λ=3e-2      λ 扫描里 retention 最好的一档，也是最大的一档。λ 越大正则对梯度
#               的贡献越大，差异越容易暴露；用 λ=0 跑这个检查毫无意义，那样正则
#               根本不进梯度，两边必然完全一样。
#   STEPS=800   必须是 accumulation_steps=8 的整数倍。这样每个窗口都是满的，
#               避开"epoch 末尾不满窗口的正则强度从 3/8 变成 1"那个**预期内**的
#               行为差异，不让它混进判据。800 步 = 100 次参数更新。
#   LOG_EVERY=1 逐步打印 task_loss，才能定位第一次分岔发生在哪一步。
#   SAVE=0      不写 adapter，这里只要 loss 轨迹。
#
# 三条各约 75 s 训练，主要开销是三次模型加载与 tokenize，一小时内跑完。
#
# 用法:
#   sbatch onereplay/slurm/23_check_reg_equivalence.slurm
#   STEPS=1600 sbatch onereplay/slurm/23_check_reg_equivalence.slurm   # 想跑更久
#
# 跑完看末尾自动打印的比对表，判据写在 scripts/compare_reg_equivalence.py 里。
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
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"

# ---- 与 06_sweep_lambda_train.slurm 逐字一致 ----
SEED="${SEED:-1}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
LAMBDA="${LAMBDA:-3e-2}"

STEPS="${STEPS:-800}"
LOG_EVERY="${LOG_EVERY:-1}"

OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/regcheck/${SLURM_JOB_ID:-local}}"
# 三条共用一个 map_cache：数据、tokenizer、max_len 完全相同，重算没有意义
MAP_CACHE="${RESULTS_ROOT}/map_cache/regcheck"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || { echo "缺协方差文件: ${COV_PATH}"; exit 1; }
mkdir -p "${OUT_DIR}" "${MAP_CACHE}"

accum_steps=$(( ACCUM / BATCH ))
if (( STEPS % accum_steps != 0 )); then
  echo "错误: STEPS=${STEPS} 不是 accumulation_steps=${accum_steps} 的整数倍。"
  echo "末尾会出现不满的窗口，其正则强度在新旧实现下本就不同，会污染判据。"
  exit 1
fi
echo "accumulation_steps=${accum_steps}, STEPS=${STEPS} -> $(( STEPS / accum_steps )) 次参数更新"

# ---------------------------------------------------------------------------
# 一条短跑。$1=标签 $2=reg_once_per_update
# 每条独立的 metrics 文件（全精度 train_task_loss / val_loss）和独立的日志
# （逐步 task_loss，精度受打印格式限制，用来定位首次分岔）。
# ---------------------------------------------------------------------------
ab_run() {
  local tag="$1"
  local reg_once="$2"
  echo "==== ${tag}: reg_once_per_update=${reg_once} ===="
  python -m onereplay.scripts.train --paradigm sft \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --epochs 1 --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TARGET_MODULES}" \
    --map_cache_dir "${MAP_CACHE}" \
    --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0 \
    --reg_once_per_update "${reg_once}" \
    --max_steps "${STEPS}" --log_every "${LOG_EVERY}" \
    --save 0 \
    --metrics_path "${OUT_DIR}/${tag}.jsonl" 2>&1 | tee "${OUT_DIR}/${tag}.log"
}

ab_run A1 0
ab_run A2 0
ab_run B 1

# ---------------------------------------------------------------------------
# 比对
# ---------------------------------------------------------------------------
echo "==== compare ===="
python -m onereplay.scripts.compare_reg_equivalence \
  --old "${OUT_DIR}/A1" --repeat "${OUT_DIR}/A2" --new "${OUT_DIR}/B"

echo "结果目录: ${OUT_DIR}"
echo "Job finished at: $(date)"
