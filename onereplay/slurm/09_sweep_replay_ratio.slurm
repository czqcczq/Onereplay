#!/usr/bin/zsh
#
#SBATCH --job-name=or_replay
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.err
#SBATCH --array=0-2

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-local} task: ${SLURM_ARRAY_TASK_ID:-0}"

set -euo pipefail
set -x

# =============================================================================
# Stage 2d: LoRA 下的 replay 基线（回放比例 r 扫描：训练 + IFEval + Multi-IF）
#
# 真 replay：保留旧知识语料，按 r 的比例把旧知识样本追加进 Commonsense 训练集，
# 用同一个交叉熵训练，不加任何协方差正则（replay_lambda=0，且不加载 C）。
# 这是 OneReplay 的对照组——我们的方法是它的 data-free 二阶近似。
#
# 【本脚本已改为自蒸馏回放】
# 旧版本用 FLAN 的 gold target 训练，那是一行式的答案键。对一个已经指令微调过的
# base 模型来说这不是"复习旧知识"而是"学一个新任务"：起始 loss 约 3.0（新任务只
# 有 0.05），少量样本就吃掉大部分梯度，把输出风格拽成 FLAN 的答案键。全量微调下
# 实测 IFEval 从 0.634 掉到 0.486、42% 的回答重复退化；LoRA 下 r=0.01 就掉到
# 0.470，r=0.10 掉到 0.397。那不是"replay 保留得差"，是模型被训坏了。
#
# 现在改用 base 模型自己对同一批 FLAN prompt 的回答作为 target，回放的初始 loss
# 接近 0，只起锚定 W0 的作用——这正是 OneReplay 的 C 所近似的东西，两条路线消费
# 的信息才对等。全量微调下同样的改动让 IFEval 回到 0.653、Multi-IF 回到 0.499
# （base 分别是 0.665 / 0.497）。
#
# 旧的 gold-target 结果仍保留在 cs_replay_r* 下，可作为消融；本脚本产出的是
# cs_replayself_r*，两者不会互相覆盖。
#
# 公平性要点：
#   1. 自蒸馏语料的 pool_size / sample_seed 与 01_collect_cov 一致，回放子集因此
#      嵌套在估计 C 的那 20k 池子里，两条路线消费的旧知识预算相同。
#   2. 除 r 外所有超参与 02/06 逐字相同。
#   3. 追加式混合：新任务样本数不变，epoch 时长涨 (1+r) 倍——这正是要暴露的代价。
#   4. 验证集只用 Commonsense，val_loss 与 λ 扫描那一列可比。
#
# 前置：自蒸馏语料必须已经生成。它只依赖 base 模型，与 LoRA / 全量无关，所以
# 13_replay_selfdistill.slurm 为全量实验生成的那份可以直接复用：
#   sbatch onereplay/slurm/13_replay_selfdistill.slurm     # 若还没生成过
# 这里不内联生成——array 的多个任务会并发写同一个 jsonl，把文件写坏。
#
# 用法:
#   sbatch onereplay/slurm/09_sweep_replay_ratio.slurm
# 只跑单个比例（例如 5%，即数组第 2 个元素）:
#   sbatch --array=1 onereplay/slurm/09_sweep_replay_ratio.slurm
# 限制并发:
#   sbatch --array=0-2%2 onereplay/slurm/09_sweep_replay_ratio.slurm
# =============================================================================

# ---- 回放比例网格。改这里时记得同步 --array 的上界 = 元素个数 - 1 ----
RATIOS=(0.01 0.05 0.10)

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
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"

# ---- 训练超参（务必与 vanilla / onereplay 完全一致，唯一变量是 RATIO）----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

# ---- 自蒸馏回放语料。POOL_SIZE / SAMPLE_SEED 只用来拼默认文件名，必须与
#      13_replay_selfdistill.slurm 生成时、以及 01_collect_cov 的取值一致 ----
POOL_SIZE="${POOL_SIZE:-20000}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"
SELF_DISTILL_FILE="${SELF_DISTILL_FILE:-${RESULTS_ROOT}/replay/flan_selfdistill_${POOL_SIZE}_seed${SAMPLE_SEED}.jsonl}"

# ---- 开销测量：短跑用 PROFILE=1 MAX_STEPS=200 SAVE=0 ----
PROFILE="${PROFILE:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
SAVE="${SAVE:-1}"

# ---- 从 array 任务号选比例（zsh 数组 1-based，array 任务号 0-based）----
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RATIO="${RATIOS[$((TASK_ID + 1))]:-}"
if [[ -z "${RATIO}" ]]; then
  echo "错误: array 任务号 ${TASK_ID} 超出 RATIOS(共 ${#RATIOS[@]} 个)；--array 上界应为 元素个数-1"
  exit 1
fi

RUN="cs_replayself_r${RATIO}_seed${SEED}"
IFEVAL_INPUT="${CODE}/third_party/instruction_following_eval/data/input_data.jsonl"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"
echo "replay sweep task ${TASK_ID}: replay_ratio=${RATIO} run=${RUN}"
echo "SELF_DISTILL_FILE=${SELF_DISTILL_FILE}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${SELF_DISTILL_FILE}" ]] || {
  echo "缺自蒸馏回放语料: ${SELF_DISTILL_FILE}"
  echo "先跑: sbatch onereplay/slurm/13_replay_selfdistill.slurm"
  echo "（该语料只依赖 base 模型，全量实验生成的那份可直接复用）"
  exit 1
}
mkdir -p "${RESULTS_ROOT}/adapters" "${RESULTS_ROOT}/metrics" \
  "${RESULTS_ROOT}/map_cache/${RUN}" "${DATASET_CACHE}"

# ---------------------------------------------------------------------------
# Step 1: 训练并保存 adapter
# replay_lambda=0 且 measure_replay_when_lambda_zero=0：不加载 C，
# 每步不算正则，训练时间才是这条基线的真实开销。
# ---------------------------------------------------------------------------
echo "==== train: ${RUN} (replay_ratio=${RATIO}) ===="
python -m onereplay.scripts.train --paradigm sft \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --profile "${PROFILE}" --max_steps "${MAX_STEPS}" \
  --replay_ratio "${RATIO}" \
  --replay_self_distill_file "${SELF_DISTILL_FILE}" \
  --replay_drop_truncated 1 \
  --replay_cache_dir "${DATASET_CACHE}" \
  --replay_lambda 0 --measure_replay_when_lambda_zero 0 \
  --save "${SAVE}" --save_path "${RESULTS_ROOT}/adapters/${RUN}" \
  --metrics_path "${RESULTS_ROOT}/metrics/${RUN}.jsonl"

# ---------------------------------------------------------------------------
# Step 2: retention 评测 IFEval + Multi-IF English
# ---------------------------------------------------------------------------
if [[ "${SAVE}" != "1" ]]; then
  echo "SAVE=0：这是开销测量短跑，跳过评测"
  echo "Job finished at: $(date)"
  exit 0
fi

METRICS="ifeval"
if [[ -f "${MULTIIF_INPUT}" ]]; then
  METRICS="${METRICS},multiif"
else
  echo "跳过 Multi-IF：未找到 ${MULTIIF_INPUT}"
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
