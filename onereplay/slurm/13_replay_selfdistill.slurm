#!/usr/bin/zsh
#
#SBATCH --job-name=or_replayself
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# 自蒸馏回放 baseline：生成 target -> 全量微调 -> IFEval/Multi-IF，一个 job 串完
#
# 为什么要有这条路线：11 号 job 的 full_replay_r0.10_seed1 用 FLAN 的 gold target
# 训练，那是一行式的答案键。对一个已经指令微调过的 base 模型来说这不是"复习旧知识"
# 而是"学一个新任务"：起始 loss 约 3.0（新任务只有 0.05），名义 10% 的样本吃掉近
# 80% 的梯度，结果 IFEval 从 0.634 掉到 0.486，42% 的回答陷入重复退化，很多回答只
# 剩下题目要求的那个格式标记而没有正文。
#
# 本脚本把同一批 FLAN prompt 的 target 换成 base 模型自己的贪心回答。回放的初始
# loss 因此接近 0，只起锚定 W0 的作用——这正是 OneReplay 的 C 所近似的东西，两条
# 路线消费的信息才对等，对比才是 apples-to-apples。
#
# 三步：
#   Step 1  生成自蒸馏语料（可断点续跑，语料已完整时会秒退）
#   Step 2  全量微调，超参与 11 号脚本的三件套逐字一致，唯一变量是回放的 target
#   Step 3  IFEval + Multi-IF
#
# 预计耗时：生成 1~3h + 训练 ~2h + 评测 ~2.5h，墙时间给了 16h。
#
# 用法:
#   sbatch onereplay/slurm/13_replay_selfdistill.slurm
#   NUM_SAMPLES=64 SKIP_TRAIN=1 sbatch onereplay/slurm/13_replay_selfdistill.slurm   # 先冒烟看生成质量
#   REPLAY_RATIO=0.05 sbatch onereplay/slurm/13_replay_selfdistill.slurm
#   SKIP_GEN=1 sbatch onereplay/slurm/13_replay_selfdistill.slurm                    # 语料已生成好
#
# 冒烟跑写进的是同一个 jsonl，正式跑会从那 64 行之后继续，不会冲突也不会重复。
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

# ---- 项目路径（与 11 号脚本保持一致）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
FLAN_DIR="${FLAN_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/flan}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"
METRICS_DIR="${METRICS_DIR:-${RESULTS_ROOT}/metrics/full}"
CKPT_DIR="${CKPT_DIR:-${RESULTS_ROOT}/checkpoints/full}"

# ---- 训练超参：必须与 11 号脚本的三件套逐字一致 ----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head}"
REPLAY_RATIO="${REPLAY_RATIO:-0.10}"

# ---- 生成设置：POOL_SIZE / SAMPLE_SEED 必须与 01_collect_cov 一致，否则回放子集
#      就不再嵌套在产生 C 的那个池子里 ----
POOL_SIZE="${POOL_SIZE:-20000}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"          # 0=整个 pool
GEN_BATCH="${GEN_BATCH:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-16}"

SELF_DISTILL_FILE="${SELF_DISTILL_FILE:-${RESULTS_ROOT}/replay/flan_selfdistill_${POOL_SIZE}_seed${SAMPLE_SEED}.jsonl}"
RUN="${RUN:-full_replayself_r${REPLAY_RATIO}_seed${SEED}}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺基座模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -d "${FLAN_DIR}/train" ]] || { echo "缺 FLAN JSONL 目录: ${FLAN_DIR}/train"; exit 1; }
mkdir -p "$(dirname "${SELF_DISTILL_FILE}")" "${METRICS_DIR}" "${CKPT_DIR}" \
  "${DATASET_CACHE}" "${RESULTS_ROOT}/map_cache/${RUN}"

echo "RUN=${RUN} REPLAY_RATIO=${REPLAY_RATIO}"
echo "SELF_DISTILL_FILE=${SELF_DISTILL_FILE}"

# ---------------------------------------------------------------------------
# Step 1: 生成自蒸馏 target
#
# 生成长度按每批最长的 prompt 逐批收缩，保证 prompt+答案+模板一定塞得进 MAXLEN，
# 训练时不会触发左截断。放不下、或者撞上预算没吐出停止符的行会被标 truncated，
# 训练时丢掉——否则就是在教模型"别结束回合"。
# 脚本自带断点续跑：语料已经齐了会直接秒退，requeue 也能接着上次的位置。
# ---------------------------------------------------------------------------
if [[ "${SKIP_GEN}" != "1" ]]; then
  echo "==== generate: ${SELF_DISTILL_FILE} ===="
  python -m onereplay.scripts.generate_replay_targets \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --data_files "${FLAN_DIR}/train/*.jsonl" --split train \
    --cache_dir "${DATASET_CACHE}" \
    --pool_size "${POOL_SIZE}" --sample_seed "${SAMPLE_SEED}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${GEN_BATCH}" --max_len "${MAXLEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --min_new_tokens "${MIN_NEW_TOKENS}" \
    --output_path "${SELF_DISTILL_FILE}" --resume 1
else
  echo "SKIP_GEN=1：直接使用已有的 ${SELF_DISTILL_FILE}"
fi
[[ -f "${SELF_DISTILL_FILE}" ]] || { echo "缺自蒸馏回放语料: ${SELF_DISTILL_FILE}"; exit 1; }

# ---------------------------------------------------------------------------
# Step 2: 训练
# replay_lambda=0 且 measure_replay_when_lambda_zero=0：这是纯 replay 路线，
# 不加 OneReplay 惩罚，也不为了记录而每步去算它。
# ---------------------------------------------------------------------------
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "SKIP_TRAIN=1：到此为止，先看生成质量"
  echo "Job finished at: $(date)"
  exit 0
fi

echo "==== train: ${RUN} ===="
python -m onereplay.scripts.train --paradigm sft --full_finetune 1 \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --replay_lambda 0 --measure_replay_when_lambda_zero 0 \
  --replay_ratio "${REPLAY_RATIO}" \
  --replay_self_distill_file "${SELF_DISTILL_FILE}" \
  --replay_drop_truncated 1 \
  --replay_cache_dir "${DATASET_CACHE}" \
  --save 1 --save_path "${CKPT_DIR}/${RUN}" \
  --metrics_path "${METRICS_DIR}/${RUN}.jsonl"

# ---------------------------------------------------------------------------
# Step 3: retention 评测
# ---------------------------------------------------------------------------
[[ -d "${CKPT_DIR}/${RUN}" ]] || { echo "缺 checkpoint: ${CKPT_DIR}/${RUN}"; exit 1; }

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

echo "Job finished at: $(date)"
