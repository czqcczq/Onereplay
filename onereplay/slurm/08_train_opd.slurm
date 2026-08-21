#!/usr/bin/zsh
#
#SBATCH --job-name=or_opd
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# Stage 4: On-Policy Distillation (OPD)
#
# 学生（Qwen3-1.7B + LoRA）对 prompt 自采样，教师（Qwen3-8B，冻结）对学生自己的
# 输出逐 token 打分，loss = reverse KL + lambda * tr(dW C dW^T)。
#
# 必跑两条线：
#   LAMBDA=0    vanilla-OPD 基线。教师本身是强 instruct 模型，其输出规范可能
#               自己就保住了指令遵循能力；没有这条线就无法把"教师带来的保持"
#               与"正则带来的保持"分开。
#   LAMBDA>0    OneReplay-OPD 主线。
# 默认一次跑完两条，训练完立即评测。
#
# 注意 LAMBDA=0 这条线默认 MEASURE_REPLAY=1：C 仍会被加载、每步仍会算正则，
# 只是结果乘 0 后不进 loss。这对训练结果没有影响（梯度贡献为 0，且正则是
# 不消耗随机数的确定性运算），但它会付掉全额的正则开销，因此**不能当成本
# 基线**。要测开销请设 MEASURE_REPLAY=0，或直接跑 17_bench_opd_vanilla.slurm。
#
# 教师必须与学生共用同一套 tokenizer/词表（同为 Qwen3 系列），否则逐 token KL 无意义。
# 教师默认放在 GPU 1（TEACHER_GPU=1），学生在 GPU 0。单卡时设 TEACHER_GPU=0。
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
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen3-8B}"
TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-${MODEL_DIR}}"
TEACHER_GPU="${TEACHER_GPU:-1}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"

# ---- 训练超参（两条线唯一差别是 LAMBDA）----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-32}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-200}"          # 训练中途的验证（OPD 下是 KL，会跑 rollout，很贵）
EVAL_MAXVAL="${EVAL_MAXVAL:-1000}"  # 训练后的 commonsense loss，与 SFT 各条线保持同一切分
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
# rollout 越长越贵；先用较短长度把流程跑通再放大
OPD_MAX_NEW_TOKENS="${OPD_MAX_NEW_TOKENS:-128}"
OPD_TEMPERATURE="${OPD_TEMPERATURE:-1.0}"
# OPD 一步要做一次采样 + 两次前向，先限制步数看曲线；0 表示跑满一个 epoch
MAX_STEPS="${MAX_STEPS:-2000}"
# PROFILE=1 分阶段计时。OPD 下 prepare_batch 就是 rollout，通常是最大头，值得先看一眼
PROFILE="${PROFILE:-0}"
SAVE="${SAVE:-1}"
# 训练和评测可以拆成两个 job：IFEval + Multi-IF 全量要跑几个小时，拆开后既容易
# 排队（评测只要 1 张卡），也能在训练结束后先看一眼再决定要不要花评测的时间。
#   第一趟  RUN_EVAL=0 sbatch --time=10:00:00 onereplay/slurm/08_train_opd.slurm
#   第二趟  RUN_TRAIN=0 sbatch --time=12:00:00 --gres=gpu:1 onereplay/slurm/08_train_opd.slurm
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
# 只在 LAMBDA=0 时起作用：1 保留 replay_reg 的观测值（能看出不加正则时权重
# 漂移多大），代价是这条线也付了正则开销；0 才是干净的 vanilla 成本基线。
MEASURE_REPLAY="${MEASURE_REPLAY:-1}"

# 依次训练的 λ；第一个 0 就是必须的 vanilla-OPD 基线，不要删
LAMBDAS=(0 1e-4)

# 冒烟测试：只跑几步、不保存、不评测。OPD 路径涉及学生采样、教师打分、正则三段，
# 第一次跑之前先用这个模式确认三段都通，再压上完整预算。
#   SMOKE=1 sbatch --time=01:00:00 onereplay/slurm/08_train_opd.slurm
# 选 λ>0 而不是 0，是为了让正则那一段也被执行到。
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  LAMBDAS=(1e-4)
  MAX_STEPS=20
  MAXVAL=8
  PROFILE=1
  SAVE=0
  RUN_EVAL=0
  echo "SMOKE=1: 只跑 ${MAX_STEPS} 步，不保存 adapter，不评测"
fi

IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺学生模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${TEACHER_MODEL_DIR}/${TEACHER_MODEL_NAME}" ]] || { echo "缺教师模型: ${TEACHER_MODEL_DIR}/${TEACHER_MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || { echo "缺协方差文件(先跑 01_collect_cov): ${COV_PATH}"; exit 1; }
mkdir -p "${RESULTS_ROOT}/adapters" "${RESULTS_ROOT}/metrics"

METRICS="commonsense,ifeval"
if [[ -f "${MULTIIF_INPUT}" ]]; then
  METRICS="${METRICS},multiif"
else
  echo "跳过 Multi-IF：未找到 ${MULTIIF_INPUT}（先在有网节点 git clone facebook/Multi-IF）"
fi

for LAMBDA in "${LAMBDAS[@]}"; do
  RUN="cs_opd_lam${LAMBDA}_seed${SEED}"
  mkdir -p "${RESULTS_ROOT}/map_cache/${RUN}"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo "==== train OPD: ${RUN} (teacher=${TEACHER_MODEL_NAME}, lambda=${LAMBDA}) ===="
    python -m onereplay.scripts.train --paradigm opd \
      --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
      --teacher_model_dir "${TEACHER_MODEL_DIR}" --teacher_model_name "${TEACHER_MODEL_NAME}" \
      --teacher_gpu "${TEACHER_GPU}" \
      --dataset_path "${CS_DATASET}" \
      --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
      --max_steps "${MAX_STEPS}" --profile "${PROFILE}" \
      --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
      --target_modules "${TARGET_MODULES}" \
      --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
      --opd_max_new_tokens "${OPD_MAX_NEW_TOKENS}" --opd_temperature "${OPD_TEMPERATURE}" \
      --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0 \
      --measure_replay_when_lambda_zero "${MEASURE_REPLAY}" \
      --save "${SAVE}" --save_path "${RESULTS_ROOT}/adapters/${RUN}" \
      --metrics_path "${RESULTS_ROOT}/metrics/${RUN}.jsonl"
  fi

  if [[ "${RUN_EVAL}" != "1" ]]; then
    echo "RUN_EVAL=0：跳过 ${RUN} 的评测"
    continue
  fi
  if [[ ! -d "${RESULTS_ROOT}/adapters/${RUN}" ]]; then
    echo "缺 adapter，跳过评测: ${RESULTS_ROOT}/adapters/${RUN}"
    continue
  fi

  echo "==== eval: ${RUN} (${METRICS}) ===="
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    --adapter_path "${RESULTS_ROOT}/adapters/${RUN}" --run_name "${RUN}" \
    --metrics "${METRICS}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${EVAL_MAXVAL}" --max_len "${MAXLEN}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
    --ifeval_input "${IFEVAL_INPUT}" \
    --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
    --multiif_limit "${MULTIIF_LIMIT}"
done

echo "Job finished at: $(date)"
