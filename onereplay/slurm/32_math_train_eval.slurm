#!/usr/bin/zsh
#
#SBATCH --job-name=or_mathtre
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
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
# 问题2 训练+评测（一个 job 串完）: 训 C_math 与 C_mix 两个 run -> 评测全部模型
#
#   DO_TRAIN=1  依次训 TRAIN_KINDS（默认 "math mix"）:
#       math -> C_math -> run cs_onereplay_math_lam${LAMBDA}_seed${SEED}_regonce
#       mix  -> C_mix  -> run cs_onereplay_ifmath_lam${LAMBDA}_seed${SEED}_regonce
#     超参与 C_if 生产 run 逐字一致（LoRA qv/lr1e-4/bs8/accum64/ep3/max_len512/reg_once），
#     唯一变量是 --cov_path，四个 OneReplay run 之间只差 C，评测才是受控对照。
#     已存在的 adapter 会跳过（除非 FORCE_TRAIN=1），方便 requeue/分批。
#
#   DO_EVAL=1   对 MODELS（默认 base vanilla cif cmath cmix）跑数学 retention:
#       math500(主) / amc / aime / gsm8k(可选)；缺哪个数据文件就跳过哪个指标。
#     WITH_IF=1 时额外补 ifeval,(multiif),commonsense 给新 run 填交叉对照表的 IF 侧。
#
# 分阶段（训练很久时建议拆两次提交）:
#   DO_EVAL=0 sbatch onereplay/slurm/32_math_train_eval.slurm     # 只训练
#   DO_TRAIN=0 sbatch onereplay/slurm/32_math_train_eval.slurm    # 只评测
#
# 常用:
#   sbatch onereplay/slurm/32_math_train_eval.slurm                       # 训两个 run + 评全部
#   TRAIN_KINDS=mix sbatch onereplay/slurm/32_math_train_eval.slurm       # 只训 C_mix
#   WITH_IF=1 MODELS="cmath cmix" sbatch onereplay/slurm/32_math_train_eval.slurm
#   LAMBDA=2e-2 sbatch onereplay/slurm/32_math_train_eval.slurm           # 用 31 号 trace 建议的 λ
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

# ---- 项目路径 ----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
MATH_DATA_DIR="${MATH_DATA_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/math}"
POOL_SIZE="${POOL_SIZE:-7500}"

# ---- 阶段开关 ----
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
TRAIN_KINDS="${TRAIN_KINDS:-math mix}"

# ---- 训练超参（与 C_if 生产 run 逐字一致，唯一变量是 C）----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
LAMBDA="${LAMBDA:-3e-2}"
IDENTITY_COV="${IDENTITY_COV:-0}"
REG_ONCE="${REG_ONCE:-1}"
SAVE="${SAVE:-1}"

MATH_COV="${MATH_COV:-${RESULTS_ROOT}/cov/cov_math_selfdistill_${POOL_SIZE}_qv.pt}"
MIX_COV="${MIX_COV:-${RESULTS_ROOT}/cov/cov_if_math_mix_qv.pt}"

# ---- 评测数据 ----
MATH500_FILE="${MATH500_FILE:-${MATH_DATA_DIR}/math500_test.jsonl}"
AMC_FILE="${AMC_FILE:-${MATH_DATA_DIR}/amc_test.jsonl}"
AIME_FILE="${AIME_FILE:-${MATH_DATA_DIR}/aime_test.jsonl}"
GSM8K_FILE="${GSM8K_FILE:-${MATH_DATA_DIR}/gsm8k_test.jsonl}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-1024}"
LIMIT="${LIMIT:-0}"
WITH_IF="${WITH_IF:-0}"

IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"

ADAPTERS="${RESULTS_ROOT}/adapters"
IF_RUN="cs_onereplay_lam${LAMBDA}_seed${SEED}_regonce"
MATH_RUN="cs_onereplay_math_lam${LAMBDA}_seed${SEED}_regonce"
MIX_RUN="cs_onereplay_ifmath_lam${LAMBDA}_seed${SEED}_regonce"
MODELS="${MODELS:-base vanilla cif cmath cmix}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
mkdir -p "${ADAPTERS}" "${RESULTS_ROOT}/metrics" "${RESULTS_ROOT}/map_cache/eval_math_seed${SEED}"

# ===========================================================================
# 训练：循环 TRAIN_KINDS
# ===========================================================================
train_one () {
  local kind=$1
  local cov_path run
  case "${kind}" in
    math) cov_path="${MATH_COV}"; run="${MATH_RUN}" ;;
    mix)  cov_path="${MIX_COV}";  run="${MIX_RUN}" ;;
    *)    echo "未知 TRAIN_KIND=${kind}，可选 math / mix"; return 1 ;;
  esac

  if [[ "${FORCE_TRAIN}" != "1" && -d "${ADAPTERS}/${run}" ]]; then
    echo "跳过训练 ${run}：adapter 已存在（FORCE_TRAIN=1 可强制重训）"
    return 0
  fi
  [[ -f "${cov_path}" ]] || { echo "缺协方差 ${cov_path}（先跑 31 号）"; return 1; }
  mkdir -p "${RESULTS_ROOT}/map_cache/${run}"

  echo "==== train ${kind}: ${run} (cov=${cov_path} lam=${LAMBDA}) ===="
  python -m onereplay.scripts.train --paradigm sft \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TARGET_MODULES}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/${run}" \
    --cov_path "${cov_path}" --replay_lambda "${LAMBDA}" \
    --identity_cov "${IDENTITY_COV}" --reg_once_per_update "${REG_ONCE}" \
    --save "${SAVE}" --save_path "${ADAPTERS}/${run}" \
    --metrics_path "${RESULTS_ROOT}/metrics/${run}.jsonl"
}

if [[ "${DO_TRAIN}" == "1" ]]; then
  for kind in ${=TRAIN_KINDS}; do
    train_one "${kind}"
  done
else
  echo "DO_TRAIN=0：跳过训练"
fi

# ===========================================================================
# 评测：数学指标（+可选 IF/CS）
# ===========================================================================
resolve_adapter () {
  case "$1" in
    base)    echo "" ;;
    vanilla) echo "${ADAPTERS}/cs_vanilla_seed${SEED}" ;;
    cif)     echo "${ADAPTERS}/${IF_RUN}" ;;
    cmath)   echo "${ADAPTERS}/${MATH_RUN}" ;;
    cmix)    echo "${ADAPTERS}/${MIX_RUN}" ;;
    *)       echo "" ;;
  esac
}

resolve_name () {
  case "$1" in
    base)    echo "base" ;;
    vanilla) echo "cs_vanilla_seed${SEED}" ;;
    cif)     echo "${IF_RUN}" ;;
    cmath)   echo "${MATH_RUN}" ;;
    cmix)    echo "${MIX_RUN}" ;;
    *)       echo "$1" ;;
  esac
}

eval_one () {
  local short=$1
  local name adapter
  name="$(resolve_name "${short}")"
  adapter="$(resolve_adapter "${short}")"

  local -a adapter_arg=()
  if [[ -n "${adapter}" ]]; then
    if [[ ! -d "${adapter}" ]]; then
      echo "跳过评测 ${short}：缺 adapter ${adapter}"
      return 0
    fi
    adapter_arg=(--adapter_path "${adapter}")
  fi

  if [[ -n "${MATH_METRICS}" ]]; then
    echo "==== eval math: ${name} (${MATH_METRICS}) ===="
    python -m onereplay.scripts.evaluate \
      --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
      "${adapter_arg[@]}" --run_name "${name}" \
      --metrics "${MATH_METRICS}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
      --math_max_new_tokens "${MATH_MAX_NEW_TOKENS}" --limit "${LIMIT}" \
      --math500_data_path "${MATH500_FILE}" \
      --amc_data_path "${AMC_FILE}" \
      --aime_data_path "${AIME_FILE}" \
      --gsm8k_data_path "${GSM8K_FILE}"
  fi

  if [[ -n "${IF_METRICS}" ]]; then
    echo "==== eval IF/CS: ${name} (${IF_METRICS}) ===="
    python -m onereplay.scripts.evaluate \
      --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
      "${adapter_arg[@]}" --run_name "${name}" \
      --metrics "${IF_METRICS}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
      --dataset_path "${CS_DATASET}" \
      --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
      --map_cache_dir "${RESULTS_ROOT}/map_cache/eval_math_seed${SEED}" \
      --ifeval_input "${IFEVAL_INPUT}" \
      --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
      --multiif_limit "${MULTIIF_LIMIT}"
  fi
}

if [[ "${DO_EVAL}" == "1" ]]; then
  MATH_METRICS=""
  [[ -f "${MATH500_FILE}" ]] && MATH_METRICS="${MATH_METRICS},math500" || echo "跳过 math500：缺 ${MATH500_FILE}"
  [[ -f "${AMC_FILE}" ]] && MATH_METRICS="${MATH_METRICS},amc" || echo "跳过 amc：缺 ${AMC_FILE}"
  [[ -f "${AIME_FILE}" ]] && MATH_METRICS="${MATH_METRICS},aime" || echo "跳过 aime：缺 ${AIME_FILE}"
  [[ -f "${GSM8K_FILE}" ]] && MATH_METRICS="${MATH_METRICS},gsm8k" || echo "跳过 gsm8k：缺 ${GSM8K_FILE}"
  MATH_METRICS="${MATH_METRICS#,}"

  IF_METRICS=""
  if [[ "${WITH_IF}" == "1" ]]; then
    IF_METRICS="ifeval,commonsense"
    [[ -f "${MULTIIF_INPUT}" ]] && IF_METRICS="${IF_METRICS},multiif" || echo "跳过 multiif：缺 ${MULTIIF_INPUT}"
  fi

  for m in ${=MODELS}; do
    eval_one "${m}"
  done

  echo "汇总 CSV: ${RESULTS_ROOT}/{math500,amc,aime,gsm8k}_summary.csv"
else
  echo "DO_EVAL=0：跳过评测"
fi

echo "Job finished at: $(date)"
