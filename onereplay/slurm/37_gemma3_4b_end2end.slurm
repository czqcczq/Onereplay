#!/usr/bin/zsh
#
#SBATCH --job-name=or_g3_e2e
#SBATCH --partition=c23g
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
# Gemma-3-4B-it 端到端：一个 24h 作业跑完第二个模型的最基本 pipeline
#
#   Stage 1  采集旧知识协方差 C = E[x x^T]      (FLAN + chat 模板, qv 两类层)
#   Stage 2a 训练 vanilla LoRA                  (replay_lambda=0，这是要修的遗忘)
#   Stage 2b 训练 OneReplay LoRA                (replay_lambda>0，加 C 正则)
#   Stage 3  评测 base / vanilla / onereplay    (commonsense held-out + IFEval + Multi-IF)
#
# 只做「最基本」那条线：不采 Fisher(不跑 EWC)、不跑 teacher/OPD、不跑 math/code。
# 每个阶段都「产物已存在就跳过」，24h 内被抢占后 requeue 可原地续跑。
#
# 结果全部落到 model 专属的 RESULTS_ROOT(默认 results_gemma3_4b)，与 Qwen 那套
# 完全隔离，C、adapter、metrics、评测都不会串味。
#
# 用法:
#   sbatch onereplay/slurm/37_gemma3_4b_end2end.slurm
#
#   只看配置不跑:            DRY_RUN=1 sbatch ...
#   跳过某阶段:              DO_COV=0 / DO_TRAIN=0 / DO_EVAL=0
#   强制重跑:                FORCE_COV=1 / FORCE_TRAIN=1 / FORCE_EVAL=1
#   换 lambda:               LAMBDA=1e-3 sbatch ...
#
# 前置条件(计算节点无外网，全部要先在有网节点备好):
#   1) 模型落到     ${MODEL_DIR}/${MODEL_NAME}      (google/gemma-3-4b-it)
#   2) FLAN JSONL   ${FLAN_DIR}/train/*.jsonl
#   3) Commonsense  ${CS_DATASET}                    (save_to_disk 的 hf 目录)
#   4) IFEval 输入   随仓库自带(third_party/…)，Multi-IF CSV 缺了会自动跳过该指标
#   5) mkdir -p onereplay/slurm/outputs
#
# 预算(H100, 4B, qv r8 估计):
#   C 采集 ~1-1.5h  ·  每个训练 ~4-6h  ·  每个模型评测 ~1-2h
#   合计约 15-20h，卡在 24h 以内；真超了就靠跳过守卫 requeue 续。
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
MODEL_NAME="${MODEL_NAME:-gemma-3-4b-it}"
FLAN_DIR="${FLAN_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/flan}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
# Gemma 专属结果树：绝不与 Qwen 的 results/ 混放
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results_gemma3_4b}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"

# ---- 训练/评测超参：三条线严格一致，唯一变量是 replay_lambda ----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
# qv：只有 q_proj/v_proj 有非零 ΔW，C 也只采这两类层，二者必须一致
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
LAMBDA="${LAMBDA:-1e-4}"

# ---- C 采集 ----
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
FLAN_SAMPLES="${FLAN_SAMPLES:-20000}"
SAMPLE_SHUFFLE="${SAMPLE_SHUFFLE:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"

# ---- 评测输入 ----
IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"

# ---- 阶段开关 ----
DO_COV="${DO_COV:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
FORCE_COV="${FORCE_COV:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ---- run 命名：与 02/03/04 保持一致，评测才找得到 adapter ----
VANILLA_RUN="cs_vanilla_seed${SEED}"
ONEREPLAY_RUN="cs_onereplay_lam${LAMBDA}_seed${SEED}"
VANILLA_ADAPTER="${RESULTS_ROOT}/adapters/${VANILLA_RUN}"
ONEREPLAY_ADAPTER="${RESULTS_ROOT}/adapters/${ONEREPLAY_RUN}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
mkdir -p "$(dirname "${COV_PATH}")" "${DATASET_CACHE}" \
  "${RESULTS_ROOT}/adapters" "${RESULTS_ROOT}/metrics" "${RESULTS_ROOT}/map_cache"

set +x
echo "========================================================================"
echo "Gemma-3-4B-it 端到端"
echo "  model          : ${MODEL_DIR}/${MODEL_NAME}"
echo "  results root   : ${RESULTS_ROOT}"
echo "  target_modules : ${TARGET_MODULES}   (C 与 LoRA 同一组)"
echo "  超参           : lr=${LR} epochs=${EPOCHS} batch=${BATCH} accum=${ACCUM} maxlen=${MAXLEN}"
echo "  lambda         : ${LAMBDA}"
echo "  cov            : ${COV_PATH}"
echo "  vanilla run    : ${VANILLA_RUN}"
echo "  onereplay run  : ${ONEREPLAY_RUN}"
echo "  阶段           : COV=${DO_COV} TRAIN=${DO_TRAIN} EVAL=${DO_EVAL}"
echo "========================================================================"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1：只打印配置，不跑。"
  echo "Job finished at: $(date)"
  exit 0
fi
set -x

# =============================================================================
# Stage 1: 采集协方差 C（Gemma 必须重采，Qwen 的 .pt 完全不能复用）
# =============================================================================
if [[ "${DO_COV}" == "1" ]]; then
  if [[ "${FORCE_COV}" != "1" && -f "${COV_PATH}" ]]; then
    echo "跳过 C 采集：${COV_PATH} 已存在（FORCE_COV=1 可强制重采）"
  else
    [[ -d "${FLAN_DIR}/train" ]] || { echo "缺 FLAN JSONL 目录: ${FLAN_DIR}/train"; exit 1; }
    echo "==== Stage 1: collect_cov (${TARGET_MODULES}) ===="
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
  fi
fi

# =============================================================================
# Stage 2a: vanilla LoRA（replay_lambda=0）
#   measure_replay_when_lambda_zero=1 顺便在 lambda=0 下测一次 replay_reg 做参考
# =============================================================================
if [[ "${DO_TRAIN}" == "1" ]]; then
  if [[ "${FORCE_TRAIN}" != "1" && -d "${VANILLA_ADAPTER}" ]]; then
    echo "跳过 vanilla 训练：${VANILLA_ADAPTER} 已存在"
  else
    mkdir -p "${RESULTS_ROOT}/map_cache/${VANILLA_RUN}"
    echo "==== Stage 2a: train vanilla ${VANILLA_RUN} ===="
    python -m onereplay.scripts.train --paradigm sft \
      --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
      --dataset_path "${CS_DATASET}" \
      --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
      --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
      --target_modules "${TARGET_MODULES}" \
      --map_cache_dir "${RESULTS_ROOT}/map_cache/${VANILLA_RUN}" \
      --cov_path "${COV_PATH}" --replay_lambda 0 \
      --measure_replay_when_lambda_zero 1 \
      --save 1 --save_path "${VANILLA_ADAPTER}" \
      --metrics_path "${RESULTS_ROOT}/metrics/${VANILLA_RUN}.jsonl"
  fi

  # ---------------------------------------------------------------------------
  # Stage 2b: OneReplay LoRA（replay_lambda>0，唯一相对 vanilla 变动的就是 lambda + C）
  # ---------------------------------------------------------------------------
  if [[ "${FORCE_TRAIN}" != "1" && -d "${ONEREPLAY_ADAPTER}" ]]; then
    echo "跳过 onereplay 训练：${ONEREPLAY_ADAPTER} 已存在"
  else
    [[ -f "${COV_PATH}" ]] || { echo "缺协方差文件(Stage 1 未产出): ${COV_PATH}"; exit 1; }
    mkdir -p "${RESULTS_ROOT}/map_cache/${ONEREPLAY_RUN}"
    echo "==== Stage 2b: train onereplay ${ONEREPLAY_RUN} ===="
    python -m onereplay.scripts.train --paradigm sft \
      --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
      --dataset_path "${CS_DATASET}" \
      --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
      --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
      --target_modules "${TARGET_MODULES}" \
      --map_cache_dir "${RESULTS_ROOT}/map_cache/${ONEREPLAY_RUN}" \
      --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0 \
      --save 1 --save_path "${ONEREPLAY_ADAPTER}" \
      --metrics_path "${RESULTS_ROOT}/metrics/${ONEREPLAY_RUN}.jsonl"
  fi
fi

# =============================================================================
# Stage 3: 评测 base / vanilla / onereplay
#   commonsense(新任务拟合) + IFEval + Multi-IF(缺 CSV 自动移除)，三指标一次加载
# =============================================================================
if [[ "${DO_EVAL}" == "1" ]]; then
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

    # 用最慢的 ifeval summary 作为「这条线已评过」的守卫
    if [[ "${FORCE_EVAL}" != "1" && -f "${RESULTS_ROOT}/ifeval/${name}/summary.json" ]]; then
      echo "跳过评测 ${name}：已有 ifeval summary"
      return 0
    fi
    echo "==== Stage 3 eval: ${name} (${METRICS}) ===="
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
  run_eval "${VANILLA_RUN}" "${VANILLA_ADAPTER}"
  run_eval "${ONEREPLAY_RUN}" "${ONEREPLAY_ADAPTER}"
fi

set +x
echo "========================================================================"
echo "产物:"
echo "  C          : ${COV_PATH}"
echo "  vanilla    : ${VANILLA_ADAPTER}"
echo "  onereplay  : ${ONEREPLAY_ADAPTER}"
echo "  评测        : ${RESULTS_ROOT}/{commonsense,ifeval,multiif}/{base,${VANILLA_RUN},${ONEREPLAY_RUN}}/summary.json"
echo "  汇总 csv    : ${RESULTS_ROOT}/{commonsense,ifeval,multiif}_summary.csv"
echo ""
echo "怎么读结论:  base 高、vanilla 掉(遗忘)、onereplay 把 IFEval/Multi-IF 拉回接近 base"
echo "             而 commonsense held-out loss 三条线相近 => 保住旧能力且没牺牲新任务。"
echo "Job finished at: $(date)"
