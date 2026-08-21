#!/usr/bin/zsh
#
#SBATCH --job-name=or_full
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
# 全量微调四件套：base（不训练）/ vanilla / OneReplay / vanilla replay
#
# 开销不另跑 bench：train_one_epoch 已把 train_sec / ms/step / 峰值显存 /
# C 与 W0 常驻量写进 metrics jsonl 和 .out。三件套必须在同一 job 内串行，
# 否则可能落到不同型号的卡上，wall time 不可比。
#
# MODE 控制本脚本做哪一步：
#   smoke      PROFILE=1 短跑 300 步（vanilla + onereplay），验证 used_layers、
#              显存、loss，并采分阶段耗时。分阶段计时的唯一合法来源。
#   calibrate  λ=0 + measure_replay=1 短跑，打印 replay_reg 量级，据此定 λ 网格。
#   train      正式三件套训练 + IFEval/Multi-IF 评测 + summarize_cost 出表（默认）
#
# 前置：COV_SCOPE=full 跑过 01_collect_cov，产物为 cov_flan_chat_20k_full.pt
#
# 用法:
#   MODE=smoke     sbatch onereplay/slurm/11_train_full_baselines.slurm
#   MODE=calibrate sbatch onereplay/slurm/11_train_full_baselines.slurm
#   sbatch onereplay/slurm/11_train_full_baselines.slurm                 # MODE=train
#   LAMBDA=1e-3 REPLAY_RATIO=0.10 sbatch onereplay/slurm/11_train_full_baselines.slurm
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

# ---- 项目路径 ----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
FLAN_DIR="${FLAN_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/flan}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_full.pt}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"
METRICS_DIR="${METRICS_DIR:-${RESULTS_ROOT}/metrics/full}"
CKPT_DIR="${CKPT_DIR:-${RESULTS_ROOT}/checkpoints/full}"

# ---- 全量超参 ----
MODE="${MODE:-train}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head}"
LAMBDA="${LAMBDA:-1e-4}"
REPLAY_RATIO="${REPLAY_RATIO:-0.10}"
REPLAY_POOL="${REPLAY_POOL:-20000}"
REPLAY_SAMPLE_SEED="${REPLAY_SAMPLE_SEED:-1}"
PROF_STEPS="${PROF_STEPS:-300}"
CALIB_STEPS="${CALIB_STEPS:-300}"

IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || {
  echo "缺全量协方差文件: ${COV_PATH}"
  echo "先跑: COV_SCOPE=full sbatch onereplay/slurm/01_collect_cov.slurm"
  exit 1
}
mkdir -p "${METRICS_DIR}" "${CKPT_DIR}" "${DATASET_CACHE}"

echo "MODE=${MODE} LR=${LR} LAMBDA=${LAMBDA} REPLAY_RATIO=${REPLAY_RATIO}"
echo "TARGET_MODULES=${TARGET_MODULES}"
echo "COV_PATH=${COV_PATH}"

# ---------------------------------------------------------------------------
# 一次全量训练。save=0 时只写 metrics，不落完整 checkpoint（约 3.4 GiB/run）。
#   $1 run 名  $2 profile  $3 max_steps  $4 save  $5.. 追加给 train 的参数
# ---------------------------------------------------------------------------
full_run() {
  local run="$1"; shift
  local profile="$1"; shift
  local max_steps="$1"; shift
  local save="$1"; shift

  echo "==== full: ${run} (profile=${profile} max_steps=${max_steps} save=${save}) ===="
  mkdir -p "${RESULTS_ROOT}/map_cache/${run}"
  python -m onereplay.scripts.train --paradigm sft --full_finetune 1 \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TARGET_MODULES}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/${run}" \
    --profile "${profile}" --max_steps "${max_steps}" \
    --save "${save}" --save_path "${CKPT_DIR}/${run}" \
    --metrics_path "${METRICS_DIR}/${run}.jsonl" \
    "$@"
}

run_eval() {
  local name="$1"
  local ckpt="$2"
  local metrics="ifeval"
  if [[ -f "${MULTIIF_INPUT}" ]]; then
    metrics="${metrics},multiif"
  else
    echo "跳过 Multi-IF：未找到 ${MULTIIF_INPUT}"
  fi
  # base 额外跑 commonsense held-out loss，便于和 LoRA 对照
  if [[ -z "${ckpt}" ]]; then
    metrics="commonsense,${metrics}"
  fi

  echo "==== eval: ${name} (${metrics}) ===="
  local -a adapter_arg=()
  if [[ -n "${ckpt}" ]]; then
    adapter_arg=(--adapter_path "${ckpt}")
  fi
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    "${adapter_arg[@]}" --run_name "${name}" \
    --metrics "${metrics}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/eval_full_seed${SEED}" \
    --ifeval_input "${IFEVAL_INPUT}" \
    --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
    --multiif_limit "${MULTIIF_LIMIT}"
}

# ---------------------------------------------------------------------------
# MODE=smoke：分阶段耗时的唯一来源；同时验证全量路径能跑通
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "smoke" ]]; then
  echo "==== MODE=smoke: profile short runs ===="
  EPOCHS=1
  # vanilla 必须连 C 都不加载，否则测出来的不是 vanilla
  full_run "full_vanilla_prof_seed${SEED}" 1 "${PROF_STEPS}" 0 \
    --cov_path "${COV_PATH}" --replay_lambda 0 --measure_replay_when_lambda_zero 0
  full_run "full_onereplay_lam${LAMBDA}_prof_seed${SEED}" 1 "${PROF_STEPS}" 0 \
    --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0
  echo "检查 .out / jsonl：used_layers 应约为 224（28 层 × 8 类），loss 应下降"
  echo "Job finished at: $(date)"
  exit 0
fi

# ---------------------------------------------------------------------------
# MODE=calibrate：读出 replay_reg 量级，据此选使 λ·reg ≈ 1e-3..1e-1 × task_loss 的 λ
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "calibrate" ]]; then
  echo "==== MODE=calibrate: measure replay_reg with lambda=0 ===="
  EPOCHS=1
  full_run "full_calib_seed${SEED}" 0 "${CALIB_STEPS}" 0 \
    --cov_path "${COV_PATH}" --replay_lambda 0 --measure_replay_when_lambda_zero 1
  echo "---- calibration hint ----"
  export METRICS_DIR SEED
  python - <<'PY'
import json, os, sys
metrics_dir = os.environ["METRICS_DIR"]
seed = os.environ["SEED"]
path = os.path.join(metrics_dir, f"full_calib_seed{seed}.jsonl")
if not os.path.isfile(path):
    print(f"missing {path}", file=sys.stderr)
    sys.exit(1)
records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
rec = records[-1]
task = float(rec["train_task_loss"])
reg = float(rec["train_replay_reg"])
print(f"task_loss={task:.6f}  replay_reg={reg:.6e}")
print("suggested lambda so that lambda*reg is roughly 1e-3 .. 1e-1 of task_loss:")
for frac in (1e-3, 1e-2, 1e-1):
    if reg <= 0:
        print("  replay_reg is zero; check cov coverage / W0 snapshot")
        break
    lam = frac * task / reg
    print(f"  frac={frac:g}  ->  lambda≈{lam:.3e}")
PY
  echo "Job finished at: $(date)"
  exit 0
fi

# ---------------------------------------------------------------------------
# MODE=train：正式三件套（同一节点串行）+ 评测 + 开销表
# ---------------------------------------------------------------------------
if [[ "${MODE}" != "train" ]]; then
  echo "未知 MODE=${MODE}，可选 smoke / calibrate / train"; exit 1
fi

VANILLA_RUN="full_vanilla_seed${SEED}"
ONEREPLAY_RUN="full_onereplay_lam${LAMBDA}_seed${SEED}"
REPLAY_RUN="full_replay_r${REPLAY_RATIO}_seed${SEED}"

# vanilla：measure_replay_when_lambda_zero=0，否则每步仍在算正则
full_run "${VANILLA_RUN}" 0 0 1 \
  --cov_path "${COV_PATH}" --replay_lambda 0 --measure_replay_when_lambda_zero 0

full_run "${ONEREPLAY_RUN}" 0 0 1 \
  --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0

[[ -d "${FLAN_DIR}/train" ]] || { echo "缺 FLAN JSONL 目录: ${FLAN_DIR}/train"; exit 1; }
full_run "${REPLAY_RUN}" 0 0 1 \
  --cov_path "${COV_PATH}" --replay_lambda 0 --measure_replay_when_lambda_zero 0 \
  --replay_ratio "${REPLAY_RATIO}" \
  --replay_data_files "${FLAN_DIR}/train/*.jsonl" --replay_split train \
  --replay_cache_dir "${DATASET_CACHE}" \
  --replay_pool_size "${REPLAY_POOL}" --replay_sample_seed "${REPLAY_SAMPLE_SEED}"

# base 不训练；评测三件套产物（adapter_path 指向完整 checkpoint 目录）
run_eval "base" ""
run_eval "${VANILLA_RUN}" "${CKPT_DIR}/${VANILLA_RUN}"
run_eval "${ONEREPLAY_RUN}" "${CKPT_DIR}/${ONEREPLAY_RUN}"
run_eval "${REPLAY_RUN}" "${CKPT_DIR}/${REPLAY_RUN}"

echo "==== cost summary (epoch 1, from training metrics; no separate bench) ===="
python -m onereplay.scripts.summarize_cost \
  --metrics_dir "${METRICS_DIR}" \
  --baseline "${VANILLA_RUN}" \
  --epoch 1 \
  --out "${METRICS_DIR}/cost_table_full.md"

echo "Job finished at: $(date)"
