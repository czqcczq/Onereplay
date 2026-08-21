#!/usr/bin/zsh
#
#SBATCH --job-name=or_opdvan
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# OPD 的「真 vanilla」成本基线：不加载 C、不计算正则。
#
# 为什么需要：08_train_opd.slurm 没有传 --measure_replay_when_lambda_zero，
# train.py 的默认值是 1，于是 cs_opd_lam0_seed1 那条「vanilla 基线」其实
# 加载了 56 个 C（0.875 GiB 常驻）并且每步都算了正则，只是把结果乘 0 后
# 没加进 loss。证据：它的 train_replay_reg=0.196 非零，而且它和 λ=1e-4 的
# peak_memory_allocated_gb 都是 28.444942951202393，小数点后 15 位一致。
#
# 后果：λ=0 付了全额正则开销，所以它不是成本基线；已报出的「λ=1e-4 比 λ=0
# 快 2.8%」纯属噪声（两者稳态单步 208.9 vs 209.1 ms，差 0.1%，那 2.8% 全部
# 来自 λ=0 作为 job 首个 run 多付的冷启动）。
# 训练结果不受影响：λ=0 时该项恒为 0，梯度贡献为 0，正则又是纯读权重的
# 确定性运算、不消耗随机数，所以数据顺序和 dropout mask 都一致。
#
# 本脚本只跑这一条 run，其余 OPD 结果沿用已有的，不重跑。
#
# 口径：除 --measure_replay_when_lambda_zero 0 之外，所有参数都与
# cs_opd_lam0_seed1 逐项对齐（batch=4 accum=32 lr=1e-5 max_steps=2000
# max_len=512 log_every=500 rollout=128 temp=1.0 seed=1），否则不可比。
# 特意不加 --exclusive：历史那次也没独占，条件要对称。
#
# 用法:
#   sbatch onereplay/slurm/17_bench_opd_vanilla.slurm
# 跑两次取最快值压噪声（要测的差异只有 4%，一次观测容易被干扰吃掉）:
#   REPEATS=2 sbatch --time=03:00:00 onereplay/slurm/17_bench_opd_vanilla.slurm
#
# -----------------------------------------------------------------------------
# 跑完拿 logs 直接对照 cs_opd_lam0_seed1（or_opd-2649408.out 第 60-107 行）。
# 注意历史 run 没有 win= 字段，下面的区间值是从累计平均反推的：
#
#   区间          λ=0 (旧)     真 vanilla 预期
#   1-500         569 ms       约 560
#   501-1000      257 ms       约 248
#   1001-1500     242 ms       约 233
#   1501-2000     208.9 ms     约 200      <- 最干净的一档，直接看 win=
#   train_sec     638.5 s      约 621      (差 8.7ms x 2000 = 17.4 s)
#   peak alloc    28.445 GiB   约 27.570   (少掉 C 的 0.875)
#   cuda:0 alloc  12.116 GiB   约 11.241   (C 在学生卡上)
#   cuda:1 alloc  16.329 GiB   16.329 不变 (教师卡不受影响)
#   cov mem       0.875 GiB    0.0         <- 这一项若不是 0 说明开关没生效
#   replay_reg    0.196        0.0         <- 同上
#   train_task_loss 1.226360   应几乎相同   <- 顺带验证「不影响训练结果」
#
# 预期结论：正则的绝对开销约 8.7 ms/step（与 batch 无关，b8 和 b32 两次
# 独立测量都是这个值），OPD 稳态单步 209 ms，所以相对开销约 +4%，明显低于
# SFT batch=8 的 +10%，因为 OPD 的单步被 rollout 和教师打分主导。
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
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen3-8B}"
TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-${MODEL_DIR}}"
TEACHER_GPU="${TEACHER_GPU:-1}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
COST_DIR="${COST_DIR:-${RESULTS_ROOT}/metrics/cost/opd_vanilla}"

# ---- 与 cs_opd_lam0_seed1 逐项一致；改任何一项都会让对比失效 ----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-32}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-200}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
OPD_MAX_NEW_TOKENS="${OPD_MAX_NEW_TOKENS:-128}"
OPD_TEMPERATURE="${OPD_TEMPERATURE:-1.0}"
MAX_STEPS="${MAX_STEPS:-2000}"
# 历史那次用的是 train.py 的默认 500；区间要对齐才能逐段比，这里显式写死。
LOG_EVERY="${LOG_EVERY:-500}"
PROFILE="${PROFILE:-0}"
REPEATS="${REPEATS:-1}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺学生模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${TEACHER_MODEL_DIR}/${TEACHER_MODEL_NAME}" ]] || { echo "缺教师模型: ${TEACHER_MODEL_DIR}/${TEACHER_MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
mkdir -p "${COST_DIR}"

for (( rep = 1; rep <= REPEATS; rep++ )); do
  if (( REPEATS > 1 )); then
    RUN="cs_opd_truevanilla_seed${SEED}_r${rep}"
  else
    RUN="cs_opd_truevanilla_seed${SEED}"
  fi
  # 独立 map cache：与 cs_opd_lam0_seed1 的目录分开，免得覆盖它的 arrow 文件。
  mkdir -p "${RESULTS_ROOT}/map_cache/${RUN}"

  echo "==== train OPD 真 vanilla: ${RUN}（不加载 C，不算正则）===="
  # 关键：--measure_replay_when_lambda_zero 0。λ=0 时它决定 C 是否被加载，
  # 传 0 才是真 vanilla。--cov_path 仍然传上，只为让参数列表与历史 run 一致；
  # λ=0 且 measure=0 时它不会被读取。
  # SAVE=0：这是纯成本 run，不需要 adapter，也不评测（已有结果沿用）。
  python -m onereplay.scripts.train --paradigm opd \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --teacher_model_dir "${TEACHER_MODEL_DIR}" --teacher_model_name "${TEACHER_MODEL_NAME}" \
    --teacher_gpu "${TEACHER_GPU}" \
    --dataset_path "${CS_DATASET}" \
    --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
    --max_steps "${MAX_STEPS}" --profile "${PROFILE}" --log_every "${LOG_EVERY}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TARGET_MODULES}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
    --opd_max_new_tokens "${OPD_MAX_NEW_TOKENS}" --opd_temperature "${OPD_TEMPERATURE}" \
    --cov_path "${COV_PATH}" --replay_lambda 0 --measure_replay_when_lambda_zero 0 \
    --identity_cov 0 \
    --save 0 \
    --metrics_path "${COST_DIR}/${RUN}.jsonl"
done

echo "==== 自检：下面两项必须都是 0，否则开关没生效、这次 run 白跑 ===="
python - "${COST_DIR}" <<'PY'
import json, sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("cs_opd_truevanilla_*.jsonl")):
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        continue
    last = records[-1]
    cov = last.get("covariance_memory_gb")
    reg = last.get("train_replay_reg")
    status = "OK" if not cov and not reg else "污染：仍在算正则"
    print(
        f"{path.name}: covariance_memory_gb={cov} train_replay_reg={reg} -> {status}\n"
        f"  train_sec={last.get('train_sec')} sec_per_step={last.get('sec_per_step')} "
        f"peak_alloc={last.get('peak_memory_allocated_gb')} "
        f"task_loss={last.get('train_task_loss')}"
    )
PY

echo "Job finished at: $(date)"
