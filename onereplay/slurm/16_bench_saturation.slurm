#!/usr/bin/zsh
#
#SBATCH --job-name=or_sat
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
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
# 只回答一个问题：micro-batch=8 是否已经吃满 GPU 吞吐？
#
# 为什么要单独做：15_bench_batch_scale 那次的 b8 档被环境拖慢了（同样 300 步、
# 同样 batch=8，or_bench-2649412 的 profile run 是 93 ms/step，那次却是
# 218 ms/step），于是同 job 内的 b8 vs b16 给出「b8 严重欠载」，跨 job 拼接
# 的干净 b8 却给出「已饱和」。两个结论相反，必须重测。
#
# 判据（比「时间比 vs token 比」更干净）：
#   把每档处理的样本数固定成 SAMPLES，让 b8 跑 SAMPLES/8 步、b16 跑
#   SAMPLES/16 步。两档吃掉同一批样本、同样多 token、同样多次优化器更新，
#   于是「总时间是否相等」就是饱和判据本身，不必再修正 token 比不是整 2 倍。
#     - 两档总时间接近  -> 时间正比于计算量 -> b8 已饱和
#     - b8 总时间明显更长 -> 小 batch 有闲置算力 -> b8 欠载
#
# 为什么这个结论重要：它直接决定 replay 用 4+4 还是 8+8。
#   4+4 + accum×2 的总时间恒为 ×2.0（单步不变、步数翻倍），与饱和无关；
#   8+8 的总时间取决于饱和程度，饱和则 ×2.2，欠载则可能只有 ×1.3。
#
# 设计要点：
#   - 全部 vanilla：判据只看 batch scaling 本身，与正则无关。
#   - ABBA 顺序（正序、逆序交替重复），位置效应对两档对称，不会只让 b8 吃冷启动。
#   - 开头有一个丢弃的 warmup run，把 GPU 时钟和 page cache 拉起来。
#   - --exclusive 独占节点。上次 b16 OneReplay 慢 120% 就是被邻居干扰的。
#     若排队太久，可以注释掉这一行，但结果的可信度会下降。
#   - 不设 max_train_samples，与历史 bench run 口径一致，稳态值可直接和
#     or_bench-2649412 的 88.0 ms/step 对照。
#
# 用法:
#   sbatch onereplay/slurm/16_bench_saturation.slurm
# 加 b32（有 OOM 风险，已用 || true 保护，不会中断整个 job）:
#   BATCHES="8 16 32" sbatch onereplay/slurm/16_bench_saturation.slurm
# 跑更多重复以压掉噪声:
#   REPEATS=4 sbatch onereplay/slurm/16_bench_saturation.slurm
#
# 跑完汇总（job 末尾会自动跑一次）:
#   python -m onereplay.scripts.summarize_saturation \
#     --metrics_dir <RESULTS_ROOT>/metrics/cost/saturation
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
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv

# ---- 项目路径 ----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
COST_DIR="${COST_DIR:-${RESULTS_ROOT}/metrics/cost/saturation}"
# tokenize 输入（数据集 / max_len / val_fraction）在所有 run 间完全相同，
# 共享一个 map cache 目录即可，不必每个 run 占一份磁盘。
MAP_CACHE="${MAP_CACHE:-${RESULTS_ROOT}/map_cache/sat_shared}"

# ---- 超参（与历史 bench run 保持一致）----
SEED="${SEED:-1}"
LR="${LR:-1e-4}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-200}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

# ---- 扫描设置 ----
# 每档都吃掉这么多训练样本。6400 能被 8/16/32/64 和 ACCUM=64 整除。
SAMPLES="${SAMPLES:-6400}"
BATCHES=(${=BATCHES:-8 16})
REPEATS="${REPEATS:-2}"
# 每个 run 打印这么多个区间速度点，便于看是否已经收敛到稳态。
LOG_POINTS="${LOG_POINTS:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || { echo "缺协方差文件: ${COV_PATH}"; exit 1; }
mkdir -p "${COST_DIR}" "${MAP_CACHE}"

# 样本数必须能被每个 batch 整除，否则各档吃的样本数不同，总时间就没法直接比。
if (( SAMPLES % ACCUM != 0 )); then
  echo "错误: SAMPLES=${SAMPLES} 不能被 ACCUM=${ACCUM} 整除，各档的优化器更新次数会不同"
  exit 1
fi
for batch in "${BATCHES[@]}"; do
  if (( SAMPLES % batch != 0 )); then
    echo "错误: SAMPLES=${SAMPLES} 不能被 BATCH=${batch} 整除"
    exit 1
  fi
  if (( batch > ACCUM )); then
    echo "错误: BATCH=${batch} > ACCUM=${ACCUM}，有效 batch 会悄悄变成 ${batch}"
    exit 1
  fi
done

echo "SAMPLES=${SAMPLES} BATCHES=${BATCHES[*]} ACCUM=${ACCUM} REPEATS=${REPEATS}"

# ---------------------------------------------------------------------------
# $1 run 名  $2 batch  $3 max_steps
# ---------------------------------------------------------------------------
sat_run() {
  local run="$1"; shift
  local batch="$1"; shift
  local max_steps="$1"; shift
  local log_every=$(( max_steps / LOG_POINTS ))
  (( log_every > 0 )) || log_every=1

  echo "==== sat: ${run} (batch=${batch} steps=${max_steps} samples=$(( batch * max_steps ))) ===="
  python -m onereplay.scripts.train --paradigm sft \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --epochs 1 --lr "${LR}" --batch_size "${batch}" --accumulation_size "${ACCUM}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TARGET_MODULES}" \
    --map_cache_dir "${MAP_CACHE}" \
    --log_every "${log_every}" \
    --profile 0 --max_steps "${max_steps}" \
    --save 0 \
    --metrics_path "${COST_DIR}/${run}.jsonl" \
    --cov_path "${COV_PATH}" --replay_lambda 0 --measure_replay_when_lambda_zero 0 \
    "$@"
}

# ---- 逆序数组，用于 ABBA ----
reversed=()
for (( i = ${#BATCHES[@]}; i >= 1; i-- )); do
  reversed+=("${BATCHES[i]}")
done

# ---- warmup：结果丢弃，只为把 GPU 时钟 / page cache 拉起来 ----
echo "======== warmup（丢弃）========"
sat_run "sat_warmup_discard" "${BATCHES[1]}" "${WARMUP_STEPS}" || true

# ---- 正式测量 ----
for (( rep = 1; rep <= REPEATS; rep++ )); do
  if (( rep % 2 == 1 )); then
    order=("${BATCHES[@]}")
  else
    order=("${reversed[@]}")
  fi
  echo "======== repeat ${rep} / ${REPEATS}（顺序: ${order[*]}）========"
  for batch in "${order[@]}"; do
    steps=$(( SAMPLES / batch ))
    # || true：大 batch 可能 OOM，不能让它掐死后面的 run（上次 b64 就是这样
    # 把整个 job 带走的）。
    sat_run "sat_b${batch}_r${rep}" "${batch}" "${steps}" || true
  done
done

echo "==== summary ===="
python -m onereplay.scripts.summarize_saturation \
  --metrics_dir "${COST_DIR}" \
  --out "${COST_DIR}/saturation_table.md"

echo "Job finished at: $(date)"
