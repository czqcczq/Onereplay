#!/usr/bin/zsh
#
#SBATCH --job-name=or_rmix
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
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
# Stage 2e: LoRA 下的 replay 基线——严格 batch 级混合（训练 + IFEval + Multi-IF）
#
# 与 09_sweep_replay_ratio.slurm 的区别只在"混合发生在哪一层"：
#   09（数据级）：把 replay 行追加进训练集再全局 shuffle。每个 micro-batch 的新旧
#                比例是二项分布的，replay 占一半时一个累积窗口里的 new task 数是
#                64 ± 5.7，只有期望正确。
#   本脚本（batch 级）：每个 micro-batch 显式拼装 —— BATCH 条 new task +
#                REPLAY_PER_BATCH 条 replay。每次更新的 new task 数是精确值。
# 老师要求的就是后者（"4+4，累积步数乘 2，保证每次更新 new task 数据量不变"）。
#
# 【口径与 02/06 完全统一：BATCH 就是 micro-batch 总大小，恒为 8】
# --replay_per_batch 是从这 8 条里切出来给 replay 的条数，剩下的 8-RPB 是 new task。
# 所以单步时间恒约 89.2ms、峰值显存恒 21.29GiB，与 vanilla / OneReplay 逐项可比，
# 三档的总时间之比就等于步数之比（x1.14 / x1.33 / x2.00）。
#
# 【ACCUM 必须随档位放大——这就是老师说的"累积步数乘 2"】
# accum_steps = ACCUM / BATCH 数的是 micro-batch 个数，而每个 micro-batch 里只有
# 8-RPB 条 new task，所以要维持"每次更新的 new task 数≈64"（与 02/06 的 8x8=64
# 一致），ACCUM 必须跟着涨：
#   50% 档 (4+4): ACCUM 64 -> 128，16 步 x 4 条 = 64 条，正好翻倍。
#   25% 档 (6+2): ACCUM -> 88，11 步 x 6 条 = 66 条。
#   12.5% 档(7+1): ACCUM -> 72， 9 步 x 7 条 = 63 条。
# 63 / 66 / 64 相对基线 64 差 -1.6% / +3.1% / 0%。差不掉到 0 是因为 64 = 2^6 不含
# 因子 7 或 3，7 条和 6 条都除不尽——这是"micro-batch 恒定"与"每次更新恒 64 条"
# 两个约束的数学冲突，只能取其一。选前者的理由是效率可比性：若改成让每次更新恒为
# 64 条（固定 new=4，micro-batch 变成 5/6/8），b5/b6 是否还在饱和区从未测过
# （2831163 只测了 8 和 16，仅能证明饱和点 <= 8），一旦欠载，测出的时间就把 GPU
# 利用率下降算进了 replay 的账上。
# 而 ±3% 这点偏差不构成风险：每个 epoch 的 new task 总量三档分别是
# 168714 / 168714 / 168712 条（只差被丢弃的不足一个 batch 的尾巴），实质完全一致，
# "新任务学得少所以 retention 才好"这个隐患并不存在，差异只在 per-update 的分组。
#
# 【replay 池是循环使用的，不要为此扩池】
# 50% 那档一个 epoch 要消耗约 16.9 万条 replay，池子只有约 1.7 万条，所以每条会被
# 用约 9.7 次。这是故意的：池子正是估计 C 的那批行，OneReplay 与 replay 消费同一
# 份旧知识、只在"怎么用"上不同。扩池到 16.9 万反而会让 replay 看到比 C 编码的更多
# 信息，对比就不公平了。代价是可能对 replay 集过拟合——那是 replay 方法该暴露的
# 固有代价，如实记录即可。
#
# 【看训练日志开头的 describe 那两行】
# 它会打印 replay 占"行"的比例和占"supervised token"的比例。两者不相等：loss 是对
# supervised token 求平均，所以"一半的行是 replay"不等于"一半的 loss 来自 replay"。
# 写结论时用 token 那个比例。
#
# 前置：自蒸馏语料必须已经生成（只依赖 base 模型，13 为全量实验生成的那份可复用）:
#   sbatch onereplay/slurm/13_replay_selfdistill.slurm
#
# 用法:
#   sbatch onereplay/slurm/18_sweep_replay_batchmix.slurm
# 只跑 50% 那档（数组第 3 个元素）:
#   sbatch --array=2 onereplay/slurm/18_sweep_replay_batchmix.slurm
# 先做开销短跑（不评测）:
#   sbatch --array=2 --export=ALL,SAVE=0,MAX_STEPS=300,PROFILE=1 \
#     onereplay/slurm/18_sweep_replay_batchmix.slurm
# =============================================================================

# ---- 三档配置：BATCH 恒为 8（与 02/06 一致），只改切给 replay 的条数 ----
# 改这里时同步 --array 上界 = 元素个数-1。
#   档位   BATCH  replay  new  ACCUM  accum_steps  new/更新  步数/epoch  时间vs vanilla
#   12.5%    8      1      7     72        9          63       24102        x1.14
#   25%      8      2      6     88       11          66       28119        x1.33
#   50%      8      4      4    128       16          64       42178        x2.00
# ACCUM 取"能被 BATCH 整除、且让 accum_steps x new 最接近 64"的值。
REPLAY_GRID=(1 2 4)
ACCUM_GRID=(72 88 128)
# 每次更新的 new task 数允许偏离基线 64 的范围；超出说明 ACCUM_GRID 配错了。
NEW_PER_UPDATE_TARGET="${NEW_PER_UPDATE_TARGET:-64}"
NEW_PER_UPDATE_TOL="${NEW_PER_UPDATE_TOL:-3}"

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

# ---- 训练超参（除 ACCUM 外与 02/06 逐字相同）----
# ACCUM 不在这里定：它必须与档位配对，从上面的网格按 array 任务号取。
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
# val_loss 是按 batch 求平均而非按 token，把验证集钉在基线的 8，这样即使将来改了
# BATCH，val_loss 仍与历史那几列可比。
EVAL_BATCH="${EVAL_BATCH:-8}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

# ---- 自蒸馏回放语料。POOL_SIZE / SAMPLE_SEED 只用来拼默认文件名，必须与
#      13_replay_selfdistill.slurm 生成时、以及 01_collect_cov 的取值一致 ----
POOL_SIZE="${POOL_SIZE:-20000}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"
SELF_DISTILL_FILE="${SELF_DISTILL_FILE:-${RESULTS_ROOT}/replay/flan_selfdistill_${POOL_SIZE}_seed${SAMPLE_SEED}.jsonl}"

# ---- 开销测量：短跑用 PROFILE=1 MAX_STEPS=300 SAVE=0 ----
PROFILE="${PROFILE:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
SAVE="${SAVE:-1}"

# ---- 从 array 任务号选档位（zsh 数组 1-based，array 任务号 0-based）----
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RPB="${REPLAY_GRID[$((TASK_ID + 1))]:-}"
ACCUM="${ACCUM_GRID[$((TASK_ID + 1))]:-}"
if [[ -z "${RPB}" || -z "${ACCUM}" ]]; then
  echo "错误: array 任务号 ${TASK_ID} 超出网格(共 ${#REPLAY_GRID[@]} 档)；--array 上界应为 档数-1"
  exit 1
fi

NEW_PER_BATCH=$(( BATCH - RPB ))
ACCUM_STEPS=$(( ACCUM / BATCH ))
NEW_PER_UPDATE=$(( ACCUM_STEPS * NEW_PER_BATCH ))
if (( NEW_PER_BATCH <= 0 )); then
  echo "错误: RPB=${RPB} 不小于 BATCH=${BATCH}，micro-batch 里没有 new task 了"
  exit 1
fi
# ACCUM 必须被 BATCH 整除，否则实际累积的 micro-batch 数与标称的 ACCUM 不符。
if (( ACCUM_STEPS * BATCH != ACCUM )); then
  echo "错误: ACCUM=${ACCUM} 不能被 BATCH=${BATCH} 整除，实际累积 ${ACCUM_STEPS} 步 x ${BATCH} = $(( ACCUM_STEPS * BATCH )) 条"
  exit 1
fi
# 每次更新的 new task 数必须贴住基线的 64，这是"retention 不是靠少学新任务换来的"
# 这件事的保证；ACCUM_GRID 配错时这里会拦下来。
DIFF=$(( NEW_PER_UPDATE - NEW_PER_UPDATE_TARGET ))
(( DIFF < 0 )) && DIFF=$(( -DIFF ))
if (( DIFF > NEW_PER_UPDATE_TOL )); then
  echo "错误: 每次更新 new task=${NEW_PER_UPDATE}（${ACCUM_STEPS} 步 x ${NEW_PER_BATCH} 条），偏离基线 ${NEW_PER_UPDATE_TARGET} 超过 ${NEW_PER_UPDATE_TOL}；请调 ACCUM_GRID"
  exit 1
fi

RUN="cs_replaymix_${NEW_PER_BATCH}n${RPB}r_seed${SEED}"
IFEVAL_INPUT="${CODE}/third_party/instruction_following_eval/data/input_data.jsonl"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"

echo "batchmix task ${TASK_ID}: micro-batch=${BATCH} = ${NEW_PER_BATCH} new + ${RPB} replay"
echo "  BATCH=${BATCH} 与 02/06 一致，故单步时间与峰值显存可直接对比"
echo "  replay 行占比=$(python -c "print(f'{${RPB}/${BATCH}:.4f}')")"
echo "  ACCUM=${ACCUM} -> accum_steps=${ACCUM_STEPS}, 每次更新 new task=${NEW_PER_UPDATE}（基线 ${NEW_PER_UPDATE_TARGET}）"
echo "  run=${RUN}"
echo "SELF_DISTILL_FILE=${SELF_DISTILL_FILE}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${SELF_DISTILL_FILE}" ]] || {
  echo "缺自蒸馏回放语料: ${SELF_DISTILL_FILE}"
  echo "先跑: sbatch onereplay/slurm/13_replay_selfdistill.slurm"
  exit 1
}
# batch 级混合是新加的开关；用旧代码跑会静默退化成不带 replay 的 vanilla。
grep -q "replay_per_batch" onereplay/scripts/train.py || {
  echo "train.py 里没有 --replay_per_batch，代码没同步到最新"
  exit 1
}
mkdir -p "${RESULTS_ROOT}/adapters" "${RESULTS_ROOT}/metrics" \
  "${RESULTS_ROOT}/map_cache/${RUN}" "${DATASET_CACHE}"

# ---------------------------------------------------------------------------
# Step 1: 训练并保存 adapter
# replay_lambda=0 且 measure_replay_when_lambda_zero=0：不加载 C、每步不算正则，
# 训练时间才是这条基线的真实开销。
# ---------------------------------------------------------------------------
echo "==== train: ${RUN} (${BATCH} new + ${RPB} replay per micro-batch) ===="
python -m onereplay.scripts.train --paradigm sft \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --eval_batch_size "${EVAL_BATCH}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --profile "${PROFILE}" --max_steps "${MAX_STEPS}" \
  --replay_per_batch "${RPB}" \
  --replay_self_distill_file "${SELF_DISTILL_FILE}" \
  --replay_drop_truncated 1 \
  --replay_cache_dir "${DATASET_CACHE}" \
  --replay_lambda 0 --measure_replay_when_lambda_zero 0 \
  --save "${SAVE}" --save_path "${RESULTS_ROOT}/adapters/${RUN}" \
  --metrics_path "${RESULTS_ROOT}/metrics/${RUN}.jsonl"

# ---- 自检：混合真的按预期生效了吗 ----
echo "==== 自检：micro-batch 构成与每次更新的 new task 数 ===="
python - "${RESULTS_ROOT}/metrics/${RUN}.jsonl" "${RPB}" "${BATCH}" "${NEW_PER_UPDATE}" <<'PY'
import json
import sys

path = sys.argv[1]
want_rpb, want_micro, want_new_per_update = (int(value) for value in sys.argv[2:5])
with open(path, encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
last = records[-1]
steps = last.get("train_steps") or 0
samples = last.get("train_samples") or 0
per_step = samples / steps if steps else 0

print(f"replay_per_batch={last.get('replay_per_batch')} (want {want_rpb})")
print(f"train_samples/train_steps={per_step:.4f} (want {want_micro})")
print(f"new_per_update={last.get('new_per_update')} (want {want_new_per_update})")
print(f"train_steps={steps}, train_samples={samples}")

if last.get("replay_per_batch") != want_rpb:
    raise SystemExit("replay_per_batch 没生效，这次 run 白跑")
# micro-batch 必须等于基线的 8，否则单步时间和显存不能与 vanilla 并排比较。
if abs(per_step - want_micro) > 1e-6:
    raise SystemExit(f"每步样本数是 {per_step}，不是 {want_micro}；混合没按预期拼装")
if last.get("new_per_update") != want_new_per_update:
    raise SystemExit(
        f"每次更新 new task={last.get('new_per_update')}，不是 {want_new_per_update}；ACCUM 配错了"
    )
print("自检通过")
PY

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
