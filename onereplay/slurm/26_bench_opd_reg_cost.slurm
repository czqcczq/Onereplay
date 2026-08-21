#!/usr/bin/zsh
#
#SBATCH --job-name=or_opdregcost
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# 修复后的 OPD 重跑：vanilla vs λ=1e-4，开销 + 效果
#
# 为什么 OPD 也要重测
# -----------------
# OPD 的 accumulation_steps 也是 8（batch 4 / accum 32），所以同样被"每个
# micro-batch 都算一遍正则"放大了 8 倍。修复前实测 198.3 -> 209.1 ms（+5.4%），
# 预期修复后约 199.6 ms（+0.7%）。
#
# 数值上从 5.4% 降到 0.7% 本来就接近零，重测的理由不是数字会变多少，而是让
# SFT-LoRA、SFT-全量、OPD 三个 paradigm 的开销都出自同一版代码，省掉"OPD 那条
# 是折算的还是实测的"这种追问。2000 步只要几分钟，代价很低。
#
# 两条都跑，虽然 vanilla 理论上不受影响
# ---------------------------------
# vanilla 那条（--measure_replay_when_lambda_zero 0）既不加载 C 也不计算正则，
# 修复完全不触及它的代码路径。同 job 再跑一遍有两个好处：分母和分子在同一节点、
# 同一环境下测出（不必跨 job 拼接），并且它顺带验证了这次改动没有意外影响 vanilla
# 路径——如果它仍然落在 198 ms 附近，就说明 training_step 的重构没有副作用。
#
# 口径
# ----
# 与 results_log 第二章一致：OPD 只有 4 个 500 步窗口，25 百分位没有意义，
# 所以取**最后一个窗口**（步 1501–2000），历史的 198.3 / 209.1 就是这个口径。
# 冷启动在 OPD 上极重（步 1–500 是 561.8 ms，稳态的 2.8 倍），全程平均不可用。
#
# 除 --reg_once_per_update 外，全部超参与 08_train_opd.slurm 逐字一致。
#
# 这次连效果一起测
# --------------
# 原计划是"只重测时间"，理由是新旧实现的累积梯度在精确算术下相同
# （verify_reg_once_per_update.py 在 float64 上比对累积 .grad，最大差 1.1e-16）。
# 24_check_reg_equivalence 在 bf16 + 真实数据上跑完后判定 DIVERGED：A2 与 A1 逐位
# 相同（噪声底 0），B 从第 21 步起分开。判据被"噪声底为 0"卡死，答不了"分数会不会
# 变"，只能交给一次真实评测。
#
# 好在 OPD 这边几乎免费：08 号记录在案的那条本来就是 EPOCHS=1 + MAX_STEPS=2000，
# 和这里的计时短跑同长同配置。所以 λ=1e-4 那条只要加上 --save 1 再评测一遍，
# 就同时是开销测量和效果确认，不需要另起一条训练。
#
# vanilla 那条不评测：修复不触及 λ=0 且 measure=0 的代码路径，cs_opd_lam0_seed1 的
# 分数继续沿用；它在这里只提供 ms/step 的分母。
#
# 【OPD 的效果对比天生比 SFT 噪声大，别过度解读】
# OPD 每步要用学生模型做一次 temperature=1.0 的采样。权重只要有一点不同，采出来的
# token 就可能分叉，之后训练数据本身就不一样了。也就是说 OPD 上新旧实现的差异里，
# 舍入只是引信，放大器是采样。这里看到的分数差不能全归给正则实现；反过来，如果连
# OPD 都看不出分数变化，那 SFT 那两条就更没理由变。
#
# 时间预算：训练两条约 0.6 h（2×2000 步），评测 commonsense+IFEval+Multi-IF 约 4 h。
# 评测只用 1 张卡，教师那张在评测阶段是空着的；想省机时就分两次投。
#
# 用法:
#   sbatch onereplay/slurm/26_bench_opd_reg_cost.slurm
#   RUN_EVAL=0 sbatch --time=01:30:00 onereplay/slurm/26_bench_opd_reg_cost.slurm
#   RUN_TRAIN=0 sbatch --time=05:00:00 --gres=gpu:1 onereplay/slurm/26_bench_opd_reg_cost.slurm
#   EVAL_METRICS=commonsense,ifeval sbatch ...   # 省掉 Multi-IF 的约 3.5 h
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

# ---- 与 08_train_opd.slurm 逐字一致 ----
SEED="${SEED:-1}"
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
LAMBDA="${LAMBDA:-1e-4}"
LOG_EVERY="${LOG_EVERY:-500}"
# 训练后的 commonsense loss 用 1000 条，与 08 号及 SFT 各条线同一切分；训练中途的
# 验证仍是 MAXVAL=200（OPD 下那是 KL，要跑 rollout，很贵）。
EVAL_MAXVAL="${EVAL_MAXVAL:-1000}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

# λ=1e-4 这条要保存并评测，所以它有正式的 run 名；_regonce 后缀避免覆盖
# 记录在案的 cs_opd_lam1e-4_seed1。
BASE_RUN="cs_opd_lam${LAMBDA}_seed${SEED}"
RUN="${BASE_RUN}_regonce"

IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"
EVAL_METRICS="${EVAL_METRICS:-}"
if [[ -z "${EVAL_METRICS}" ]]; then
  EVAL_METRICS="commonsense,ifeval"
  if [[ -f "${MULTIIF_INPUT}" ]]; then
    EVAL_METRICS="${EVAL_METRICS},multiif"
  else
    echo "跳过 Multi-IF：未找到 ${MULTIIF_INPUT}"
  fi
fi

OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/regcost_opd/${SLURM_JOB_ID:-local}}"
MAP_CACHE="${RESULTS_ROOT}/map_cache/regcost_opd"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺学生模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${TEACHER_MODEL_DIR}/${TEACHER_MODEL_NAME}" ]] || { echo "缺教师模型"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || { echo "缺协方差文件: ${COV_PATH}"; exit 1; }
mkdir -p "${OUT_DIR}" "${MAP_CACHE}" "${RESULTS_ROOT}/adapters" \
  "${RESULTS_ROOT}/metrics" "${RESULTS_ROOT}/map_cache/${RUN}"

# ---------------------------------------------------------------------------
# 一条 OPD run。$1=标签 $2=λ $3=measure_replay_when_lambda_zero
#
# tag=reg 那条是 λ=1e-4，同时承担开销测量和效果确认，所以它保存权重、metrics 写进
# 正式目录（再拷一份到 OUT_DIR/reg.jsonl 给下面的汇总读）。vanilla 那条只要计时，
# 不保存。
# ---------------------------------------------------------------------------
opd_run() {
  local tag="$1"
  local lam="$2"
  local measure="$3"
  local save=0
  # save=0 时 train.py 收到的是空 save_path，什么都不会写；这里指一个显然用不到的
  # 路径，免得读者以为 vanilla 也会往正式 adapter 目录里落东西。
  local save_path="${OUT_DIR}/unused_${tag}_adapter"
  local metrics_path="${OUT_DIR}/${tag}.jsonl"
  local map_cache="${MAP_CACHE}"
  if [[ "${tag}" == "reg" ]]; then
    save=1
    save_path="${RESULTS_ROOT}/adapters/${RUN}"
    metrics_path="${RESULTS_ROOT}/metrics/${RUN}.jsonl"
    map_cache="${RESULTS_ROOT}/map_cache/${RUN}"
  fi
  echo "==== ${tag}: lambda=${lam} measure=${measure} save=${save} ===="
  python -m onereplay.scripts.train --paradigm opd \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --teacher_model_dir "${TEACHER_MODEL_DIR}" --teacher_model_name "${TEACHER_MODEL_NAME}" \
    --teacher_gpu "${TEACHER_GPU}" \
    --dataset_path "${CS_DATASET}" \
    --epochs 1 --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
    --max_steps "${MAX_STEPS}" --profile 0 \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TARGET_MODULES}" \
    --map_cache_dir "${map_cache}" \
    --opd_max_new_tokens "${OPD_MAX_NEW_TOKENS}" --opd_temperature "${OPD_TEMPERATURE}" \
    --cov_path "${COV_PATH}" --replay_lambda "${lam}" --identity_cov 0 \
    --measure_replay_when_lambda_zero "${measure}" \
    --reg_once_per_update 1 \
    --log_every "${LOG_EVERY}" \
    --save "${save}" --save_path "${save_path}" \
    --metrics_path "${metrics_path}" 2>&1 | tee "${OUT_DIR}/${tag}.log" || true
  if [[ "${tag}" == "reg" && -f "${metrics_path}" ]]; then
    cp -f "${metrics_path}" "${OUT_DIR}/${tag}.jsonl"
  fi
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  # vanilla 先跑：它是两条里冷启动位置更不利的那个，这样正则的开销不会被
  # "第二条白捡热环境"高估（2649408 的 λ=1e-4 就是因为跑在第二位而显得偏快）。
  opd_run vanilla 0 0
  opd_run reg "${LAMBDA}" 1
else
  echo "RUN_TRAIN=0：跳过训练，直接评测已有的 ${RESULTS_ROOT}/adapters/${RUN}"
fi

# ---------------------------------------------------------------------------
# 评测：口径与 08 号逐字一致（commonsense 用 EVAL_MAXVAL=1000），只有 run_name 不同
# ---------------------------------------------------------------------------
if [[ "${RUN_EVAL}" == "1" && -d "${RESULTS_ROOT}/adapters/${RUN}" ]]; then
  echo "==== eval: ${RUN} (${EVAL_METRICS}) ===="
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    --adapter_path "${RESULTS_ROOT}/adapters/${RUN}" --run_name "${RUN}" \
    --metrics "${EVAL_METRICS}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${EVAL_MAXVAL}" --max_len "${MAXLEN}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
    --ifeval_input "${IFEVAL_INPUT}" \
    --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
    --multiif_limit "${MULTIIF_LIMIT}"
elif [[ "${RUN_EVAL}" == "1" ]]; then
  echo "缺 adapter，跳过评测: ${RESULTS_ROOT}/adapters/${RUN}"
else
  echo "RUN_EVAL=0：跳过评测。可用 RUN_TRAIN=0 --gres=gpu:1 单独提交评测"
fi

# ---------------------------------------------------------------------------
# 开销汇总：取最后一个 500 步窗口，与 results_log 第二章口径一致
# RUN_TRAIN=0 的评测专场里没有计时日志，跳过。
# ---------------------------------------------------------------------------
if [[ ! -f "${OUT_DIR}/vanilla.log" || ! -f "${OUT_DIR}/reg.log" ]]; then
  echo "跳过开销汇总：${OUT_DIR} 里没有这次的计时日志"
else
echo "==== 开销汇总 ===="
python - "${OUT_DIR}/vanilla" "${OUT_DIR}/reg" "${ACCUM}" "${BATCH}" <<'PY'
import json
import re
import sys
from pathlib import Path

WINDOW = re.compile(r"step (\d+)/\d+ .*?win=([\d.]+)ms/step")

# Measured before the fix (jobs 2832597 and 2649408), same last-window basis.
OLD_VANILLA = 198.3
OLD_REG = 209.1


def windows(prefix: str) -> list[tuple[int, float]]:
    path = Path(f"{prefix}.log")
    if not path.is_file():
        raise SystemExit(f"missing log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    found = [(int(m.group(1)), float(m.group(2))) for m in WINDOW.finditer(text)]
    if not found:
        raise SystemExit(f"no win= fields in {path}")
    return found


def peak(prefix: str) -> float | None:
    path = Path(f"{prefix}.jsonl")
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1]).get("peak_memory_allocated_gb")


vanilla_prefix, reg_prefix, accum, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
accumulation_steps = max(accum // batch, 1)

for label, prefix in (("vanilla", vanilla_prefix), ("lambda>0", reg_prefix)):
    print(f"{label} windows: " + ", ".join(f"step {s}: {ms:.1f}ms" for s, ms in windows(prefix)))

vanilla_ms = windows(vanilla_prefix)[-1][1]
reg_ms = windows(reg_prefix)[-1][1]
delta = reg_ms - vanilla_ms
print(
    f"\nlast window (steady state): vanilla {vanilla_ms:.1f} -> regularized {reg_ms:.1f} ms/step, "
    f"{delta:+.1f} ms ({delta / vanilla_ms * 100:+.2f}%)"
)
print(
    f"before the fix: {OLD_VANILLA} -> {OLD_REG} ms "
    f"({(OLD_REG - OLD_VANILLA) / OLD_VANILLA * 100:+.2f}%)"
)

# The old per-micro-batch overhead was the cost of one evaluation, which is what
# one optimizer step now pays; these two should agree.
per_update = delta * accumulation_steps
old_delta = OLD_REG - OLD_VANILLA
drift = abs(per_update - old_delta) / old_delta
print(
    f"one evaluation: {per_update:.1f} ms/update now vs {old_delta:.1f} ms/micro-batch before"
    f" -> {'consistent' if drift < 0.35 else f'off by {drift * 100:.0f}%, investigate'}"
)

# vanilla is untouched by the fix, so drifting far from 198.3 points at the
# environment or at an unintended side effect of the refactor, not at the change.
if abs(vanilla_ms - OLD_VANILLA) / OLD_VANILLA > 0.10:
    print(
        f"WARNING: vanilla moved from {OLD_VANILLA} to {vanilla_ms:.1f} ms although the fix "
        "does not touch its code path; check node contention or the refactor"
    )

vanilla_peak, reg_peak = peak(vanilla_prefix), peak(reg_prefix)
if vanilla_peak is not None and reg_peak is not None:
    print(
        f"peak allocated: {vanilla_peak:.2f} -> {reg_peak:.2f} GiB ({reg_peak - vanilla_peak:+.2f}); "
        "expected about +0.88, unchanged by the fix"
    )
PY
fi

# ---------------------------------------------------------------------------
# 效果并排：没有判据，就是把新旧两边的数字放在同一屏里
# ---------------------------------------------------------------------------
echo "==== 效果并排: 新实现 ${RUN} vs 记录在案的 ${BASE_RUN} ===="
echo "---- 训练 metrics（新）----"
[[ -f "${RESULTS_ROOT}/metrics/${RUN}.jsonl" ]] \
  && cat "${RESULTS_ROOT}/metrics/${RUN}.jsonl" \
  || echo "缺 ${RESULTS_ROOT}/metrics/${RUN}.jsonl"
echo "---- 训练 metrics（旧，记录在案）----"
[[ -f "${RESULTS_ROOT}/metrics/${BASE_RUN}.jsonl" ]] \
  && cat "${RESULTS_ROOT}/metrics/${BASE_RUN}.jsonl" \
  || echo "缺 ${RESULTS_ROOT}/metrics/${BASE_RUN}.jsonl"
for metric in ${(s:,:)EVAL_METRICS}; do
  echo "---- ${metric} summary（新）----"
  [[ -f "${RESULTS_ROOT}/${metric}/${RUN}/summary.json" ]] \
    && cat "${RESULTS_ROOT}/${metric}/${RUN}/summary.json" \
    || echo "缺 ${RESULTS_ROOT}/${metric}/${RUN}/summary.json"
  echo "---- ${metric} summary（旧，记录在案）----"
  [[ -f "${RESULTS_ROOT}/${metric}/${BASE_RUN}/summary.json" ]] \
    && cat "${RESULTS_ROOT}/${metric}/${BASE_RUN}/summary.json" \
    || echo "缺 ${RESULTS_ROOT}/${metric}/${BASE_RUN}/summary.json"
done

echo "结果目录: ${OUT_DIR}"
echo "Job finished at: $(date)"
