#!/usr/bin/zsh
#
#SBATCH --job-name=or_mathcov
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
# 问题2 数据链（一个 job 串完）: MATH 自蒸馏 -> 采 C_math -> 混合 C_mix
#
# 三步本来就是串行依赖，合并成一个脚本；每步有独立 SKIP 开关，避免长链中途失败
# 要从头重跑。
#
#   Step 1  自蒸馏      base 对 MATH train 7500 的 problem 贪心生成自己的 CoT 解答，
#                       作为采 C_math 的语料（不能用 gold solution，理由见下）。
#   Step 2  采 C_math   用 {problem, 自蒸馏解答} 采二阶矩，qv scope，与 C_if 层对齐。
#   Step 3  混合 C_mix  C_mix = w_if*C_if + w_math*C_math（加权求和，无 trace 归一化）。
#
# 为什么自蒸馏而不是 gold solution
#   C=E[x x^T] 锚定"产生这些激活的行为"。gold solution 是教科书风格，不是 1.7B 自己
#   会产生的东西；用它采 C 等于保护一个模型本来不具备的行为。自蒸馏让 C 捕捉模型自己
#   的数学推理激活——那才是要保住的历史能力。与 19 号对 FLAN 的做法同理。
#
# MAXLEN 一致性
#   自蒸馏(Step1)与采 C(Step2)必须用同一个 MAXLEN，否则 C 看到的上下文长度和语料不符。
#   本脚本用同一个 ${MAXLEN} 贯穿两步。它不必等于 34/32 号训练的 max_len——C 是与序列
#   长无关的 d×d 二阶矩。
#
# 用法:
#   DIAG=1 sbatch onereplay/slurm/31_math_data_cov.slurm          # 先诊断长度定 MAXLEN
#   sbatch onereplay/slurm/31_math_data_cov.slurm                 # 全链: 自蒸馏->C_math->C_mix
#   MAXLEN=1536 MAX_NEW_TOKENS=1280 sbatch onereplay/slurm/31_math_data_cov.slurm
#   SKIP_GEN=1 sbatch onereplay/slurm/31_math_data_cov.slurm      # 语料已生成，从采 C 起
#   SKIP_GEN=1 SKIP_COV=1 sbatch onereplay/slurm/31_math_data_cov.slurm   # 只做混合
#   W_IF=1 W_MATH=0.3 sbatch onereplay/slurm/31_math_data_cov.slurm       # 按 trace 反比设权重
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
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"
MATH_DATA_DIR="${MATH_DATA_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/math}"
MATH_TRAIN_FILE="${MATH_TRAIN_FILE:-${MATH_DATA_DIR}/math_train_inputs_targets.jsonl}"

# ---- 自蒸馏 + 采 C 共用 ----
SEED="${SEED:-1}"
POOL_SIZE="${POOL_SIZE:-7500}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"
GEN_BATCH="${GEN_BATCH:-32}"
MAXLEN="${MAXLEN:-1280}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-16}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

# ---- 阶段开关 ----
DIAG="${DIAG:-0}"
DIAG_SAMPLES="${DIAG_SAMPLES:-200}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_COV="${SKIP_COV:-0}"
SKIP_MIX="${SKIP_MIX:-0}"

# ---- 文件 ----
SELF_DISTILL_FILE="${SELF_DISTILL_FILE:-${RESULTS_ROOT}/replay/math_selfdistill_${POOL_SIZE}_seed${SAMPLE_SEED}.jsonl}"
DIAG_FILE="${DIAG_FILE:-${RESULTS_ROOT}/replay/math_selfdistill_diag_${DIAG_SAMPLES}_seed${SAMPLE_SEED}.jsonl}"
MATH_COV="${MATH_COV:-${RESULTS_ROOT}/cov/cov_math_selfdistill_${POOL_SIZE}_qv.pt}"
IF_COV="${IF_COV:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
MIX_COV="${MIX_COV:-${RESULTS_ROOT}/cov/cov_if_math_mix_qv.pt}"
W_IF="${W_IF:-0.5}"
W_MATH="${W_MATH:-0.5}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺基座模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
mkdir -p "$(dirname "${SELF_DISTILL_FILE}")" "$(dirname "${MATH_COV}")" "${DATASET_CACHE}"

echo "MATH_TRAIN_FILE=${MATH_TRAIN_FILE}"
echo "DIAG=${DIAG} MAXLEN=${MAXLEN} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "MATH_COV=${MATH_COV}  MIX_COV=${MIX_COV}  weights if=${W_IF} math=${W_MATH}"

# ---------------------------------------------------------------------------
# 长度诊断分析器：读一份自蒸馏 jsonl，打印 token 分位数 + 截断率 + MAXLEN 建议
# ---------------------------------------------------------------------------
analyze_lengths () {
  local jsonl_file="$1"
  python - "${jsonl_file}" "${MAXLEN}" <<'PY'
import json
import sys

path, max_len = sys.argv[1], int(sys.argv[2])
prompt_toks, target_toks, total_toks = [], [], []
truncated = 0
n = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        p = int(row.get("prompt_tokens", 0))
        t = int(row.get("target_tokens", 0))
        prompt_toks.append(p)
        target_toks.append(t)
        total_toks.append(p + t)
        if row.get("truncated"):
            truncated += 1

def pct(values, q):
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))
    return s[idx]

print(f"==== length diagnostic on {n} rows (MAXLEN={max_len}) ====")
for label, values in (("prompt", prompt_toks), ("target", target_toks), ("prompt+target", total_toks)):
    print(f"{label:>14}: P50={pct(values,50)} P90={pct(values,90)} P95={pct(values,95)} max={max(values) if values else 0}")
frac = truncated / n if n else 0.0
print(f"truncated fraction: {truncated}/{n} = {frac:.3f}")
p90 = pct(total_toks, 90)
if frac <= 0.05 and p90 <= max_len:
    print(f"verdict: MAXLEN={max_len} 够用（截断率低且 P90={p90}<=MAXLEN），可直接全量")
else:
    print(f"verdict: 建议调大 —— 截断率 {frac:.3f} 或 P90={p90} 逼近/超过 MAXLEN={max_len}；"
          f"提高 MAX_NEW_TOKENS / MAXLEN 后重跑（自蒸馏与采 C 的 MAXLEN 必须一致）")
PY
}

# ---------------------------------------------------------------------------
# DIAG 模式：小样本自蒸馏 + 长度诊断，然后退出
# ---------------------------------------------------------------------------
if [[ "${DIAG}" == "1" ]]; then
  [[ -f "${MATH_TRAIN_FILE}" ]] || { echo "缺 MATH train: ${MATH_TRAIN_FILE}（先跑 download_math_data / prepare_math_data）"; exit 1; }
  echo "==== DIAG: self-distill ${DIAG_SAMPLES} samples -> ${DIAG_FILE} ===="
  python -m onereplay.scripts.generate_replay_targets \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --data_files "${MATH_TRAIN_FILE}" --split train \
    --cache_dir "${DATASET_CACHE}" \
    --input_column inputs --target_column targets \
    --pool_size "${POOL_SIZE}" --sample_seed "${SAMPLE_SEED}" \
    --num_samples "${DIAG_SAMPLES}" \
    --batch_size "${GEN_BATCH}" --max_len "${MAXLEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --min_new_tokens "${MIN_NEW_TOKENS}" \
    --output_path "${DIAG_FILE}" --resume 0
  analyze_lengths "${DIAG_FILE}"
  echo "DIAG 完成：据上面分位数与建议定 MAXLEN / MAX_NEW_TOKENS，再不带 DIAG 跑全链"
  echo "Job finished at: $(date)"
  exit 0
fi

# ---------------------------------------------------------------------------
# Step 1: 全量自蒸馏（断点续跑；SKIP_GEN=1 直接用已有语料）
# ---------------------------------------------------------------------------
if [[ "${SKIP_GEN}" != "1" ]]; then
  [[ -f "${MATH_TRAIN_FILE}" ]] || { echo "缺 MATH train: ${MATH_TRAIN_FILE}（先跑 download_math_data / prepare_math_data）"; exit 1; }
  echo "==== Step1 self-distill -> ${SELF_DISTILL_FILE} ===="
  python -m onereplay.scripts.generate_replay_targets \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --data_files "${MATH_TRAIN_FILE}" --split train \
    --cache_dir "${DATASET_CACHE}" \
    --input_column inputs --target_column targets \
    --pool_size "${POOL_SIZE}" --sample_seed "${SAMPLE_SEED}" \
    --num_samples 0 \
    --batch_size "${GEN_BATCH}" --max_len "${MAXLEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --min_new_tokens "${MIN_NEW_TOKENS}" \
    --output_path "${SELF_DISTILL_FILE}" --resume 1
  analyze_lengths "${SELF_DISTILL_FILE}"
else
  echo "SKIP_GEN=1：使用已有自蒸馏语料 ${SELF_DISTILL_FILE}"
fi
[[ -f "${SELF_DISTILL_FILE}" ]] || { echo "缺自蒸馏语料: ${SELF_DISTILL_FILE}"; exit 1; }

# ---------------------------------------------------------------------------
# Step 2: 采 C_math（与自蒸馏同源：require_target + 左截断）+ 对 C_if 出 trace 比值
# ---------------------------------------------------------------------------
if [[ "${SKIP_COV}" != "1" ]]; then
  echo "==== Step2 collect C_math -> ${MATH_COV} ===="
  python -m onereplay.scripts.collect_cov \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    --data_files "${SELF_DISTILL_FILE}" --dataset_split train \
    --cache_dir "${DATASET_CACHE}" \
    --input_column inputs --target_column targets \
    --use_chat_template 1 --include_target_in_chat 1 \
    --require_target 1 \
    --max_samples 0 \
    --sample_shuffle 0 --sample_seed "${SAMPLE_SEED}" \
    --batch_size 2 --max_len "${MAXLEN}" \
    --truncation_side left \
    --target_modules "${TARGET_MODULES}" \
    --output_path "${MATH_COV}"

  if [[ -f "${IF_COV}" ]]; then
    echo "==== compare C_math against C_if: ${IF_COV} ===="
    python - "${MATH_COV}" "${IF_COV}" <<'PY'
import sys

import torch

from onereplay.core.covariance import load_covariance_file

math_cov = load_covariance_file(sys.argv[1])
if_cov = load_covariance_file(sys.argv[2])
shared = sorted(set(math_cov) & set(if_cov))
print(f"layers: {len(math_cov)} C_math, {len(if_cov)} C_if, {len(shared)} shared")
only_math = sorted(set(math_cov) - set(if_cov))
only_if = sorted(set(if_cov) - set(math_cov))
if only_math or only_if:
    print(f"WARNING: layer sets differ. only-math={only_math[:5]} only-if={only_if[:5]}")

ratios = []
for name in shared:
    m = torch.diagonal(math_cov[name].float()).sum().item()
    f = torch.diagonal(if_cov[name].float()).sum().item()
    if f > 0:
        ratios.append(m / f)
if ratios:
    r = torch.tensor(ratios)
    mean_ratio = r.mean().item()
    print(
        "trace(C_math)/trace(C_if) per layer: "
        f"mean={mean_ratio:.3f} min={r.min().item():.3f} max={r.max().item():.3f}"
    )
    print(
        "混合建议: 要两个 C 贡献相当，权重可设 w_if:w_math ≈ 1 : "
        f"{1.0 / mean_ratio:.3f}（按 trace 反比），否则大 trace 一方主导。"
    )
else:
    print("no comparable layers; skipping scale report")
PY
  else
    echo "跳过 C_math/C_if 尺度对比：未找到 C_if ${IF_COV}"
  fi
else
  echo "SKIP_COV=1：使用已有 C_math ${MATH_COV}"
fi
[[ -f "${MATH_COV}" ]] || { echo "缺 C_math: ${MATH_COV}"; exit 1; }

# ---------------------------------------------------------------------------
# Step 3: 混合 C_mix = w_if*C_if + w_math*C_math + trace 诊断（供 32 号定 λ）
# ---------------------------------------------------------------------------
if [[ "${SKIP_MIX}" != "1" ]]; then
  [[ -f "${IF_COV}" ]] || { echo "缺 C_if(混合需要): ${IF_COV}"; exit 1; }
  echo "==== Step3 mix -> ${MIX_COV} (if=${W_IF} math=${W_MATH}) ===="
  python -m onereplay.mix_covariances \
    --inputs "${IF_COV},${MATH_COV}" \
    --weights "${W_IF},${W_MATH}" \
    --output_path "${MIX_COV}" \
    --normalize_weights 1

  echo "==== trace diagnostic: C_mix vs C_if / C_math ===="
  python - "${MIX_COV}" "${IF_COV}" "${MATH_COV}" <<'PY'
import sys

import torch

from onereplay.core.covariance import load_covariance_file

mix = load_covariance_file(sys.argv[1])
c_if = load_covariance_file(sys.argv[2])
c_math = load_covariance_file(sys.argv[3])
shared = sorted(set(mix) & set(c_if) & set(c_math))
print(f"shared layers: {len(shared)}")


def trace(cov, name):
    return torch.diagonal(cov[name].float()).sum().item()


ratio_mix_if, ratio_mix_math = [], []
for name in shared:
    ti, tm, tx = trace(c_if, name), trace(c_math, name), trace(mix, name)
    if ti > 0:
        ratio_mix_if.append(tx / ti)
    if tm > 0:
        ratio_mix_math.append(tx / tm)


def report(label, values):
    if not values:
        print(f"{label}: n/a")
        return
    t = torch.tensor(values)
    print(f"{label}: mean={t.mean().item():.3f} min={t.min().item():.3f} max={t.max().item():.3f}")


report("trace(C_mix)/trace(C_if)  ", ratio_mix_if)
report("trace(C_mix)/trace(C_math)", ratio_mix_math)
mean_mix_if = torch.tensor(ratio_mix_if).mean().item() if ratio_mix_if else 1.0
print(
    "定 λ: C_if 单独 run 用 λ=3e-2；要让 C_mix 上 IF 保护强度相当，"
    f"可用 λ_mix ≈ 3e-2 / {mean_mix_if:.3f} = {3e-2 / mean_mix_if:.2e}（32 号可据此覆盖 LAMBDA）"
)
PY
else
  echo "SKIP_MIX=1：跳过混合"
fi

echo "下一步: COV_KIND 由 32 号内部循环 math/mix 训练并评测"
echo "  sbatch onereplay/slurm/32_math_train_eval.slurm"
echo "Job finished at: $(date)"
