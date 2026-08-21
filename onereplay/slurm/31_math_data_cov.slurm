#!/usr/bin/zsh
#
#SBATCH --job-name=or_mathcov
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
# 问题2 数据链（一个 job 串完）: MetaMath 池 -> 自蒸馏 -> 采 C_math -> 混合 C_mix
#
#   Step 0  建池      MetaMathQA -> 每道唯一原题各出 2 个视角:
#                       canonical  原题逐字（与 MATH-500/AMC/AIME 的题面一致）
#                       rephrased  一条 MetaMath 改写（同题不同措辞）
#                     每题恰好 2 条 => 每题等权，无任何"哪些题选哪种增强"的任意性。
#                     不用 AnsAug（其 query 就是原题，等于复制 canonical）；
#                     不用 SV/FOBAR（backward 形式在所有评测集里没有对应，且
#                     pilot 显示其 prompt 有模板缺陷、截断率高达 ~14%）。
#   Step 1  自蒸馏    base 对每个视角分别贪心生成自己的解答（THINKING 可开）。
#                     分视角写文件：generate_replay_targets 只回写固定字段，
#                     合成一个文件会丢掉 view 标签，之后就没法按视角统计。
#   Step 2  采 C_math 用 {prompt, 自蒸馏解答} 采二阶矩，两个视角一起 glob 读入。
#   Step 3  混合      C_mix = w_if*C_if + w_math*C_math（默认等权 0.5/0.5）。
#
# 为什么自蒸馏而不是 MetaMath 自带 response
#   C=E[x x^T] 锚定"产生这些激活的行为"。MetaMath 的 response 不是 1.7B 自己会
#   产生的东西；用它采 C 等于保护一个模型本来不具备的行为。也正因为丢弃 response，
#   AnsAug 的价值（同题多份参考解）在我们的设置里完全失效。
#   注意：不按正确性筛选自蒸馏轨迹——采 C 要的是模型做数学时的代表性激活，
#   按正确性过滤会让 C 偏向简单题。
#
# THINKING=1 的三个连带要求
#   1) 生成侧 --enable_thinking 1（打开 <think> 块）。
#   2) 采 C 侧 --concat_prompt_target 1：Qwen3 的 chat template 会把 assistant
#      消息里的 <think>...</think> 剥掉，直接重渲染等于把思考轨迹丢掉。改成
#      "生成提示 + 原始 target" 拼接，才与真实生成的序列逐 token 一致。
#      本脚本用 DEBUG_EXAMPLES 打印渲染结果，务必确认 contains <think>: True。
#   3) 预算要大得多：非 thinking 的 pilot 中位数只有 ~260 token，开 thinking 后
#      通常上千。MAXLEN/MAX_NEW_TOKENS 不提上去，大量行会因未吐 EOS 被判截断、
#      写成空 target 并丢弃。**先跑 DIAG=1 定预算**，别直接全量。
#
# MAXLEN 一致性
#   自蒸馏(Step1)与采 C(Step2)必须用同一个 MAXLEN，否则 C 看到的上下文长度和
#   语料不符。本脚本用同一个 ${MAXLEN} 贯穿两步。
#
# 用法:
#   DIAG=1 sbatch onereplay/slurm/31_math_data_cov.slurm         # 先诊断长度定预算
#   sbatch onereplay/slurm/31_math_data_cov.slurm                # 全链
#   SKIP_PREP=1 SKIP_GEN=1 sbatch ...                            # 语料已好，从采 C 起
#   SKIP_PREP=1 SKIP_GEN=1 SKIP_COV=1 sbatch ...                 # 只做混合
#   THINKING=0 sbatch ...                                        # 关掉 thinking
#   自蒸馏支持 --resume 1：作业超时后原样再投一次即可接着跑。
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
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"

# ---- MetaMath 池 ----
METAMATH_PATH="${METAMATH_PATH:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/MetaMathQA}"
VIEW_DIR="${VIEW_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/MetaMathQA_views}"
VIEW_PREFIX="${VIEW_PREFIX:-metamath_cmath_30k}"
AUG_WEIGHTS="${AUG_WEIGHTS:-Rephrased=1.0}"
SOURCES="${SOURCES:-GSM,MATH}"
VIEWS="${VIEWS:-canonical rephrased}"

# ---- 自蒸馏 + 采 C 共用 ----
SEED="${SEED:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"
GEN_BATCH="${GEN_BATCH:-16}"
THINKING="${THINKING:-1}"
# thinking 的轨迹长得多，默认预算按 thinking 给；THINKING=0 时可自行调小。
MAXLEN="${MAXLEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-16}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
COV_BATCH="${COV_BATCH:-1}"
DEBUG_EXAMPLES="${DEBUG_EXAMPLES:-1}"

# ---- 阶段开关 ----
DIAG="${DIAG:-0}"
DIAG_SAMPLES="${DIAG_SAMPLES:-100}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_COV="${SKIP_COV:-0}"
SKIP_MIX="${SKIP_MIX:-0}"

# ---- 文件 ----
# thinking 与非 thinking 的语料/协方差必须分开存，否则两种模式会互相覆盖。
if [[ "${THINKING}" == "1" ]]; then THINK_SUFFIX="_think"; else THINK_SUFFIX=""; fi
TAG="${TAG:-metamath30k${THINK_SUFFIX}}"
DISTILL_DIR="${DISTILL_DIR:-${RESULTS_ROOT}/replay}"
MATH_COV="${MATH_COV:-${RESULTS_ROOT}/cov/cov_math_${TAG}_qv.pt}"
IF_COV="${IF_COV:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
MIX_COV="${MIX_COV:-${RESULTS_ROOT}/cov/cov_if_math_${TAG}_mix_qv.pt}"
W_IF="${W_IF:-0.5}"
W_MATH="${W_MATH:-0.5}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺基座模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
mkdir -p "${DISTILL_DIR}" "$(dirname "${MATH_COV}")" "${DATASET_CACHE}" "${VIEW_DIR}"

echo "METAMATH_PATH=${METAMATH_PATH}"
echo "THINKING=${THINKING} MAXLEN=${MAXLEN} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "MATH_COV=${MATH_COV}  MIX_COV=${MIX_COV}  weights if=${W_IF} math=${W_MATH}"

distill_file () {
  echo "${DISTILL_DIR}/${VIEW_PREFIX}_$1_selfdistill_seed${SAMPLE_SEED}${THINK_SUFFIX}.jsonl"
}

# ---------------------------------------------------------------------------
# 长度诊断分析器：读一份自蒸馏 jsonl，打印 token 分位数 + 截断率 + 预算建议
# ---------------------------------------------------------------------------
analyze_lengths () {
  local jsonl_file="$1"
  python - "${jsonl_file}" "${MAXLEN}" "${MAX_NEW_TOKENS}" <<'PY'
import json
import sys

path, max_len, max_new = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
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
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


print(f"==== length diagnostic on {n} rows (MAXLEN={max_len} MAX_NEW={max_new}) ====")
for label, values in (
    ("prompt", prompt_toks),
    ("target", target_toks),
    ("prompt+target", total_toks),
):
    print(
        f"{label:>14}: P50={pct(values,50)} P90={pct(values,90)} "
        f"P95={pct(values,95)} P99={pct(values,99)} max={max(values) if values else 0}"
    )
frac = truncated / n if n else 0.0
print(f"truncated fraction: {truncated}/{n} = {frac:.3f}  (截断行会被写空并在采 C 时丢弃)")
p99 = pct(total_toks, 99)
if frac <= 0.05 and p99 <= max_len:
    print(f"verdict: 预算够用（截断率 {frac:.3f} 且 P99={p99}<=MAXLEN={max_len}），可全量")
else:
    print(
        f"verdict: 建议调大 —— 截断率 {frac:.3f} 或 P99={p99} 逼近/超过 MAXLEN={max_len}。"
        f"把 MAXLEN 提到 >= {int(p99 * 1.15)} 并相应提高 MAX_NEW_TOKENS 后重跑"
        "（自蒸馏与采 C 的 MAXLEN 必须一致）"
    )
PY
}

# ---------------------------------------------------------------------------
# Step 0: 建 MetaMath 池（canonical + rephrased，每题各一条）
# ---------------------------------------------------------------------------
if [[ "${SKIP_PREP}" != "1" ]]; then
  echo "==== Step0 build MetaMath pool -> ${VIEW_DIR}/${VIEW_PREFIX}_*.jsonl ===="
  python -m onereplay.scripts.prepare_metamath_cmath \
    --metamath_path "${METAMATH_PATH}" \
    --out_dir "${VIEW_DIR}" \
    --prefix "${VIEW_PREFIX}" \
    --sources "${SOURCES}" \
    --aug_weights "${AUG_WEIGHTS}" \
    --seed "${SAMPLE_SEED}" \
    --tokenizer_path "${MODEL_DIR}/${MODEL_NAME}"
else
  echo "SKIP_PREP=1：使用已有池 ${VIEW_DIR}/${VIEW_PREFIX}_*.jsonl"
fi

# ---------------------------------------------------------------------------
# Step 1: 分视角自蒸馏（断点续跑）。DIAG=1 时只跑小样本并诊断后退出。
# ---------------------------------------------------------------------------
self_distill () {
  local view=$1
  local num_samples=$2
  local out_file=$3
  local pool_file="${VIEW_DIR}/${VIEW_PREFIX}_${view}.jsonl"
  [[ -f "${pool_file}" ]] || { echo "缺池文件: ${pool_file}"; exit 1; }

  echo "==== self-distill [${view}] thinking=${THINKING} -> ${out_file} ===="
  python -m onereplay.scripts.generate_replay_targets \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --data_files "${pool_file}" --split train \
    --cache_dir "${DATASET_CACHE}" \
    --input_column inputs --target_column targets \
    --pool_size 0 --sample_seed "${SAMPLE_SEED}" \
    --num_samples "${num_samples}" \
    --batch_size "${GEN_BATCH}" --max_len "${MAXLEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --min_new_tokens "${MIN_NEW_TOKENS}" \
    --enable_thinking "${THINKING}" \
    --output_path "${out_file}" --resume 1
}

if [[ "${DIAG}" == "1" ]]; then
  for view in ${=VIEWS}; do
    diag_file="${DISTILL_DIR}/${VIEW_PREFIX}_${view}_diag${DIAG_SAMPLES}.jsonl"
    rm -f "${diag_file}"
    self_distill "${view}" "${DIAG_SAMPLES}" "${diag_file}"
    analyze_lengths "${diag_file}"
  done
  echo "DIAG 完成：据上面 P99/截断率定 MAXLEN 与 MAX_NEW_TOKENS，再不带 DIAG 跑全链"
  echo "Job finished at: $(date)"
  exit 0
fi

if [[ "${SKIP_GEN}" != "1" ]]; then
  for view in ${=VIEWS}; do
    out_file="$(distill_file "${view}")"
    self_distill "${view}" 0 "${out_file}"
    analyze_lengths "${out_file}"
  done
else
  echo "SKIP_GEN=1：使用已有自蒸馏语料"
fi

DISTILL_GLOB="${DISTILL_DIR}/${VIEW_PREFIX}_*_selfdistill_seed${SAMPLE_SEED}${THINK_SUFFIX}.jsonl"
for view in ${=VIEWS}; do
  f="$(distill_file "${view}")"
  [[ -f "${f}" ]] || { echo "缺自蒸馏语料: ${f}"; exit 1; }
done

# ---------------------------------------------------------------------------
# Step 2: 采 C_math（两个视角一起读）+ 对 C_if 出 trace 比值
#   THINKING=1 时必须 --concat_prompt_target 1，否则 <think> 会被模板剥掉。
#   DEBUG_EXAMPLES 会打印渲染结果，确认 "contains <think>: True"。
# ---------------------------------------------------------------------------
if [[ "${SKIP_COV}" != "1" ]]; then
  echo "==== Step2 collect C_math -> ${MATH_COV} ===="
  python -m onereplay.scripts.collect_cov \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    --data_files "${DISTILL_GLOB}" --dataset_split train \
    --cache_dir "${DATASET_CACHE}" \
    --input_column inputs --target_column targets \
    --use_chat_template 1 --include_target_in_chat 1 \
    --enable_thinking "${THINKING}" \
    --concat_prompt_target "${THINKING}" \
    --debug_print_examples "${DEBUG_EXAMPLES}" \
    --require_target 1 \
    --max_samples 0 \
    --sample_shuffle 0 --sample_seed "${SAMPLE_SEED}" \
    --batch_size "${COV_BATCH}" --max_len "${MAXLEN}" \
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
        "注：本轮按要求用等权 0.5/0.5 混合。若 trace 比值远离 1，"
        f"等权实际上会让大 trace 一方主导；按 trace 反比应为 w_if:w_math ≈ 1 : {1.0 / mean_ratio:.3f}"
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
# Step 3: 混合 C_mix = w_if*C_if + w_math*C_math（等权）+ trace 诊断（供 32 号定 λ）
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
echo "  MATH_COV=${MATH_COV} MIX_COV=${MIX_COV} sbatch onereplay/slurm/32_math_train_eval.slurm"
echo "Job finished at: $(date)"
