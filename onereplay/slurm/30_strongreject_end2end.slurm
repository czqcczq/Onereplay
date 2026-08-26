#!/usr/bin/zsh
#
#SBATCH --job-name=or_strongreject
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
# 问题 1 - StrongREJECT 越狱补充评测（生成 + 评分 + 分析，一次跑完）
#   目的：用 StrongREJECT 官方微调 autograder（0-1 有害强度分）复核越狱表现。
#   该分数对拒答/空/坍缩输出都给 ~0，天然不受 track B 的“输出坍缩伪影”影响，
#   是对 WildGuard 三分类的正交、更公平的对照。
#
#   SR track 每条 forbidden prompt 生成两类输入：
#     direct              直接问（无攻击基线，用于算 attack gain）
#     + 越狱外壳           与 track B 完全相同的 5 个壳（2 手写 + 3 In-The-Wild）
#
#   生成结果写到独立的 SR_ROOT，避免覆盖 A/B/C 的 responses.jsonl。
#   先在有网节点用 scripts/download_safety_data.py 下好：
#     - strongreject_dataset.csv（ungated raw GitHub）
#     - StrongREJECT grader adapter qylu4156/strongreject-15k-v1（LoRA，ungated）
#     - 它的 base google/gemma-2b（gated，需接受授权；缓存进 HF cache 供离线解析）
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
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
SEED="${SEED:-1}"

# ---- StrongREJECT 数据 + 越狱外壳来源（有网节点先下好）----
SAFETY_RAW="${SAFETY_RAW:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/safety}"
STRONGREJECT_CSV="${STRONGREJECT_CSV:-${SAFETY_RAW}/strongreject_dataset.csv}"
ITW_PATH="${ITW_PATH:-${SAFETY_RAW}/in_the_wild_jailbreak.csv}"
NUM_ITW_TEMPLATES="${NUM_ITW_TEMPLATES:-3}"
NUM_SR="${NUM_SR:-0}"            # 0 = 全部 313 条
SR_WRAP="${SR_WRAP:-1}"          # 1 = 同时生成越狱外壳版本
SR_PROMPTS="${SR_PROMPTS:-${RESULTS_ROOT}/safety_prompts/strongreject_prompts.jsonl}"
SAFETY_MAX_NEW_TOKENS="${SAFETY_MAX_NEW_TOKENS:-256}"

# ---- StrongREJECT grader（有网节点先下 qylu4156/strongreject-15k-v1）----
SR_MODEL_DIR="${SR_MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
SR_MODEL_NAME="${SR_MODEL_NAME:-strongreject-grader}"
SR_BATCH="${SR_BATCH:-16}"
# grader 的 base 权重位置。留空 = 用 adapter_config.json 里的 google/gemma-2b，
# 靠 HF cache 离线解析；若 cache 里没有，就设成本地 base 模型目录。
SR_BASE="${SR_BASE:-}"

# ---- 生成结果与评分/分析输出（独立目录，避免覆盖 A/B/C）----
SR_ROOT="${SR_ROOT:-${RESULTS_ROOT}/strongreject}"
SR_JUDGE_OUT="${SR_JUDGE_OUT:-${RESULTS_ROOT}/strongreject_scored}"
SR_REPORT="${SR_REPORT:-${SR_JUDGE_OUT}/strongreject_analysis.json}"

FORCE_GEN="${FORCE_GEN:-0}"     # 1 = 即使已有 responses 也重新生成

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -f "${STRONGREJECT_CSV}" ]] || { echo "缺 StrongREJECT CSV: ${STRONGREJECT_CSV}"; exit 1; }
# 查实际权重文件而不仅仅是目录——空目录会让 offline 加载假通过后报错。
# grader（qylu4156/strongreject-15k-v1）是 LoRA adapter，只有 adapter_model.safetensors，
# 没有 config.json；它的 base（gated google/gemma-2b）要单独缓存好，由 judge 脚本报错兜底。
SR_GRADER_DIR="${SR_MODEL_DIR}/${SR_MODEL_NAME}"
[[ -f "${SR_GRADER_DIR}/adapter_model.safetensors" || -f "${SR_GRADER_DIR}/config.json" ]] || {
  echo "缺 grader 权重: ${SR_GRADER_DIR}/(adapter_model.safetensors|config.json)"
  echo "先在有网节点下好（接受 Gemma 授权 + 登录后）:"
  echo "  python -m onereplay.scripts.download_safety_data --only_sr 1"
  exit 1
}

# ---- 构建 StrongREJECT prompts 文件（缺失时才生成）----
if [[ ! -f "${SR_PROMPTS}" ]]; then
  echo "==== build strongreject prompts -> ${SR_PROMPTS} ===="
  itw_arg=(); [[ "${SR_WRAP}" == "1" && -f "${ITW_PATH}" ]] && itw_arg=(--itw_path "${ITW_PATH}" --num_itw_templates "${NUM_ITW_TEMPLATES}")
  python -m onereplay.scripts.prepare_safety_data \
    --strongreject_path "${STRONGREJECT_CSV}" \
    --num_sr "${NUM_SR}" --sr_wrap "${SR_WRAP}" \
    "${itw_arg[@]}" \
    --output_path "${SR_PROMPTS}"
fi

# ---- 待评模型（run_name;adapter_path，adapter 为空即 base）----
RUNS="${RUNS:-base;|vanilla_seed${SEED};${RESULTS_ROOT}/adapters/cs_vanilla_seed${SEED}|onereplay_lam3e-2_seed${SEED};${RESULTS_ROOT}/adapters/cs_onereplay_lam3e-2_seed${SEED}_regonce}"

run_sr_gen () {
  local name=$1
  local adapter=$2
  local -a adapter_arg=()
  if [[ -n "${adapter}" ]]; then adapter_arg=(--adapter_path "${adapter}"); fi

  local resp="${SR_ROOT}/safety/${name}/responses.jsonl"
  if [[ "${FORCE_GEN}" != "1" && -s "${resp}" ]]; then
    echo "==== strongreject gen: ${name} (跳过，已有 ${resp}) ===="
    return 0
  fi

  echo "==== strongreject gen: ${name} ===="
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    "${adapter_arg[@]}" --run_name "${name}" \
    --metrics safety --out_dir "${SR_ROOT}" --seed "${SEED}" \
    --safety_prompts "${SR_PROMPTS}" \
    --safety_max_new_tokens "${SAFETY_MAX_NEW_TOKENS}"
}

# ---- 阶段 1：逐模型生成，收集各自的 responses 路径 ----
RESP_LIST=""
RUN_NAMES=""
IFS='|' read -rA run_specs <<< "${RUNS}"
for spec in "${run_specs[@]}"; do
  name="${spec%%;*}"
  adapter="${spec#*;}"
  run_sr_gen "${name}" "${adapter}"
  resp="${SR_ROOT}/safety/${name}/responses.jsonl"
  if [[ -z "${RESP_LIST}" ]]; then RESP_LIST="${resp}"; else RESP_LIST="${RESP_LIST},${resp}"; fi
  if [[ -z "${RUN_NAMES}" ]]; then RUN_NAMES="${name}"; else RUN_NAMES="${RUN_NAMES},${name}"; fi
done

# ---- 阶段 2：StrongREJECT grader 一次性评分本次生成的所有 run ----
echo "==== strongreject grade: ${RESP_LIST} ===="
base_arg=()
if [[ -n "${SR_BASE}" ]]; then base_arg=(--judge_base "${SR_BASE}"); fi
python -m onereplay.scripts.judge_strongreject \
  --responses "${RESP_LIST}" \
  --judge_model_dir "${SR_MODEL_DIR}" --judge_model_name "${SR_MODEL_NAME}" \
  "${base_arg[@]}" \
  --out_dir "${SR_JUDGE_OUT}" \
  --batch_size "${SR_BATCH}"

# ---- 阶段 3：配对统计分析（均值差 bootstrap + 二值化 McNemar + 攻击增益）----
echo "==== strongreject analyze -> ${SR_REPORT} ===="
python -m onereplay.scripts.analyze_strongreject \
  --scored "${SR_JUDGE_OUT}/*/scored_sr.jsonl" \
  --runs "${RUN_NAMES}" \
  --out_path "${SR_REPORT}"

echo "Job finished at: $(date)"
