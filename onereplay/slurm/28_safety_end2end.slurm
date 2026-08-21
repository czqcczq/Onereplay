#!/usr/bin/zsh
#
#SBATCH --job-name=or_safety
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
# 问题 1 - 安全性评测（生成 + 判分，一次跑完）
#   默认只测 base / vanilla / onereplay 三条线，三条 track 共用 HarmBench：
#     A  直接问              -> 良性微调是否冲掉了 base 对齐
#     B  behavior × 越狱外壳 -> 固定 transfer attack，所有模型吃相同输入
#     C  XSTest              -> 过度拒绝对照
#   分步版见 28_eval_safety.slurm（生成）+ 29_judge_safety.slurm（判分）。
#   先在有网节点下载 HarmBench / In-The-Wild / XSTest 及 allenai/wildguard。
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

# ---- 安全数据原始文件（有网节点先下好）----
SAFETY_RAW="${SAFETY_RAW:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/safety}"
HARMBENCH_CSV="${HARMBENCH_CSV:-${SAFETY_RAW}/harmbench_behaviors_text_all.csv}"
ITW_PATH="${ITW_PATH:-${SAFETY_RAW}/in_the_wild_jailbreak.csv}"
XSTEST_PATH="${XSTEST_PATH:-${SAFETY_RAW}/xstest.csv}"
SAFETY_PROMPTS="${SAFETY_PROMPTS:-${RESULTS_ROOT}/safety_prompts/safety_prompts.jsonl}"
NUM_BEHAVIORS="${NUM_BEHAVIORS:-200}"
NUM_ITW_TEMPLATES="${NUM_ITW_TEMPLATES:-3}"
SAFETY_MAX_NEW_TOKENS="${SAFETY_MAX_NEW_TOKENS:-256}"

# ---- judge 模型（有网节点先下 allenai/wildguard）----
JUDGE_MODEL_DIR="${JUDGE_MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-wildguard}"
JUDGE_BATCH="${JUDGE_BATCH:-16}"
JUDGE_OUT="${JUDGE_OUT:-${RESULTS_ROOT}/safety_judged}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -f "${HARMBENCH_CSV}" ]] || { echo "缺 HarmBench CSV: ${HARMBENCH_CSV}"; exit 1; }
[[ -d "${JUDGE_MODEL_DIR}/${JUDGE_MODEL_NAME}" ]] || {
  echo "缺 judge 模型: ${JUDGE_MODEL_DIR}/${JUDGE_MODEL_NAME}（先下 allenai/wildguard）"; exit 1;
}

# ---- 构建统一 prompts 文件（缺失时才生成）----
if [[ ! -f "${SAFETY_PROMPTS}" ]]; then
  echo "==== build safety prompts -> ${SAFETY_PROMPTS} ===="
  itw_arg=(); [[ -f "${ITW_PATH}" ]] && itw_arg=(--itw_path "${ITW_PATH}")
  xstest_arg=(); [[ -f "${XSTEST_PATH}" ]] && xstest_arg=(--xstest_path "${XSTEST_PATH}")
  python -m onereplay.scripts.prepare_safety_data \
    --harmbench_csv "${HARMBENCH_CSV}" \
    "${itw_arg[@]}" "${xstest_arg[@]}" \
    --num_behaviors "${NUM_BEHAVIORS}" \
    --num_itw_templates "${NUM_ITW_TEMPLATES}" \
    --output_path "${SAFETY_PROMPTS}"
fi

# ---- 待评模型（run_name;adapter_path，adapter 为空即 base）----
RUNS="${RUNS:-base;|vanilla_seed${SEED};${RESULTS_ROOT}/adapters/cs_vanilla_seed${SEED}|onereplay_lam3e-2_seed${SEED};${RESULTS_ROOT}/adapters/cs_onereplay_lam3e-2_seed${SEED}_regonce}"

run_safety () {
  local name=$1
  local adapter=$2
  local -a adapter_arg=()
  if [[ -n "${adapter}" ]]; then adapter_arg=(--adapter_path "${adapter}"); fi

  echo "==== safety gen: ${name} ===="
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    "${adapter_arg[@]}" --run_name "${name}" \
    --metrics safety --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
    --safety_prompts "${SAFETY_PROMPTS}" \
    --safety_max_new_tokens "${SAFETY_MAX_NEW_TOKENS}"
}

# ---- 阶段 1：逐模型生成，收集各自的 responses 路径 ----
RESP_LIST=""
IFS='|' read -rA run_specs <<< "${RUNS}"
for spec in "${run_specs[@]}"; do
  name="${spec%%;*}"
  adapter="${spec#*;}"
  run_safety "${name}" "${adapter}"
  resp="${RESULTS_ROOT}/safety/${name}/responses.jsonl"
  if [[ -z "${RESP_LIST}" ]]; then RESP_LIST="${resp}"; else RESP_LIST="${RESP_LIST},${resp}"; fi
done

# ---- 阶段 2：WildGuard 一次性判分本次生成的所有 run ----
echo "==== safety judge: ${RESP_LIST} ===="
python -m onereplay.scripts.judge_safety \
  --responses "${RESP_LIST}" \
  --judge_model_dir "${JUDGE_MODEL_DIR}" --judge_model_name "${JUDGE_MODEL_NAME}" \
  --out_dir "${JUDGE_OUT}" \
  --batch_size "${JUDGE_BATCH}"

echo "Job finished at: $(date)"
