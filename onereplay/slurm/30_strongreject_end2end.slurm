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
#
#   待评的五条线与 28 号一致：base / vanilla / EWC(λ=3e2) / Replay(4n4r) /
#   OneReplay(λ=3e-2)。生成约 2.7h/run，全部跑满约 14h，所以断点续跑是必需的：
#   已有 summary.json 的 run 跳过生成，已有 scored_sr.jsonl 的 run 跳过评分
#   （FORCE_GEN=1 / FORCE_SCORE=1 强制重跑），超时被砍后原样再投即可接着跑。
#   完成标记从 responses.jsonl 换成 summary.json：前者是边生成边追加写的，作业
#   被砍时会留下"非空但只写了一半"的文件，用它当标记会把截断结果当成正式数据。
#
# 用法:
#   sbatch onereplay/slurm/30_strongreject_end2end.slurm
#   先看该做什么、不真跑: DRY_RUN=1 sbatch onereplay/slurm/30_strongreject_end2end.slurm
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

FORCE_GEN="${FORCE_GEN:-0}"       # 1 = 即使已有 responses 也重新生成
FORCE_SCORE="${FORCE_SCORE:-0}"   # 1 = 即使已有 scored_sr.jsonl 也重新评分
DRY_RUN="${DRY_RUN:-0}"

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
#   顺序即报告顺序，base 第一位。run 名与 28 号逐字相同，两套判分口径才能对上。
RUNS="${RUNS:-base;|vanilla_seed${SEED};${RESULTS_ROOT}/adapters/cs_vanilla_seed${SEED}|ewc_lam3e2_seed${SEED};${RESULTS_ROOT}/adapters/cs_ewc_lam3e2_seed${SEED}|replaymix_4n4r_seed${SEED};${RESULTS_ROOT}/adapters/cs_replaymix_4n4r_seed${SEED}|onereplay_lam3e-2_seed${SEED};${RESULTS_ROOT}/adapters/cs_onereplay_lam3e-2_seed${SEED}_regonce}"

IFS='|' read -rA run_specs <<< "${RUNS}"

# ---------------------------------------------------------------------------
# 前置检查：adapter 目录名 + 各 run 已完成到哪一步
# 生成一条要 2.7h，名字猜错要等很久才报错，所以先把五条核一遍。
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "现有 adapter（${RESULTS_ROOT}/adapters）:"
ls -1 "${RESULTS_ROOT}/adapters" 2>/dev/null || echo "  (目录不存在)"
echo ""
printf "%-30s %-8s %-8s %s\n" "run_name" "生成" "评分" "adapter"
missing_adapter=0
for spec in "${run_specs[@]}"; do
  name="${spec%%;*}"
  adapter="${spec#*;}"
  gen_state="待跑"
  if [[ "${FORCE_GEN}" != "1" && -f "${SR_ROOT}/safety/${name}/summary.json" ]]; then
    gen_state="已有"
  fi
  score_state="待跑"
  if [[ "${FORCE_SCORE}" != "1" && -s "${SR_JUDGE_OUT}/${name}/scored_sr.jsonl" ]]; then
    score_state="已有"
  fi
  if [[ -n "${adapter}" && ! -d "${adapter}" ]]; then
    printf "%-30s %-8s %-8s %s\n" "${name}" "${gen_state}" "${score_state}" "!! 缺 ${adapter}"
    missing_adapter=1
  else
    printf "%-30s %-8s %-8s %s\n" "${name}" "${gen_state}" "${score_state}" "${adapter:-<裸 base>}"
  fi
done
echo "========================================================================"

if [[ "${missing_adapter}" == "1" ]]; then
  echo "有 adapter 目录不存在。核对 run 名后用 RUNS=... 覆盖："
  echo "  EWC 的 λ 网格   见 onereplay/slurm/23_sweep_ewc_lambda.slurm"
  echo "  Replay 的 n/r 档 见 onereplay/slurm/18_sweep_replay_batchmix.slurm"
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1：只做前置检查，不生成也不评分。"
  echo "Job finished at: $(date)"
  exit 0
fi

run_sr_gen () {
  local name=$1
  local adapter=$2
  local -a adapter_arg=()
  if [[ -n "${adapter}" ]]; then adapter_arg=(--adapter_path "${adapter}"); fi

  # 完成标记用 summary.json（只在跑完所有 prompt 后写），不用边写边追加的
  # responses.jsonl，否则被砍掉的半截结果会被当成完成。
  local done_marker="${SR_ROOT}/safety/${name}/summary.json"
  if [[ "${FORCE_GEN}" != "1" && -f "${done_marker}" ]]; then
    echo "==== strongreject gen: ${name} (跳过，已有 ${done_marker}) ===="
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

# ---- 阶段 1：逐模型生成；只把还没评分的 run 收进评分清单 ----
RESP_LIST=""
RUN_NAMES=""
for spec in "${run_specs[@]}"; do
  name="${spec%%;*}"
  adapter="${spec#*;}"
  run_sr_gen "${name}" "${adapter}"
  if [[ -z "${RUN_NAMES}" ]]; then RUN_NAMES="${name}"; else RUN_NAMES="${RUN_NAMES},${name}"; fi
  if [[ "${FORCE_SCORE}" != "1" && -s "${SR_JUDGE_OUT}/${name}/scored_sr.jsonl" ]]; then
    echo "已有评分结果，跳过 grade: ${name}"
    continue
  fi
  resp="${SR_ROOT}/safety/${name}/responses.jsonl"
  if [[ -z "${RESP_LIST}" ]]; then RESP_LIST="${resp}"; else RESP_LIST="${RESP_LIST},${resp}"; fi
done

# ---- 阶段 2：StrongREJECT grader 一次性评分待评的所有 run ----
if [[ -n "${RESP_LIST}" ]]; then
  echo "==== strongreject grade: ${RESP_LIST} ===="
  base_arg=()
  if [[ -n "${SR_BASE}" ]]; then base_arg=(--judge_base "${SR_BASE}"); fi
  python -m onereplay.scripts.judge_strongreject \
    --responses "${RESP_LIST}" \
    --judge_model_dir "${SR_MODEL_DIR}" --judge_model_name "${SR_MODEL_NAME}" \
    "${base_arg[@]}" \
    --out_dir "${SR_JUDGE_OUT}" \
    --batch_size "${SR_BATCH}"
else
  echo "==== strongreject grade: 所有 run 都已评分，跳过（FORCE_SCORE=1 可强制重评）===="
fi

# ---- 阶段 2b：重建总汇总（judge 只写本次评的那几个 run）----
echo "==== rebuild ${SR_JUDGE_OUT}/strongreject_all_runs.json ===="
python - "${SR_JUDGE_OUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
overall = {}
for summary_path in sorted(root.glob("*/strongreject_summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_name = summary.pop("run_name", summary_path.parent.name)
    # 与 judge_strongreject 写出的形状一致：只留 summarize_run 的字段
    summary.pop("num_scored", None)
    summary.pop("binary_threshold", None)
    overall[run_name] = summary
(root / "strongreject_all_runs.json").write_text(
    json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"汇总了 {len(overall)} 个 run: {sorted(overall)}")
PY

# ---- 阶段 3：配对统计分析（均值差 bootstrap + 二值化 McNemar + 攻击增益）----
echo "==== strongreject analyze -> ${SR_REPORT} ===="
python -m onereplay.scripts.analyze_strongreject \
  --scored "${SR_JUDGE_OUT}/*/scored_sr.jsonl" \
  --runs "${RUN_NAMES}" \
  --out_path "${SR_REPORT}"

echo "Job finished at: $(date)"
