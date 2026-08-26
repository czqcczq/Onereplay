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
# 问题 1 - 安全性评测（生成 + 判分 + 配对统计，一次跑完）
#   五条线共用同一个 prompts 文件，三条 track 共用 HarmBench：
#     A  直接问              -> 良性微调是否冲掉了 base 对齐
#     B  behavior × 越狱外壳 -> 固定 transfer attack，所有模型吃相同输入
#     C  XSTest              -> 过度拒绝对照
#   base / vanilla / onereplay 已在 2026-08-18 定稿（作业 3049367，结论见
#   results_log/2026-08-18_安全性保留_问题1定稿.md），本次补 EWC(λ=3e2) 与
#   Replay(batch-mix 4n4r) 两条正则化/回放家族的对照。
#
#   断点续跑：已有 summary.json 的 run 跳过生成（每条约 2.3h），已有 judged.jsonl
#   的 run 跳过判分，所以直接重投只会补新增的那几条，旧那三条的数字原样保留、
#   不会被重算。FORCE_GEN=1 / FORCE_JUDGE=1 可强制重跑。
#   判定生成完成看 summary.json 而不是 responses.jsonl：后者是边生成边追加写的，
#   作业被墙时间砍掉时会留下一个"非空但只写了一半"的文件，拿它当完成标记会把
#   截断的结果当成正式数据；summary.json 只在全部 prompt 跑完后才落盘。
#
#   判分脚本每次只把「本次判的 run」写进 safety_all_runs.json，所以判完之后由本
#   脚本从各 run 的 safety_summary.json 重建这个汇总文件，跳过的 run 不会掉。
#
#   分步版见 29_judge_safety.slurm（只重判分，改了 judge 逻辑时用）。
#   先在有网节点下载 HarmBench / In-The-Wild / XSTest 及 allenai/wildguard。
#
# 用法:
#   sbatch onereplay/slurm/28_safety_end2end.slurm
#   先看该做什么、不真跑:  DRY_RUN=1 sbatch onereplay/slurm/28_safety_end2end.slurm
#   只补一条线:
#     RUNS="ewc_lam3e2_seed1;/hpcwork/.../adapters/cs_ewc_lam3e2_seed1" sbatch ...
#   （只传一条时统计分析会退化成单 run，配对结论仍看默认的五条一起跑）
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
SAFETY_REPORT="${SAFETY_REPORT:-${JUDGE_OUT}/safety_analysis.json}"

FORCE_GEN="${FORCE_GEN:-0}"       # 1 = 即使已有 responses 也重新生成
FORCE_JUDGE="${FORCE_JUDGE:-0}"   # 1 = 即使已有 judged.jsonl 也重新判分
DO_ANALYZE="${DO_ANALYZE:-1}"     # 判分后跑 McNemar / bootstrap / 退化诊断
DRY_RUN="${DRY_RUN:-0}"

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
#   顺序即报告顺序，base 放第一位（配对比较里它是"未受损"的参照）。
#   EWC 的 λ=3e2 是 23 号 λ 网格的最优档，Replay 的 4n4r 是 18 号的 50% 档，
#   两者与 vanilla / onereplay 同数据、同 3 epoch、同 seed，详见
#   results_log/结果汇总_性能与开销.md 第一章 1.0。
RUNS="${RUNS:-base;|vanilla_seed${SEED};${RESULTS_ROOT}/adapters/cs_vanilla_seed${SEED}|ewc_lam3e2_seed${SEED};${RESULTS_ROOT}/adapters/cs_ewc_lam3e2_seed${SEED}|replaymix_4n4r_seed${SEED};${RESULTS_ROOT}/adapters/cs_replaymix_4n4r_seed${SEED}|onereplay_lam3e-2_seed${SEED};${RESULTS_ROOT}/adapters/cs_onereplay_lam3e-2_seed${SEED}_regonce}"

IFS='|' read -rA run_specs <<< "${RUNS}"

# ---------------------------------------------------------------------------
# 前置检查
# adapter 目录名猜错是这条线上最容易踩的坑（EWC 的名字由 λ 拼出来，Replay 的由
# n/r 拼出来），生成一条要 2.3h，所以先把五条全部核一遍再动手，并把 adapters/
# 的实际内容打出来，方便一眼看出该传什么 run 名。
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "现有 adapter（${RESULTS_ROOT}/adapters）:"
ls -1 "${RESULTS_ROOT}/adapters" 2>/dev/null || echo "  (目录不存在)"
echo ""
printf "%-30s %-8s %-8s %s\n" "run_name" "生成" "判分" "adapter"
missing_adapter=0
for spec in "${run_specs[@]}"; do
  name="${spec%%;*}"
  adapter="${spec#*;}"
  gen_state="待跑"
  if [[ "${FORCE_GEN}" != "1" && -f "${RESULTS_ROOT}/safety/${name}/summary.json" ]]; then
    gen_state="已有"
  fi
  judge_state="待跑"
  if [[ "${FORCE_JUDGE}" != "1" && -s "${JUDGE_OUT}/${name}/judged.jsonl" ]]; then
    judge_state="已有"
  fi
  if [[ -n "${adapter}" && ! -d "${adapter}" ]]; then
    printf "%-30s %-8s %-8s %s\n" "${name}" "${gen_state}" "${judge_state}" "!! 缺 ${adapter}"
    missing_adapter=1
  else
    printf "%-30s %-8s %-8s %s\n" "${name}" "${gen_state}" "${judge_state}" "${adapter:-<裸 base>}"
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
  echo "DRY_RUN=1：只做前置检查，不生成也不判分。"
  echo "Job finished at: $(date)"
  exit 0
fi

run_safety () {
  local name=$1
  local adapter=$2
  local -a adapter_arg=()
  if [[ -n "${adapter}" ]]; then adapter_arg=(--adapter_path "${adapter}"); fi

  # 完成标记用 summary.json（只在跑完所有 prompt 后写），不用边写边追加的
  # responses.jsonl，否则被砍掉的半截结果会被当成完成。
  local done_marker="${RESULTS_ROOT}/safety/${name}/summary.json"
  if [[ "${FORCE_GEN}" != "1" && -f "${done_marker}" ]]; then
    echo "==== safety gen: ${name} (跳过，已有 ${done_marker}) ===="
    return 0
  fi

  echo "==== safety gen: ${name} ===="
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    "${adapter_arg[@]}" --run_name "${name}" \
    --metrics safety --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
    --safety_prompts "${SAFETY_PROMPTS}" \
    --safety_max_new_tokens "${SAFETY_MAX_NEW_TOKENS}"
}

# ---- 阶段 1：逐模型生成；只把还没判分的 run 收进判分清单 ----
JUDGE_LIST=""
RUN_NAMES=""
for spec in "${run_specs[@]}"; do
  name="${spec%%;*}"
  adapter="${spec#*;}"
  run_safety "${name}" "${adapter}"
  if [[ -z "${RUN_NAMES}" ]]; then RUN_NAMES="${name}"; else RUN_NAMES="${RUN_NAMES},${name}"; fi
  if [[ "${FORCE_JUDGE}" != "1" && -s "${JUDGE_OUT}/${name}/judged.jsonl" ]]; then
    echo "已有判分结果，跳过 judge: ${name}"
    continue
  fi
  resp="${RESULTS_ROOT}/safety/${name}/responses.jsonl"
  if [[ -z "${JUDGE_LIST}" ]]; then JUDGE_LIST="${resp}"; else JUDGE_LIST="${JUDGE_LIST},${resp}"; fi
done

# ---- 阶段 2：WildGuard 一次性判分待判的所有 run（约 5 min/run）----
if [[ -n "${JUDGE_LIST}" ]]; then
  echo "==== safety judge: ${JUDGE_LIST} ===="
  python -m onereplay.scripts.judge_safety \
    --responses "${JUDGE_LIST}" \
    --judge_model_dir "${JUDGE_MODEL_DIR}" --judge_model_name "${JUDGE_MODEL_NAME}" \
    --out_dir "${JUDGE_OUT}" \
    --batch_size "${JUDGE_BATCH}"
else
  echo "==== safety judge: 所有 run 都已判分，跳过（FORCE_JUDGE=1 可强制重判）===="
fi

# ---- 阶段 2b：从各 run 的 summary 重建总汇总（judge 只写本次判的那几个）----
echo "==== rebuild ${JUDGE_OUT}/safety_all_runs.json ===="
python - "${JUDGE_OUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
overall = {}
for summary_path in sorted(root.glob("*/safety_summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overall[summary.get("run_name", summary_path.parent.name)] = summary["per_track"]
(root / "safety_all_runs.json").write_text(
    json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"汇总了 {len(overall)} 个 run: {sorted(overall)}")
PY

# ---- 阶段 3：配对统计（McNemar 精确检验 + bootstrap CI + 退化 / 净率诊断）----
if [[ "${DO_ANALYZE}" == "1" ]]; then
  echo "==== safety analyze -> ${SAFETY_REPORT} ===="
  python -m onereplay.scripts.analyze_safety \
    --judged "${JUDGE_OUT}/*/judged.jsonl" \
    --runs "${RUN_NAMES}" \
    --out_path "${SAFETY_REPORT}" || true
fi

echo "Job finished at: $(date)"
