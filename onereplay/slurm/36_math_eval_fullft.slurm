#!/usr/bin/zsh
#
#SBATCH --job-name=or_math_full
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

# =============================================================================
# 全参微调下的数学 retention：base vs full_vanilla vs full_onereplay
#   只评测，不训练。checkpoint 是 11 号 (11_train_full_baselines.slurm) 的产物。
#
# -----------------------------------------------------------------------------
# 这个作业要回答什么
#   LoRA 侧（r=8, q_proj/v_proj, 0.093% 参数）已经测清楚：Commonsense170k SFT
#   **没有**损害数学解题能力。在 vanilla 正常作答的子集上 72.6% vs 72.2%
#   (MATH-500, p=0.905)、77.9% vs 80.1% (GSM8K, p=0.094)；被摧毁的是输出格式与
#   终止行为（GSM8K 缺 '####' 从 0.2% 涨到 22.4%）。详见
#   results_log/08-25_math_结果.md。
#
#   于是"OneReplay 能不能保住数学"这个问题无法回答——要保护的东西从没被破坏过。
#   本作业就是去找一个**数学真的会掉**的配置。全参微调是最直接的候选：同样的
#   数据、同样的 3 epoch，但动全部 1.72B 参数而不是 1.6M。
#
#   判定门槛（必须事先定死，且要高于量程）:
#     GSM8K   (1319 题) 配对可分辨下限 2.26pp -> 遗忘需 >= 8pp  才算这条路走得通
#     MATH-500 (500 题) 配对可分辨下限 3.51pp -> 遗忘需 >= 10pp
#   达不到这个量级，就说明问题不在"参数量太少"，而在 Commonsense170k 的梯度里
#   本来就没有破坏算术的成分，那时候该换的是新任务而不是 LoRA 配置。
#
# -----------------------------------------------------------------------------
# 为什么写到独立的 OUT_DIR 而不是 ${RESULTS_ROOT}
#   RWTH 的 ${RESULTS_ROOT}/{math500,gsm8k}/ 下已经有一批**不可用**的旧结果:
#   作业 3073173 跑在 MATH_MAX_NEW_TOKENS=1024，而且早于 2026-08-24 的 gsm8k
#   '####' 抽取修复（commit c91d634），那一版 predicted_answer 直接取
#   split("####")[-1]，撞上 Qwen 的 "#### 18 ####" 会拿到空串判成没作答 ——
#   所以那次 base 的 GSM8K 只有 3.8%（正确值约 77%）。
#
#   evaluate 的断点续跑逻辑是"summary.json 存在就跳过"，如果写进同一个目录，
#   base 会被那批坏结果挡住、静默跳过，整个对比作废。所以本作业写到
#   ${RESULTS_ROOT}/math_fullft/，全部数字由本作业自己产出，同 cap、同代码版本，
#   自成闭环，也不去动旧目录。
#
# -----------------------------------------------------------------------------
# MATH_MAX_NEW_TOKENS = 4096
#   与 Hopper 侧的 LoRA 结果对齐。MATH-500 判分靠抽最后一个 \boxed{...}，被砍断
#   即记错；1792 下实测截断率 12-13%，放开到 4096 后绝对准确率约 +1.2pt。
#   本作业内所有 run 用同一个值，这是可比的前提。
#
# -----------------------------------------------------------------------------
# MODELS 的顺序是有意的
#   base -> full_vanilla -> full_onereplay，前两个就是决定性的那一对。24h 墙时间
#   下三个模型约 20h（每模型 GSM8K 约 2-3h + MATH-500 约 4h），万一超时被砍，
#   已完成的 run 有 summary.json、原样再 sbatch 一次即可接着跑，而且最先拿到的
#   一定是能下结论的那两行。
#
#   lora_vanilla 是可选的第四行：同集群下 LoRA 与全参的损伤幅度对比。默认不评
#   （Hopper 上已有它的数字，只是跨集群），需要时加进 MODELS。
#
# -----------------------------------------------------------------------------
# 用法:
#   sbatch onereplay/slurm/36_math_eval_fullft.slurm
#
#   只跑决定性的那一对:   MODELS="base full_vanilla" sbatch ...
#   加上同集群 LoRA 对照: MODELS="base full_vanilla full_onereplay lora_vanilla" sbatch ...
#   冒烟（每指标前 20 题，十几分钟）: LIMIT=20 sbatch ...
#   只看前置检查不评测:   DRY_RUN=1 sbatch ...
#   全参 OneReplay 换 λ:  FULL_LAMBDA=1e-3 sbatch ...
#
# 前置条件:
#   1) checkpoint 就位（11 号 MODE=train 时 save=1 才会落盘，约 3.4 GiB/run）:
#        ${RESULTS_ROOT}/checkpoints/full/full_vanilla_seed1
#        ${RESULTS_ROOT}/checkpoints/full/full_onereplay_lam1e-4_seed1
#      全参 checkpoint 是完整模型（config.json + safetensors + tokenizer），不是
#      LoRA adapter；evaluate 靠目录里有没有 adapter_config.json 自动区分，所以
#      仍然通过 --adapter_path 传进去，与 12_eval_full_replay.slurm 同一套路。
#   2) 数学数据: ${MATH_DATA_DIR}/{math500,gsm8k}_test.jsonl
#   3) 诊断脚本: onereplay/scripts/diagnose_math_pairs.py（评完自动跑配对检验）
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

# ---- 项目路径（与 11/12 号保持一致）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
CKPT_DIR="${CKPT_DIR:-${RESULTS_ROOT}/checkpoints/full}"
ADAPTERS="${ADAPTERS:-${RESULTS_ROOT}/adapters}"
MATH_DATA_DIR="${MATH_DATA_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/math}"

# 见文件头：独立输出根，避免撞上 1024 预算 + 旧抽取逻辑留下的坏结果
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/math_fullft}"

# ---- run 名 ----
SEED="${SEED:-1}"
FULL_LAMBDA="${FULL_LAMBDA:-1e-4}"
FULL_VANILLA_RUN="${FULL_VANILLA_RUN:-full_vanilla_seed${SEED}}"
FULL_ONEREPLAY_RUN="${FULL_ONEREPLAY_RUN:-full_onereplay_lam${FULL_LAMBDA}_seed${SEED}}"
LORA_VANILLA_RUN="${LORA_VANILLA_RUN:-cs_vanilla_seed${SEED}}"
MODELS="${MODELS:-base full_vanilla full_onereplay}"

# ---- 评测参数 ----
MATH500_FILE="${MATH500_FILE:-${MATH_DATA_DIR}/math500_test.jsonl}"
GSM8K_FILE="${GSM8K_FILE:-${MATH_DATA_DIR}/gsm8k_test.jsonl}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-4096}"
LIMIT="${LIMIT:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
DO_DIAG="${DO_DIAG:-1}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺基座模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
mkdir -p "${OUT_DIR}"

# gsm8k 放前面：它比 MATH-500 快得多，超时被砍时先拿到的是完整的那个指标
ALL_METRICS=""
if [[ -f "${GSM8K_FILE}" ]]; then ALL_METRICS="${ALL_METRICS} gsm8k"; else echo "跳过 gsm8k：缺 ${GSM8K_FILE}"; fi
if [[ -f "${MATH500_FILE}" ]]; then ALL_METRICS="${ALL_METRICS} math500"; else echo "跳过 math500：缺 ${MATH500_FILE}"; fi
ALL_METRICS="${ALL_METRICS# }"
[[ -n "${ALL_METRICS}" ]] || { echo "两个数学数据文件都不在: ${MATH_DATA_DIR}"; exit 1; }

if [[ "${LIMIT}" != "0" ]]; then
  echo "!! LIMIT=${LIMIT}：冒烟模式，产出的 summary.json 只覆盖前 ${LIMIT} 题，"
  echo "!! 不是正式结果。验证完请删掉 ${OUT_DIR} 再用 LIMIT=0 重跑。"
fi

# ---------------------------------------------------------------------------
# 模型解析
#   base         -> adapter_path 为空，加载裸 Qwen3-1.7B
#   full_*       -> adapter_path 指向全参 checkpoint 目录。evaluate 里的
#                   is_lora_adapter_dir() 看目录有没有 adapter_config.json，
#                   没有就把整个目录当完整模型加载（tokenizer 也在里面），
#                   此时传进去的 --model_dir/--model_name 会被忽略。
#   lora_vanilla -> adapter_path 指向 LoRA adapter，走 PeftModel 挂载
# ---------------------------------------------------------------------------
resolve_path () {
  case "$1" in
    base)           echo "" ;;
    full_vanilla)   echo "${CKPT_DIR}/${FULL_VANILLA_RUN}" ;;
    full_onereplay) echo "${CKPT_DIR}/${FULL_ONEREPLAY_RUN}" ;;
    lora_vanilla)   echo "${ADAPTERS}/${LORA_VANILLA_RUN}" ;;
    *)              echo "" ;;
  esac
}

resolve_name () {
  case "$1" in
    base)           echo "base" ;;
    full_vanilla)   echo "${FULL_VANILLA_RUN}" ;;
    full_onereplay) echo "${FULL_ONEREPLAY_RUN}" ;;
    lora_vanilla)   echo "${LORA_VANILLA_RUN}" ;;
    *)              echo "$1" ;;
  esac
}

# ---------------------------------------------------------------------------
# 前置检查
# checkpoint 名字猜错是这条线上最容易踩的坑（11 号的 run 名由 λ 拼出来，14 号
# 扫过 1e-3/3e-3/1e-2/3e-2 四档），路径不存在时 evaluate 会直接报错退出，所以
# 这里先列出 CKPT_DIR 的实际内容，方便一眼看出该传什么 FULL_LAMBDA。
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "现有全参 checkpoint（${CKPT_DIR}）:"
ls -1 "${CKPT_DIR}" 2>/dev/null || echo "  (目录不存在)"
echo ""
echo "前置检查  METRICS=${ALL_METRICS}  cap=${MATH_MAX_NEW_TOKENS}  OUT_DIR=${OUT_DIR}"
printf "%-16s %-40s %s\n" "short" "run_name" "待评指标 / 跳过原因"
for short in ${=MODELS}; do
  name="$(resolve_name "${short}")"
  path="$(resolve_path "${short}")"
  if [[ -n "${path}" && ! -d "${path}" ]]; then
    printf "%-16s %-40s %s\n" "${short}" "${name}" "!! 缺模型目录: ${path}"
    continue
  fi
  todo=""
  for m in ${=ALL_METRICS}; do
    if [[ "${FORCE_EVAL}" != "1" && -f "${OUT_DIR}/${m}/${name}/summary.json" ]]; then
      continue
    fi
    todo="${todo},${m}"
  done
  todo="${todo#,}"
  if [[ -z "${todo}" ]]; then
    printf "%-16s %-40s %s\n" "${short}" "${name}" "全部已有结果，跳过"
  else
    printf "%-16s %-40s %s\n" "${short}" "${name}" "${todo}"
  fi
done
echo "========================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1：只做前置检查，不评测。"
  echo "Job finished at: $(date)"
  exit 0
fi

set -x

# ---------------------------------------------------------------------------
# 评测：每个 (指标, run) 已有 summary.json 就跳过，超时后原样再投可以接着跑，
# 也避免 *_summary.csv 里追加出重复行。
# ---------------------------------------------------------------------------
pending_metrics () {
  local name=$1
  local out="" m
  for m in ${=ALL_METRICS}; do
    if [[ "${FORCE_EVAL}" != "1" && -f "${OUT_DIR}/${m}/${name}/summary.json" ]]; then
      echo "已有结果，跳过 ${m}/${name}" >&2
      continue
    fi
    out="${out},${m}"
  done
  echo "${out#,}"
}

eval_one () {
  local short=$1
  local name path metrics
  name="$(resolve_name "${short}")"
  path="$(resolve_path "${short}")"

  if [[ -n "${path}" && ! -d "${path}" ]]; then
    echo "跳过 ${short}：缺模型目录 ${path}"
    return 0
  fi

  metrics="$(pending_metrics "${name}")"
  if [[ -z "${metrics}" ]]; then
    echo "跳过 ${name}：所有指标都已有结果（FORCE_EVAL=1 可强制重评）"
    return 0
  fi

  echo "==== eval math: ${name} (${metrics}) ===="
  echo "model path: ${path:-<裸 base>}"
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    --adapter_path "${path}" --run_name "${name}" \
    --metrics "${metrics}" --out_dir "${OUT_DIR}" --seed "${SEED}" \
    --math_max_new_tokens "${MATH_MAX_NEW_TOKENS}" --limit "${LIMIT}" \
    --math500_data_path "${MATH500_FILE}" \
    --gsm8k_data_path "${GSM8K_FILE}"
}

for short in ${=MODELS}; do
  eval_one "${short}"
done

# ---------------------------------------------------------------------------
# 后置诊断：配对 McNemar + 抽取健康度 + 增益来源分解
# 只读 responses.jsonl，几分钟。--model_path 用于统计 token 长度，顺便确认所有
# run 的 token max 一致（预算不一致是这条线上出过一次的错）。
# ---------------------------------------------------------------------------
if [[ "${DO_DIAG}" == "1" ]]; then
  DIAG_RUNS=""
  for short in ${=MODELS}; do
    DIAG_RUNS="${DIAG_RUNS},$(resolve_name "${short}")"
  done
  DIAG_RUNS="${DIAG_RUNS#,}"

  echo "==== 诊断 A：以 base 为基准（遗忘量看这个）===="
  python -m onereplay.scripts.diagnose_math_pairs \
    --results_root "${OUT_DIR}" --metrics "${ALL_METRICS// /,}" \
    --runs "${DIAG_RUNS}" --baseline "base" \
    --cap "${MATH_MAX_NEW_TOKENS}" --model_path "${MODEL_DIR}/${MODEL_NAME}" || true

  if [[ "${MODELS}" == *full_onereplay* && "${MODELS}" == *full_vanilla* ]]; then
    echo "==== 诊断 B：以 full_vanilla 为基准（保护效果看这个）===="
    python -m onereplay.scripts.diagnose_math_pairs \
      --results_root "${OUT_DIR}" --metrics "${ALL_METRICS// /,}" \
      --runs "${FULL_VANILLA_RUN},${FULL_ONEREPLAY_RUN}" --baseline "${FULL_VANILLA_RUN}" \
      --cap "${MATH_MAX_NEW_TOKENS}" --model_path "${MODEL_DIR}/${MODEL_NAME}" || true
  fi
fi

set +x
echo "========================================================================"
echo "产物:"
echo "  ${OUT_DIR}/{gsm8k,math500}/<run_name>/{summary.json,responses.jsonl}"
echo "  ${OUT_DIR}/{gsm8k,math500}_summary.csv"
echo ""
echo "怎么下结论 —— 看诊断 A 里 full_vanilla 相对 base 的 Δacc 和 p:"
echo "  GSM8K   遗忘 >= 8pp  且显著  => 全参这条路走得通，按此配置重开一轮"
echo "  MATH-500 遗忘 >= 10pp 且显著  => 同上"
echo "  两个都够不到量程（GSM8K 2.26pp / MATH-500 3.51pp）=> 问题不在参数量，"
echo "    而在 Commonsense170k 的梯度里没有破坏算术的成分，该换的是新任务。"
echo ""
echo "同时看「抽取健康度」里的 no-marker 一列：LoRA 侧 vanilla 是 22.4%（base 0.2%），"
echo "如果全参下这个数更高而 accuracy 仍不掉，那就再次印证'被摧毁的是格式不是数学'。"
echo "Job finished at: $(date)"
