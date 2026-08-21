#!/usr/bin/zsh
#
#SBATCH --job-name=or_covsd
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# Stage 1-sd: 用自蒸馏语料重采一份 C = E[x x^T]（"我们的方法 + 自蒸馏"）
#
# 为什么要这一步
# -------------
# 01_collect_cov 采的 C 用的是 FLAN prompt + FLAN gold target。gold target 是
# 一行式答案键，对一个已经指令微调过的 base 模型来说那是"另一个任务"的输出风格，
# 不是它自己的行为。而 replay 自蒸馏 baseline 训练的是 base 模型自己的回答。
# 两条路线因此在"要保住什么"上口径不同：
#
#   replay 自蒸馏   锚定 base 模型在 FLAN prompt 上的真实行为
#   OneReplay(gold) C 里混进了 gold 答案键的 hidden state
#
# 本脚本把 C 的文本源换成 13 号 job 生成的那份自蒸馏语料（同一批 prompt，
# target 是 base 模型自己的贪心回答），于是两条路线消费完全同一份数据，
# 差别只剩"怎么用"：压成二阶矩一次性带走，还是留着语料反复重训。
#
# 与 replay 严格同源的三个保证
# ---------------------------
#   1. 行集合相同  语料本身就是 shuffle(seed=1)+select(20000) 的产物，
#      --require_target 1 丢掉 target 为空的行（撞上生成预算没吐停止符的，
#      以及 prompt 过长直接跳过的），与 replay 侧 --replay_drop_truncated 1
#      加 to_sft_schema 的非空过滤等价。--max_samples 0 表示全都要，不再二次抽样。
#   2. 截断方向相同  训练侧是 full_input_ids[-max_length:]，即左截断、保住尾部
#      答案。collect_cov 默认走 tokenizer 的右截断，会先把 assistant 回答砍掉。
#      gold target 只有一行，几乎不会溢出 512，所以这个差异以前不显形；自蒸馏
#      答案最长 384 token，不对齐的话 C 可能根本没看到答案。--truncation_side left。
#   3. 渲染方式相同  仍然 chat 模板 + include_target_in_chat 1，user=inputs、
#      assistant=targets，和训练侧 apply_train_template 的消息结构一致。
#
# 其余采集参数（模型、max_len、batch_size、target_modules）与 01 号逐字一致，
# 保证和 gold C 之间唯一的变量就是文本源。
#
# 用法:
#   sbatch onereplay/slurm/19_collect_cov_selfdistill.slurm                  # 默认 qv（配 LoRA）
#   COV_SCOPE=full sbatch onereplay/slurm/19_collect_cov_selfdistill.slurm   # 配全量微调
#
# 跑完先看末尾那张 trace 对比表，再决定 20 号脚本的 λ，理由见那一段的说明。
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

# ---- 项目路径（与 01 号一致）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"
MAXLEN="${MAXLEN:-512}"

# ---- 自蒸馏语料：POOL_SIZE / SAMPLE_SEED 必须与 13 号 job 生成时一致 ----
POOL_SIZE="${POOL_SIZE:-20000}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"
SELF_DISTILL_FILE="${SELF_DISTILL_FILE:-${RESULTS_ROOT}/replay/flan_selfdistill_${POOL_SIZE}_seed${SAMPLE_SEED}.jsonl}"

# ---- 覆盖范围预设，与 01 号相同；文件名加 selfdistill 以区分数据源 ----
COV_SCOPE="${COV_SCOPE:-qv}"
case "${COV_SCOPE}" in
  qv)
    TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
    COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_selfdistill_20k_qv.pt}"
    GOLD_COV="${GOLD_COV:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
    ;;
  full)
    TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head}"
    COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_selfdistill_20k_full.pt}"
    GOLD_COV="${GOLD_COV:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_full.pt}"
    ;;
  *)
    echo "未知 COV_SCOPE=${COV_SCOPE}，可选 qv / full"; exit 1
    ;;
esac
echo "COV_SCOPE=${COV_SCOPE} TARGET_MODULES=${TARGET_MODULES}"
echo "SELF_DISTILL_FILE=${SELF_DISTILL_FILE}"
echo "COV_PATH=${COV_PATH}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -f "${SELF_DISTILL_FILE}" ]] || {
  echo "缺自蒸馏语料: ${SELF_DISTILL_FILE}"
  echo "先跑 13_replay_selfdistill.slurm（SKIP_TRAIN=1 就够，只要生成那一步）"
  exit 1
}
mkdir -p "$(dirname "${COV_PATH}")" "${DATASET_CACHE}"

# ---------------------------------------------------------------------------
# Step 1: 采集 C
#
# --sample_shuffle 0 且 --max_samples 0：语料已经是 seed=1 抽样后的池子，
# 这里要的就是它过滤后的全部行，不做二次抽样。C 是求和统计量，读入顺序不影响结果。
# ---------------------------------------------------------------------------
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
  --output_path "${COV_PATH}"

# ---------------------------------------------------------------------------
# Step 2: 与 gold C 对比尺度
#
# 惩罚是 λ · (1/L) Σ_l tr(ΔW_l C_l ΔW_l^T)，对 C 的绝对尺度是线性的。也就是说
# 换一份 C 之后，同一个 λ 的实际正则强度会按 trace 的比值等比变化：如果自蒸馏 C
# 的平均 trace 是 gold 的 1.5 倍，那 λ=1e-2 打出来的效果相当于 gold 下的 1.5e-2。
#
# 所以这张表是给 20 号脚本选 λ 用的：
#   ratio 落在 0.8~1.25   直接用 λ=1e-2，就是同一强度的干净对照
#   ratio 偏离更多        要么额外跑一档 λ=1e-2/ratio 做等效强度锚点，
#                         要么按 20 号脚本注释里的办法扫 3e-3 / 1e-2 / 3e-2
#
# 缺 gold C 不算失败，只跳过对比。
# ---------------------------------------------------------------------------
if [[ -f "${GOLD_COV}" ]]; then
  echo "==== compare against gold C: ${GOLD_COV} ===="
  python - "${COV_PATH}" "${GOLD_COV}" <<'PY'
import sys

import torch

from onereplay.core.covariance import load_covariance_file

new_path, gold_path = sys.argv[1], sys.argv[2]
new_cov = load_covariance_file(new_path)
gold_cov = load_covariance_file(gold_path)

shared = sorted(set(new_cov) & set(gold_cov))
only_new = sorted(set(new_cov) - set(gold_cov))
only_gold = sorted(set(gold_cov) - set(new_cov))
print(f"layers: {len(new_cov)} new, {len(gold_cov)} gold, {len(shared)} shared")
if only_new or only_gold:
    # Different layer sets mean the two C files cover different penalties, so a
    # run using one is not a controlled comparison against a run using the other.
    print(f"WARNING: layer sets differ. only-new={only_new[:5]} only-gold={only_gold[:5]}")

ratios = []
for name in shared:
    new_trace = torch.diagonal(new_cov[name].float()).sum().item()
    gold_trace = torch.diagonal(gold_cov[name].float()).sum().item()
    if gold_trace > 0:
        ratios.append(new_trace / gold_trace)

if not ratios:
    print("no comparable layers; skipping the scale report")
    raise SystemExit(0)

ratios_tensor = torch.tensor(ratios)
mean_ratio = ratios_tensor.mean().item()
print(
    "trace(selfdistill C) / trace(gold C) per layer: "
    f"mean={mean_ratio:.3f} min={ratios_tensor.min().item():.3f} "
    f"max={ratios_tensor.max().item():.3f}"
)
print(f"equal-strength lambda for the new C = 1e-2 / {mean_ratio:.3f} = {1e-2 / mean_ratio:.2e}")
if 0.8 <= mean_ratio <= 1.25:
    print("verdict: scales match closely enough, run 20_ with LAMBDA=1e-2 as is")
else:
    print(
        "verdict: scales differ enough that lambda=1e-2 is no longer the same "
        "regularization strength; add the equal-strength lambda above as a second run"
    )
PY
else
  echo "跳过尺度对比：未找到 gold C ${GOLD_COV}"
fi

echo "Job finished at: $(date)"
