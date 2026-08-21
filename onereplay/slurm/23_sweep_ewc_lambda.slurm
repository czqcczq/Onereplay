#!/usr/bin/zsh
#
#SBATCH --job-name=or_ewc
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
#SBATCH --array=0-3

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-local} task: ${SLURM_ARRAY_TASK_ID:-0}"

set -euo pipefail
set -x

# =============================================================================
# Stage 2f: EWC 基线的 λ 扫描（训练 + IFEval + Multi-IF）
#
# 这是 regularization 这一族里除 OneReplay 之外的对照方法。除了 --regularizer ewc
# 和 --fisher_path，所有训练超参与 02/06 逐字相同，唯一变量是 λ。
#
# 罚项：R = (1/L) Σ_l Σ_ij F_l,ij (ΔW_l,ij)²，ΔW = scale · B A
# 对照 OneReplay：R = (1/L) Σ_l tr(ΔW_l C_l ΔW_l^T)
# 两个矩阵都在冻结 base 模型上、同一批 FLAN 20k 行上估一次，全程不更新。
#
# 【λ 网格必须重新探，OneReplay 的网格搬过来一定全废】
# 22 号实测（job 2944210）: F 的均值 9.978e-02，最大 1.268e+05。
# C 是 hidden state 的二阶矩，过了 RMSNorm 之后在 1e0~1e1。
# 两个罚项的差距不止元素量级，还差一次维度收缩：
#   EWC       R = (1/L) Σ_l Σ_ij F_ij ΔW_ij²        —— F 逐元素乘，不收缩
#   OneReplay R = (1/L) Σ_l tr(ΔW C ΔW^T)          —— 沿 hidden 维收缩一次
# 所以 R_or / R_ewc 大致是 d·c / mean(F) ≈ 2048×5 / 0.1 ≈ 1e5，反过来
# λ_ewc ≈ λ_or × 1e5 = 3e-2 × 1e5 ≈ 3e3。下面的网格 (1e2 … 1e4) 就是围着这个
# 估计铺的，前后各留一个数量级。
# 注意 22 号日志里那句 "start near 1 / mean"（≈10）漏掉了上面这次收缩，别照着抄。
#
# 上面的估计有正负一个数量级的不确定度，而整组是 4×5.7 h。先花十几分钟探一档:
#   1) 只看罚项量级，不保存不评测:
#        sbatch --array=1 --export=ALL,SAVE=0,MAX_STEPS=300 \
#          onereplay/slurm/23_sweep_ewc_lambda.slurm
#      探针要用网格里最小的那档（现在是 index 1 = 1e2）：λ 给大了 ΔW 会被压住，
#      R 跟着变小，你会误判成"还得再加大 λ"。
#      日志里的 replay_reg 是未乘 λ 的原始 R（loss = task_loss + λ·R），
#      目标是让 λ·R 在训练中期占 task_loss（约 1.1）的 1%–10%。
#   2) 按探到的中心值改下面的 LAMBDAS，把 --array 上界改成 元素个数-1，再整组提交。
#
# 【结果怎么读：一定按等新任务性能比】
# EWC-LoRA (ICLR 2026, arXiv 2602.17559) 在多任务设定下实测 precomputed 固定
# Fisher 的塑性最低——也就是说 EWC 很可能不是"retention 更好"，而是"新任务学不动"。
# 所以别各挑一个 λ 报点，要画在相同 commonsense val_loss 处的 IFEval 曲线。
# Step 3 的自检会把 val_loss 和 vanilla 的基线值比一下并提示。
#
# 前置：
#   sbatch onereplay/slurm/22_collect_fisher.slurm
#
# 用法:
#   sbatch onereplay/slurm/23_sweep_ewc_lambda.slurm
#   sbatch --array=0-3%2 onereplay/slurm/23_sweep_ewc_lambda.slurm   # 限并发
# =============================================================================

# ---- λ 网格。改这里时同步 --array 上界 = 元素个数-1 ----
LAMBDAS=(3e2 1e2 1e3 1e4)

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

# ---- 项目路径（与 06 一致）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
FISHER_PATH="${FISHER_PATH:-${RESULTS_ROOT}/fisher/fisher_flan_chat_20k_qv.pt}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"

# ---- 训练超参：与 02/06 逐字相同，唯一变量是 LAMBDA ----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
EVAL_BATCH="${EVAL_BATCH:-8}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

# ---- 探针短跑：SAVE=0 MAX_STEPS=300 只看罚项量级，不评测 ----
SAVE="${SAVE:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
PROFILE="${PROFILE:-0}"
# vanilla 的 3-epoch val_loss，用于 Step 3 的塑性提示。留空则跳过比较。
VANILLA_VAL_LOSS="${VANILLA_VAL_LOSS:-}"

# ---- 从 array 任务号选 λ（zsh 数组 1-based，array 任务号 0-based）----
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LAMBDA="${LAMBDAS[$((TASK_ID + 1))]:-}"
if [[ -z "${LAMBDA}" ]]; then
  echo "错误: array 任务号 ${TASK_ID} 在 LAMBDAS(共 ${#LAMBDAS[@]} 个) 中没有对应的 λ；--array 上界应为 元素个数-1"
  exit 1
fi

RUN="cs_ewc_lam${LAMBDA}_seed${SEED}"
IFEVAL_INPUT="${CODE}/third_party/instruction_following_eval/data/input_data.jsonl"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"
echo "ewc sweep task ${TASK_ID}: replay_lambda=${LAMBDA} run=${RUN}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${FISHER_PATH}" ]] || {
  echo "缺 Fisher 文件: ${FISHER_PATH}"
  echo "先跑: sbatch onereplay/slurm/22_collect_fisher.slurm"
  exit 1
}
# --regularizer 是新加的开关；用旧代码跑会直接报 unrecognized arguments，但显式
# 检查能给出更有用的提示。
grep -q "regularizer" onereplay/scripts/train.py || {
  echo "train.py 里没有 --regularizer，代码没同步到最新"
  exit 1
}
mkdir -p "${RESULTS_ROOT}/adapters" "${RESULTS_ROOT}/metrics" \
  "${RESULTS_ROOT}/map_cache/${RUN}"

# ---------------------------------------------------------------------------
# Step 0: 确认 F 和 C 出自同一批旧知识行
# 两个文件的 metadata 里都记了池子指纹。不一致就说明 EWC 和 OneReplay 消费的
# 旧知识不同，这次对比不成立。现有的 C 若是在加指纹之前采的，这里只警告不拦。
# ---------------------------------------------------------------------------
echo "==== Step 0: 校验 F 与 C 的旧知识池一致 ===="
python - "${FISHER_PATH}" "${COV_PATH}" <<'PY'
import sys
from pathlib import Path

import torch

fisher_path, cov_path = sys.argv[1], sys.argv[2]
fisher_meta = torch.load(fisher_path, map_location="cpu")["metadata"]
fisher_hash = fisher_meta.get("pool_fingerprint", "")
print(f"fisher pool: rows={fisher_meta.get('pool_rows')} fingerprint={fisher_hash}")
print(f"fisher estimator: {fisher_meta.get('estimator')}")

if not Path(cov_path).is_file():
    print(f"警告: 找不到 {cov_path}，跳过与 C 的比对")
    raise SystemExit(0)

cov_meta = torch.load(cov_path, map_location="cpu").get("metadata", {})
cov_hash = cov_meta.get("pool_fingerprint", "")
if not cov_hash:
    print(
        "警告: C 的 metadata 里没有 pool_fingerprint，说明它是在加指纹打印之前采的。"
        " 重跑一次 01_collect_cov 就能补上；在那之前这次比对只能靠 22 的 Step 0。"
    )
    raise SystemExit(0)

print(f"cov pool: rows={cov_meta.get('pool_rows')} fingerprint={cov_hash}")
if cov_hash != fisher_hash:
    raise SystemExit(
        f"F 与 C 的旧知识池不同 ({fisher_hash} vs {cov_hash})；EWC 与 OneReplay 的对比"
        " 会被数据差异污染，重采其中一个"
    )
print("池子一致")
PY

# ---------------------------------------------------------------------------
# Step 1: 训练并保存 adapter
# 新任务验证 loss 每个 epoch 写进 metrics/*.jsonl，不用另跑 commonsense 评测。
# ---------------------------------------------------------------------------
echo "==== train: ${RUN} (ewc, lambda=${LAMBDA}) ===="
python -m onereplay.scripts.train --paradigm sft \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --eval_batch_size "${EVAL_BATCH}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --profile "${PROFILE}" --max_steps "${MAX_STEPS}" \
  --regularizer ewc --fisher_path "${FISHER_PATH}" \
  --replay_lambda "${LAMBDA}" \
  --save "${SAVE}" --save_path "${RESULTS_ROOT}/adapters/${RUN}" \
  --metrics_path "${RESULTS_ROOT}/metrics/${RUN}.jsonl"

# ---------------------------------------------------------------------------
# Step 2: 自检 + 塑性提示
# ---------------------------------------------------------------------------
echo "==== Step 2: 自检 ===="
python - "${RESULTS_ROOT}/metrics/${RUN}.jsonl" "${LAMBDA}" "${VANILLA_VAL_LOSS}" <<'PY'
import json
import sys

path, want_lambda, vanilla_val = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with open(path, encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
last = records[-1]

reg = last.get("train_replay_reg") or 0.0
lambda_reg = last.get("train_lambda_reg") or 0.0
task_loss = last.get("train_task_loss") or 0.0
share = lambda_reg / task_loss if task_loss else 0.0

print(f"regularizer      = {last.get('regularizer')}")
print(f"replay_lambda    = {last.get('replay_lambda')} (want {want_lambda})")
print(f"train_replay_reg = {reg:.6e}")
print(f"lambda * reg     = {lambda_reg:.6e}  ({share:.2%} of task_loss {task_loss:.4f})")
print(f"val_loss         = {last.get('val_loss')}")

if last.get("regularizer") != "ewc":
    raise SystemExit("regularizer 不是 ewc，这次 run 白跑")
# 罚项恒为 0 的唯一原因是 Fisher 的层名一个都没匹配上 LoRA 模块（训练日志第一步
# 打印的 "regularizer covers N layers" 会是 0），或者 F 文件本身是空的。
if reg == 0.0:
    raise SystemExit("罚项恒为 0，检查训练日志里 regularizer covers 了几层")

# 罚项占比落在两端说明这一档的 lambda 探错了，跑完也只是浪费一个网格点。
if share < 1e-3:
    print(f"提示: 罚项只占 task_loss 的 {share:.3%}，这一档基本等于 vanilla，往上调 lambda")
elif share > 0.5:
    print(f"提示: 罚项占 task_loss 的 {share:.1%}，新任务很可能学不动了，往下调 lambda")

if vanilla_val:
    gap = (last.get("val_loss") or 0.0) - float(vanilla_val)
    print(f"val_loss 相对 vanilla({vanilla_val}) 的差值 = {gap:+.4f}")
    if gap > 0.05:
        print(
            "提示: 新任务明显学得更差。这正是 EWC-LoRA(2602.17559) 观察到的"
            " precomputed 固定 Fisher 塑性偏低。写结论时要按等 val_loss 比 IFEval，"
            "不能直接说 EWC retention 更好。"
        )
print("自检通过")
PY

# ---------------------------------------------------------------------------
# Step 3: retention 评测 IFEval + Multi-IF English
# ---------------------------------------------------------------------------
if [[ "${SAVE}" != "1" ]]; then
  echo "SAVE=0：这是探针短跑，跳过评测"
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
