#!/usr/bin/zsh
#
#SBATCH --job-name=or_sdcov
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.err
#SBATCH --array=0

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-local} task: ${SLURM_ARRAY_TASK_ID:-0}"

set -euo pipefail
set -x

# =============================================================================
# Stage 2d: OneReplay + 自蒸馏（训练 + IFEval/Multi-IF）
#
# 这一格补的是什么
# ---------------
# 旧知识有两种定义，用法也有两种，四格里之前一直空着右下角：
#
#                        gold 旧知识              自蒸馏旧知识
#   replay（留语料）     FLAN 答案键当 target     base 模型自己的回答当 target
#   OneReplay（压成 C）  cov_flan_chat_20k_qv     ← 本脚本（cov_flan_selfdistill_20k_qv）
#
# 缺这一格的话，"OneReplay vs replay 自蒸馏"就跨了两种旧知识定义：replay 自蒸馏
# 锚的是 base 模型自己的行为，而 gold C 里混着 FLAN 答案键的 hidden state。补上
# 之后两条路线吃的是同一份数据（同一批 prompt、同一批自蒸馏回答、同样丢掉那
# 2565 行截断的），差别只剩"压成二阶矩一次带走"还是"留着语料反复重训"，效率
# 对比也才站得住。
#
# 注意口径：本脚本是 LoRA。要和 replay 侧对比，请用同为 LoRA 的 18 号脚本
# （batch 级混合三档）的结果，不要拿全量微调的 full_replayself_r0.10 来比。
#
# 与 06 号 λ=1e-2 那次的唯一差别就是 --cov_path，其余超参逐字照搬（LoRA q_proj/
# v_proj、3 epoch、lr=1e-4、batch=8、accum=64、max_len=512、seed=1），所以
# cs_onereplay_lam1e-2_seed1 直接就是它的对照组，不用重跑。
#
# 关于 λ
# ------
# 默认 1e-2，即 λ 扫描里 retention 最好的那一档。但惩罚对 C 的绝对尺度是线性的
# （λ · (1/L) Σ_l tr(ΔW_l C_l ΔW_l^T)），换一份 C 之后同一个 λ 的实际强度会按
# trace 比值漂移。19 号脚本跑完会打一张 trace 对比表并给出等效强度的 λ：
#   - 比值接近 1（0.8~1.25）：本脚本原样跑即可，array=0 一个 job 就够
#   - 比值偏离较多：把等效 λ 填进 LAMBDAS，然后 sbatch --array=0-2 一起跑，
#     用等效 λ 那一档回答"是 C 的数据源变好了，还是只是正则变强了"
#
# 用法:
#   sbatch onereplay/slurm/20_train_onereplay_selfdistill.slurm                      # 只跑 λ=1e-2
#   sbatch --array=0-2 onereplay/slurm/20_train_onereplay_selfdistill.slurm          # 连带两个旁档
#   COV_SCOPE=full sbatch ... （需自行改成全量微调的超参，本脚本按 LoRA 配的）
# =============================================================================

# ---- λ 网格。任务 0 是主实验；1、2 是需要时才用的旁档 ----
LAMBDAS=(1e-2 3e-3 3e-2)

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

# ---- 项目路径（与 06 号一致）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"

# ---- 唯一的变量：C 来自自蒸馏语料，由 19_collect_cov_selfdistill 产出 ----
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_selfdistill_20k_qv.pt}"

# ---- 训练超参：与 06_sweep_lambda_train.slurm 逐字一致 ----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LAMBDA="${LAMBDA:-${LAMBDAS[$((TASK_ID + 1))]}}"
if [[ -z "${LAMBDA}" ]]; then
  echo "错误: array 任务号 ${TASK_ID} 在 LAMBDAS(共 ${#LAMBDAS[@]} 个) 中没有对应的 λ"
  exit 1
fi

# sdcov = C 采自自蒸馏语料，和 cs_onereplay_lam*（gold C）区分开，两边互不覆盖
RUN="cs_onereplay_sdcov_lam${LAMBDA}_seed${SEED}"
IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"
echo "task ${TASK_ID}: replay_lambda=${LAMBDA} run=${RUN}"
echo "COV_PATH=${COV_PATH}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_PATH}" ]] || {
  echo "缺自蒸馏协方差文件: ${COV_PATH}"
  echo "先跑 19_collect_cov_selfdistill.slurm"
  exit 1
}
mkdir -p "${RESULTS_ROOT}/adapters" "${RESULTS_ROOT}/metrics" \
  "${RESULTS_ROOT}/map_cache/${RUN}"

# ---------------------------------------------------------------------------
# Step 0: 确认这份 C 真的是自蒸馏的那份
#
# 整个实验的意义全压在"C 换了数据源"这一件事上。COV_PATH 一旦被环境变量指错，
# 跑出来的就是 cs_onereplay_lam1e-2_seed1 的一次重复，而名字却写着 sdcov —— 那是
# 会污染结论的错，且要 16 小时以后才可能被发现。三个 metadata 字段全部对上才继续。
# ---------------------------------------------------------------------------
echo "==== check cov provenance: ${COV_PATH} ===="
python - "${COV_PATH}" <<'PY'
import sys

import torch

path = sys.argv[1]
payload = torch.load(path, map_location="cpu", weights_only=False)
metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
source = f"{metadata.get('data_files', '')} {metadata.get('dataset_path', '')}"

problems = []
if "selfdistill" not in source:
    problems.append(f"data source does not look self-distilled: {source!r}")
if metadata.get("require_target") != 1:
    problems.append(
        f"require_target={metadata.get('require_target')!r}, so C covers rows replay drops"
    )
if metadata.get("truncation_side") != "left":
    problems.append(
        f"truncation_side={metadata.get('truncation_side')!r}, so long answers were cut "
        "from the wrong end and C may not have seen them"
    )

print(f"cov source        : {source.strip()}")
print(f"require_target    : {metadata.get('require_target')}")
print(f"truncation_side   : {metadata.get('truncation_side')}")
print(f"include_target    : {metadata.get('include_target_in_chat')}")
print(f"max_len           : {metadata.get('max_len')}")
print(f"target_modules    : {metadata.get('target_modules')}")

if problems:
    print("FAIL: " + "; ".join(problems))
    print("regenerate with 19_collect_cov_selfdistill.slurm, or point COV_PATH at the right file")
    raise SystemExit(1)
print("OK: C comes from the self-distilled corpus with replay-matched filtering and truncation")
PY

# ---------------------------------------------------------------------------
# Step 1: 训练并保存 adapter
# 新任务验证 loss 由训练脚本每个 epoch 写进 metrics/*.jsonl。
# ---------------------------------------------------------------------------
echo "==== train: ${RUN} (lambda=${LAMBDA}) ===="
python -m onereplay.scripts.train --paradigm sft \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}" --identity_cov 0 \
  --save 1 --save_path "${RESULTS_ROOT}/adapters/${RUN}" \
  --metrics_path "${RESULTS_ROOT}/metrics/${RUN}.jsonl"

# ---------------------------------------------------------------------------
# Step 2: retention 评测 IFEval + Multi-IF English（一次模型加载跑完）
# ---------------------------------------------------------------------------
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
