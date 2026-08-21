#!/usr/bin/zsh
#
#SBATCH --job-name=or_fisher
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
# Stage 1b: 采集 EWC 基线用的对角 empirical Fisher
#
# 这是 01_collect_cov 的姊妹脚本。OneReplay 用 C = E[x x^T] 加权 ΔW，EWC 用
# 对角 Fisher F 加权 ΔW，两者都在冻结的 base 模型上估一次、全程不更新，所以
# 对比里唯一变的就是"用什么矩阵加权"。
#
# 【取数参数必须和 01 逐字相同】
# 这是整个对比的公平性根基。下面的 FLAN_DIR / FLAN_SAMPLES / SAMPLE_SHUFFLE /
# SAMPLE_SEED / MAXLEN / use_chat_template / include_target_in_chat 全部照抄 01，
# 并且都不传 --truncation_side、不传 --require_target（01 也没传，用的是默认值）。
#
# Step 0 会用 check_old_knowledge_pool.py 分别按 cov 和 fisher 的参数算一遍选中
# 20k 行的 sha256，不相等就直接退出。这一步纯 CPU、不加载模型、几秒钟，用来把
# "两边应该拿到同一批行"从假设变成事实——datasets 升级换了置换算法、或者哪天有人
# 改了其中一个脚本的 flag，都会在这里被拦住而不是变成一个没人发现的混淆变量。
#
# 【为什么右截断，不传 left】
# 上一轮讨论里考虑过传 --truncation_side left 与训练对齐，最后选了"与 C 对齐"：
# 01 没传，用的是 tokenizer 默认的右截断。FLAN 的 gold target 是一行短句，512
# 下几乎不溢出，所以右截断砍掉 assistant 回答这件事基本不发生。真溢出的那几行
# 会落到 zero_supervision_rows 里被计数，脚本末尾会打出来；占比若超过 1% 就该查。
#
# 【为什么 fp32】
# --use_bf16 0。bf16 的指数位和 fp32 一样宽所以不会下溢，但 8 位尾数会给每个
# 梯度带约 0.4% 相对误差，平方后约 0.8%。累加器本来就是 fp32（那部分是必须的，
# bf16 累加两万项会把尾巴吞光），但已经发生在 backward 里的舍入是 widen 不回来的。
# 这是个一次性的 job，慢一倍换掉一整类质疑很划算。
#
# 【为什么 batch_size=1】
# Fisher 要的是 per-example 梯度：先平方再累加。梯度累积算出来的是 (Σg)² 而不是
# Σg²，会静默地估成另一个矩阵。所以 collect_fisher.py 把 batch_size 写死为 1，
# 不给开关。20k 条 fp32 前向+反向，估计 2–4 小时。
#
# 前置：无（只依赖 base 模型和 FLAN dump，和 01 一样）
#
# 用法:
#   sbatch onereplay/slurm/22_collect_fisher.slurm                    # 默认 qv
#   COV_SCOPE=full sbatch onereplay/slurm/22_collect_fisher.slurm     # 全量八类层
# 只做指纹校验、不采集（几秒钟）:
#   CHECK_ONLY=1 sbatch onereplay/slurm/22_collect_fisher.slurm
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

# ---- 项目路径（与 01 一致）----
REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
FLAN_DIR="${FLAN_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/flan}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"
MAXLEN="${MAXLEN:-512}"

# ---- 覆盖范围。注意 01 的默认是 full；LoRA 那条线用的是 qv，两边必须选同一个 ----
# FISHER_DEVICE: qv 档的累加器约 0.7 GiB，放 GPU；full 档约 7 GiB，再叠上 fp32
# 模型和它的梯度就吃紧了，所以自动落到 cpu。
COV_SCOPE="${COV_SCOPE:-qv}"
case "${COV_SCOPE}" in
  qv)
    TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
    FISHER_PATH="${FISHER_PATH:-${RESULTS_ROOT}/fisher/fisher_flan_chat_20k_qv.pt}"
    FISHER_DEVICE="${FISHER_DEVICE:-cuda}"
    ;;
  full)
    TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head}"
    FISHER_PATH="${FISHER_PATH:-${RESULTS_ROOT}/fisher/fisher_flan_chat_20k_full.pt}"
    FISHER_DEVICE="${FISHER_DEVICE:-cpu}"
    ;;
  *)
    echo "未知 COV_SCOPE=${COV_SCOPE}，可选 qv / full"; exit 1
    ;;
esac
echo "COV_SCOPE=${COV_SCOPE} TARGET_MODULES=${TARGET_MODULES} FISHER_DEVICE=${FISHER_DEVICE}"

# ---- 取数参数：逐字照抄 01_collect_cov.slurm ----
FLAN_SAMPLES="${FLAN_SAMPLES:-20000}"
SAMPLE_SHUFFLE="${SAMPLE_SHUFFLE:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"
USE_BF16="${USE_BF16:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${FLAN_DIR}/train" ]] || { echo "缺 FLAN JSONL 目录: ${FLAN_DIR}/train"; exit 1; }
mkdir -p "$(dirname "${FISHER_PATH}")" "${DATASET_CACHE}"

# ---------------------------------------------------------------------------
# Step 0: 指纹校验——C 和 F 必须看到同一批 20k 行
# 两次调用传的是完全相同的 flag，所以这一步在正常情况下必然通过；它存在的意义是
# 在"某天有人只改了其中一个脚本"或"datasets 换了 shuffle 实现"时炸出来。
# ---------------------------------------------------------------------------
echo "==== Step 0: 旧知识池指纹校验 ===="
POOL_ARGS=(
  --data_files "${FLAN_DIR}/train/*.jsonl" --dataset_split train
  --cache_dir "${DATASET_CACHE}"
  --max_samples "${FLAN_SAMPLES}"
  --sample_shuffle "${SAMPLE_SHUFFLE}" --sample_seed "${SAMPLE_SEED}"
)
COV_FINGERPRINT=$(python -m onereplay.scripts.check_old_knowledge_pool \
  "${POOL_ARGS[@]}" --label cov | tail -n 1)
FISHER_FINGERPRINT=$(python -m onereplay.scripts.check_old_knowledge_pool \
  "${POOL_ARGS[@]}" --label fisher | tail -n 1)
echo "${COV_FINGERPRINT}"
echo "${FISHER_FINGERPRINT}"

# 输出形如 "cov rows=20000 fingerprint=abc..."，只要 = 后面那截；用 ##* 会连
# "fingerprint=" 一起留下，和 metadata 里存的裸 hash 对不上。
COV_HASH="${COV_FINGERPRINT##*fingerprint=}"
FISHER_HASH="${FISHER_FINGERPRINT##*fingerprint=}"
if [[ "${COV_HASH}" != "${FISHER_HASH}" ]]; then
  echo "错误: 两套取数参数选出的行不同"
  echo "  cov    ${COV_HASH}"
  echo "  fisher ${FISHER_HASH}"
  echo "EWC 与 OneReplay 必须消费同一批旧知识，否则对比被数据差异污染。"
  exit 1
fi
echo "指纹一致: ${COV_HASH}"
echo "把这个值和 01_collect_cov 日志里的 'Stage 1: pool rows=... fingerprint=...' 对一下；"
echo "现有的 C 是在加指纹打印之前采的，日志里没有这一行，重跑一次 01 即可补上。"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "CHECK_ONLY=1：只做指纹校验，跳过采集"
  echo "Job finished at: $(date)"
  exit 0
fi

# ---------------------------------------------------------------------------
# Step 1: 采集 F
# ---------------------------------------------------------------------------
echo "==== Step 1: collect Fisher -> ${FISHER_PATH} ===="
python -m onereplay.scripts.collect_fisher \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
  --use_bf16 "${USE_BF16}" \
  --data_files "${FLAN_DIR}/train/*.jsonl" --dataset_split train \
  --cache_dir "${DATASET_CACHE}" \
  --use_chat_template 1 --include_target_in_chat 1 \
  --max_samples "${FLAN_SAMPLES}" \
  --sample_shuffle "${SAMPLE_SHUFFLE}" --sample_seed "${SAMPLE_SEED}" \
  --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --fisher_device "${FISHER_DEVICE}" \
  --seed "${SAMPLE_SEED}" \
  --output_path "${FISHER_PATH}"

# ---------------------------------------------------------------------------
# Step 2: 自检——把估计器的实测诊断读出来
# length_exponent 和 effective_sample_size 是要写进论文附录的两个数，别只看日志。
# ---------------------------------------------------------------------------
echo "==== Step 2: 自检 ===="
python - "${FISHER_PATH}" "${COV_HASH}" <<'PY'
import json
import sys

import torch

path, want_hash = sys.argv[1], sys.argv[2]
payload = torch.load(path, map_location="cpu")
meta = payload["metadata"]
report = meta["length_weighting"]

print(json.dumps({key: meta[key] for key in
                  ("estimator", "num_examples", "pool_rows", "pool_fingerprint",
                   "use_bf16", "truncation_side", "prompt_prefix_mismatches")}, indent=2))
print(json.dumps(report, indent=2))
print(json.dumps(meta["fisher_scale"], indent=2))

if meta["pool_fingerprint"] != want_hash:
    raise SystemExit(
        f"F 的池子指纹 {meta['pool_fingerprint']} 与校验值 {want_hash} 不符"
    )
if meta["loss_reduction"] != "sum":
    raise SystemExit("loss 不是 sum 归约，这不是 sequence-level Fisher")
if not payload["fishers"]:
    raise SystemExit("F 是空的，target_modules 没匹配上任何层")

zero_share = report["zero_supervision_rows"] / max(report["examples"], 1)
if zero_share > 0.01:
    print(
        f"警告: {report['zero_supervision_rows']:.0f} 行（{zero_share:.2%}）没有受监督 token。"
        " 右截断吃掉回答的比例超过 1%，查一下 max_len 和 FLAN 的 target 长度分布。"
    )

print("\n写论文用这两个数:")
print(f"  length_exponent      = {report['length_exponent']:.3f}  (1=token 梯度不相关, 2=完全对齐)")
print(f"  effective_sample_size= {report['effective_sample_size']:.0f} / {report['examples']:.0f}"
      f"  (ratio {report['ess_ratio']:.3f})")
print("自检通过")
PY

echo "Job finished at: $(date)"
