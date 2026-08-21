#!/usr/bin/zsh
#
#SBATCH --job-name=or_probe
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%A_%a.err
#SBATCH --array=0-2

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-local} task: ${SLURM_ARRAY_TASK_ID:-0}"

set -euo pipefail
set -x

# =============================================================================
# 验证性实验：训练期间旧知识 loss 的曲线形状
#
# 【要回答的问题】
# replay 的池子是 17429 条（20k 去掉截断行），50% 档一个 epoch 要消耗约 16.9 万条，
# 3 个 epoch 下来每条被用了约 29 遍。在这种重复强度下，"IFEval 只掉 1.1 个点"到底
# 是保住了旧能力，还是把这 1.7 万条背下来了？现有的所有指标都是训练结束后的一个
# 快照，回答不了这个问题，也看不见中间过程。
#
# 【三条探针曲线，缺一不可】
#   flan_heldout  replay 训练时碰不到的 FLAN 切片 -> 旧知识的泛化
#   flan_inpool   从 replay 实际训练的池子里等距抽 1000 条 -> 旧知识的记忆
#   cs_val        已有的 Commonsense 验证集 -> 新任务学得怎样
# 只看 held-out 一条是读不出结论的：曲线上升既可能是过拟合到池子，也可能是整个模型
# 被新任务拽走。两条 FLAN 曲线之间的 gap 把这两种解释分开，而 gap 有意义的前提是
# 两边用完全相同的 target 口径（都是 base 模型自己的答案）和完全相同的 tokenize。
#
# 【held-out 怎么取：同分布，且过滤后零重叠】
# 池子的取法是 shuffle(seed=SAMPLE_SEED) 后取前 POOL_SIZE 行。所以在同一个 shuffle
# 上从第 POOL_SIZE 行往后接着取，就拿到了来自同一次随机排列因而同分布的一段。这就是
# generate_replay_targets.py 的 --pool_offset。
#
# 但行号不重叠推不出 prompt 文本不重叠：FLAN 自己就有重复 prompt——2948829 那次的
# 日志里，20000 行的池子只有 18725 个不同的 prompt，重复率 6.4%，而 1500 行 held-out
# 与池子的文本交集是 98 行，正好 6.5%，两个数字是同一件事。所以下面的自检不是要求
# 交集为 0（在这份语料上不可能），而是把撞上的行剔掉后写出一份 _disjoint.jsonl，
# 探针只读那一份。
# 它真正要抓的故障是 --pool_offset 被静默忽略：那时 held-out 退化成池子的前
# HELDOUT_SIZE 行，交集接近 100%，与 6.5% 差十几倍，按比例一眼分得开。
# 过滤只能去掉字面相同的 prompt，同模板不同内容的近似重复仍然在——那正是 held-out
# 该有的样子，它要的是同分布而不是同样本。
#
# 【为什么 target 必须是自蒸馏而不是 FLAN 的 gold】
# 训练用的 replay 是自蒸馏版，探针也用自蒸馏 target，两条 FLAN 曲线才在同一口径上、
# 才能相减。更要紧的是语义：自蒸馏 target 就是 base 模型自己的贪心回答，所以 step 0
# 的 loss 接近 0，之后每一点直接读作"模型相对 W0 漂移了多少"。换成 gold target，
# 起点会在 3.0 附近，曲线里混着"模型正在学 FLAN 一行式答案风格"的信号，遗忘和风格
# 迁移就分不开了。
#
# 【为什么必须带上 vanilla 和 OneReplay】
# 1) vanilla 是遗忘的上界。没有它，"replay 的 held-out 涨了 0.3"是多是少无从判断。
# 2) vanilla 还是一个免费的对照：它两个 FLAN 切片都没训练过，所以两条曲线本该完全
#    重合。真重合了，就证明 replay run 上张开的 gap 只可能来自 replay 本身，而不是
#    两个切片本来难度就不同。这个校验比任何论证都硬。
# 3) OneReplay 没有池子可以过拟合，它的两条曲线也该重合但整体抬升。三组放一起，
#    "压缩成 C" 与 "留着语料反复训" 的区别是直接看得见的。
#
# 【x 轴是 optimizer update，不是 micro-batch】
# replay 每个 micro-batch 只装 4 条新任务、accum 128；vanilla 装 8 条、accum 64。
# 两者每次更新都是 64 条新任务，3 个 epoch 都是 7908 次更新，但 micro-batch 数差一倍。
# 所以 --probe_every_updates 数的是更新数，三条 run 的探针点在 x 轴上逐点对齐，
# epoch 边界也落在同样的 2636 / 5272 / 7908。
#
# 【这几个 run 的训练轨迹应当与历史 run 逐位相同】
# 探针只做 forward，在 model.eval() 下 dropout 是恒等、不消耗全局 RNG，探针 loader
# 也是 shuffle=False 不抽 RNG。所以权重轨迹不受影响，末尾 val_loss 应与历史那几个
# run 对得上。Step 4 会去比对——对不上就说明探针干扰了训练，曲线不能用。
# 探针耗时已在 trainer 里从 train_sec / ms/step / win= 中扣除，但 elapsed_sec 是墙钟
# 仍然含它；这几个 run 本来也不用于报开销数字。
#
# 【结果直接看日志，不需要额外画图】
# Step 3 会把曲线等间隔采样成一张表打出来，并给出形状判定（一直不变 / 单调上升 /
# 先降后升 / 先升后降）。原始逐点数据在 metrics jsonl 里，record_type=probe 那些行。
#
# 前置: 13 号生成的训练池自蒸馏语料。held-out 语料本脚本会自己生成（约 3 分钟）。
#
# 用法:
#   sbatch onereplay/slurm/27_probe_heldout_curve.slurm
# 只跑 replay 那条:
#   sbatch --array=0 onereplay/slurm/27_probe_heldout_curve.slurm
# 先冒烟（约 15 分钟，确认探针点写得出来）:
#   sbatch --array=0 --export=ALL,MAX_STEPS=600,PROBE_EVERY_UPDATES=4,EPOCHS=1,HELDOUT_SIZE=64,PROBE_HELDOUT_SIZE=32,PROBE_INPOOL_SIZE=32,PROBE_CS_VAL_SIZE=32 \
#     onereplay/slurm/27_probe_heldout_curve.slurm
# =============================================================================

# ---- 三条 run。除 replay / 正则的开关外，其余超参与 02 / 06 / 18 逐字一致 ----
#   task  方法                RPB  ACCUM  LAMBDA  accum_steps  步数/epoch  更新/epoch
#   0     Replay 50% (4n4r)    4    128    0          16         42178       2636
#   1     Vanilla              0     64    0           8         21090       2636
#   2     OneReplay λ=3e-2     0     64    3e-2        8         21090       2636
NAME_GRID=(replaymix_4n4r vanilla onereplay_lam3e-2)
RPB_GRID=(4 0 0)
ACCUM_GRID=(128 64 64)
LAMBDA_GRID=(0 0 3e-2)

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

REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
FLAN_DIR="${FLAN_DIR:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/flan}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
DATASET_CACHE="${DATASET_CACHE:-${RESULTS_ROOT}/dataset_cache}"
COV_PATH="${COV_PATH:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"

SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-8}"
EVAL_BATCH="${EVAL_BATCH:-8}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
MAX_STEPS="${MAX_STEPS:-0}"
# adapter 三条都已经有了（2849732_2 / 2170645 / 2856542_1），这次只要曲线。
SAVE="${SAVE:-0}"

# ---- 池子设置：必须与 01_collect_cov / 13_replay_selfdistill 逐字一致，
#      否则 held-out 不是"同一次 shuffle 的后半段"，与池子的同分布关系就断了 ----
POOL_SIZE="${POOL_SIZE:-20000}"
SAMPLE_SEED="${SAMPLE_SEED:-1}"
SELF_DISTILL_FILE="${SELF_DISTILL_FILE:-${RESULTS_ROOT}/replay/flan_selfdistill_${POOL_SIZE}_seed${SAMPLE_SEED}.jsonl}"

# ---- held-out 语料生成设置 ----
# 从第 POOL_SIZE 行开始取 HELDOUT_SIZE 行。生成 1500 而不是 1000，是因为撞上生成
# 预算、没吐出停止符的行会被标 truncated 并丢掉；20k 那次丢了 12.8%，1500 条留够
# 余量后仍有约 1300 条可用，探针再等距抽到 1000 条。
HELDOUT_OFFSET="${HELDOUT_OFFSET:-${POOL_SIZE}}"
HELDOUT_SIZE="${HELDOUT_SIZE:-1500}"
HELDOUT_FILE="${HELDOUT_FILE:-${RESULTS_ROOT}/replay/flan_selfdistill_heldout_off${HELDOUT_OFFSET}_${HELDOUT_SIZE}_seed${SAMPLE_SEED}.jsonl}"
# 剔掉与训练池 prompt 文本相撞的行之后的那一份，探针只读它。
PROBE_HELDOUT_FILE="${PROBE_HELDOUT_FILE:-${HELDOUT_FILE%.jsonl}_disjoint.jsonl}"
# 交集比例超过这个数就判定 --pool_offset 失效而不是语料自身重复（实测 6.5%）。
OVERLAP_ABORT_RATIO="${OVERLAP_ABORT_RATIO:-0.25}"
GEN_BATCH="${GEN_BATCH:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-16}"
SKIP_GEN="${SKIP_GEN:-0}"

# ---- 探针设置 ----
# 64 次更新一评：3 个 epoch 共 7908 次更新 -> 约 124 个点，形状看得清，
# 代价约 35 分钟（124 x 2500 条 forward），叠在最长那条 3.84h 上约 +15%。
PROBE_EVERY_UPDATES="${PROBE_EVERY_UPDATES:-64}"
PROBE_HELDOUT_SIZE="${PROBE_HELDOUT_SIZE:-1000}"
# 与 held-out 取同样的条数：两条曲线是要相减的，样本量不等会给两边不同的噪声。
PROBE_INPOOL_SIZE="${PROBE_INPOOL_SIZE:-1000}"
PROBE_CS_VAL_SIZE="${PROBE_CS_VAL_SIZE:-1000}"

# ---- 取本次档位（zsh 数组 1-based，array 任务号 0-based）----
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NAME="${NAME_GRID[$((TASK_ID + 1))]:-}"
RPB="${RPB_GRID[$((TASK_ID + 1))]:-}"
ACCUM="${ACCUM_GRID[$((TASK_ID + 1))]:-}"
LAMBDA="${LAMBDA_GRID[$((TASK_ID + 1))]:-}"
if [[ -z "${NAME}" || -z "${RPB}" || -z "${ACCUM}" || -z "${LAMBDA}" ]]; then
  echo "错误: array 任务号 ${TASK_ID} 超出网格(共 ${#NAME_GRID[@]} 条)；--array 上界应为 条数-1"
  exit 1
fi

ACCUM_STEPS=$(( ACCUM / BATCH ))
NEW_PER_BATCH=$(( BATCH - RPB ))
NEW_PER_UPDATE=$(( ACCUM_STEPS * NEW_PER_BATCH ))
# 三条 run 的曲线要叠在同一个 x 轴上，前提是每次更新消费的新任务数相同。
if (( NEW_PER_UPDATE != 64 )); then
  echo "错误: ${NAME} 每次更新的 new task=${NEW_PER_UPDATE}，不是 64；三条曲线的 x 轴对不齐"
  exit 1
fi

RUN="cs_${NAME}_seed${SEED}_probe"
BASELINE_METRICS="${RESULTS_ROOT}/metrics/cs_${NAME}_seed${SEED}.jsonl"
METRICS_PATH="${RESULTS_ROOT}/metrics/${RUN}.jsonl"

echo "probe task ${TASK_ID}: ${NAME}"
echo "  RPB=${RPB} ACCUM=${ACCUM} accum_steps=${ACCUM_STEPS} new/更新=${NEW_PER_UPDATE} LAMBDA=${LAMBDA}"
echo "  探针每 ${PROBE_EVERY_UPDATES} 次更新一评 = $(( PROBE_EVERY_UPDATES * ACCUM_STEPS )) 个 micro-batch"
echo "  run=${RUN}"
echo "  HELDOUT_FILE=${HELDOUT_FILE}"
echo "  PROBE_HELDOUT_FILE=${PROBE_HELDOUT_FILE}"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -d "${FLAN_DIR}/train" ]] || { echo "缺 FLAN JSONL 目录: ${FLAN_DIR}/train"; exit 1; }
[[ -f "${SELF_DISTILL_FILE}" ]] || {
  echo "缺训练池自蒸馏语料: ${SELF_DISTILL_FILE}"
  echo "先跑: sbatch onereplay/slurm/13_replay_selfdistill.slurm"
  exit 1
}
# 用旧代码跑会静默忽略这两组参数：--pool_offset 没生效会让 held-out 变成训练池本身
# （两条曲线完美重合，看上去像个干净的阴性结论，非常容易被当真；下面那步过滤还会按
# 交集比例兜底拦一次），--probe_* 没生效则是白跑 12 小时没有曲线。
grep -q "pool_offset" onereplay/scripts/generate_replay_targets.py || {
  echo "generate_replay_targets.py 里没有 --pool_offset，代码没同步到最新"
  exit 1
}
grep -q "probe_every_updates" onereplay/scripts/train.py || {
  echo "train.py 里没有 --probe_every_updates，代码没同步到最新"
  exit 1
}
if [[ "${LAMBDA}" != "0" ]]; then
  [[ -f "${COV_PATH}" ]] || { echo "缺协方差: ${COV_PATH}"; exit 1; }
fi
mkdir -p "$(dirname "${HELDOUT_FILE}")" "${RESULTS_ROOT}/adapters" \
  "${RESULTS_ROOT}/metrics" "${RESULTS_ROOT}/map_cache/${RUN}" "${DATASET_CACHE}"

# ---------------------------------------------------------------------------
# Step 1: 准备 held-out 自蒸馏语料
#
# 三个 array task 都需要这份语料，但同时往一个 jsonl 里 append 会把行写串。这里让
# 每个 task 生成到自己的临时文件，完成后用 mv 原子替换到最终路径。三份内容必然相同
# （贪心解码、同 seed、同模型、批次按 prompt 长度排序，全程确定性），谁先谁后都无所
# 谓，读者拿到的要么是旧 inode 要么是新 inode，不会读到半截文件。
# 代价是三个 task 各花约 3 分钟重复生成一遍，换来的是不需要任何锁或轮询等待——后者
# 在 array task 启动顺序不确定时很容易变成死等。
# 文件已经在了就直接跳过，所以重跑不会再付这个代价。
# ---------------------------------------------------------------------------
if [[ -f "${HELDOUT_FILE}" || "${SKIP_GEN}" == "1" ]]; then
  echo "==== held-out 语料已存在，跳过生成 ===="
else
  echo "==== 检查 FLAN 语料够不够长 ===="
  # offset 落在语料末尾时 select 会静默截断，给出比要的少的行数。
  python - "${FLAN_DIR}/train" "${DATASET_CACHE}" "${HELDOUT_OFFSET}" "${HELDOUT_SIZE}" <<'PY'
import sys
from datasets import load_dataset

flan_dir, cache_dir, offset, size = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = len(load_dataset("json", data_files=f"{flan_dir}/*.jsonl", split="train", cache_dir=cache_dir))
need = offset + size
print(f"FLAN 全量 {rows} 行，需要至少 {need} 行")
if rows < need:
    raise SystemExit(
        f"语料只有 {rows} 行，取不到 [{offset}, {need}) 这一段；"
        f"调小 HELDOUT_SIZE，或确认 FLAN dump 是否完整"
    )
print(f"held-out 之后还剩 {rows - need} 行未使用")
PY

  GEN_TMP="${HELDOUT_FILE}.tmp.${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
  rm -f "${GEN_TMP}"
  echo "==== generate held-out self-distilled targets -> ${GEN_TMP} ===="
  python -m onereplay.scripts.generate_replay_targets \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --data_files "${FLAN_DIR}/train/*.jsonl" --split train \
    --cache_dir "${DATASET_CACHE}" \
    --pool_size "${HELDOUT_SIZE}" --pool_offset "${HELDOUT_OFFSET}" \
    --sample_seed "${SAMPLE_SEED}" \
    --batch_size "${GEN_BATCH}" --max_len "${MAXLEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --min_new_tokens "${MIN_NEW_TOKENS}" \
    --output_path "${GEN_TMP}" --resume 1
  mv -f "${GEN_TMP}" "${HELDOUT_FILE}"
  echo "已写入 ${HELDOUT_FILE}"
fi
[[ -f "${HELDOUT_FILE}" ]] || { echo "缺 held-out 语料: ${HELDOUT_FILE}"; exit 1; }

# ---- 自检 + 过滤：把与训练池 prompt 文本相撞的行剔掉 ----
# 这是整个实验成立的前提，所以用 prompt 文本硬校验，而不是相信参数传对了。但判据是
# 交集的"比例"而不是"是否为 0"，理由见开头那段：FLAN 自身有 6.5% 的重复 prompt。
# 过滤结果写成 _disjoint.jsonl，训练那一步只读这一份。
# 三个 array task 会各算一遍并写同一个路径，所以照样先写临时文件再 mv 原子替换：
# 三份内容必然相同（同样的输入、纯确定性的集合运算），读者拿到的要么是旧 inode
# 要么是新 inode，不会读到半截文件。
echo "==== 自检 + 过滤：held-out 与训练池的 prompt 交集 ===="
FILTER_TMP="${PROBE_HELDOUT_FILE}.tmp.${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
rm -f "${FILTER_TMP}"
python - "${HELDOUT_FILE}" "${SELF_DISTILL_FILE}" "${FILTER_TMP}" \
  "${PROBE_HELDOUT_SIZE}" "${OVERLAP_ABORT_RATIO}" <<'PY'
import json
import sys
from pathlib import Path


def load(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


heldout_path, pool_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
want, abort_ratio = int(sys.argv[4]), float(sys.argv[5])

heldout = load(heldout_path)
pool_prompts = {row["inputs"] for row in load(pool_path)}
rest = [row for row in heldout if row["inputs"] not in pool_prompts]
overlap = len(heldout) - len(rest)
ratio = overlap / max(len(heldout), 1)
print(f"held-out {len(heldout)} 行；训练池 {len(pool_prompts)} 个不同 prompt")
print(f"prompt 文本交集 {overlap} 行（{ratio:.1%}）")

# 语料自身的重复率量级（约 6.5%）与"offset 没生效"（约 100%）差十几倍，所以这条
# 比例判据既不会被前者误杀，也不会漏掉后者。
if ratio >= abort_ratio:
    raise SystemExit(
        f"交集 {ratio:.1%} 远超 FLAN 自身的重复 prompt 率，说明 held-out 基本就是训练池"
        f"那一段；检查 --pool_offset 是否生效、HELDOUT_FILE 是否指到了池子的语料"
    )

usable = [row for row in rest if not row.get("truncated") and row.get("targets", "").strip()]
print(
    f"剔掉交集 {overlap} 行；剩下的 {len(rest)} 行里再剔掉截断/空 target "
    f"{len(rest) - len(usable)} 行 -> {len(usable)} 行可用"
)
if len(usable) < want:
    raise SystemExit(
        f"可用行 {len(usable)} < 探针要的 {want} 条；调大 HELDOUT_SIZE 重新生成 held-out"
    )

with open(out_path, "w", encoding="utf-8") as handle:
    for row in usable:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"已写出 {len(usable)} 行到 {out_path}")
print("自检通过：过滤后的 held-out 与训练池零 prompt 重叠")
PY
mv -f "${FILTER_TMP}" "${PROBE_HELDOUT_FILE}"
echo "探针 held-out 语料: ${PROBE_HELDOUT_FILE}"

# ---------------------------------------------------------------------------
# Step 2: 训练 + 探针
# ---------------------------------------------------------------------------
# 追加写的 jsonl：重跑前先挪走，否则两次的探针点会串在一起。
if [[ -f "${METRICS_PATH}" ]]; then
  mv "${METRICS_PATH}" "${METRICS_PATH}.$(date +%Y%m%d_%H%M%S).bak"
  echo "已把上一次的 ${METRICS_PATH} 改名备份"
fi

# 按档位拼装 replay 与正则的开关。
EXTRA_ARGS=()
if (( RPB > 0 )); then
  EXTRA_ARGS+=(--replay_per_batch "${RPB}"
               --replay_self_distill_file "${SELF_DISTILL_FILE}"
               --replay_drop_truncated 1)
fi
if [[ "${LAMBDA}" == "0" ]]; then
  # 真 vanilla / 纯 replay：不加载 C，也不为了记录而每步去算它。
  EXTRA_ARGS+=(--replay_lambda 0 --measure_replay_when_lambda_zero 0)
else
  EXTRA_ARGS+=(--cov_path "${COV_PATH}" --replay_lambda "${LAMBDA}")
fi

# --probe_inpool_file 显式传：vanilla 和 OneReplay 不做 replay，不显式给的话它们就
# 没有池内曲线，也就丢掉了"两条 FLAN 曲线本该重合"这个对照。
echo "==== train + probe: ${RUN} ===="
python -m onereplay.scripts.train --paradigm sft \
  --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
  --dataset_path "${CS_DATASET}" \
  --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
  --eval_batch_size "${EVAL_BATCH}" \
  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
  --target_modules "${TARGET_MODULES}" \
  --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
  --max_steps "${MAX_STEPS}" \
  --replay_cache_dir "${DATASET_CACHE}" \
  "${EXTRA_ARGS[@]}" \
  --probe_every_updates "${PROBE_EVERY_UPDATES}" \
  --probe_heldout_file "${PROBE_HELDOUT_FILE}" \
  --probe_inpool_file "${SELF_DISTILL_FILE}" \
  --probe_heldout_size "${PROBE_HELDOUT_SIZE}" \
  --probe_inpool_size "${PROBE_INPOOL_SIZE}" \
  --probe_cs_val_size "${PROBE_CS_VAL_SIZE}" \
  --save "${SAVE}" --save_path "${RESULTS_ROOT}/adapters/${RUN}" \
  --metrics_path "${METRICS_PATH}"

# ---------------------------------------------------------------------------
# Step 3: 自检 + 曲线摘要
#
# step 0 的两条 FLAN 曲线用的都是 base 模型自己的贪心回答做 target，所以此刻的 loss
# 必须很小。如果它不小，说明 target 口径串了（比如误用了 gold），后面整条曲线的解读
# 都会错，先在这里拦下来。
# 然后把曲线等间隔采样成一张表打出来，并给出形状判定——老师要的就是这个答案。
# ---------------------------------------------------------------------------
echo "==== 自检 + 曲线摘要 ===="
python - "${METRICS_PATH}" "${PROBE_EVERY_UPDATES}" "${RUN}" <<'PY'
import json
import sys

path, every, run = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path, encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]

probes = [r for r in records if r.get("record_type") == "probe"]
epoch_rows = [r for r in records if r.get("record_type") == "epoch"]
if not probes:
    raise SystemExit("一个探针点都没有；--probe_every_updates 没生效，这次 run 白跑")

names = sorted(key[len("probe_"):] for key in probes[0] if key.startswith("probe_"))
print(f"run={run}  探针集={names}")
print(f"探针点 {len(probes)} 个，update {probes[0]['update']} -> {probes[-1]['update']}")
print(f"探针总耗时 {sum(r['probe_sec'] for r in probes) / 60:.1f} min")

first = probes[0]
if first["update"] != 0:
    raise SystemExit(f"第一个探针点在 update {first['update']}，不是训练前的 0")
for name in ("flan_heldout", "flan_inpool"):
    key = f"probe_{name}"
    if key not in first:
        continue
    print(f"step 0 {name} = {first[key]:.6f}")
    # base 模型对自己的贪心回答，交叉熵应远小于 1。0.5 是给 LoRA 初始化和左截断留的
    # 余量；真出现大于它的值就是 target 口径出了问题。
    if first[key] > 0.5:
        raise SystemExit(
            f"step 0 的 {name}={first[key]:.4f} 太高。自蒸馏 target 下 base 模型对自己的"
            f"回答应接近 0；检查 --probe_*_file 是不是指到了 gold target 的语料"
        )

gaps = {probes[i]["update"] - probes[i - 1]["update"] for i in range(1, len(probes))}
if gaps and gaps != {every}:
    print(f"警告: 探针间隔不均匀 {sorted(gaps)}，期望恒为 {every}")

# ---- 采样成一张表 ----
has_gap = "probe_flan_heldout" in first and "probe_flan_inpool" in first
header = f"{'update':>8} {'epoch':>6}"
for name in names:
    header += f" {name:>12}"
if has_gap:
    header += f" {'gap(h-i)':>10}"
print()
print(header)
stride = max(len(probes) // 24, 1)
shown = list(range(0, len(probes), stride))
if shown[-1] != len(probes) - 1:
    shown.append(len(probes) - 1)
for index in shown:
    record = probes[index]
    line = f"{record['update']:>8} {record['epoch']:>6}"
    for name in names:
        line += f" {record[f'probe_{name}']:>12.6f}"
    if has_gap:
        line += f" {record['probe_flan_heldout'] - record['probe_flan_inpool']:>10.6f}"
    print(line)

# ---- 形状判定 ----
# 逐点有噪声，先做窗口 3 的滑动平均再看走势，否则相邻两点的抖动会被当成拐点。
def smooth(values, window=3):
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


print()
for name in names:
    series = [r[f"probe_{name}"] for r in probes]
    if len(series) < 5:
        continue
    updates = [r["update"] for r in probes]
    # step 0 是训练前的锚点，先摘掉再平滑，不能平滑完再切。它与第一个训练点之间
    # 必然有一次跳变（自蒸馏 target 下起点接近 0），平滑窗口跨过这个跳变会把第一个
    # 训练点压低一截，于是一条单调下降的曲线会被判成"先升后降"。
    body = smooth(series[1:])
    body_updates = updates[1:]
    lo = min(range(len(body)), key=lambda i: body[i])
    hi = max(range(len(body)), key=lambda i: body[i])
    span = max(body) - min(body)
    scale = max(abs(body[0]), 1e-9)
    start, end = body[0], body[-1]

    if span < 0.02 * scale:
        shape = "基本不变"
    elif lo == 0 and hi == len(body) - 1:
        shape = "单调上升"
    elif hi == 0 and lo == len(body) - 1:
        shape = "单调下降"
    elif 0 < lo < len(body) - 1 and (start - body[lo]) > 0.1 * span and (end - body[lo]) > 0.1 * span:
        shape = f"先降后升，最低点在 update {body_updates[lo]}（{body[lo]:.6f}）"
    elif 0 < hi < len(body) - 1 and (body[hi] - start) > 0.1 * span and (body[hi] - end) > 0.1 * span:
        shape = f"先升后降，最高点在 update {body_updates[hi]}（{body[hi]:.6f}）"
    elif end > start:
        shape = "总体上升（非单调）"
    else:
        shape = "总体下降（非单调）"
    print(
        f"{name:>12}: step0={series[0]:.6f} 训练首点={start:.6f} 末点={end:.6f} "
        f"极差={span:.6f} -> {shape}"
    )

if has_gap:
    gap_series = [r["probe_flan_heldout"] - r["probe_flan_inpool"] for r in probes]
    print(
        f"{'gap':>12}: 起始={gap_series[1]:.6f} 结束={gap_series[-1]:.6f} "
        f"最大={max(gap_series):.6f}"
    )
    print(
        "  gap 随训练张开 = replay 在背池子（池内降、池外不降）；"
        "gap 始终贴近 0 = 两个切片同步变化，没有过拟合到具体样本。"
    )

print()
for record in epoch_rows:
    print(
        f"epoch {record['epoch']}: val_loss={record['val_loss']:.6f} "
        f"train_task_loss={record['train_task_loss']:.6f} "
        f"train_sec={record.get('train_sec', 0):.0f}（已扣探针） "
        f"probe_sec_total={record.get('probe_sec_total', 0):.0f}"
    )
print("自检通过")
PY

# ---------------------------------------------------------------------------
# Step 4: 与历史 run 对轨
#
# 探针只做 forward，eval 模式下 dropout 恒等、不抽全局 RNG，探针 loader 也是
# shuffle=False，所以权重轨迹本该与不带探针的历史 run 逐位相同。对不上就说明探针
# 以某种方式干扰了训练，这时曲线画出来也不代表原来那条 baseline。
# 只告警不中断：cs_vanilla_seed1 是 7 月的旧代码跑的，本来就不保证逐位可比。
# ---------------------------------------------------------------------------
if [[ -f "${BASELINE_METRICS}" && "${MAX_STEPS}" == "0" ]]; then
  echo "==== 对轨：${BASELINE_METRICS} ===="
  python - "${METRICS_PATH}" "${BASELINE_METRICS}" <<'PY'
import json
import sys


def epoch_records(path):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [r for r in rows if r.get("record_type", "epoch") == "epoch"]


probed = epoch_records(sys.argv[1])
baseline = epoch_records(sys.argv[2])
if not baseline:
    raise SystemExit(0)

for new, old in zip(probed, baseline):
    for field in ("train_task_loss", "val_loss"):
        a, b = new.get(field), old.get(field)
        if a is None or b is None:
            continue
        delta = abs(a - b)
        flag = "一致" if delta < 1e-9 else f"差 {delta:.3e}"
        print(f"epoch {new['epoch']} {field}: {a:.9f} vs {b:.9f}  {flag}")
        if delta >= 1e-6:
            print(f"  警告: {field} 与历史 run 相差 {delta:.3e}，探针可能干扰了训练轨迹")
PY
else
  echo "跳过对轨：没有 ${BASELINE_METRICS}，或这是 MAX_STEPS 短跑"
fi

echo "逐点数据: grep '\"record_type\": \"probe\"' ${METRICS_PATH}"
echo "Job finished at: $(date)"
