#!/usr/bin/zsh
#
#SBATCH --job-name=or_lora_lam2e2
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# 修复后的重跑：LoRA 与全量，开销和效果一次拿全
#
# 为什么要重跑
# ----------
# 修复前正则在每个 micro-batch 都算一次，而它只依赖权重、窗口内权重不变，所以
# accumulation_steps=8 里有 7 次是白算。旧的开销数字因此系统性放大：
#
#   LoRA  88.1 -> 97.5 ms（+10.7%），预期修复后约 +1.3%
#   全量  96.5 -> 552  ms（×5.72），预期修复后约 ×1.59
#
# 更要紧的是旧数字不具备不变性——冗余次数等于累积次数，所以同一个方法换个
# accumulation_size 就会测出不同的"开销"。修复后正则固定为每次更新一次，
# 这个量才是方法本身的性质。
#
# 为什么这次连效果一起测
# ------------------
# 原计划是"只重测时间"，理由是新旧实现的累积梯度在精确算术下相同
# （scripts/verify_reg_once_per_update.py 在 float64 上直接比对累积 .grad，最大差
# 1.1e-16）。24_check_reg_equivalence 在 bf16 + 真实模型 + 真实数据上跑完后，这个
# 理由只剩一半：A2 与 A1 逐位相同（噪声底 0），B 从第 21 步起与 A1 分开，逐步
# task_loss 最大相对差 1.1e-1，epoch 级 val_loss 相对差 8.4e-4。判据被"噪声底为 0"
# 卡死了——只要新实现不是逐位复现，那个测试就永远给不出 EQUIVALENT，所以
# "分数会不会变"它答不了，只能交给一次真实的评测。
#
# 设计：一条 run 同时给出开销和效果
# ---------------------------
# OneReplay 那两条直接用 06 / 14 的生产配置跑满 3 个 epoch（λ=3e-2，是两边 λ 扫描
# 各自最好也最大的一档，正则值最大，开销是保守上界）：
#   - 开销不需要额外的短跑。trainer 每 LOG_EVERY 步打一次 win=（500 步窗口速度），
#     结束时打 ---- cost ---- 块，metrics jsonl 里还有 sec_per_step 和 peak memory。
#   - 效果必须在生产配置上测。5000 步只有约 0.24 个 epoch，那样的 adapter 和记录在案
#     的 3-epoch 分数不可比，测出来的差异无法归因给这次修复。
#   - 一条 run 出两个数，也就不存在"计时那条和评测那条不是同一次训练"的追问。
#
# vanilla 那两条只跑 ${VANILLA_STEPS} 步，且不评测
# ------------------------------------------
# 它们只提供 ms/step 的分母。修复完全不触及 λ=0 且 measure=0 的代码路径，vanilla 的
# 效果无从改变，记录在案的分数继续沿用，再花四小时评测一遍是纯浪费。
# 但分母必须在同一个 job、同一个节点上测出来，不能沿用历史值：全量那个 96.5 ms 早于
# `win=` 字段，是全程平均而不是稳态，混着冷启动，拿它做分母会把开销算小。
# 两边窗口数不等（vanilla 10 个，OneReplay 约 125 个）不影响可比性：win= 都是 500 步
# 窗口速度，取 25 百分位本来就是为了剔掉偶发的慢窗口。
#
# vanilla 一律 --measure_replay_when_lambda_zero 0。λ=0 但仍加载 C 并求值的 run 付的
# 是全额正则开销，当基线会把开销吃掉一半（这是 OPD 那次踩过的坑）。
#
# --exclusive 独占节点。共享节点上邻居抢 CPU 会直接体现在 ms/step 上，2649412 的
# vanilla 后段就因此凭空慢了 208 s。
#
# 开销汇总：沿用 scripts/summarize_reg_cost.py，不新写比对代码
# --------------------------------------------------
# 它按 <mode>_rep1.log 找 win=、取 25 百分位，和修复前的 88.1/97.5、96.5/552 逐项
# 对照，并检查 (OneReplay-vanilla)×accum_steps 是否落回旧的每-micro-batch 开销
# （约 9 ms / 455 ms）——落回了就说明两次测量互相印证，变的只是求值频率。
# 所以生产 run 的日志和 metrics 会额外拷一份成 <mode>_rep1.*，喂给它。
# 它原来会在两条 run 的 train_tokens 不一致时告警，而现在 vanilla 故意跑得短，
# 那个告警成了必然的误报，所以判据收紧成"步数相同才检查 token 数"。
#
# 效果对比：也不需要脚本
# ------------------
# 训练每个 epoch 的 val_loss 写在 metrics jsonl 里，评测把 summary.json 打印出来，
# 脚本末尾把新旧两份直接并排 cat，肉眼比即可。记录值：
#   LoRA λ=3e-2 (job 2856542_1)  IFEval strict_prompt 0.6617，Multi-IF avg 0.4706
#   全量 λ=3e-2 (14 号那批)       见 results/ifeval_summary.csv 里
#                                full_onereplay_lam3e-2_seed1 那一行
# 注意这次评测的分辨率：IFEval 541 条 prompt、Multi-IF 每轮 909 条，二项标准差各约
# 2.0 和 1.6 个点。它能证伪"分数变了两个点以上"，不可能证明"一点没变"。真正灵敏的
# 仪器是同一次训练顺带产出的 val_loss（1000 条、连续量），24 号在 800 步上给的界是
# 8.4e-4 相对差，这次把它延长到 3 个 epoch。
#
# 时间预算（当前只跑 LoRA λ=2e-2）
#   200 步 warmup       不到 1 min（不保存、不评测）
#   LoRA 3 epoch        约 1.6 h
#   IFEval + Multi-IF   约 4 h
#   --time 给到 8 h 留余量。
#
# 当前直接提交配置:
#   sbatch onereplay/slurm/25_bench_reg_cost.slurm
#   只运行 LoRA OneReplay λ=2e-2，完成 3 epoch 训练、IFEval 和 Multi-IF；
#   不重复运行 5000 步 Vanilla 开销基线，保留最终 adapter。
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

REPO_ROOT="${REPO_ROOT:-/home/xsz96350/K-RES}"
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
CS_DATASET="${CS_DATASET:-/hpcwork/xsz96350/Chen_logs/onereplay/datasets/commonsense_170k_hf}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"
COV_QV="${COV_QV:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_qv.pt}"
COV_FULL="${COV_FULL:-${RESULTS_ROOT}/cov/cov_flan_chat_20k_full.pt}"
# 全量那批的 metrics 和 checkpoint 与 14 号脚本同目录，这样新旧两条落在一起
METRICS_DIR_FULL="${METRICS_DIR_FULL:-${RESULTS_ROOT}/metrics/full}"
CKPT_DIR_FULL="${CKPT_DIR_FULL:-${RESULTS_ROOT}/checkpoints/full}"

# ---- 共享超参：与 06 / 14 逐字一致 ----
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-64}"
MAXLEN="${MAXLEN:-512}"
VAL_FRAC="${VAL_FRAC:-0.01}"
MAXVAL="${MAXVAL:-1000}"
LAMBDA="2e-2"

# ---- 各自与历史训练 run 一致的部分 ----
LR_LORA="${LR_LORA:-1e-4}"
TM_LORA="${TM_LORA:-q_proj,v_proj}"
LR_FULL="${LR_FULL:-1e-5}"
TM_FULL="${TM_FULL:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head}"

# ---- 计时口径 ----
# LOG_EVERY=500 -> win= 是 500 步窗口速度，与 results_log 第二章和
# summarize_reg_cost 的 25 百分位口径一致。
LOG_EVERY="${LOG_EVERY:-500}"
VANILLA_STEPS=0
WARMUP_STEPS="${WARMUP_STEPS:-200}"

MODES="lora"
RUN_TRAIN=1
RUN_EVAL=1
# 全量 checkpoint 一个约 3.4 GiB；指标落盘后不再复用权重时设 0 让 job 自己清掉
KEEP_CKPT=1

IFEVAL_INPUT="${IFEVAL_INPUT:-${CODE}/third_party/instruction_following_eval/data/input_data.jsonl}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"
EVAL_METRICS="ifeval,multiif"

OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/regcost/${SLURM_JOB_ID:-local}}"
MAP_CACHE="${RESULTS_ROOT}/map_cache/regcost"

cd "${REPO_ROOT}"
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${CS_DATASET}" ]] || { echo "缺 commonsense_170k: ${CS_DATASET}"; exit 1; }
[[ -f "${COV_QV}" ]] || { echo "缺 qv 协方差: ${COV_QV}"; exit 1; }
# 用旧代码跑会直接报 unrecognized arguments，但显式检查能给出更有用的提示。
grep -q "reg_once_per_update" onereplay/scripts/train.py || {
  echo "train.py 里没有 --reg_once_per_update，代码没同步到最新"
  exit 1
}
mkdir -p "${OUT_DIR}" "${MAP_CACHE}" "${RESULTS_ROOT}/adapters" \
  "${RESULTS_ROOT}/metrics" "${METRICS_DIR_FULL}" "${CKPT_DIR_FULL}"

echo "MODES=${MODES} RUN_TRAIN=${RUN_TRAIN} RUN_EVAL=${RUN_EVAL} EVAL_METRICS=${EVAL_METRICS}"

# ---------------------------------------------------------------------------
# 按档位摊开配置。zsh 函数里赋的是全局变量，调用后直接用。
#   RUN        本次的新 run 名，带 _regonce 后缀，避免覆盖记录在案的那条
#   BASE_RUN   记录在案的旧实现 run 名，末尾用来并排打印
#   VAN_TAG / REG_TAG   summarize_reg_cost 认的文件名前缀
# ---------------------------------------------------------------------------
mode_settings() {
  case "$1" in
    lora)
      FULL=0
      LR="${LR_LORA}"
      TM="${TM_LORA}"
      COV="${COV_QV}"
      BASE_RUN="cs_onereplay_lam${LAMBDA}_seed${SEED}"
      RUN="${BASE_RUN}_regonce"
      SAVE_DIR="${RESULTS_ROOT}/adapters/${RUN}"
      METRICS_FILE="${RESULTS_ROOT}/metrics/${RUN}.jsonl"
      BASE_METRICS="${RESULTS_ROOT}/metrics/${BASE_RUN}.jsonl"
      VAN_TAG="lora_van"
      REG_TAG="lora_or"
      # IFEval 和 Multi-IF 不读这几个参数（06 号当时也没传），带上只是为了
      # EVAL_METRICS 里加 commonsense 时也能用，顺带避免空数组展开。
      EVAL_EXTRA=(--dataset_path "${CS_DATASET}"
                  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}"
                  --max_len "${MAXLEN}"
                  --map_cache_dir "${RESULTS_ROOT}/map_cache/eval_lora_seed${SEED}")
      ;;
    full)
      FULL=1
      LR="${LR_FULL}"
      TM="${TM_FULL}"
      COV="${COV_FULL}"
      BASE_RUN="full_onereplay_lam${LAMBDA}_seed${SEED}"
      RUN="${BASE_RUN}_regonce"
      SAVE_DIR="${CKPT_DIR_FULL}/${RUN}"
      METRICS_FILE="${METRICS_DIR_FULL}/${RUN}.jsonl"
      BASE_METRICS="${METRICS_DIR_FULL}/${BASE_RUN}.jsonl"
      VAN_TAG="full_van"
      REG_TAG="full_or"
      # 14 号评测时传了这几个，保持一致
      EVAL_EXTRA=(--dataset_path "${CS_DATASET}"
                  --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}"
                  --max_len "${MAXLEN}"
                  --map_cache_dir "${RESULTS_ROOT}/map_cache/eval_full_seed${SEED}")
      ;;
    *)
      echo "未知档位: $1（可选 lora / full）"
      return 1
      ;;
  esac
}

# ---------------------------------------------------------------------------
# vanilla 计时短跑：只为拿同节点的 ms/step 分母，不保存、不评测。
# $1=档位 $2=标签 $3=步数
# 全量那档单步数百毫秒，显存接近 53 GiB；|| true 保证单条失败不掐死整个 job
# （15_bench_batch_scale 曾因 b64 的 OOM 丢掉后面所有 run）。
# ---------------------------------------------------------------------------
vanilla_run() {
  local mode="$1" tag="$2" steps="$3"
  mode_settings "${mode}" || return 1
  echo "==== vanilla 计时 ${tag}: mode=${mode} steps=${steps} full_finetune=${FULL} ===="
  python -m onereplay.scripts.train --paradigm sft --full_finetune "${FULL}" \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --epochs 1 --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TM}" \
    --map_cache_dir "${MAP_CACHE}" \
    --cov_path "${COV}" --replay_lambda 0 --identity_cov 0 \
    --measure_replay_when_lambda_zero 0 \
    --reg_once_per_update 1 \
    --max_steps "${steps}" --log_every "${LOG_EVERY}" \
    --save 0 \
    --metrics_path "${OUT_DIR}/${tag}.jsonl" 2>&1 | tee "${OUT_DIR}/${tag}.log" || true
}

# ---------------------------------------------------------------------------
# OneReplay 生产 run：配置与 06 / 14 逐字一致，唯一的差别是
# --reg_once_per_update 1 和为了看窗口速度而显式给的 --log_every。
# 训练结束后把日志和 metrics 拷成 <REG_TAG>_rep1.*，供 summarize_reg_cost 读。
# ---------------------------------------------------------------------------
prod_run() {
  local mode="$1"
  mode_settings "${mode}" || return 1
  echo "==== 生产 run ${RUN}: full_finetune=${FULL} lambda=${LAMBDA} epochs=${EPOCHS} ===="
  python -m onereplay.scripts.train --paradigm sft --full_finetune "${FULL}" \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" --seed "${SEED}" \
    --dataset_path "${CS_DATASET}" \
    --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" --accumulation_size "${ACCUM}" \
    --val_fraction "${VAL_FRAC}" --max_val_samples "${MAXVAL}" --max_len "${MAXLEN}" \
    --target_modules "${TM}" \
    --map_cache_dir "${RESULTS_ROOT}/map_cache/${RUN}" \
    --cov_path "${COV}" --replay_lambda "${LAMBDA}" --identity_cov 0 \
    --reg_once_per_update 1 \
    --log_every "${LOG_EVERY}" \
    --save 1 --save_path "${SAVE_DIR}" \
    --metrics_path "${METRICS_FILE}" 2>&1 | tee "${OUT_DIR}/${REG_TAG}_rep1.log" || true
  # || true：全量那条显存接近 53 GiB，单条失败不该掐死后面的档位和评测
  # （15_bench_batch_scale 曾因 b64 的 OOM 丢掉后面所有 run）。
  if [[ -f "${METRICS_FILE}" ]]; then
    cp -f "${METRICS_FILE}" "${OUT_DIR}/${REG_TAG}_rep1.jsonl"
  fi
}

# ---------------------------------------------------------------------------
# 评测：口径与记录在案的那条逐字一致，只有 run_name 不同。
# ---------------------------------------------------------------------------
eval_run() {
  local mode="$1"
  mode_settings "${mode}" || return 1
  if [[ ! -d "${SAVE_DIR}" ]]; then
    echo "缺权重，跳过评测: ${SAVE_DIR}"
    return 0
  fi
  echo "==== eval ${RUN} (${EVAL_METRICS}) ===="
  python -m onereplay.scripts.evaluate \
    --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
    --adapter_path "${SAVE_DIR}" --run_name "${RUN}" \
    --metrics "${EVAL_METRICS}" --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
    "${EVAL_EXTRA[@]}" \
    --ifeval_input "${IFEVAL_INPUT}" \
    --multiif_input "${MULTIIF_INPUT}" --multiif_language "${MULTIIF_LANG}" \
    --multiif_limit "${MULTIIF_LIMIT}"
}

# ---------------------------------------------------------------------------
# 新旧并排打印。没有任何判据，就是把两边的数字放在同一屏里。
# ---------------------------------------------------------------------------
show_pair() {
  local mode="$1"
  mode_settings "${mode}" || return 1
  echo "==== ${mode}: 新实现 ${RUN} vs 记录在案的 ${BASE_RUN} ===="
  echo "---- 每个 epoch 的 val_loss / train_task_loss（新）----"
  [[ -f "${METRICS_FILE}" ]] && cat "${METRICS_FILE}" || echo "缺 ${METRICS_FILE}"
  echo "---- 每个 epoch 的 val_loss / train_task_loss（旧，记录在案）----"
  [[ -f "${BASE_METRICS}" ]] && cat "${BASE_METRICS}" || echo "缺 ${BASE_METRICS}"
  local metric
  for metric in ${(s:,:)EVAL_METRICS}; do
    echo "---- ${metric} summary（新）----"
    [[ -f "${RESULTS_ROOT}/${metric}/${RUN}/summary.json" ]] \
      && cat "${RESULTS_ROOT}/${metric}/${RUN}/summary.json" \
      || echo "缺 ${RESULTS_ROOT}/${metric}/${RUN}/summary.json"
    echo "---- ${metric} summary（旧，记录在案）----"
    [[ -f "${RESULTS_ROOT}/${metric}/${BASE_RUN}/summary.json" ]] \
      && cat "${RESULTS_ROOT}/${metric}/${BASE_RUN}/summary.json" \
      || echo "缺 ${RESULTS_ROOT}/${metric}/${BASE_RUN}/summary.json"
  done
}

mode_list=(${=MODES})

if [[ "${RUN_TRAIN}" == "1" ]]; then
  # warmup 结果丢弃，只为把 GPU 时钟和 page cache 拉起来，不让第一条 vanilla
  # 背上冷启动（那会抬高分母，把正则开销算小）。
  vanilla_run "${mode_list[1]}" warmup "${WARMUP_STEPS}"
  rm -f "${OUT_DIR}/warmup.log" "${OUT_DIR}/warmup.jsonl"

  if (( VANILLA_STEPS > 0 )); then
    for mode in ${mode_list}; do
      mode_settings "${mode}"
      vanilla_run "${mode}" "${VAN_TAG}_rep1" "${VANILLA_STEPS}"
    done
  else
    echo "VANILLA_STEPS=0：跳过分母，summarize_reg_cost 只会打印 OneReplay 那一侧"
  fi

  for mode in ${mode_list}; do
    prod_run "${mode}"
  done
else
  echo "RUN_TRAIN=0：跳过训练，只做评测和并排打印"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  for mode in ${mode_list}; do
    eval_run "${mode}"
  done
else
  echo "RUN_EVAL=0：跳过评测。跑完训练后可用 RUN_TRAIN=0 单独提交评测（只要 1 张卡）"
fi

# ---------------------------------------------------------------------------
# 开销汇总（已有代码）+ 效果并排（纯 cat）
# ---------------------------------------------------------------------------
echo "==== 开销汇总 ===="
python -m onereplay.scripts.summarize_reg_cost --out_dir "${OUT_DIR}" || true

echo "==== 效果并排 ===="
for mode in ${mode_list}; do
  show_pair "${mode}"
done

if [[ "${KEEP_CKPT}" != "1" ]]; then
  for mode in ${mode_list}; do
    mode_settings "${mode}"
    if (( FULL == 1 )); then
      echo "KEEP_CKPT=0：删掉全量 checkpoint ${SAVE_DIR}"
      rm -rf "${SAVE_DIR}"
    fi
  done
fi

echo "结果目录: ${OUT_DIR}"
echo "Job finished at: $(date)"
