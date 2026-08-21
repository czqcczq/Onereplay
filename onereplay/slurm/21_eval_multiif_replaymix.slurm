#!/usr/bin/zsh
#
#SBATCH --job-name=or_mifredo
#SBATCH --partition=c25g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --chdir=/home/xsz96350/K-RES
#SBATCH --output=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.out
#SBATCH --error=/home/xsz96350/K-RES/onereplay/slurm/outputs/%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

set -euo pipefail
set -x

# =============================================================================
# 补测 Multi-IF English：cs_replaymix_4n4r_seed1（18 号脚本 50% 档，作业 2849732_2）
#
# 为什么需要这个脚本
# ----------------
# 那次 job 训练正常结束、adapter 已保存、IFEval 也出了结果，只有 Multi-IF 在生成到
# 约 470/909 时被打断：
#   File ".../onereplay/eval/metrics/multiif.py", line 168, in run
#     print(f"multiif generated {idx}/{len(rows)}")
#   OSError: [Errno 116] Stale file handle
# 报错点是 print，不是评测逻辑——挂掉的是**写 slurm .out 这个句柄**。.out 落在
# /home（NFS），长时间运行后句柄失效就会抛 ESTALE。所以补跑只需重跑 multiif 一项，
# 同时把这条路径绕开。
#
# 三处针对性处理
# -------------
# 1. python 的 stdout/stderr 整体重定向到 /hpcwork 上的日志文件，那 909 行进度不再
#    经过 /home 的 .out 句柄。slurm 的 .out 只留下少量 echo 和最终 summary。
# 2. 加重试（默认 3 次）。ESTALE 是瞬时故障，不该让整个 job 白跑。
#    注意 MultiIFMetric 没有断点续跑，重试是从头再来，所以墙时间按 3 次估。
# 3. 跑完做完整性自检（909 组、turn 计数 909/909/896），避免把半截结果当成最终值。
#
# 与原 run 的口径一致性
# -------------------
# 评测只依赖 adapter，不依赖训练环境，且 --seed 1、--multiif_language English、
# --multiif_limit 0 与 18 号脚本里那次调用逐字相同，所以结果与其余各 run 可比。
#
# 用法:
#   sbatch onereplay/slurm/21_eval_multiif_replaymix.slurm
# 换一个 adapter:
#   RUN=cs_replaymix_6n2r_seed1 sbatch onereplay/slurm/21_eval_multiif_replaymix.slurm
# 先小样本验证链路通不通（约 2 分钟）:
#   MULTIIF_LIMIT=20 MAX_ATTEMPTS=1 sbatch onereplay/slurm/21_eval_multiif_replaymix.slurm
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
CODE="${CODE:-onereplay}"
MODEL_DIR="${MODEL_DIR:-/hpcwork/xsz96350/Chen_logs/checkpoints/}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
RESULTS_ROOT="${RESULTS_ROOT:-/hpcwork/xsz96350/Chen_logs/onereplay/results}"

# ---- 要补测的 run（adapters/ 下的目录名，同时用作 run_name）----
RUN="${RUN:-cs_replaymix_4n4r_seed1}"
ADAPTER="${ADAPTER:-${RESULTS_ROOT}/adapters/${RUN}}"

# ---- Multi-IF 设置：与 18 号脚本那次调用逐字相同 ----
SEED="${SEED:-1}"
MULTIIF_INPUT="${MULTIIF_INPUT:-${CODE}/third_party/Multi-IF/multiIF_20241018.csv}"
MULTIIF_LANG="${MULTIIF_LANG:-English}"
MULTIIF_LIMIT="${MULTIIF_LIMIT:-0}"        # 0=全量 909 组
EXPECT_CONVERSATIONS="${EXPECT_CONVERSATIONS:-909}"

# ---- 重试与日志 ----
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/logs}"
LOG="${LOG_DIR}/${RUN}_multiif_${SLURM_JOB_ID:-local}.log"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 前置检查：三样缺一不可，都在几秒内查完，不要等 3 小时后才发现
# ---------------------------------------------------------------------------
[[ -d "${MODEL_DIR}/${MODEL_NAME}" ]] || { echo "缺模型: ${MODEL_DIR}/${MODEL_NAME}"; exit 1; }
[[ -d "${ADAPTER}" ]] || {
  echo "缺 adapter: ${ADAPTER}"
  echo "这次补测只跑评测，不重训；adapter 必须是 2849732_2 那次保存下来的那份。"
  exit 1
}
[[ -f "${MULTIIF_INPUT}" ]] || {
  echo "缺 Multi-IF 数据: ${MULTIIF_INPUT}"
  echo "请先在有网节点: cd ${CODE}/third_party && git clone https://huggingface.co/datasets/facebook/Multi-IF"
  exit 1
}

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}/multiif/${RUN}"

# ---------------------------------------------------------------------------
# 把上次中断留下的半截 responses.jsonl 挪走
#
# MultiIFMetric 用 "w" 打开该文件，本来就会截断，挪走只是为了留证据：万一新结果和
# 旧的半截对不上，还能回去查。summary.json 上次没写出来（崩在生成阶段），所以不存在
# 「读到旧 summary 误以为已完成」的风险。
# ---------------------------------------------------------------------------
PARTIAL="${RESULTS_ROOT}/multiif/${RUN}/responses.jsonl"
if [[ -f "${PARTIAL}" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  mv "${PARTIAL}" "${PARTIAL}.partial_${STAMP}"
  echo "已把上次的半截结果挪到 ${PARTIAL}.partial_${STAMP}"
fi

echo "==== Multi-IF 补测 ===="
echo "  run            = ${RUN}"
echo "  adapter        = ${ADAPTER}"
echo "  language       = ${MULTIIF_LANG}, limit = ${MULTIIF_LIMIT}, seed = ${SEED}"
echo "  输出目录        = ${RESULTS_ROOT}/multiif/${RUN}"
echo "  详细日志        = ${LOG}"
echo "  进度不打到 slurm .out，是为了绕开上次崩掉的那个 NFS 句柄；要看进度请 tail 上面这个文件"

# ---------------------------------------------------------------------------
# 评测（带重试）
# 909 组 × 3 轮 ≈ 2700 次生成，节点正常时约 3 h，被抢占时可能翻倍。
# ---------------------------------------------------------------------------
attempt=1
ok=0
while (( attempt <= MAX_ATTEMPTS )); do
  echo "---- 第 ${attempt}/${MAX_ATTEMPTS} 次尝试，开始于 $(date) ----"
  if python -m onereplay.scripts.evaluate \
      --model_dir "${MODEL_DIR}" --model_name "${MODEL_NAME}" \
      --adapter_path "${ADAPTER}" --run_name "${RUN}" \
      --metrics multiif --out_dir "${RESULTS_ROOT}" --seed "${SEED}" \
      --multiif_input "${MULTIIF_INPUT}" \
      --multiif_language "${MULTIIF_LANG}" \
      --multiif_limit "${MULTIIF_LIMIT}" \
      >> "${LOG}" 2>&1; then
    ok=1
    echo "---- 第 ${attempt} 次尝试成功，结束于 $(date) ----"
    break
  fi
  echo "---- 第 ${attempt} 次尝试失败，日志末尾 ----"
  tail -n 40 "${LOG}" || true
  attempt=$(( attempt + 1 ))
  # 写成 if 而不是 `(( ... )) && sleep`：算术表达式取值为 0 时返回非零状态，
  # 在 set -e 下会把整个 job 掐掉。
  if (( attempt <= MAX_ATTEMPTS )); then
    sleep 60
  fi
done

if (( ok != 1 )); then
  echo "${MAX_ATTEMPTS} 次都失败了，完整日志见 ${LOG}"
  exit 1
fi

# ---------------------------------------------------------------------------
# 完整性自检 + 打印结果
# 只有全部 909 组、且 turn 计数与其余 run 一致（909 / 909 / 896），这份数才能进汇总表。
# ---------------------------------------------------------------------------
echo "==== 结果 ===="
python - "${RESULTS_ROOT}/multiif/${RUN}/summary.json" "${EXPECT_CONVERSATIONS}" "${MULTIIF_LIMIT}" <<'PY'
import json
import sys

path, expect, limit = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8") as handle:
    summary = json.load(handle)

print(json.dumps(summary, ensure_ascii=False, indent=2))

if limit > 0:
    print(f"\nMULTIIF_LIMIT={limit}，这是小样本验证跑，不做完整性检查，结果不要写进汇总表")
    raise SystemExit(0)

problems = []
got = summary.get("num_conversations")
if got != expect:
    problems.append(f"num_conversations={got}，应为 {expect}")

# turn 3 少 13 组是数据本身如此（有些对话只有两轮），其余 run 也都是 896。
for turn, want in (("turn_1", 909), ("turn_2", 909), ("turn_3", 896)):
    n = summary.get("per_turn", {}).get(turn, {}).get("num_prompts")
    if n != want:
        problems.append(f"{turn}.num_prompts={n}，应为 {want}")

unsupported = summary.get("unsupported_instructions") or {}
if unsupported:
    problems.append(f"出现未支持的指令类型: {unsupported}")

if problems:
    print("\nFAIL: " + "; ".join(problems))
    print("这份结果不完整，不要填进汇总表")
    raise SystemExit(1)

print("\nOK: 909 组全部完成，turn 计数与其余 run 一致，可以填进汇总表")
print(f"  avg_strict_prompt_accuracy      = {summary['avg_strict_prompt_accuracy'] * 100:.2f}%")
print(f"  avg_strict_instruction_accuracy = {summary['avg_strict_instruction_accuracy'] * 100:.2f}%")
for turn in ("turn_1", "turn_2", "turn_3"):
    t = summary["per_turn"][turn]
    print(
        f"  {turn}: strict prompt {t['strict_prompt_accuracy'] * 100:.2f}%  "
        f"strict inst {t['strict_instruction_accuracy'] * 100:.2f}%"
    )
PY

echo "Job finished at: $(date)"
