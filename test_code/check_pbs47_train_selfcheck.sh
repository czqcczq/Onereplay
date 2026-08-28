#!/bin/bash
# Throwaway check for 47_if_fisher.pbs's Step 4 self-check, extracted verbatim.
# The block guards a 5.7-hour training run, and the failure it exists to catch
# (penalty identically zero, i.e. the Fisher matched no LoRA layer and the
# checkpoint is really vanilla) only shows up in the metrics file. So the block
# itself has to be exercised on a good record and on each defect before a job
# relies on it. No GPU, no cluster.
set -uo pipefail

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# The cluster has `python`; a bare WSL shell often only has `python3`. The block
# under test is copied verbatim and calls `python`, so bridge it here rather than
# editing the copy. Only stdlib json/sys are needed.
if ! command -v python >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    python () { python3 "$@"; }
    echo "note: no 'python' on PATH, using python3"
  else
    echo "SKIP: neither python nor python3 found"; exit 0
  fi
fi

FISHER_PATH="/results/fisher/fisher_flan_chat_20k_qv.pt"
LAMBDA="3e2"
ACCUM="64"

# --- the block under test, byte-identical to the PBS Step 4 ---
selfcheck () {
  local metrics="$1" vanilla="${2:-}"
  python - "${metrics}" "${LAMBDA}" "${FISHER_PATH}" "${ACCUM}" "${vanilla}" <<'PY'
import json
import sys

path = sys.argv[1]
want_lambda = float(sys.argv[2])
want_penalty_path = sys.argv[3]
want_rows_per_update = int(sys.argv[4])
vanilla_val = sys.argv[5]
with open(path, encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
if not records:
    raise SystemExit(f"{path} 是空的，训练没有写出任何一条 metrics")
last = records[-1]

reg = last.get("train_replay_reg") or 0.0
lambda_reg = last.get("train_lambda_reg") or 0.0
task_loss = last.get("train_task_loss") or 0.0
share = lambda_reg / task_loss if task_loss else 0.0

print(f"regularizer      = {last.get('regularizer')}")
print(f"replay_lambda    = {last.get('replay_lambda')} (want {want_lambda})")
print(f"penalty_path     = {last.get('penalty_path')}")
print(f"batch/accum      = {last.get('batch_size')} / {last.get('accumulation_size')} (行/更新)")
print(f"train_replay_reg = {reg:.6e}")
print(f"lambda * reg     = {lambda_reg:.6e}  ({share:.2%} of task_loss {task_loss:.4f})")
print(f"val_loss         = {last.get('val_loss')}")

failures = []
if last.get("regularizer") != "ewc":
    failures.append(f"regularizer={last.get('regularizer')!r}，应为 'ewc'，这次 run 白跑")
if float(last.get("replay_lambda") or 0) != want_lambda:
    failures.append(f"replay_lambda={last.get('replay_lambda')}，应为 {want_lambda}")
got_penalty = last.get("penalty_path")
if got_penalty is None:
    print("提示: metrics 里没有 penalty_path（train.py 版本较旧），跳过这一项")
elif got_penalty != want_penalty_path:
    failures.append(f"penalty_path={got_penalty!r}，应为 {want_penalty_path!r}")
rows_per_update = last.get("accumulation_size")
if rows_per_update is not None and int(rows_per_update) != want_rows_per_update:
    failures.append(
        f"accumulation_size={rows_per_update}，应为 {want_rows_per_update}。"
        "每次更新的行数变了，与旧 run 不再可比"
    )
if reg == 0.0:
    failures.append(
        "罚项恒为 0：Fisher 的层名一个都没匹配上 LoRA 模块，或 F 文件本身是空的。"
        " 看训练日志里 'regularizer covers N layers' 是不是 0。这个 checkpoint 实际是 vanilla。"
    )

if share and share < 1e-3:
    print(f"提示: 罚项只占 task_loss 的 {share:.3%}，这一档基本等于 vanilla，λ 偏小")
elif share > 0.5:
    print(f"提示: 罚项占 task_loss 的 {share:.1%}，新任务很可能学不动了，λ 偏大")

if vanilla_val:
    gap = (last.get("val_loss") or 0.0) - float(vanilla_val)
    print(f"val_loss 相对 vanilla({vanilla_val}) 的差值 = {gap:+.4f}")
    if gap > 0.05:
        print(
            "提示: 新任务明显学得更差。这正是 EWC-LoRA(2602.17559) 观察到的 precomputed"
            " 固定 Fisher 塑性偏低。写结论时要按等 val_loss 比 IFEval，"
            "不能直接说 EWC retention 更好。"
        )

if failures:
    print("\n自检失败:")
    for item in failures:
        print(f"  - {item}")
    raise SystemExit(1)
print("\n自检通过")
PY
}

# --- synthetic metrics records, one line per epoch like trainers/base.py writes ---
write_metrics () {
  local out="$1" reg="$2" regularizer="$3" lam="$4" penalty="$5" accum="$6"
  python - "${out}" "${reg}" "${regularizer}" "${lam}" "${penalty}" "${accum}" <<'PY'
import json
import sys

out, reg, regularizer, lam, penalty, accum = sys.argv[1:7]
reg = float(reg)
lam = float(lam)
with open(out, "w", encoding="utf-8") as handle:
    for epoch in (1, 2, 3):
        handle.write(json.dumps({
            "epoch": epoch,
            "train_task_loss": 1.1,
            "train_replay_reg": reg,
            "train_lambda_reg": lam * reg,
            "val_loss": 1.05,
            "regularizer": regularizer if regularizer != "none" else None,
            "replay_lambda": lam,
            "penalty_path": penalty if penalty != "none" else None,
            "batch_size": 8,
            "accumulation_size": int(accum),
        }) + "\n")
PY
}

FISHER="${FISHER_PATH}"
fail=0
expect_pass () {
  local label="$1"; shift
  if selfcheck "$@" > "${TMP}/out.log" 2>&1; then
    echo "PASS (expected): ${label}"
  else
    echo "FAIL: ${label} should have passed"; cat "${TMP}/out.log"; fail=1
  fi
}
expect_reject () {
  local label="$1" needle="$2"; shift 2
  if selfcheck "$@" > "${TMP}/out.log" 2>&1; then
    echo "FAIL: ${label} should have been rejected"; cat "${TMP}/out.log"; fail=1
  elif grep -q "${needle}" "${TMP}/out.log"; then
    echo "REJECT (expected): ${label}"
  else
    echo "FAIL: ${label} rejected but never mentioned '${needle}'"; cat "${TMP}/out.log"; fail=1
  fi
}

echo "==== a healthy EWC run ===="
write_metrics "${TMP}/good.jsonl" 2.0e-4 ewc 3e2 "${FISHER}" 64
selfcheck "${TMP}/good.jsonl"
echo

# lambda*reg / task_loss = 3e2*2e-4/1.1 = 5.5%, inside the sane band.
expect_pass "healthy run" "${TMP}/good.jsonl"

# The expensive silent failure: 5.7h of training that produced a vanilla model.
write_metrics "${TMP}/zeroreg.jsonl" 0.0 ewc 3e2 "${FISHER}" 64
expect_reject "penalty identically zero" "罚项恒为 0" "${TMP}/zeroreg.jsonl"

# --regularizer defaulted back to onereplay, so this run used C, not F.
write_metrics "${TMP}/wrongreg.jsonl" 2.0e-4 onereplay 3e2 "${FISHER}" 64
expect_reject "regularizer not ewc" "regularizer=" "${TMP}/wrongreg.jsonl"

# Trained against some other Fisher than the one this job just collected.
write_metrics "${TMP}/wrongf.jsonl" 2.0e-4 ewc 3e2 "/results/fisher/fisher_math.pt" 64
expect_reject "penalty_path points elsewhere" "penalty_path=" "${TMP}/wrongf.jsonl"

# Rows per optimizer update drifted, so the run is not comparable to the old one.
write_metrics "${TMP}/wrongaccum.jsonl" 2.0e-4 ewc 3e2 "${FISHER}" 128
expect_reject "accumulation_size drifted" "accumulation_size=" "${TMP}/wrongaccum.jsonl"

# Wrong lambda.
write_metrics "${TMP}/wronglam.jsonl" 2.0e-4 ewc 1e2 "${FISHER}" 64
expect_reject "lambda not 3e2" "replay_lambda=" "${TMP}/wronglam.jsonl"

# An older train.py that does not record penalty_path must degrade to a note.
write_metrics "${TMP}/nopath.jsonl" 2.0e-4 ewc 3e2 none 64
expect_pass "metrics without penalty_path" "${TMP}/nopath.jsonl"
grep -q "没有 penalty_path" "${TMP}/out.log" || { echo "FAIL: no note about the missing key"; fail=1; }

# Empty metrics file (training died before the first epoch ended).
: > "${TMP}/empty.jsonl"
expect_reject "empty metrics file" "是空的" "${TMP}/empty.jsonl"

echo
echo "==== the plasticity hint fires on a worse val_loss ===="
selfcheck "${TMP}/good.jsonl" 0.95 > "${TMP}/vanilla.log" 2>&1
grep -q "val_loss 相对 vanilla" "${TMP}/vanilla.log" || { echo "FAIL: no gap line"; fail=1; }
grep -q "塑性偏低" "${TMP}/vanilla.log" || { echo "FAIL: no plasticity hint"; fail=1; }
grep "val_loss 相对 vanilla\|塑性偏低" "${TMP}/vanilla.log"

# And stays quiet when the new task was learned just as well.
selfcheck "${TMP}/good.jsonl" 1.04 > "${TMP}/vanilla2.log" 2>&1
grep -q "塑性偏低" "${TMP}/vanilla2.log" && { echo "FAIL: hint fired on a 0.01 gap"; fail=1; }
echo "quiet on a 0.01 gap (expected)"

# Penalty share at the extremes should hint, not fail.
write_metrics "${TMP}/tiny.jsonl" 1.0e-9 ewc 3e2 "${FISHER}" 64
expect_pass "negligible penalty share" "${TMP}/tiny.jsonl"
grep -q "λ 偏小" "${TMP}/out.log" || { echo "FAIL: no small-lambda hint"; fail=1; }
write_metrics "${TMP}/huge.jsonl" 1.0e-1 ewc 3e2 "${FISHER}" 64
expect_pass "overwhelming penalty share" "${TMP}/huge.jsonl"
grep -q "λ 偏大" "${TMP}/out.log" || { echo "FAIL: no large-lambda hint"; fail=1; }

echo
if [[ ${fail} -eq 0 ]]; then
  echo "ALL CHECKS PASSED"
else
  echo "SOME CHECKS FAILED"
fi
exit ${fail}
