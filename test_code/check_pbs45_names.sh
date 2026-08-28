#!/bin/bash
# Throwaway check for 45_math_replay_baselines.pbs's naming and batch arithmetic.
# Runs the helper definitions in isolation, no cluster and no python needed.
set -euo pipefail

SEED=1
BATCH=16
REPLAY_PER_BATCH=8
ACCUM=128
ADAPTERS="/adapters"
RUN_PREFIX="cs_replaymix_ifmath"
W_IF_ROWS=0.5;   W_MATH_ROWS=0.5
W_IF_TOKENS=0.8; W_MATH_TOKENS=0.2
ARMS="rows tokens"
MODELS=""
EXTRA_MODELS=""

NEW_PER_BATCH=$(( BATCH - REPLAY_PER_BATCH ))
SHARE_TAG="${NEW_PER_BATCH}n${REPLAY_PER_BATCH}r"
NEW_PER_UPDATE=$(( (ACCUM / BATCH) * NEW_PER_BATCH ))

# The whole point of doubling the micro-batch: this number must not move, and it
# must survive halving the micro-batch again (the OOM fallback).
echo "batch=${BATCH} accum_steps=$(( ACCUM / BATCH )) new/update=${NEW_PER_UPDATE}"
[[ ${NEW_PER_UPDATE} -eq 64 ]] || { echo "FAIL new/update != 64"; exit 1; }
[[ $(( (ACCUM / 8) * (8 - 4) )) -eq 64 ]] || { echo "FAIL batch=8 fallback != 64"; exit 1; }
[[ $(( (64 / BATCH) * BATCH )) -eq 64 ]] || { echo "FAIL vanilla-style accum"; exit 1; }
echo "batch=8/replay=4 fallback keeps new/update=64 with the same ACCUM"

arm_weights () {
  case "$1" in
    rows)   echo "${W_IF_ROWS},${W_MATH_ROWS}" ;;
    tokens) echo "${W_IF_TOKENS},${W_MATH_TOKENS}" ;;
    *)      echo "" ;;
  esac
}
weight_tag () {
  awk -F, -v spec="$1" 'BEGIN{
    n = split(spec, w, ",")
    s = 0
    for (i = 1; i <= n; i++) s += w[i]
    out = ""
    for (i = 1; i <= n; i++) out = out (i > 1 ? "-" : "") sprintf("%d", w[i] / s * 100 + 0.5)
    print out
  }'
}
run_name () {
  local weights
  weights="$(arm_weights "$1")"
  [[ -n "${weights}" ]] || { echo ""; return; }
  echo "${RUN_PREFIX}_${SHARE_TAG}_rows$(weight_tag "${weights}")_seed${SEED}"
}

got_rows="$(run_name rows)"
got_tokens="$(run_name tokens)"
echo "rows arm   : ${got_rows}"
echo "tokens arm : ${got_tokens}"
[[ "${got_rows}"   == "cs_replaymix_ifmath_8n8r_rows50-50_seed1" ]] || { echo "FAIL rows name"; exit 1; }
[[ "${got_tokens}" == "cs_replaymix_ifmath_8n8r_rows80-20_seed1" ]] || { echo "FAIL tokens name"; exit 1; }
[[ -z "$(run_name bogus)" ]] || { echo "FAIL unknown arm should be empty"; exit 1; }

# Unnormalized weights must land on the same name, matching train.py's handling.
W_IF_TOKENS=4; W_MATH_TOKENS=1
[[ "$(run_name tokens)" == "${got_tokens}" ]] || { echo "FAIL 4:1 != 0.8:0.2"; exit 1; }
W_IF_TOKENS=0.8; W_MATH_TOKENS=0.2
echo "4:1 normalizes to the same name as 0.8:0.2"

EVAL_SPECS=""
append_spec () {
  if [[ -z "${EVAL_SPECS}" ]]; then EVAL_SPECS="$1"; else EVAL_SPECS="${EVAL_SPECS}|$1"; fi
}
if [[ -n "${MODELS}" ]]; then
  for m in ${MODELS}; do append_spec "${m};${ADAPTERS}/${m}"; done
else
  for arm in ${ARMS}; do
    run="$(run_name "${arm}")"
    append_spec "${run};${ADAPTERS}/${run}"
  done
fi
EXTRA_MODELS="cs_replaymix_4n4r_seed1;/safety/adapters/cs_replaymix_4n4r_seed1"
if [[ -n "${EXTRA_MODELS}" ]]; then
  IFS='|' read -ra extra <<< "${EXTRA_MODELS}"
  for spec in "${extra[@]}"; do append_spec "${spec}"; done
fi

IFS='|' read -ra specs <<< "${EVAL_SPECS}"
[[ ${#specs[@]} -eq 3 ]] || { echo "FAIL expected 3 eval specs, got ${#specs[@]}"; exit 1; }
for spec in "${specs[@]}"; do
  echo "eval spec  : name=${spec%%;*} adapter=${spec#*;}"
done
[[ "${specs[2]}" == "cs_replaymix_4n4r_seed1;/safety/adapters/cs_replaymix_4n4r_seed1" ]] \
  || { echo "FAIL EXTRA_MODELS absolute path lost"; exit 1; }

echo "ALL CHECKS PASSED"
