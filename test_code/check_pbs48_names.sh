#!/bin/bash
# Throwaway check for 48_math_ewc_mix.pbs's variant/lambda naming, the PROBE grid
# narrowing, and EVAL_SPECS expansion. Runs the helper definitions in isolation,
# no cluster and no python needed.
set -euo pipefail

SEED=1
BATCH=8
ACCUM=64
ADAPTERS="/adapters"
RUN_PREFIX="cs_ewcmix"
RESULTS_ROOT="/results"
FISHER_MIX_EQUAL="${RESULTS_ROOT}/fisher/fisher_mix_equal_qv.pt"
FISHER_MIX_HALF="${RESULTS_ROOT}/fisher/fisher_mix_half_qv.pt"
VARIANTS="equal half"
LAMBDAS="1 3 10 30"
MODELS=""
EXTRA_MODELS=""

# ACCUM must divide by BATCH, else accum_steps truncates and new/update drifts.
(( ACCUM % BATCH == 0 )) || { echo "FAIL ACCUM not divisible by BATCH"; exit 1; }
# EWC has no replay slot, so rows/update == ACCUM, matching 23/47's 64.
[[ ${ACCUM} -eq 64 ]] || { echo "FAIL rows/update != 64"; exit 1; }

variant_fisher () {
  case "$1" in
    equal) echo "${FISHER_MIX_EQUAL}" ;;
    half)  echo "${FISHER_MIX_HALF}" ;;
    *)     echo "" ;;
  esac
}
variant_mix_args () {
  case "$1" in
    equal) echo "--equalize mean" ;;
    half)  echo "--weights 0.5,0.5" ;;
    *)     echo "" ;;
  esac
}
run_name () {
  echo "${RUN_PREFIX}_$1_lam$2_seed${SEED}"
}

[[ "$(variant_fisher equal)" == "/results/fisher/fisher_mix_equal_qv.pt" ]] || { echo "FAIL equal fisher"; exit 1; }
[[ "$(variant_fisher half)"  == "/results/fisher/fisher_mix_half_qv.pt"  ]] || { echo "FAIL half fisher"; exit 1; }
[[ -z "$(variant_fisher bogus)" ]] || { echo "FAIL unknown variant should be empty"; exit 1; }
[[ "$(variant_mix_args equal)" == "--equalize mean" ]]   || { echo "FAIL equal args"; exit 1; }
[[ "$(variant_mix_args half)"  == "--weights 0.5,0.5" ]] || { echo "FAIL half args"; exit 1; }

[[ "$(run_name equal 3e2)" == "cs_ewcmix_equal_lam3e2_seed1" ]] || { echo "FAIL equal name"; exit 1; }
[[ "$(run_name half 1e3)"  == "cs_ewcmix_half_lam1e3_seed1"  ]] || { echo "FAIL half name"; exit 1; }
echo "variant/lambda names ok: $(run_name equal 3e2) / $(run_name half 1e3)"

# PROBE narrows LAMBDAS to the numerically smallest entry (give-it-big hides R).
PROBE_LAMBDAS="$(printf '%s\n' ${LAMBDAS} | sort -g | head -n 1)"
[[ "${PROBE_LAMBDAS}" == "1" ]] || { echo "FAIL probe should pick 1, got ${PROBE_LAMBDAS}"; exit 1; }
# also holds for exponent notation and when the smallest is not first in the list
[[ "$(printf '%s\n' 1e3 1e2 3e3 | sort -g | head -n 1)" == "1e2" ]] || { echo "FAIL probe min"; exit 1; }
echo "PROBE narrows [${LAMBDAS}] -> ${PROBE_LAMBDAS}"

# EVAL_SPECS: one spec per variant x lambda, plus any EXTRA_MODELS absolute pairs.
EVAL_SPECS=""
append_spec () {
  if [[ -z "${EVAL_SPECS}" ]]; then EVAL_SPECS="$1"; else EVAL_SPECS="${EVAL_SPECS}|$1"; fi
}
if [[ -n "${MODELS}" ]]; then
  for m in ${MODELS}; do append_spec "${m};${ADAPTERS}/${m}"; done
else
  for v in ${VARIANTS}; do
    for lam in ${LAMBDAS}; do
      run="$(run_name "${v}" "${lam}")"
      append_spec "${run};${ADAPTERS}/${run}"
    done
  done
fi
EXTRA_MODELS="cs_ewc_lam3e2_seed1;/safety/adapters/cs_ewc_lam3e2_seed1"
if [[ -n "${EXTRA_MODELS}" ]]; then
  IFS='|' read -ra extra <<< "${EXTRA_MODELS}"
  for spec in "${extra[@]}"; do append_spec "${spec}"; done
fi

IFS='|' read -ra specs <<< "${EVAL_SPECS}"
# 2 variants x 4 lambdas + 1 extra = 9
[[ ${#specs[@]} -eq 9 ]] || { echo "FAIL expected 9 eval specs, got ${#specs[@]}"; exit 1; }
for spec in "${specs[@]}"; do
  echo "eval spec  : name=${spec%%;*} adapter=${spec#*;}"
done
[[ "${specs[0]}" == "cs_ewcmix_equal_lam1_seed1;/adapters/cs_ewcmix_equal_lam1_seed1" ]] \
  || { echo "FAIL first spec"; exit 1; }
[[ "${specs[8]}" == "cs_ewc_lam3e2_seed1;/safety/adapters/cs_ewc_lam3e2_seed1" ]] \
  || { echo "FAIL EXTRA_MODELS absolute path lost"; exit 1; }

echo "ALL CHECKS PASSED"
