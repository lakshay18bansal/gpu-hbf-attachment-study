#!/usr/bin/env bash
# Run the whole Qwen3 plan (README "Running Qwen3-1.7B"), at most N simulations at a time.
# usage: models/run_qwen_all.sh [--jobs N] [main|sweeps|all]      (default: --jobs 10 all)
#   main    A. the full token on C1, C2, C3, C4, C4 without LHB (about a day each)
#   sweeps  B. 25 runs on decoder layer 13: baselines, parity, credits, tR 3/5/20, flash bw = HBM
# Each run needs about 0.3 GB of RAM and one core. A run whose folder already holds rc=0 is
# skipped, so the script can be restarted after an interruption. Progress: runs/qwen_all.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ST="$(cd "$HERE/.." && pwd)"; cd "$ST"
JOBS=10; WHAT=all
while (($#)); do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    main|sweeps|all) WHAT="$1"; shift ;;
    *) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
  esac
done
R="${HBF_RUNS:-$ST/runs}"; mkdir -p "$R"; L=qwen3_layer13
# Each entry: <out folder>|<run.sh arguments>
MAIN=(
  "C1__qwen3_full__pfoff|C1 qwen3_full"
  "C2__qwen3_full__pfoff|C2 qwen3_full"
  "C3__qwen3_full__pfoff|C3 qwen3_full"
  "C4__qwen3_full__pfon|C4 qwen3_full"
  "C4__qwen3_full__pfoff|C4 qwen3_full --prefetch off"
)
SWEEPS=(
  "C1__${L}__pfoff|C1 $L"
  "C2__${L}__pfoff|C2 $L"
  "C3__${L}__pfoff|C3 $L"
  "C4__${L}__pfon|C4 $L"
  "C4__${L}__pfoff|C4 $L --prefetch off"
  "C2__${L}__pfon|C2 $L --prefetch on"
  "C3__${L}__pfon|C3 $L --prefetch on"
  "C2__${L}__pfoff__l2cache|C2 $L -- -gpgpu_hbf_l2_policy 1"
  "C3__${L}__pfoff__l2cache|C3 $L -- -gpgpu_hbf_l2_policy 1"
  "C3__${L}__pfoff__slots257|C3 $L -- -gpgpu_hbf_credits 257"
  "C2__${L}__pfoff__cred4096|C2 $L -- -gpgpu_dram_max_outstanding 4096"
  "C4__${L}__pfon__cred4096|C4 $L -- -gpgpu_dram_max_outstanding 4096"
  "C4__${L}__pfoff__cred4096|C4 $L --prefetch off -- -gpgpu_dram_max_outstanding 4096"
)
for t in 3 5 20; do
  SWEEPS+=("C2__${L}__pfoff__tR$t|C2 $L --tR-us $t" "C3__${L}__pfoff__tR$t|C3 $L --tR-us $t" "C4__${L}__pfon__tR$t|C4 $L --tR-us $t")
done
SWEEPS+=(
  "C2__${L}__pfoff__bwmatch|C2 $L -- -gpgpu_hbf_max_active 1418 -gpgpu_hbf_num_subarrays 1424"
  "C3__${L}__pfoff__bwmatch|C3 $L -- -gpgpu_hbf_max_active 1418 -gpgpu_hbf_num_subarrays 1424"
  "C4__${L}__pfon__bwmatch|C4 $L -- -gpgpu_hbf_max_active 1418 -gpgpu_hbf_num_subarrays 1424"
)
RUNS=()
[[ "$WHAT" == main || "$WHAT" == all ]] && RUNS+=("${MAIN[@]}")
[[ "$WHAT" == sweeps || "$WHAT" == all ]] && RUNS+=("${SWEEPS[@]}")

LOG="$R/qwen_all.log"
echo "$(date '+%F %T') start: ${#RUNS[@]} runs, $JOBS at a time" >> "$LOG"
one() {
  local out="$1" args="$2" rest="" pre="$2"
  if [[ "$args" == *" -- "* ]]; then pre="${args%% -- *}"; rest="${args#* -- }"; fi
  echo "$(date '+%F %T') begin $out" >> "$LOG"
  # shellcheck disable=SC2086
  models/run.sh $pre --out "$R/$out" --force -- $rest > "$R/$out.out" 2>&1
  echo "$(date '+%F %T') end   $out $(cat "$R/$out/rc" 2>/dev/null || echo rc=?)" >> "$LOG"
}
for r in "${RUNS[@]}"; do
  out="${r%%|*}"; args="${r#*|}"
  if [[ "$(cat "$R/$out/rc" 2>/dev/null)" == rc=0 ]]; then echo "$(date '+%F %T') skip  $out (done)" >> "$LOG"; continue; fi
  while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 20; done
  one "$out" "$args" &
  sleep 2
done
wait
echo "$(date '+%F %T') all done" >> "$LOG"
models/compare.py qwen3_full >> "$LOG" 2>&1
models/compare.py qwen3_layer13 >> "$LOG" 2>&1
