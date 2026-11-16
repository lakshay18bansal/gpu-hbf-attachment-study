#!/usr/bin/env bash
# Run the study's experiments, at most N simulations at a time.
#   models/run_experiments.sh [--jobs N] [main|layer|validation|all]     (default: --jobs 4 all)
#
#   main        E1/E2/E10: the full decode token on C1-C4 (C4 with its 16 MiB LHB) and C4 with
#               LHBs of 10, 40 and 200 MiB; vectorAdd on C1-C4. A full token takes 5-21 h.
#   layer       decoder layer 13 (one of the 28 identical layers, 10-90 min per run):
#                 E4  tR 3/5/10/15/20 us for C1-C4
#                 E5  C4 LHB size 2/4/8/10/16/32/40/200 MiB x tR 3/5/10/15/20 us
#                 E6  GPU memory slots: C2 with 4096, C3 with 257
#                 E9  flash parallelism: 256/512/1024 sub-arrays for C2, C3, C4
#   validation  the request-level vectorAdd checks (models/validate/run_validation.sh)
#
# Each run needs about 0.3 GB of RAM and one core. A run whose folder already holds rc=0 is
# skipped, so the script can be restarted. Progress: runs/experiments.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ST="$(cd "$HERE/.." && pwd)"; cd "$ST"
JOBS=4; WHAT=all
while (($#)); do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    main|layer|validation|all) WHAT="$1"; shift ;;
    *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
  esac
done
R="${HBF_RUNS:-$ST/runs}"; mkdir -p "$R/logs"; L=qwen3_layer13; LOG="$R/experiments.log"
tag() { local t=""; [[ "$1" != 15 ]] && t+="__tR$1"; [[ "$2" != 16 ]] && t+="__lhb$2"; echo "$t"; }

# Each entry: <run folder>|<run.sh arguments>.  Longest runs first.
MAIN=(
  "C2__qwen3_full|C2 qwen3_full" "C3__qwen3_full|C3 qwen3_full" "C1__qwen3_full|C1 qwen3_full"
  "C4__qwen3_full|C4 qwen3_full"
  "C4__qwen3_full__lhb10|C4 qwen3_full --lhb-mib 10"
  "C4__qwen3_full__lhb40|C4 qwen3_full --lhb-mib 40"
  "C4__qwen3_full__lhb200|C4 qwen3_full --lhb-mib 200"
  "C1__vectoradd|C1 vectoradd" "C2__vectoradd|C2 vectoradd" "C3__vectoradd|C3 vectoradd"
  "C4__vectoradd|C4 vectoradd"
)
LAYER=(
  # E9 and the C2/C3 tR runs are the longest
  "C2__${L}__sa256|C2 $L --subarrays 256" "C2__${L}__sa1024|C2 $L --subarrays 1024"
  "C2__${L}|C2 $L" "C3__${L}|C3 $L" "C1__${L}|C1 $L"
  "C3__${L}__sa256|C3 $L --subarrays 256" "C3__${L}__sa1024|C3 $L --subarrays 1024"
  "C4__${L}__sa256|C4 $L --subarrays 256" "C4__${L}__sa1024|C4 $L --subarrays 1024"
  "C2__${L}__slots4096|C2 $L -- -gpgpu_dram_max_outstanding 4096"
  "C3__${L}__slots257|C3 $L -- -gpgpu_hbf_credits 257"
)
for t in 3 5 10 20; do LAYER+=("C2__${L}__tR$t|C2 $L --tR-us $t" "C3__${L}__tR$t|C3 $L --tR-us $t"); done
for lhb in 2 4 8 10 16 32 40 200; do
  for t in 3 5 10 15 20; do LAYER+=("C4__${L}$(tag $t $lhb)|C4 $L --tR-us $t --lhb-mib $lhb"); done
done

case "$WHAT" in
  main) LIST=("${MAIN[@]}") ;;
  layer) LIST=("${LAYER[@]}") ;;
  validation) LIST=() ;;
  all) LIST=("${MAIN[@]}" "${LAYER[@]}") ;;
esac

run_one() {
  local out="$1"; shift
  if [[ -f "$R/$out/rc" ]] && grep -q 'rc=0' "$R/$out/rc"; then echo "skip (done) $out" >> "$LOG"; return; fi
  echo "$(date '+%F %T') start $out" >> "$LOG"
  # shellcheck disable=SC2086
  "$HERE/run.sh" $@ --out "$R/$out" --force > "$R/logs/$out.out" 2>&1
  echo "$(date '+%F %T') end   $out rc=$?" >> "$LOG"
}
for e in "${LIST[@]}"; do
  out="${e%%|*}"; args="${e#*|}"
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
  # a run already in progress elsewhere (its folder exists without rc) is left alone
  if [[ -d "$R/$out" && ! -f "$R/$out/rc" ]] && pgrep -f -- "$R/$out " > /dev/null; then
    echo "skip (running elsewhere) $out" >> "$LOG"; continue
  fi
  run_one "$out" $args &
  sleep 2
done
wait
if [[ "$WHAT" == validation || "$WHAT" == all ]]; then
  echo "$(date '+%F %T') start validation" >> "$LOG"
  "$HERE/validate/run_validation.sh" --jobs "$JOBS" > "$R/logs/validation.out" 2>&1
  echo "$(date '+%F %T') end   validation rc=$?" >> "$LOG"
fi
echo "$(date '+%F %T') all done ($WHAT)" >> "$LOG"
