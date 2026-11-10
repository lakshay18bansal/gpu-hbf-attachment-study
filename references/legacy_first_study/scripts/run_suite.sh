#!/usr/bin/env bash
# Run a named set of simulations (the study matrix) with a parallel-job limit.
#
# usage: scripts/run_suite.sh <suite> [--jobs N] [--dry-run] [--force] [--results DIR] [--timeout SEC]
# suites:
#   smoke   vectorAdd on every config, tR 15 us, prefetch off and on
#   qproj   Qwen3 layer-13 q_proj GEMV (kernel 4667) on every config, tR 15 us, prefetch off and on
#   layer   Qwen3 layer 13 (64 kernels): C1 once; C2, C3 x2, C4 with prefetch at tR {3,5,15,20};
#           without prefetch at tR 15 (all HBF configs are latency-bound and equal there, see the
#           smoke suite) plus a tR {3,5,20} sweep on C3-iso-stack
#   edges   Qwen3 prologue (18 kernels): C1 + every HBF config at tR 15, prefetch off/on;
#           tail (9 kernels incl. the 622 MB lm_head GEMV): C1 + every HBF config at tR 15, prefetch on
#           (prefetch-off tails take ~10 h each; tools/report.py scales them from the layer, 'est.')
#   hop     C4 base-die/D2D hop sensitivity on layer 13: +50/+200 ns (default +10), tR 15, prefetch on
#   paper   C2-legacy (HBF-Sim as published, 8 SMs) on layer 13 at tR 15, prefetch off
#   study   layer + edges + hop + paper (everything the report needs)
#   fullmodel  all 28 decoder layers simulated one by one (same as the HBF-Sim paper's full decode
#           step, split per layer so the runs are parallel), tR 15: C1; C2, C3 x2, C4 prefetch on;
#           FULL_PF_OFF=1 adds prefetch off for the HBF configs. Prologue and tail come from 'edges'.
# Finished runs (metrics.json with ok=true) are skipped unless --force. Default --jobs 4
# (each run needs ~0.2-0.7 GB RAM; this laptop has 7 GB). Results: results/<suite>/<run>/.
set -euo pipefail
ST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:-}"; shift || true
JOBS=4; DRY=0; FORCE=0; RES=""; TIMEOUT=0
while (($#)); do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --force) FORCE=1; shift ;;
    --results) RES="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

HBF_CFGS=(C2_hbf_dram C3_hbf_hbm_direct/iso_stack C3_hbf_hbm_direct/iso_shoreline C4_hbf_behind_hbm)
RUNS=()   # entries: "<config dir>|<workload>|<tR>|<prefetch>|<run name>|<extra opts>"
add() { RUNS+=("$1|$2|$3|$4|$5|${6:-}"); }
tag() { (NAME=; source "$ST/$1/config.env"; echo "$NAME"); }  # run names use config.env NAME
matrix() {  # workload, tR list, prefetch list
  local w="$1" trs="$2" pfs="$3" c t p
  add C1_hbm_only "$w" 15 off "C1_hbm_only__${w//:/_}"
  for c in "${HBF_CFGS[@]}"; do for t in $trs; do for p in $pfs; do
    add "$c" "$w" "$t" "$p" "$(tag "$c")__${w//:/_}__tR${t}__pf$p"
  done; done; done
}
suite() {
  local c t
  case "$1" in
    smoke) matrix vectoradd 15 "off on" ;;
    qproj) matrix qwen3_kernel:4667 15 "off on" ;;
    layer)
      matrix qwen3_layer13 "3 5 15 20" on
      for c in "${HBF_CFGS[@]}"; do add "$c" qwen3_layer13 15 off "$(tag "$c")__qwen3_layer13__tR15__pfoff"; done
      for t in 3 5 20; do
        add C3_hbf_hbm_direct/iso_stack qwen3_layer13 $t off "C3_iso_stack__qwen3_layer13__tR${t}__pfoff"
      done ;;
    edges) matrix qwen3_prologue 15 "off on"; matrix qwen3_tail 15 on ;;
    hop) for h in 50 200; do
           add C4_hbf_behind_hbm qwen3_layer13 15 on "C4_behind_hbm__qwen3_layer13__tR15__pfon__hop$h" \
             "-gpgpu_hbf_frontend_latency_ns $(python3 -c "print(52.49+$h)")"
         done ;;
    paper) add C2_hbf_dram/legacy_paper qwen3_layer13 15 off "C2_legacy_paper__qwen3_layer13__tR15__pfoff" ;;
    study) suite edges; suite layer; suite paper; suite hop ;;  # longest (lm_head tails) first
    fullmodel)
      for n in $(seq 0 27); do
        add C1_hbm_only "qwen3_layer:$n" 15 off "C1_hbm_only__qwen3_layer_$n"
        for c in "${HBF_CFGS[@]}"; do
          add "$c" "qwen3_layer:$n" 15 on "$(tag "$c")__qwen3_layer_${n}__tR15__pfon"
          if [[ "${FULL_PF_OFF:-0}" == 1 ]]; then
            add "$c" "qwen3_layer:$n" 15 off "$(tag "$c")__qwen3_layer_${n}__tR15__pfoff"
          fi
        done
      done ;;
    *) echo "unknown suite '$1' (smoke qproj layer edges hop paper study fullmodel)" >&2; exit 2 ;;
  esac
}
suite "$SUITE"
[[ -n "$RES" ]] || RES="$ST/results/$SUITE"
mkdir -p "$RES"

TODO=()
for r in "${RUNS[@]}"; do
  IFS='|' read -r cfg wl tr pf name extra <<< "$r"
  out="$RES/$name"
  if ((!FORCE)) && [[ -f "$out/metrics.json" ]] &&
     python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/metrics.json')).get('ok') else 1)"; then
    echo "done  $name"; continue
  fi
  TODO+=("$r")
  echo "todo  $name"
done
echo "${#TODO[@]} of ${#RUNS[@]} runs to do in $RES (jobs=$JOBS)"
((DRY)) && exit 0

run_one() {
  IFS='|' read -r cfg wl tr pf name extra <<< "$1"
  local x=()
  [[ -n "$extra" ]] && read -r -a x <<< "$extra"
  "$ST/scripts/run_config.sh" "$ST/$cfg" "$wl" --tR-us "$tr" --prefetch "$pf" --out "$RES/$name" \
    --force --timeout "$TIMEOUT" -- "${x[@]}" > "$RES/$name.out" 2>&1 \
    && echo "ok    $name ($(date +%H:%M))" || echo "FAIL  $name (see $RES/$name.out)"
}
for r in "${TODO[@]}"; do
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
  run_one "$r" &
done
wait
echo "suite $SUITE finished: $RES"
