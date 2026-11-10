#!/usr/bin/env bash
# Run one workload on one of the four memory organizations (HBF-Sim GPU and HBF device).
#
# usage: models/run.sh <C1|C2|C3|C4> <workload> [--prefetch on|off] [--out DIR] [--force]
#                            [--timeout SEC] [--dry-run] [-- -opt val ...]
#   C1  GPU + HBM (Ramulator HBM2E)                 C1_hbm_only/
#   C2  GPU + HBF + DRAM, HBF-Sim exactly           C2_hbf_dram/
#   C3  GPU + HBM and HBF both direct               C3_hbf_hbm_direct/
#   C4  GPU + HBM, HBF behind the HBM base die (H3) C4_h3_hbf_behind_hbm/
#   <workload>  vectoradd | vectoradd_nohbf | qwen3_full | qwen3_layer13 | qwen3_prologue
#               | qwen3_tail | qwen3_layer:<0-27> | qwen3_kernel:<kernel id> | <path to kernelslist.g>
#     vectoradd        A and B (4 MiB each) in the HBF window, C in HBM/DRAM
#     vectoradd_nohbf  the original trace: no address in the HBF window
#     qwen3_full       the whole decode token (1,819 kernels), HBF-Sim's paper experiment
#   --prefetch  H3's LHB + prefetcher. Default: on for C4, off for C1-C3 (HBF-Sim has none).
#   --tR-us X   NAND page read time in us (default 15, HBF-Sim's); the LHB and prefetch window are
#               resized with it (H3 eq. 1: LHB = 2 x BW x tR). The run folder gets a __tR<X> tag.
#   --out       run folder (default runs/<C>__<workload>__pf<on|off>[__tR<X>])
#   options after "--" are appended last (they override everything).
#
# Config assembly (later lines override earlier ones):
#   base/hbfsim_qwen3.config (HBF-Sim's Qwen3 config, verbatim) -> base/common.config
#   -> <C>/config.overrides -> prefetch lines (if on) -> options after "--"
# The run folder keeps gpgpusim.config, memory.yaml, run.log, run_meta.json, summary.txt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST="$(cd "$HERE/.." && pwd)"
SIM="$ST/simulator"
RUNS="${HBF_RUNS:-$ST/runs}"   # where new runs go (results/ holds the shipped ones)
usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
(($# >= 2)) || usage
case "$1" in
  C1) CFG=C1_hbm_only ;;
  C2) CFG=C2_hbf_dram ;;
  C3) CFG=C3_hbf_hbm_direct ;;
  C4) CFG=C4_h3_hbf_behind_hbm ;;
  *) echo "unknown configuration $1 (C1|C2|C3|C4)" >&2; usage ;;
esac
C="$1"; WL="$2"; shift 2
PF=""; OUT=""; FORCE=0; TIMEOUT=0; DRY=0; EXTRA=(); TR_US=15
while (($#)); do
  case "$1" in
    --prefetch) PF="$2"; shift 2 ;;
    --tR-us) TR_US="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option $1" >&2; usage ;;
  esac
done
[[ -n "$PF" ]] || { [[ "$C" == C4 ]] && PF=on || PF=off; }
[[ "$PF" == on || "$PF" == off ]] || { echo "--prefetch must be on or off" >&2; exit 2; }
[[ "$PF" == on && "$C" == C1 ]] && { echo "C1 has no HBF tier: --prefetch on is meaningless" >&2; exit 2; }

# ---- workload -> kernel list --------------------------------------------------------------------
QT="$ST/workloads/qwen3_1p7b/trace"
case "$WL" in
  vectoradd) KL="$ST/workloads/vectoradd/hbf/kernelslist.g" ;;
  vectoradd_nohbf) KL="$ST/workloads/vectoradd/original/kernelslist.g" ;;
  qwen3_full|qwen3_layer13|qwen3_prologue|qwen3_tail) KL="$QT/list_${WL#qwen3_}.g" ;;
  qwen3_layer:*)   # decoder layer N (0-27): kernels 18+64N .. 18+64N+63 of the full list
    N="${WL#qwen3_layer:}"; KL="$QT/list_layer$N.g"
    [[ -f "$KL" ]] || sed -n "$((19 + 64 * N)),$((82 + 64 * N))p" "$QT/list_full.g" > "$KL" ;;
  qwen3_kernel:*)
    KID="${WL#qwen3_kernel:}"; KL="$QT/list_k$KID.g"
    if [[ ! -f "$KL" ]]; then
      f="$(grep -m1 "^kernel-$KID-" "$QT/kernelslist.g")" || { echo "no kernel $KID" >&2; exit 2; }
      echo "$f" > "$KL"
    fi ;;
  *) KL="$(cd "$(dirname "$WL")" && pwd)/$(basename "$WL")" ;;
esac
[[ -f "$KL" ]] || { echo "kernel list not found: $KL" >&2; exit 2; }
WLTAG="${WL//[:\/]/_}"
[[ "$WL" == */* ]] && WLTAG="$(basename "$(dirname "$KL")")"

TRTAG=""; [[ "$TR_US" != 15 ]] && TRTAG="__tR${TR_US}"
[[ -n "$OUT" ]] || OUT="$RUNS/${C}__${WLTAG}__pf${PF}${TRTAG}"
mkdir -p "$(dirname "$OUT")"; OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
if [[ -e "$OUT" ]]; then
  if ((FORCE)); then rm -rf "$OUT"; else echo "exists (use --force): $OUT" >&2; exit 3; fi
fi
mkdir -p "$OUT"

# ---- prefetcher / LHB (H3 eq. 1: capacity = 2 x BW x tR, BW = what the HBF can deliver to
# the GPU = the HBM link, 8 x 48.384 = 387.07 GB/s; tR = 15 us) ---------------------------------
# At tR = 15 us: LHB 2835 pages = 11.6 MB, window 1418 prefetches in flight for full bandwidth.
read -r LHB_PAGES PF_WINDOW TR_NS < <(python3 -c "
import math, sys; tr = float(sys.argv[1]) * 1e-6; bw = 387.072e9
print(math.ceil(2 * bw * tr / 4096), math.ceil(bw * tr / 4096), round(tr * 1e9, 3))" "$TR_US")
HINTS=""
if [[ "$PF" == on ]]; then
  HD="$(dirname "$KL")/hints"; mkdir -p "$HD"
  HINTS="$HD/$(basename "$KL" .g).hints"
  [[ -s "$HINTS" ]] || python3 "$ST/tools/hbf_prefetch_hints.py" "$KL" -o "$HINTS" --jobs 4 >&2
fi

# ---- gpgpusim.config -----------------------------------------------------------------------------
G="$OUT/gpgpusim.config"
{
  echo "# assembled by models/run.sh for $C ($CFG) / $WL / prefetch=$PF"
  echo "# ---- base/hbfsim_qwen3.config (HBF-Sim artifact/configs/qwen3.config, verbatim)"
  cat "$HERE/base/hbfsim_qwen3.config"
  echo; echo "# ---- base/common.config"; cat "$HERE/base/common.config"
  echo; echo "# ---- $CFG/config.overrides"; cat "$HERE/$CFG/config.overrides"
  if [[ "$PF" == on ]]; then
    echo; echo "# ---- prefetcher + LHB (run.sh --prefetch on)"
    echo "-gpgpu_hbf_prefetch_file $HINTS"
    echo "-gpgpu_hbf_lhb_pages $LHB_PAGES"
    echo "-gpgpu_hbf_prefetch_window $PF_WINDOW"
    # Continuous (H3: "continuously bringing the data required ... to the LHB"): every kernel's
    # hints are queued; the prefetcher runs at most lhb_pages ahead of demand, so the LHB
    # capacity is the only limit on how far ahead it reads.
    echo "-gpgpu_hbf_prefetch_lookahead 1000000"
  fi
  if [[ "$TR_US" != 15 ]]; then
    echo; echo "# ---- NAND read time (run.sh --tR-us $TR_US)"
    echo "-gpgpu_hbf_tR_ns $TR_NS"
  fi
} > "$G"

# ---- Ramulator memory.yaml (exported from memory/hbm2e_slice8.py at every run) -------------------
RAM_ROOT="${RAM_ROOT:-$ST/simulator/ramulator2}"
cp "$HERE/memory/hbm2e_slice8.py" "$OUT/memory_USED.py"
( export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$RAM_ROOT/python:$HERE/memory${PYTHONPATH:+:$PYTHONPATH}"
  python3 -m ramulator export "$HERE/memory/hbm2e_slice8.py" -o "$OUT/memory.yaml" > "$OUT/ramulator_export.log" 2>&1 &&
  python3 -c 'import runpy, sys; print(runpy.run_path(sys.argv[1])["SUMMARY"])' "$HERE/memory/hbm2e_slice8.py" \
    > "$OUT/memory_summary.txt" ) || { echo "ERROR: Ramulator export failed:" >&2; cat "$OUT/ramulator_export.log" >&2; exit 2; }

ARGS=(-config ./gpgpusim.config -gpgpu_ramulator_enabled 1 -gpgpu_ramulator_config ./memory.yaml
      "${EXTRA[@]}" -trace "$KL")

LAUNCH="$(grep -E '^-gpgpu_kernel_launch_latency' "$G" | tail -1 | awk '{print $2}')"
CORE_MHZ="$(grep -E '^-gpgpu_clock_domains' "$G" | tail -1 | awk '{print $2}' | cut -d: -f1)"
ATTACH="$( (grep -E '^-gpgpu_hbf_attach' "$G" || true) | tail -1 | awk '{print $2}')"
HBF_ON="$( (grep -E '^-gpgpu_hbf_enabled' "$G" || true) | tail -1 | awk '{print $2}')"
python3 - "$OUT/run_meta.json" <<EOF
import json, sys
json.dump({
  "study": "hbfsim_4arch", "config": "$C", "config_dir": "$HERE/$CFG", "workload": "$WL",
  "kernelslist": "$KL", "prefetch": "$PF", "tR_us": float("$TR_US"), "memory": "hbm2e_slice8",
  "hbf": "$HBF_ON" == "1", "attach": int("${ATTACH:-0}"), "hbf_channels": 16,
  "hbf_link_gbps_per_channel": 192.0, "hbf_link_gbps_total": 3072.0,
  "near_peak_gbps": 387.072, "core_mhz": float("$CORE_MHZ"),
  "launch_latency_cycles": int("${LAUNCH:-0}"), "ramulator_tx_bytes": 32,
  "lhb_pages": $LHB_PAGES if "$PF" == "on" else 0, "prefetch_window": $PF_WINDOW if "$PF" == "on" else 0,
  "hints": "$HINTS", "extra": "${EXTRA[*]}", "args": "${ARGS[*]}"
}, open(sys.argv[1], "w"), indent=1)
EOF
if ((DRY)); then echo "dry run: $OUT"; cat "$OUT/run_meta.json"; exit 0; fi

echo "run: $C $WL pf=$PF -> $OUT"
set +e
# The simulator output goes to run.log and to a second copy (run.log.copy): one validation run
# lost its run.log (truncated after the run, cause not found); the copy guards long runs.
TIMECMD=(); [[ -x /usr/bin/time ]] && TIMECMD=(/usr/bin/time -v -o "$OUT/time.txt")
SIM_TIMEOUT="$TIMEOUT" "${TIMECMD[@]}" "$SIM/run_sim.sh" "$OUT" "${ARGS[@]}" 2>&1 \
  | tee "$OUT/run.log.copy" > "$OUT/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "rc=$rc" > "$OUT/rc"
if ! grep -q '^gpu_tot_sim_cycle' "$OUT/run.log"; then
  if grep -q '^gpu_tot_sim_cycle' "$OUT/run.log.copy"; then
    echo "WARNING: $OUT/run.log lost its content; restored from run.log.copy" >&2
    cp "$OUT/run.log.copy" "$OUT/run.log"
  else
    echo "WARNING: no statistics in $OUT/run.log (rc=$rc)" >&2
  fi
fi
cmp -s "$OUT/run.log" "$OUT/run.log.copy" && rm -f "$OUT/run.log.copy"
python3 "$HERE/validate/summarize.py" "$OUT" > "$OUT/summary.txt" 2>&1 || true
cat "$OUT/summary.txt"
exit $rc
