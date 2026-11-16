#!/usr/bin/env bash
# Run one workload on one of the four memory organizations (HBF-Sim GPU and HBF device).
#
# usage: models/run.sh <C1|C2|C3|C4> <workload> [--tR-us X] [--lhb-mib N] [--subarrays N]
#                            [--out DIR] [--force] [--timeout SEC] [--dry-run] [-- -opt val ...]
#   C1  GPU + HBM (Ramulator HBM2E)                 C1_hbm_only/
#   C2  GPU + HBF + DRAM, HBF-Sim exactly           C2_hbf_dram/
#   C3  GPU + HBM and HBF both direct               C3_hbf_hbm_direct/
#   C4  GPU + HBM, HBF behind the HBM base die (H3) C4_h3_hbf_behind_hbm/
#   <workload>  vectoradd | vectoradd_nohbf | qwen3_full | qwen3_layer13 | qwen3_prologue
#               | qwen3_tail | qwen3_layer:<0-27> | qwen3_kernel:<kernel id> | <path to kernelslist.g>
#     vectoradd        A and B (4 MiB each) in the HBF window, C in HBM/DRAM
#     vectoradd_nohbf  the original trace: no address in the HBF window
#     qwen3_full       the whole decode token (1,819 kernels), HBF-Sim's paper experiment
#   --tR-us X     NAND page read time in us (default 15, HBF-Sim's). Folder tag __tR<X>.
#   --lhb-mib N   C4 only: size of H3's latency-hiding buffer (LHB) in the HBM base die, in MiB
#                 (default 16 = H3's 40 MB per 1 TB/s stack, scaled to our 387 GB/s). The LHB is
#                 part of H3 and always present in C4; it is filled ahead of use from per-kernel
#                 page lists, half of it at a time (double buffering). Folder tag __lhb<N>.
#   --subarrays N flash sub-arrays that can work in parallel (default 512, HBF-Sim's). Folder tag __sa<N>.
#   --out         run folder (default runs/<C>__<workload>[__tR<X>][__lhb<N>][__sa<N>])
#   options after "--" are appended last (they override everything).
#
# Config assembly (later lines override earlier ones):
#   base/hbfsim_qwen3.config (HBF-Sim's Qwen3 config, verbatim) -> base/common.config
#   -> <C>/config.overrides -> LHB lines (C4 only) -> options after "--"
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
OUT=""; FORCE=0; TIMEOUT=0; DRY=0; EXTRA=(); TR_US=15; LHB_MIB=16; SUBARRAYS=512
while (($#)); do
  case "$1" in
    --lhb-mib) LHB_MIB="$2"; shift 2 ;;
    --subarrays) SUBARRAYS="$2"; shift 2 ;;
    --tR-us) TR_US="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option $1" >&2; usage ;;
  esac
done
# The LHB belongs to H3 (C4) only; C1-C3 follow HBF-Sim, which has none.
[[ "$C" == C4 ]] && PF=on || PF=off
[[ "$LHB_MIB" =~ ^[0-9]+$ && "$LHB_MIB" -ge 1 ]] || { echo "--lhb-mib must be a whole number of MiB" >&2; exit 2; }
[[ "$SUBARRAYS" =~ ^[0-9]+$ && $((SUBARRAYS % 16)) -eq 0 && "$SUBARRAYS" -ge 16 ]] ||
  { echo "--subarrays must be a multiple of 16 (16 flash channels)" >&2; exit 2; }
[[ "$C" != C4 && "$LHB_MIB" != 16 ]] && { echo "--lhb-mib applies to C4 only" >&2; exit 2; }

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
# every trace file the list names must exist: the simulator skips a missing one and still exits 0
NKERN=0
while read -r kf; do
  [[ -f "$(dirname "$KL")/$kf" ]] || { echo "trace file missing: $(dirname "$KL")/$kf" >&2; exit 2; }
  NKERN=$((NKERN + 1))
done < <(grep -E '^kernel-' "$KL")
WLTAG="${WL//[:\/]/_}"
[[ "$WL" == */* ]] && WLTAG="$(basename "$(dirname "$KL")")"

TAG=""; [[ "$TR_US" != 15 ]] && TAG+="__tR${TR_US}"
[[ "$C" == C4 && "$LHB_MIB" != 16 ]] && TAG+="__lhb${LHB_MIB}"
[[ "$SUBARRAYS" != 512 ]] && TAG+="__sa${SUBARRAYS}"
[[ -n "$OUT" ]] || OUT="$RUNS/${C}__${WLTAG}${TAG}"
mkdir -p "$(dirname "$OUT")"; OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
if [[ -e "$OUT" ]]; then
  if ((FORCE)); then rm -rf "$OUT"; else echo "exists (use --force): $OUT" >&2; exit 3; fi
fi
mkdir -p "$OUT"

# ---- C4 LHB: fixed size (independent of tR), filled half at a time (double buffering) ---------
LHB_PAGES=$((LHB_MIB * 256)); PF_WINDOW=$((LHB_PAGES / 2))
TR_NS="$(python3 -c 'import sys; print(round(float(sys.argv[1]) * 1e3, 3))' "$TR_US")"
HINTS=""
if [[ "$PF" == on ]]; then
  HD="$(dirname "$KL")/hints"; mkdir -p "$HD"
  HINTS="$HD/$(basename "$KL" .g).hints"
  if [[ "$WL" == vectoradd_nohbf ]]; then
    # the original trace (binary .tracez) has no address in the HBF window: nothing to list
    [[ -s "$HINTS" ]] || printf '# hbf-prefetch-v1 base=0x4000000000 page=4096\n# %s: no HBF addresses\n' "$KL" > "$HINTS"
  else
    [[ -s "$HINTS" ]] || python3 "$ST/tools/hbf_prefetch_hints.py" "$KL" -o "$HINTS" --jobs 4 >&2
  fi
fi

# ---- gpgpusim.config -----------------------------------------------------------------------------
G="$OUT/gpgpusim.config"
{
  echo "# assembled by models/run.sh for $C ($CFG) / $WL / tR=${TR_US}us / lhb=$([[ $PF == on ]] && echo ${LHB_MIB}MiB || echo none) / subarrays=$SUBARRAYS"
  echo "# ---- base/hbfsim_qwen3.config (HBF-Sim artifact/configs/qwen3.config, verbatim)"
  cat "$HERE/base/hbfsim_qwen3.config"
  echo; echo "# ---- base/common.config"; cat "$HERE/base/common.config"
  echo; echo "# ---- $CFG/config.overrides"; cat "$HERE/$CFG/config.overrides"
  if [[ "$PF" == on ]]; then
    echo; echo "# ---- H3 latency-hiding buffer in the HBM base die: ${LHB_MIB} MiB (run.sh --lhb-mib)"
    echo "-gpgpu_hbf_prefetch_file $HINTS"
    echo "-gpgpu_hbf_lhb_pages $LHB_PAGES"
    echo "-gpgpu_hbf_prefetch_window $PF_WINDOW"
    # H3: the LHB is continuously filled with the data the next kernels need; it runs at most
    # one LHB ahead of the GPU, so its capacity is the limit on how far ahead it reads.
    echo "-gpgpu_hbf_prefetch_lookahead 1000000"
  fi
  if [[ "$SUBARRAYS" != 512 ]]; then
    echo; echo "# ---- flash parallelism (run.sh --subarrays $SUBARRAYS)"
    echo "-gpgpu_hbf_num_subarrays $SUBARRAYS"
    echo "-gpgpu_hbf_max_active $SUBARRAYS"
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
  "kernelslist": "$KL", "lhb": "$PF" == "on", "lhb_mib": int("$LHB_MIB") if "$PF" == "on" else 0,
  "subarrays": int("$SUBARRAYS"), "tR_us": float("$TR_US"), "memory": "hbm2e_slice8",
  "hbf": "$HBF_ON" == "1", "attach": int("${ATTACH:-0}"), "hbf_channels": 16,
  "hbf_link_gbps_per_channel": 192.0, "hbf_link_gbps_total": 3072.0,
  "near_peak_gbps": 387.072, "core_mhz": float("$CORE_MHZ"),
  "launch_latency_cycles": int("${LAUNCH:-0}"), "ramulator_tx_bytes": 32,
  "lhb_pages": $LHB_PAGES if "$PF" == "on" else 0, "lhb_fill_pages": $PF_WINDOW if "$PF" == "on" else 0,
  "hints": "$HINTS", "extra": "${EXTRA[*]}", "args": "${ARGS[*]}"
}, open(sys.argv[1], "w"), indent=1)
EOF
if ((DRY)); then echo "dry run: $OUT"; cat "$OUT/run_meta.json"; exit 0; fi

echo "run: $C $WL tR=${TR_US}us$([[ $PF == on ]] && echo " lhb=${LHB_MIB}MiB") -> $OUT"
set +e
# The simulator output goes to run.log and to a second copy (run.log.copy): one validation run
# lost its run.log (truncated after the run, cause not found); the copy guards long runs.
TIMECMD=(); [[ -x /usr/bin/time ]] && TIMECMD=(/usr/bin/time -v -o "$OUT/time.txt")
SIM_TIMEOUT="$TIMEOUT" "${TIMECMD[@]}" "$SIM/run_sim.sh" "$OUT" "${ARGS[@]}" 2>&1 \
  | tee "$OUT/run.log.copy" > "$OUT/run.log"
rc=${PIPESTATUS[0]}
set -e
# a run that stopped early (e.g. a trace file vanished during the run) is a failure, whatever rc says
NDONE="$(grep -c '^kernel_name = ' "$OUT/run.log.copy" || true)"
if ((rc == 0 && NDONE != NKERN)); then
  echo "ERROR: $NDONE of $NKERN kernels completed in $OUT" >&2; rc=4
fi
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
