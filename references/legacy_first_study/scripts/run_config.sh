#!/usr/bin/env bash
# Run one workload on one memory configuration.
#
# usage: scripts/run_config.sh <config dir> <workload> [--tR-us X] [--prefetch off|on]
#                              [--out DIR] [--force] [--timeout SEC] [--dry-run] [-- -opt val ...]
#   <config dir>  a folder with config.env (+ config.overrides), e.g. C3_hbf_hbm_direct/iso_stack
#   <workload>    vectoradd | qwen3_layer13 | qwen3_layer:<0-27> | qwen3_prologue | qwen3_tail | qwen3_full
#                 | qwen3_kernel:<kernel id> | <path to a kernelslist.g>
#   --tR-us       NAND page read time (default 15). Media parallelism, credits and the LHB follow it.
#   --prefetch    HBF prefetcher + latency-hiding buffer (default off)
#   --ram-set     extra HBM/DRAM (Ramulator) overrides for this run, e.g. "DATA_RATE_MBPS=12800,ns.nCL=25"
#   --set VAR=VAL override one config.env variable for this run (repeatable), e.g. --set HBF_LINK_GBPS=25.6
#   --out         run folder (default results/adhoc/<config>__<workload>__tR<X>__pf<on|off>)
#   options after "--" are appended last (they override everything).
#
# Config assembly (later options override earlier ones):
#   common/gpu/qwen3_slice.config (GPU only)  ->  common/memory/builtin_qwen3.config (if MEM=builtin)
#   ->  common/hbf_base.config (if HBF=1)  ->  tR-derived HBF lines
#   ->  <config dir>/config.overrides  ->  extra options
# The run folder keeps the exact gpgpusim.config, memory.yaml, run.log, run_meta.json, metrics.json.
set -euo pipefail
ST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM="$ST/sim"
usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
(($# >= 2)) || usage
CFG="$(cd "$1" && pwd)"; WL="$2"; shift 2
TR_US=15; PF=off; OUT=""; FORCE=0; TIMEOUT=0; DRY=0; EXTRA=(); CLI_RAM_SET=""; SETS=()
while (($#)); do
  case "$1" in
    --tR-us) TR_US="$2"; shift 2 ;;
    --prefetch) PF="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --ram-set) CLI_RAM_SET="$2"; shift 2 ;;
    --set) SETS+=("$2"); shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option $1" >&2; usage ;;
  esac
done
[[ "$PF" == on || "$PF" == off ]] || { echo "--prefetch must be on or off" >&2; exit 2; }

# ---- configuration -----------------------------------------------------------------------------
[[ -f "$CFG/config.env" ]] || { echo "no $CFG/config.env" >&2; exit 2; }
NAME= MEM= RAM_SET= NEAR_PEAK_GBPS=0 HBF=0 ATTACH=0 HBF_CHANNELS=0 HBF_LINK_GBPS=0 PHY_GBPS=0 FRONTEND_NS=0 DERIVE=1
# LHB (prefetch buffer) = one fixed SRAM size in every config: 3000 pages (12 MB) = H3's 2 x BW x tR
# at the reference point (409.6 GB/s, 15 us). It must hold a GEMV's concurrently demanded pages, which
# does not shrink with tR; only the in-flight window (BW x tR) does.
LHB_PAGES=3000
# shellcheck disable=SC1091
source "$CFG/config.env"
[[ -n "$NAME" && -n "$MEM" ]] || { echo "config.env must set NAME and MEM" >&2; exit 2; }
for kv in "${SETS[@]}"; do
  k="${kv%%=*}"; v="${kv#*=}"
  case "$k" in
    MEM|RAM_SET|HBF|ATTACH|HBF_CHANNELS|HBF_LINK_GBPS|PHY_GBPS|FRONTEND_NS|LHB_PAGES|DERIVE) printf -v "$k" '%s' "$v" ;;
    *) echo "--set: unknown config.env variable '$k'" >&2; exit 2 ;;
  esac
done

# ---- workload -> kernel list --------------------------------------------------------------------
QT="$ST/workloads/qwen3_1p7b/trace"
case "$WL" in
  vectoradd) KL="$ST/workloads/vectoradd/hbf/kernelslist.g" ;;
  qwen3_layer13|qwen3_prologue|qwen3_tail|qwen3_full) KL="$QT/list_${WL#qwen3_}.g" ;;
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

[[ -n "$OUT" ]] || OUT="$ST/results/adhoc/${NAME}__${WLTAG}__tR${TR_US}__pf${PF}"
mkdir -p "$(dirname "$OUT")"; OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
if [[ -e "$OUT" ]]; then
  if ((FORCE)); then rm -rf "$OUT"; else echo "exists (use --force): $OUT" >&2; exit 3; fi
fi

# ---- derived HBF parameters (02_design.md sections 2 and 4) --------------------------------------
read -r BW MAXACT SPC NSUB CREDITS LHB WINDOW < <(python3 - "$HBF_CHANNELS" "$HBF_LINK_GBPS" "$TR_US" "$LHB_PAGES" <<'EOF'
import math, sys
ch, link, tr_us, lhb = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
bw = ch * link                                    # GB/s of the slice's HBF link
if ch == 0:
    print(0, 0, 0, 0, 0, 0, 0); sys.exit()
maxact = math.ceil(bw * 1e9 * tr_us * 1e-6 / 4096)  # Little's law: pages in flight for full BW
spc = max(32, math.ceil(maxact / ch))               # sub-arrays per channel
credits = max(1024, math.ceil(bw / 8 * 1e9 * (tr_us + 5) * 1e-6 / 32))  # 32 B requests per partition
window = min(maxact, lhb)                           # prefetch reads in flight: BW x tR
print(f"{bw:g}", maxact, spc, ch * spc, credits, lhb, window)
EOF
)

mkdir -p "$OUT"
C="$OUT/gpgpusim.config"
{
  echo "# assembled by scripts/run_config.sh for $NAME / $WL / tR=$TR_US us / prefetch=$PF"
  cat "$ST/common/gpu/qwen3_slice.config"
  if [[ "$MEM" == builtin ]]; then
    echo; echo "# ---- common/memory/builtin_qwen3.config (GPGPU-Sim built-in DRAM)"; cat "$ST/common/memory/builtin_qwen3.config"
  fi
  if ((HBF)); then
    echo; echo "# ---- common/hbf_base.config"; cat "$ST/common/hbf_base.config"
    echo; echo "# ---- derived from config.env and tR"
    echo "-gpgpu_hbf_tR_ns $(python3 -c "print($TR_US*1000)")"
    echo "-gpgpu_hbf_attach $ATTACH"
    echo "-gpgpu_hbf_frontend_latency_ns $FRONTEND_NS"
    echo "-gpgpu_hbf_phy_gbps $PHY_GBPS"
    if ((DERIVE)); then
      echo "-gpgpu_hbf_num_channels $HBF_CHANNELS"
      echo "-gpgpu_hbf_channel_bw_gbps $HBF_LINK_GBPS"
      echo "-gpgpu_hbf_max_active $MAXACT"
      echo "-gpgpu_hbf_num_subarrays $NSUB"
      echo "-gpgpu_hbf_credits $CREDITS"
    fi
  fi
  [[ -f "$CFG/config.overrides" ]] && { echo; echo "# ---- $CFG/config.overrides"; cat "$CFG/config.overrides"; }
} > "$C"

ARGS=(-config ./gpgpusim.config)
if [[ "$MEM" != builtin ]]; then
  # Export the Ramulator YAML from the Python config at every run, so edits to
  # common/memory/<MEM>.py and RAM_SET (config.env) / --ram-set always take effect.
  RS="$RAM_SET${RAM_SET:+${CLI_RAM_SET:+,}}$CLI_RAM_SET"
  cp "$ST/common/memory/$MEM.py" "$OUT/memory_USED.py"
  RAM_ROOT="${RAM_ROOT:-<btp_work>/accelsim-work/ramulator}"
  ( export PYTHONDONTWRITEBYTECODE=1 RAMULATOR_SET="$RS"
    export PYTHONPATH="$RAM_ROOT/python:$ST/common/memory${PYTHONPATH:+:$PYTHONPATH}"
    python3 -m ramulator export "$ST/common/memory/$MEM.py" -o "$OUT/memory.yaml" > "$OUT/ramulator_export.log" 2>&1 &&
    python3 -c 'import runpy, sys; print(runpy.run_path(sys.argv[1])["SUMMARY"])' "$ST/common/memory/$MEM.py" \
      > "$OUT/memory_summary.txt" ) || { echo "ERROR: Ramulator export failed:" >&2; cat "$OUT/ramulator_export.log" >&2; exit 2; }
  # Peak bandwidth for the report comes from the exported config, not from config.env.
  PK="$(grep -oE '= [0-9.]+ GB/s peak' "$OUT/memory_summary.txt" | grep -oE '[0-9.]+' | head -1)"
  [[ -n "$PK" ]] && NEAR_PEAK_GBPS="$PK"
  ARGS+=(-gpgpu_ramulator_enabled 1 -gpgpu_ramulator_config ./memory.yaml)
fi
HINTS=""
if [[ "$PF" == on ]]; then
  ((HBF)) || { echo "--prefetch on needs an HBF configuration" >&2; exit 2; }
  HD="$(dirname "$KL")/hints"; mkdir -p "$HD"
  HINTS="$HD/$(basename "$KL" .g).hints"
  [[ -s "$HINTS" ]] || python3 "$ST/tools/hbf_prefetch_hints.py" "$KL" -o "$HINTS" --jobs 4 >&2
  ARGS+=(-gpgpu_hbf_prefetch_file "$HINTS" -gpgpu_hbf_lhb_pages "$LHB"
         -gpgpu_hbf_prefetch_window "$WINDOW" -gpgpu_hbf_prefetch_lookahead 1)
fi
ARGS+=("${EXTRA[@]}" -trace "$KL")

LAUNCH="$(grep -E '^-gpgpu_kernel_launch_latency' "$C" | tail -1 | awk '{print $2}')"
CORE_MHZ="$(grep -E '^-gpgpu_clock_domains' "$C" | tail -1 | awk '{print $2}' | cut -d: -f1)"
python3 - "$OUT/run_meta.json" <<EOF
import json, sys
json.dump({
  "config": "$NAME", "config_dir": "$CFG", "workload": "$WL", "kernelslist": "$KL",
  "tR_us": float("$TR_US"), "prefetch": "$PF", "memory": "$MEM", "hbf": bool($HBF),
  "attach": int("$ATTACH"), "hbf_channels": int("$HBF_CHANNELS"),
  "hbf_link_gbps_per_channel": float("$HBF_LINK_GBPS"), "hbf_link_gbps_total": float("$BW"),
  "phy_gbps_per_partition": float("$PHY_GBPS"), "frontend_ns": float("$FRONTEND_NS"),
  "near_peak_gbps": float("$NEAR_PEAK_GBPS"), "ram_set": "${RS:-}", "sets": "${SETS[*]}", "core_mhz": float("$CORE_MHZ"),
  "launch_latency_cycles": int("${LAUNCH:-0}"), "ramulator_tx_bytes": 32,
  "derived": {"max_active": int("$MAXACT"), "subarrays_per_channel": int("$SPC"),
              "num_subarrays": int("$NSUB"), "credits_per_partition": int("$CREDITS"),
              "lhb_pages": int("$LHB") if "$PF" == "on" else 0,
              "prefetch_window": int("$WINDOW") if "$PF" == "on" else 0},
  "hints": "$HINTS", "extra": "${EXTRA[*]}", "args": "${ARGS[*]}"
}, open(sys.argv[1], "w"), indent=1)
EOF
if ((DRY)); then echo "dry run: $OUT"; cat "$OUT/run_meta.json"; exit 0; fi

echo "run: $NAME $WL tR=$TR_US pf=$PF -> $OUT"
set +e
SIM_TIMEOUT="$TIMEOUT" /usr/bin/time -v -o "$OUT/time.txt" "$SIM/run_sim.sh" "$OUT" "${ARGS[@]}" > "$OUT/run.log" 2>&1
rc=$?
set -e
echo "rc=$rc" > "$OUT/rc"
python3 "$ST/tools/parse_metrics.py" "$OUT" --print || true
exit $rc
