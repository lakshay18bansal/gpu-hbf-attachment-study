#!/usr/bin/env bash
# Make every vectorAdd run that validate/check.py checks, then run the checks.
# usage: models/validate/run_validation.sh [--jobs N] [--only ID,ID]  (default 5 runs at a time)
# Output: runs/validation/<run>/ and runs/validation/VALIDATION.md. About 3 h with 5 jobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; ST="$(cd "$HERE/.." && pwd)"
V="${HBF_RUNS:-$ST/runs}/validation"; mkdir -p "$V"
JOBS=5; ONLY=""
while (($#)); do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --only) ONLY=",$2,"; shift 2 ;;   # rerun only these run ids (comma list), then check
    *) echo "usage: run_validation.sh [--jobs N] [--only ID,ID,...]" >&2; exit 2 ;;
  esac
done
acc() { echo "-gpgpu_mem_access_trace_file $V/$1/access.txt -gpgpu_mem_access_trace_detail 1"; }
csv() { echo "-gpgpu_hbf_trace_file $V/$1/hbf.csv -gpgpu_hbf_trace_level 1"; }
RW="$ST/workloads/vectoradd/hbf_rw/kernelslist.g"
if [[ ! -f "$RW" ]]; then  # A, B and C (vectorAdd's output) moved into the HBF window
  python3 "$ST/tools/remap_ranges.py" "$ST/workloads/vectoradd/work/kernelslist.g" -o "$(dirname "$RW")" \
    --range 0x7d8162a00000:4194304 --range 0x7d8162e00000:4194304 --range 0x7d8163200000:4194304
fi
# Same, but only the first 512 KiB of C in HBF: C2/C3 make every store a non-posted flash write
# (L2 bypass), so the full C takes ~12 h there; 512 KiB = 16,384 store sectors, 128 pages.
RWS="$ST/workloads/vectoradd/hbf_rw512k/kernelslist.g"
if [[ ! -f "$RWS" ]]; then
  python3 "$ST/tools/remap_ranges.py" "$ST/workloads/vectoradd/work/kernelslist.g" -o "$(dirname "$RWS")" \
    --range 0x7d8162a00000:4194304 --range 0x7d8162e00000:4194304 --range 0x7d8163200000:524288
fi
RUNS=(
  "C1|C1 vectoradd|$(acc C1)"
  "C2|C2 vectoradd|$(acc C2) $(csv C2)"
  "C3|C3 vectoradd|$(acc C3) $(csv C3)"
  "C4|C4 vectoradd|$(acc C4) $(csv C4)"
  # C2 without the GPU front end on HBF requests: HBF-Sim's own assumption, compared with HBF-Sim
  "E_C2_fe0|C2 vectoradd|-gpgpu_hbf_frontend_latency_ns 0"
  "T_C1|C1 vectoradd_nohbf|"
  "T_C2|C2 vectoradd_nohbf|"
  "T_C3|C3 vectoradd_nohbf|"
  "T_C4|C4 vectoradd_nohbf --prefetch off|"
  "T_C4_link0|C4 vectoradd_nohbf --prefetch off|-gpgpu_hbf_phy_gbps 0"
  "A_C4_pfoff|C4 vectoradd --prefetch off|$(csv A_C4_pfoff)"
  "A_C4_pfoff_hop1000|C4 vectoradd --prefetch off|$(csv A_C4_pfoff_hop1000) -gpgpu_hbf_d2d_hop_ns 1000"
  "A_C4_hop1000|C4 vectoradd|$(csv A_C4_hop1000) -gpgpu_hbf_d2d_hop_ns 1000"
  "A_C4_link5|C4 vectoradd|$(acc A_C4_link5) -gpgpu_hbf_phy_gbps 5"
  "A_C4_q1|C4 vectoradd|-gpgpu_hbf_h3_queue_size 1"
  # deeper GPU memory controller: 4096 requests in flight per partition instead of 257
  "A_C2_cred4096|C2 vectoradd|$(acc A_C2_cred4096) $(csv A_C2_cred4096) -gpgpu_dram_max_outstanding 4096"
  "A_C4_pfoff_cred4096|C4 vectoradd --prefetch off|$(acc A_C4_pfoff_cred4096) $(csv A_C4_pfoff_cred4096) -gpgpu_dram_max_outstanding 4096"
  # deadlock check still active: GPGPU-Sim's own 50,000-cycle window must (falsely) fire on HBF writes
  "D_W_C4_win50k|C4 $RW --timeout 3600|-gpgpu_deadlock_window 50000"
  # parity variants (same mechanism in C2/C3 as in C4) and the tR option
  "P_C2_pf|C2 vectoradd --prefetch on|$(acc P_C2_pf) $(csv P_C2_pf)"
  "P_C3_pf|C3 vectoradd --prefetch on|$(acc P_C3_pf) $(csv P_C3_pf)"
  "P_C2_l2|C2 vectoradd|$(acc P_C2_l2) $(csv P_C2_l2) -gpgpu_hbf_l2_policy 1"
  "P_C3_l2|C3 vectoradd|$(acc P_C3_l2) $(csv P_C3_l2) -gpgpu_hbf_l2_policy 1"
  "P_C3_cred257|C3 vectoradd|$(acc P_C3_cred257) $(csv P_C3_cred257) -gpgpu_hbf_credits 257"
  "S_C4_tR3_pfoff|C4 vectoradd --prefetch off --tR-us 3|$(csv S_C4_tR3_pfoff)"
  "S_C4_tR3|C4 vectoradd --tR-us 3|$(csv S_C4_tR3)"
  "S_C3_bwmatch|C3 vectoradd|$(csv S_C3_bwmatch) -gpgpu_hbf_max_active 1418 -gpgpu_hbf_num_subarrays 1424"
  # HBF write path: A, B and C all in the HBF window (workloads/vectoradd/hbf_rw)
  "W_C2|C2 $RWS --timeout 14400|$(acc W_C2) $(csv W_C2)"
  "W_C3|C3 $RWS --timeout 14400|$(acc W_C3) $(csv W_C3)"
  "W_C4|C4 $RW --timeout 3600|$(acc W_C4) $(csv W_C4)"
  "W_C4_pfoff|C4 $RW --prefetch off --timeout 3600|$(acc W_C4_pfoff) $(csv W_C4_pfoff)"
)
one() {
  IFS='|' read -r id args extra <<< "$1"
  # shellcheck disable=SC2086
  "$HERE/run.sh" $args --out "$V/$id" --force -- $extra > "$V/$id.out" 2>&1 || echo "FAILED: $id"
}
ref() {  # HBF-Sim exactly as published: its qwen3.config verbatim, GPGPU-Sim's built-in DRAM
  rm -rf "$V/REF_hbfsim"; mkdir -p "$V/REF_hbfsim"
  "$ST/simulator/run_sim.sh" "$V/REF_hbfsim" -config "$HERE/base/hbfsim_qwen3.config" \
    -trace "$ST/workloads/vectoradd/hbf/kernelslist.g" > "$V/REF_hbfsim/run.log" 2>&1 || echo "FAILED: REF_hbfsim"
}
[[ -z "$ONLY" || "$ONLY" == *,REF_hbfsim,* ]] && ref &
for r in "${RUNS[@]}"; do
  [[ -n "$ONLY" && "$ONLY" != *",${r%%|*},"* ]] && continue
  while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 5; done
  one "$r" &
done
wait
python3 "$HERE/validate/check.py" "$V"
