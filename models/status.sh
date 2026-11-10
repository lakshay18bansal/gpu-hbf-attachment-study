#!/usr/bin/env bash
# Status of the Qwen runs: is each one still running, finished OK, or failed?
# usage: models/status.sh [pattern]     (default pattern: qwen3_full; e.g. qwen3_layer13, vectoradd, or '' for all)
# Looks at runs/ (or $HBF_RUNS).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; R="${HBF_RUNS:-$(cd "$HERE/.." && pwd)/runs}"
PAT="${1-qwen3_full}"
# Folders whose simulator is alive right now (from each accel-sim process's working directory)
declare -A ALIVE
for p in $(pgrep -f 'accel-sim.out -config' 2>/dev/null); do
  d="$(readlink "/proc/$p/cwd" 2>/dev/null)" && ALIVE["$d"]=$p
done
printf '%-34s %-10s %-11s %-15s %-9s %s\n' RUN STATE KERNELS SIM_CYCLES LOG_AGE NOTE
for d in "$R"/C*"$PAT"*/; do
  d="${d%/}"; n="$(basename "$d")"; log="$d/run.log"; [[ -f "$log" ]] || continue
  case "$n" in *qwen3_full*) total=1819 ;; *qwen3_layer*) total=64 ;; *qwen3_prologue*) total=18 ;; *qwen3_tail*) total=9 ;; *vectoradd*) total=1 ;; *) total='?' ;; esac
  k=$(grep -c '^gpu_tot_sim_cycle' "$log")
  cyc=$(grep '^gpu_tot_sim_cycle' "$log" | tail -1 | awk '{print $3}')
  age=$(( $(date +%s) - $(stat -c %Y "$log") ))
  err=$(grep -v '^-' "$log" | grep -m1 -iE 'deadlock detected|GPGPU-Sim uArch: ERROR|Segmentation fault|Assertion|core dumped|Aborted|terminate called|out of memory|Killed' | cut -c1-60)
  rc=$(cat "$d/rc" 2>/dev/null)
  if [[ -n "${ALIVE[$d]}" ]]; then
    st=running; ((age > 1800)) && st=STUCK?
  elif [[ "$rc" == rc=0 ]]; then st=DONE
  elif [[ -n "$rc" ]]; then st="FAILED"
  else st="DIED"   # no simulator and no rc file: killed (PC shut down, OOM, kill)
  fi
  [[ "$st" == DONE ]] && k=$total
  printf '%-34s %-10s %-11s %-15s %-9s %s\n' "$n" "$st" "$k/$total" "${cyc:-0}" "$((age/60))m" "${err:-$rc}"
done
echo
echo "running = fine | DONE = finished OK (read summary.txt) | FAILED/DIED = see run.log, restart with --force"
echo "STUCK? = log silent for 30+ min while the process is alive (normal during lm_head; worry after a few hours)"
