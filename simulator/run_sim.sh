#!/usr/bin/env bash
# Run the simulator in a run folder.
#   simulator/run_sim.sh <run dir> <accel-sim args...>      (stdout/stderr pass through)
# SIM_BIN      binary (default simulator/build/accel-sim.out)
# SIM_TIMEOUT  seconds, 0 = none (default)
# HBF_CONTAINER=/path/to/accelsim.sif runs it inside Apptainer; otherwise natively.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
D="$1"; shift
cd "$D"
BIN="${SIM_BIN:-$HERE/build/accel-sim.out}"
[[ -x "$BIN" ]] || { echo "simulator binary $BIN not found: run simulator/build.sh first" >&2; exit 2; }
T=()
[[ "${SIM_TIMEOUT:-0}" != 0 ]] && T=(timeout "$SIM_TIMEOUT")
if [[ -n "${HBF_CONTAINER:-}" ]]; then
  B=(--bind "$ROOT" --bind /tmp)
  [[ "$PWD" == "$ROOT"* ]] || B+=(--bind "$PWD")
  exec "${T[@]}" nice -n 10 apptainer exec --cleanenv "${B[@]}" "$HBF_CONTAINER" "$BIN" "$@"
fi
exec "${T[@]}" nice -n 10 "$BIN" "$@"
