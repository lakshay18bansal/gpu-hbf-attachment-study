#!/usr/bin/env bash
# Regenerate the Ramulator YAML files the simulator reads (-gpgpu_ramulator_config) from the
# Python configs in this folder, exactly as accelsim-work/scripts/simulate.sh does it
# (python3 -m ramulator export <cfg.py> -o <cfg.yaml>), and print each config's SUMMARY.
#
# usage: common/memory/export.sh [cfg ...]        (default: every *_slice8*.py here)
#        RAMULATOR_SET="DATA_RATE_MBPS=..." common/memory/export.sh hbm3_slice8 -o /path/x.yaml
#          (one config with overrides, written to another file; do not overwrite the
#           committed YAMLs with an override)
#
# Runs on the host: the prebuilt Ramulator Python module needs Python 3.12 (host python3).
# Ramulator's package is used in place from accelsim-work (read-only): PYTHONDONTWRITEBYTECODE
# keeps Python from writing __pycache__ into that tree.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAM_ROOT="${RAM_ROOT:-<btp_work>/accelsim-work/ramulator}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RAM_ROOT/python:$HERE${PYTHONPATH:+:$PYTHONPATH}"

OUT_OVERRIDE=""
CFGS=()
while (($#)); do
  case "$1" in
    -o) OUT_OVERRIDE="$2"; shift 2 ;;
    *) CFGS+=("${1%.py}"); shift ;;
  esac
done
((${#CFGS[@]})) || CFGS=(hbm3_slice8 hbm3_slice8_half gddr6_slice8)
if [[ -n "$OUT_OVERRIDE" && ${#CFGS[@]} -ne 1 ]]; then
  echo "ERROR: -o needs exactly one config" >&2; exit 1
fi
if [[ -n "${RAMULATOR_SET:-}" && -z "$OUT_OVERRIDE" ]]; then
  echo "ERROR: RAMULATOR_SET is set; pass -o <file> so the reference YAMLs stay unchanged" >&2
  exit 1
fi

for c in "${CFGS[@]}"; do
  py="$HERE/$c.py"
  yaml="${OUT_OVERRIDE:-$HERE/$c.yaml}"
  python3 -m ramulator export "$py" -o "$yaml"
  echo "$c -> $yaml"
  python3 -c 'import runpy, sys; print("  " + runpy.run_path(sys.argv[1])["SUMMARY"])' "$py"
done
