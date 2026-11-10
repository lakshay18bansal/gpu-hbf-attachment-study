#!/usr/bin/env bash
# Build results/REPORT.md (+ runs.csv, tpot.json, plots/) and results/PROJECTION.md from finished runs.
# usage: scripts/make_report.sh [results dirs ...]   (default: results/{smoke,qproj,study,fullmodel})
set -euo pipefail
ST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIRS=("$@"); ((${#DIRS[@]})) || DIRS=("$ST/results/smoke" "$ST/results/qproj" "$ST/results/study" "$ST/results/fullmodel")
EX=(); for d in "${DIRS[@]}"; do [[ -d "$d" ]] && EX+=("$d"); done
PY=python3; [[ -x <btp_work>/accel_sim_hbf/.venv/bin/python ]] && PY=<btp_work>/accel_sim_hbf/.venv/bin/python
for d in "${EX[@]}"; do for r in "$d"/*/; do [[ -f "$r/run_meta.json" ]] && python3 "$ST/tools/parse_metrics.py" "$r"; done; done
"$PY" "$ST/tools/report.py" "${EX[@]}" -o "$ST/results/REPORT.md" --csv "$ST/results/runs.csv" --plots "$ST/results/plots"
python3 "$ST/tools/project_model.py" --results "${EX[@]}" -o "$ST/results/PROJECTION.md"
