#!/usr/bin/env bash
# Runs ON the GPU box (after accelsim-work/ssh_traces_build/remote_setup.sh has built the tracer).
# Captures one decode step / one layer with capture_qwen3_layer.py under the NVBit tracer, post-
# processes the traces and packs <ROOT>/out/capture/<name>.tar.gz.
#
# usage: remote_capture.sh <ROOT> <name> [capture_qwen3_layer.py args ...]
#   e.g. remote_capture.sh /root/ssh_traces_build qwen3_b8_c2048_l13 --batch 8 --context 2048 --probe-layer 13
set -Eeuo pipefail
ROOT="$1"; NAME="$2"; shift 2
REPO="$ROOT/accel-sim-framework"
TOOL="$REPO/util/tracer_nvbit/tracer_tool/tracer_tool.so"
POST="$REPO/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/env.sh" ]] || { echo "ERROR: $ROOT/env.sh missing: run remote_setup.sh first" >&2; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/env.sh"
[[ -f "$TOOL" && -x "$POST" ]] || { echo "ERROR: tracer not built in $REPO" >&2; exit 1; }

# A python with torch + transformers (the vLLM image has both; env.sh's PYTHON needs vllm).
PY=""
for p in ${PYTHON:-} python3 /usr/bin/python3 /venv/main/bin/python; do
  if [[ -x "$(command -v "$p" 2>/dev/null)" ]] && "$p" -c 'import torch, transformers' >/dev/null 2>&1; then
    PY="$(command -v "$p")"; break
  fi
done
[[ -n "$PY" ]] || { echo "ERROR: no python with torch + transformers" >&2; exit 1; }
CC="$("$PY" -c 'import torch; print("%d%d" % torch.cuda.get_device_capability())')"
if [[ "$CC" == 89 ]]; then
  echo "WARNING: sm_89 (RTX 4090 / L4 / L40S): Accel-Sim's front end rejects binary version 89 unless the" >&2
  echo "         libraries ship sm_80/86 SASS. Prefer an A100 (sm_80) or H100 (sm_90)." >&2
fi
# Free GPU memory taken by the image's own vLLM server (Vast vLLM images).
if command -v supervisorctl >/dev/null && supervisorctl status vllm 2>/dev/null | grep -q RUNNING; then
  supervisorctl stop vllm || true; sleep 5
fi

T="$ROOT/out/capture/$NAME"
rm -rf "$T"; mkdir -p "$T/traces"
echo "=== native (untraced) run: timing reference ==="
"$PY" "$HERE/capture_qwen3_layer.py" --out "$T/native" --no-trace "$@"
echo "=== traced run (only the cudaProfilerStart/Stop region) ==="
CUDA_INJECTION64_PATH="$TOOL" ACTIVE_FROM_START=0 USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$T/traces" \
  "$PY" "$HERE/capture_qwen3_layer.py" --out "$T/capture" "$@" 2>&1 | tee "$T/trace.log"
mv "$T/capture/metadata" "$T/metadata"; rmdir "$T/capture"
ls "$T"/traces/kernelslist* >/dev/null 2>&1 || { echo "ERROR: no kernels traced" >&2; exit 1; }
"$POST" "$T/traces" -j "$(nproc)"
[[ -s "$T/traces/kernelslist.g" ]] || { echo "ERROR: post-processing produced no kernelslist.g" >&2; exit 1; }
rm -f "$T"/traces/*.trace "$T"/traces/*.trace.xz "$T"/traces/kernelslist_ctx_*
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv > "$T/gpu.txt"
tar -C "$ROOT/out/capture" -czf "$ROOT/out/capture/$NAME.tar.gz" "$NAME"
echo "CAPTURE_DONE $NAME: $(grep -c '^kernel' "$T/traces/kernelslist.g") kernels -> $ROOT/out/capture/$NAME.tar.gz"
