#!/usr/bin/env bash
# Laptop side: capture the batch / context matrix on one rented GPU session and bring it home.
# Uses the SSH settings and the remote setup of accelsim-work/ssh_traces_build (remote_trace.conf,
# remote_setup.sh), so start the instance and update SSH_HOST / SSH_PORT there first.
#
# usage: scripts/capture/capture_suite.sh [--dry-run] [--only NAME] [--conf FILE]
# Captures (Qwen3-1.7B, BF16, eager attention, StaticCache, decode step 2):
#   qwen3_b{1,8,32}_c{512,2048}_l13   one decoder layer (layer 13) of one decode step
#   qwen3_b1_c512_full                the whole decode step (cross-check with the bundled HBF-Sim trace)
# Rough cost on an A100/H100: each layer capture 5-20 min (tracing slows the step ~1000x;
# batch 32 x 2048 is the largest), the full step ~1-2 h; disk ~0.2-3 GB per capture (compressed).
# Afterwards, locally:  tools/place_tensors.py staging/<name> -o workloads/<name> --hbf weights
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST="$(cd "$HERE/../.." && pwd)"
SSHB=<btp_work>/accelsim-work/ssh_traces_build
CONF="$SSHB/remote_trace.conf"; DRY=0; ONLY=""
while (($#)); do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --conf) CONF="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
CAPS=()
for b in 1 8 32; do for c in 512 2048; do
  CAPS+=("qwen3_b${b}_c${c}_l13|--batch $b --context $c --probe-layer 13")
done; done
CAPS+=("qwen3_b1_c512_full|--batch 1 --context 512 --probe-layer -1")

# shellcheck disable=SC1090
source "$CONF"
: "${SSH_HOST:?Set SSH_HOST in $CONF}"; SSH_PORT="${SSH_PORT:-22}"; SSH_USER="${SSH_USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/ssh_traces_build}"
COMMON=(-o StrictHostKeyChecking=accept-new); [[ -n "${SSH_KEY:-}" ]] && COMMON+=(-i "$SSH_KEY")
SSH=(ssh -p "$SSH_PORT" "${COMMON[@]}" "$SSH_USER@$SSH_HOST")
SCP=(scp -P "$SSH_PORT" "${COMMON[@]}")
for c in "${CAPS[@]}"; do
  name="${c%%|*}"; [[ -z "$ONLY" || "$ONLY" == "$name" ]] && echo "capture $name: ${c#*|}"
done
((DRY)) && exit 0

"${SSH[@]}" "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv"
"${SSH[@]}" "mkdir -p $REMOTE_ROOT/capture"
"${SCP[@]}" "$SSHB/remote_setup.sh" "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/"
"${SCP[@]}" "$HERE/remote_capture.sh" "$HERE/capture_qwen3_layer.py" "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/capture/"
"${SSH[@]}" "bash $REMOTE_ROOT/remote_setup.sh $REMOTE_ROOT ${ACCELSIM_BRANCH:-dev} ${REMOTE_PYTHON:-}"
"${SSH[@]}" "python3 -c 'import transformers' 2>/dev/null || python3 -m pip install -q 'transformers>=4.56' accelerate"
mkdir -p "$ST/scripts/capture/staging"
for c in "${CAPS[@]}"; do
  name="${c%%|*}"; args="${c#*|}"
  [[ -z "$ONLY" || "$ONLY" == "$name" ]] || continue
  echo "=== $name ==="
  "${SSH[@]}" "${HF_TOKEN:+HF_TOKEN=$HF_TOKEN} bash $REMOTE_ROOT/capture/remote_capture.sh $REMOTE_ROOT $name $args"
  "${SCP[@]}" "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/out/capture/$name.tar.gz" "$ST/scripts/capture/staging/"
  tar -C "$ST/scripts/capture/staging" -xzf "$ST/scripts/capture/staging/$name.tar.gz"
  echo "staged: $ST/scripts/capture/staging/$name"
done
echo "Next: python3 tools/place_tensors.py scripts/capture/staging/<name> -o workloads/<name> --hbf weights"
