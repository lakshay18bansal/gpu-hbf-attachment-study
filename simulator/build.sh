#!/usr/bin/env bash
# Build Ramulator 2 and the merged simulator (Accel-Sim + GPGPU-Sim + Ramulator bridge + HBF-Sim
# HBF tier + the C1-C4 attachment modes).
#
#   simulator/build.sh [-jN]
#
# Output: simulator/ramulator2/libramulator.so and simulator/build/accel-sim.out
# (the binary finds libramulator.so through an absolute RUNPATH: rebuild after moving the repo).
#
# Native build: needs the CUDA toolkit (nvcc; CUDA_INSTALL_PATH, default /usr/local/cuda), cmake >= 3.17,
#   g++ >= 12 (C++20 for the Ramulator bridge), bison, flex, zlib, libzstd, make and Python >= 3.10 with
#   its headers (Ramulator's Python bindings). setup/install_ubuntu.sh installs all of them.
# Container build: HBF_CONTAINER=/path/to/accelsim.sif simulator/build.sh  (Apptainer/Singularity;
#   setup/accelsim.def builds a suitable image).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
JOBS="${1:--j$(nproc 2>/dev/null || echo 4)}"

if [[ -n "${HBF_CONTAINER:-}" && -z "${HBF_IN_CONTAINER:-}" ]]; then
  [[ -f "$HBF_CONTAINER" ]] || { echo "HBF_CONTAINER=$HBF_CONTAINER not found" >&2; exit 2; }
  exec apptainer exec --bind "$ROOT" "$HBF_CONTAINER" env HBF_IN_CONTAINER=1 bash "$HERE/build.sh" "$JOBS"
fi

export CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda}"
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
command -v nvcc >/dev/null || { echo "nvcc not found: install the CUDA toolkit or set CUDA_INSTALL_PATH" >&2; exit 2; }

echo "== 1/2 Ramulator 2 (simulator/ramulator2)"
( cd "$HERE/ramulator2" && BUILD_DIR="$HERE/ramulator2/build" JOBS="${JOBS#-j}" sh ./build.sh >/dev/null )
ls -la "$HERE/ramulator2/libramulator.so"

echo "== 2/2 Accel-Sim / GPGPU-Sim with Ramulator and HBF (simulator/build)"
set +u   # setup_environment.sh reads unset variables
cd "$HERE/accel-sim-framework"
source ./gpu-simulator/setup_environment.sh >/dev/null
set -u
cmake -S ./gpu-simulator -B "$HERE/build" -DGPGPUSIM_USE_RAMULATOR=ON \
  -DRAMULATOR_ROOT="$HERE/ramulator2" >/dev/null
nice -n 10 cmake --build "$HERE/build" --target accel-sim.out "$JOBS"
ls -la "$HERE/build/accel-sim.out"
echo "build OK"
