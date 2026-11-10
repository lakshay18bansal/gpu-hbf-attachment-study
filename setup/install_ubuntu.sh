#!/usr/bin/env bash
# Install everything needed to build and run this repository natively on Ubuntu 24.04 (tested
# environment: Ubuntu 24.04, g++ 13, cmake 3.28, Python 3.12, CUDA 12.8). Ubuntu 22.04 also works
# with g++-12 (Ramulator 2 needs C++20).
#
#   setup/install_ubuntu.sh             apt packages + Python venv (.venv) with requirements.txt
#   setup/install_ubuntu.sh --cuda      also install the CUDA 12.8 toolkit from NVIDIA's apt repo
#
# No GPU is needed: the simulator replays traces; the CUDA toolkit (nvcc + headers) is only
# needed to build it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sudo apt-get update
sudo apt-get install -y build-essential g++ cmake make git bison flex xutils-dev zlib1g-dev libzstd-dev \
  libglu1-mesa-dev python3 python3-dev python3-pip python3-venv time xz-utils bc wget

if [[ "${1:-}" == --cuda ]]; then
  . /etc/os-release
  REPO="ubuntu${VERSION_ID//./}"          # ubuntu2404 or ubuntu2204
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb" \
    -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-8
fi

if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  echo "CUDA toolkit: $(/usr/local/cuda/bin/nvcc --version | tail -1)"
else
  echo "WARNING: no CUDA toolkit in /usr/local/cuda. Run 'setup/install_ubuntu.sh --cuda', or install"
  echo "         one yourself and set CUDA_INSTALL_PATH to it before simulator/build.sh."
fi

python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"
echo "done. Activate the Python environment with:  source .venv/bin/activate"
