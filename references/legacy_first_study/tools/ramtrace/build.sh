#!/usr/bin/env bash
# Build the standalone Ramulator replay driver against accelsim-work's prebuilt libramulator.so
# (read-only; nothing in the Ramulator tree is rebuilt). Host g++ 13.3 is the compiler the library
# was built with; set CXX to use another one (e.g. inside accelsim.sif).
# Origin of ramtrace.cpp: research-phase copy
#   scratchpad/code-ramulator-bridge/driver/ramtrace.cpp (docs/research/ramulator_standalone_and_bridge.md 1.6)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAM_ROOT="${RAM_ROOT:-<btp_work>/accelsim-work/ramulator}"
"${CXX:-g++}" -std=c++20 -O2 -I"$RAM_ROOT/src" "$HERE/ramtrace.cpp" -o "$HERE/ramtrace" \
  "$RAM_ROOT/libramulator.so" -Wl,-rpath,"$RAM_ROOT"
echo "built $HERE/ramtrace"
