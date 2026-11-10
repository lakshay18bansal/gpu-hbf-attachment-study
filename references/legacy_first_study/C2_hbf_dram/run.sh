#!/usr/bin/env bash
# Run one workload on this configuration. usage: ./run.sh <workload> [--tR-us X] [--prefetch on|off]
#   [--out DIR] [--force] [-- -extra_opt val ...]   (all options: scripts/run_config.sh)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../scripts/run_config.sh" "$HERE" "$@"
