#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${1:-system_config.yaml}"
if [ $# -gt 0 ]; then
    shift
fi

export CHITU_EXP_NAME="${CHITU_EXP_NAME:-fpp_warmup_cooldown}"
export CHITU_EXP_CACHE_STRATEGIES="${CHITU_EXP_CACHE_STRATEGIES:-}"
export CHITU_EXP_FPP_SCHEDULES="${CHITU_EXP_FPP_SCHEDULES:-1:0 2:0 3:0 4:0 5:0 6:0 7:0 8:0 9:0 10:0 1:1 2:1 3:1 4:1 4:3 5:1 5:3 5:5 6:1 7:1}"
# export CHITU_EXP_FPP_SCHEDULES="${CHITU_EXP_FPP_SCHEDULES:-1:0 2:0 3:0 4:0 5:0 6:0 7:0 6:1 7:1}"

export CHITU_EXP_SKIP_BASELINE="${CHITU_EXP_SKIP_BASELINE:-0}"
export CHITU_EXP_FPP_DEBUG="${CHITU_EXP_FPP_DEBUG:-1}"

cd "$PROJECT_ROOT"
exec bash "$PROJECT_ROOT/script/run_flexcache_quality_experiments.sh" "$CONFIG_FILE" "$@"
