#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_NAME="Wan2.1-T2V-1.3B"
MODEL_CKPT_DIR="/home/dataset/Wan2.1-T2V-1.3B"
NUM_NODES=1

if [ -z "$MODEL_NAME" ]; then
    echo "Error: please set MODEL_NAME in $0" >&2
    exit 1
fi

if [ -z "$MODEL_CKPT_DIR" ]; then
    echo "Error: please set MODEL_CKPT_DIR in $0" >&2
    exit 1
fi

run_case() {
    local tag="$1"
    local gpus_per_node="$2"
    local cfg_size="$3"
    local cp_size="$4"
    local fpp_size="$5"
    local patch_num="$6"

    "$PROJECT_ROOT/run.sh" \
        --num-nodes "$NUM_NODES" \
        --gpus-per-node "$gpus_per_node" \
        --cfg-size "$cfg_size" \
        --cp-size "$cp_size" \
        --fpp-size "$fpp_size" \
        --patch-num "$patch_num" \
        --model-name "$MODEL_NAME" \
        --model-ckpt-dir "$MODEL_CKPT_DIR" \
        --tag "$tag"
}

# Missing latency rows in experiments/data_latency.md.
run_case "small_baseline" 1 1 1 1 1
run_case "small_4cp" 4 1 4 1 1
run_case "small_4fpp" 4 1 1 4 7
run_case "small_2cfp2cp" 4 2 2 1 1
run_case "small_2cfp2pp" 4 2 1 2 3
run_case "small_2cp2pp" 4 1 2 2 3

run_case "small_8cp" 8 1 8 1 1
run_case "small_8pp" 8 1 1 8 12
run_case "small_2cfp4cp" 8 2 4 1 1
run_case "small_2cfp4pp" 8 2 1 4 7
run_case "small_2cfp2cp2pp" 8 2 2 2 3



