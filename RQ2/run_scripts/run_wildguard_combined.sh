#!/usr/bin/env bash


set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="logs/wildguard_combined_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "logging to $LOG_DIR"

GPU0_FILES=(
    "RQ2/data/mutated/fairness_bias/fairness_bias_fuzzed_prompt_combined_seeds.json"
)
GPU1_FILES=(
    "RQ2/data/mutated/harmful_violent_content/harmful_violent_content_fuzzed_prompt_combined_seeds.json"
    "RQ2/data/mutated/mental_health_self_harm/mental_health_self_harm_fuzzed_prompt_combined_seeds.json"
)

run_queue() {
    local device="$1"
    shift
    local f stem log_file
    for f in "$@"; do
        stem="$(basename "$f" .json)"
        log_file="$LOG_DIR/${stem}_${device//:/_}.log"
        echo "[$(date +%H:%M:%S)] (${device}) ${f} -> ${log_file}"
        DEVICE="$device" python RQ2/scripts/wildguard_evaluation/run_wildguard.py "$f" \
            > "$log_file" 2>&1
    done
}

run_queue cuda:0 "${GPU0_FILES[@]}" &
pid0=$!
run_queue cuda:1 "${GPU1_FILES[@]}" &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1

if [ "$status" -eq 0 ]; then
    echo "all wildguard runs complete. logs in $LOG_DIR"
else
    echo "one or more wildguard runs failed -- check $LOG_DIR"
fi
exit "$status"
