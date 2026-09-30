#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CATEGORIES=(fairness_bias harmful_violent_content mental_health_self_harm)
SEEDS=(1 24 42 64 120)

LOG_DIR="logs/wildguard_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "logging to $LOG_DIR"

# full list of input files, in (category, seed) order
FILES=()
for category in "${CATEGORIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        FILES+=("RQ2/data/mutated/${category}/${category}_approved_prompts_seed${seed}.json")
    done
done

GPU0_FILES=()
GPU1_FILES=()
for i in "${!FILES[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_FILES+=("${FILES[$i]}")
    else
        GPU1_FILES+=("${FILES[$i]}")
    fi
done

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
