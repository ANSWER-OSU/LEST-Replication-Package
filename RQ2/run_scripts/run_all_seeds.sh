#!/usr/bin/env bash


set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CATEGORIES=(fairness_bias harmful_violent_content mental_health_self_harm)
SEEDS=(1 24 42 64 120)
MINUTES=20

LOG_DIR="logs/experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "logging to $LOG_DIR"

JOBS=()
for category in "${CATEGORIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${category}:${seed}")
    done
done

run_jobs() {
    local device="$1"
    shift
    for job in "$@"; do
        local category="${job%%:*}"
        local seed="${job##*:}"
        local log_file="$LOG_DIR/${category}_seed${seed}.log"
        echo "[$(date +%H:%M:%S)] [${device}] ${category} seed=${seed} (${MINUTES} min) -> ${log_file}"
        DEVICE="${device}" python main.py -c "${category}.json" -seed "${seed}" -min "${MINUTES}" \
            > "$log_file" 2>&1
    done
}

GPU0_JOBS=()
GPU1_JOBS=()
for i in "${!JOBS[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_JOBS+=("${JOBS[$i]}")
    else
        GPU1_JOBS+=("${JOBS[$i]}")
    fi
done

run_jobs cuda:0 "${GPU0_JOBS[@]}" &
pid0=$!
run_jobs cuda:1 "${GPU1_JOBS[@]}" &
pid1=$!

wait "$pid0" "$pid1"

echo "all ${#JOBS[@]} runs complete. logs in $LOG_DIR"
