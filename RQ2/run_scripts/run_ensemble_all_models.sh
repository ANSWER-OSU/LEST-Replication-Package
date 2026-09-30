#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CATEGORIES=(fairness_bias harmful_violent_content mental_health_self_harm)

FAILED=()

run_queue() {
    local device="$1"
    local model="$2"
    local category input
    for category in "${CATEGORIES[@]}"; do
        input="RQ2/data/evaluation/model_response/${category}_mutated_prompt_result_${model}_wildguard_result.json"
        echo "[$(date +%H:%M:%S)] (${device}) ${category}/${model} <- ${input}"
        if ! DEVICE="$device" ./RQ2/run_scripts/run_ensemble_all.sh "$input"; then
            echo "[$(date +%H:%M:%S)] ${category}/${model} FAILED -- continuing with the rest of this queue" >&2
            FAILED+=("${category}/${model}")
        fi
    done
}

run_queue cuda:0 claude &
pid0=$!
run_queue cuda:1 gpt &
pid1=$!

wait "$pid0"
wait "$pid1"

if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] all category/model ensembles completed"
    exit 0
else
    echo "[$(date +%H:%M:%S)] completed with failures: ${FAILED[*]} -- rerun this script to resume them" >&2
    exit 1
fi
