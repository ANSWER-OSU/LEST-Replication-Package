#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CATEGORIES=(fairness_bias harmful_violent_content mental_health_self_harm)

run_queue() {
    local device="$1"
    local model="$2"
    local category input
    for category in "${CATEGORIES[@]}"; do
        input="RQ2/data/evaluation/model_response/${category}_mutated_prompt_result_${model}.json"
        echo "[$(date +%H:%M:%S)] (${device}) ${category}/${model} <- ${input}"
        DEVICE="$device" python RQ2/scripts/wildguard_evaluation/wildguard_analyze_model_response.py "$input"
    done
}

run_queue cuda:0 claude &
pid0=$!
run_queue cuda:1 gpt &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1

if [ "$status" -eq 0 ]; then
    echo "all wildguard response-classification runs complete"
else
    echo "one or more runs failed -- check the output above, rerun to resume" >&2
fi
exit "$status"
