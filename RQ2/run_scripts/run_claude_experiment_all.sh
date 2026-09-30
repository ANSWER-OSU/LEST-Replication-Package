#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-mutated}"
export MODE

CATEGORIES=(fairness_bias harmful_violent_content mental_health_self_harm)

FAILED=()

for category in "${CATEGORIES[@]}"; do
    input="RQ2/data/evaluation/wildguard/${category}/${category}_fuzzed_prompt_combined_seeds_wildguard_without_sampled.json"
    echo "[$(date +%H:%M:%S)] ${category} (mode=${MODE}) <- ${input}"
    if ! INPUT_PATH="$input" python RQ2/scripts/model_experiment/claude_experiment.py; then
        echo "[$(date +%H:%M:%S)] ${category} FAILED -- continuing with the remaining categories" >&2
        FAILED+=("$category")
    fi
done

if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] all categories completed"
    exit 0
else
    echo "[$(date +%H:%M:%S)] completed with failures: ${FAILED[*]} -- rerun this script to resume them" >&2
    exit 1
fi
