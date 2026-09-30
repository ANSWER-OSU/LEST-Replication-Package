#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_PATH="${1:-data/evaluation/new_test_mutated_prompt_result_claude_wildguard_result.json}"

if [[ "$INPUT_PATH" != *_wildguard_result.json ]]; then
    echo "[$(date +%H:%M:%S)] wildguard (response classification)"
    if ! python RQ2/scripts/wildguard_evaluation/wildguard_analyze_model_response.py "$INPUT_PATH"; then
        echo "[$(date +%H:%M:%S)] wildguard FAILED -- not running the judges on a partial file; rerun this script to resume" >&2
        exit 1
    fi
    # same output path wildguard_analyze_model_response.py writes to
    INPUT_PATH="${INPUT_PATH%.json}_wildguard_result.json"
fi

FAILED=()

run_step() {
    local label="$1"
    shift
    echo "[$(date +%H:%M:%S)] $label"
    if ! "$@"; then
        echo "[$(date +%H:%M:%S)] $label FAILED -- continuing with the remaining judges" >&2
        FAILED+=("$label")
    fi
}

run_step "guardreasoner"           python RQ2/scripts/judge_ensemble/run_guardreasoner.py "$INPUT_PATH"
run_step "qwen3guard"              python RQ2/scripts/judge_ensemble/run_qwen3guard.py "$INPUT_PATH"
run_step "sorrybench (non-voting)" python RQ2/scripts/judge_ensemble/run_sorrybench_judge.py "$INPUT_PATH"
run_step "strongreject (non-voting)" python RQ2/scripts/judge_ensemble/run_strongreject.py "$INPUT_PATH"

run_step "aggregate" python RQ2/scripts/judge_ensemble/aggregate_ensemble.py "$INPUT_PATH"

if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] all steps completed"
    exit 0
else
    echo "[$(date +%H:%M:%S)] completed with failures: ${FAILED[*]} -- rerun this script to resume them" >&2
    exit 1
fi
