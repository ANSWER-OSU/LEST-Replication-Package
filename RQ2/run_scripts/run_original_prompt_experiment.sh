#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BASE="RQ2/data/evaluation/original_prompt_experiment"
CATEGORIES=(fairness_bias harmful_violent_content mental_health_self_harm)
LOG_DIR="$BASE/logs"
mkdir -p "$LOG_DIR"

stamp() { date +%H:%M:%S; }

echo "[$(stamp)] step 0: building inputs"
python RQ2/scripts/utils/original_prompt_experiment.py build || exit 1

# ---- step 1: responses ------------------------------------------------------
get_responses() {
    local model="$1" category input rc=0
    for category in "${CATEGORIES[@]}"; do
        input="$BASE/inputs/${category}/${category}_original_prompts.json"
        echo "[$(stamp)] ${model} responses: ${category}"
        MODE=original INPUT_PATH="$input" OUTPUT_DIR="$BASE/responses" \
            python "RQ2/scripts/model_experiment/${model}_experiment.py" || rc=1
    done
    return $rc
}

echo "[$(stamp)] step 1: claude + gpt responses"
get_responses claude > "$LOG_DIR/responses_claude.out" 2>&1 &
pid0=$!
get_responses gpt > "$LOG_DIR/responses_gpt.out" 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then
    echo "[$(stamp)] step 1 had failures -- see $LOG_DIR/responses_*.out. Not continuing; rerun to resume." >&2
    exit 1
fi

# ---- steps 2 and 3: judges, one GPU per model ------------------------------
judge_queue() {
    local device="$1" model="$2" category responses rc=0
    for category in "${CATEGORIES[@]}"; do
        responses="$BASE/responses/${category}_original_prompt_result_${model}"
        echo "[$(stamp)] (${device}) wildguard response ${category}/${model}"
        DEVICE="$device" python RQ2/scripts/wildguard_evaluation/wildguard_analyze_model_response.py \
            "${responses}.json" || rc=1
    done
    for category in "${CATEGORIES[@]}"; do
        responses="$BASE/responses/${category}_original_prompt_result_${model}"
        echo "[$(stamp)] (${device}) ensemble ${category}/${model}"
        ENSEMBLE_OUTPUT_ROOT="$BASE/judge_ensemble" DEVICE="$device" \
            ./RQ2/run_scripts/run_ensemble_all.sh "${responses}_wildguard_result.json" || rc=1
    done
    return $rc
}

echo "[$(stamp)] steps 2-3: wildguard response classification, then judge ensemble"
judge_queue cuda:0 claude > "$LOG_DIR/judges_claude.out" 2>&1 &
pid0=$!
judge_queue cuda:1 gpt > "$LOG_DIR/judges_gpt.out" 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [ "$status" -ne 0 ]; then
    echo "[$(stamp)] a judge step had failures -- see $LOG_DIR/judges_*.out. Rerun to resume." >&2
fi

# ---- step 4: verify ---------------------------------------------------------
echo "[$(stamp)] step 4: verifying every stage still holds the original prompts"
python RQ2/scripts/utils/original_prompt_experiment.py verify || status=1

if [ "$status" -eq 0 ]; then
    echo "[$(stamp)] original-prompt experiment complete: $BASE"
else
    echo "[$(stamp)] finished with problems -- see above" >&2
fi
exit "$status"
