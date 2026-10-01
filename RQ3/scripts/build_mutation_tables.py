import json
import os
import sys

CATEGORIES = ["fairness_bias", "harmful_violent_content", "mental_health_self_harm"]
MODELS = ["claude", "gpt"]
OUT_DIR = "RQ3/data/analysis"
HARMFULNESS_FILTERED_CATEGORIES = {"harmful_violent_content", "mental_health_self_harm"}

MAIN_ENSEMBLE = "RQ2/data/evaluation/judge_ensemble/{category}_mutated_prompt_result_{model}_wildguard_result_ensemble.json"
BASELINE_ENSEMBLE = "RQ2/data/evaluation/original_prompt_experiment/judge_ensemble/{category}_original_prompt_result_{model}_wildguard_result_ensemble.json"
COMBINED = "RQ2/data/mutated/{category}/{category}_fuzzed_prompt_combined_seeds.json"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def result_index(category, model):
    # the pilot run's results are already merged into the main ensemble files
    return {row["prompt_id"]: row for row in load(MAIN_ENSEMBLE.format(category=category, model=model))}


def baseline_index(category, model):
    """{question_id: ensemble row} for the original prompts."""
    return {row["question_id"]: row for row in load(BASELINE_ENSEMBLE.format(category=category, model=model))}


def build(category, model):
    approved = [r for r in load(COMBINED.format(category=category)) if r["approved"] is True]
    results = result_index(category, model)
    baselines = baseline_index(category, model)
    response_key = f"{model}_response"

    rows = []
    for r in approved:
        result = results.get(r["prompt_id"])
        if result is None:
            sys.exit(f"{category}/{model}: no result for {r['prompt_id']}")
        if result["mutated_prompt"] != r["mutated_prompt"]:
            sys.exit(f"{category}/{model}: result for {r['prompt_id']} has a different prompt")
        base = baselines.get(r["question_id"])
        if base is None:
            sys.exit(f"{category}/{model}: no baseline for question {r['question_id']}")

        if (
            category in HARMFULNESS_FILTERED_CATEGORIES
            and result.get("wildguard_prompt_harmfulness") == "unharmful"
        ):
            continue

        rows.append(
            {
                "prompt_id": r["prompt_id"],
                "question_id": r["question_id"],
                "category_name": category,
                "model": model,
                "model_id": result["model"],
                "fuzzer_seed": r["seed"],
                "generation": r["generation"],
                "last_operator": r["mutation"],
                "op_path": r["op_path"],
                "original_prompt": r["original_prompt"],
                "parent_prompt": r["parent_prompt"],
                "mutated_prompt": r["mutated_prompt"],
                "semantic_similarity": r["semantic_similarity"],
                "lexical_difference": r["lexical_difference"],
                "wildguard_prompt_harmfulness": result.get("wildguard_prompt_harmfulness"),
                "response": result.get(response_key),
                "label": result["ensemble_refusal"],
                "status": result["ensemble_refusal_status"],
                "votes": result["ensemble_refusal_votes"],
                "baseline_response": base.get(response_key),
                "baseline_label": base["ensemble_refusal"],
                "baseline_status": base["ensemble_refusal_status"],
            }
        )
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for category in CATEGORIES:
        for model in MODELS:
            rows = build(category, model)
            path = f"{OUT_DIR}/{category}_mutation_all_experiment_{model}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            print(f"{category}/{model}: {len(rows)} mutations -> {path}")


if __name__ == "__main__":
    main()
