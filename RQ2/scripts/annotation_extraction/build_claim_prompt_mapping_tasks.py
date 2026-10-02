import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CLAIMS = ROOT / "RQ1/outputs/annotated_claims.csv"
PROMPTS_DIR = ROOT / "RQ2/data/evaluation/original_prompt_experiment/inputs"
OUT_DIR = ROOT / "RQ2/annotations/claim_to_prompt"
CATEGORIES = ["fairness_bias", "harmful_violent_content", "mental_health_self_harm"]


def load_prompts():
    prompts = {}
    for cat in CATEGORIES:
        with open(PROMPTS_DIR / cat / f"{cat}_original_prompts.json", encoding="utf-8") as f:
            prompts[cat] = json.load(f)
    return prompts


def main():
    claims = pd.read_csv(CLAIMS)
    testable = claims[claims["label_testable"] == "Testable"]
    prompts = load_prompts()

    tasks = []
    for _, c in testable.iterrows():
        options = [
            # The choice value is what gets stored, so it leads with the
            # question id to keep results parseable ("q21: ...").
            {"value": f"q{p['question_id']}: {p['original_prompt']}"}
            for p in prompts[c["category"]]
        ]
        tasks.append({
            "data": {
                "claim_id": int(c["id"]),
                "claim_text": c["claim_text"],
                "category": c["category"],
                "section": c["section"].strip(),
                "source_doc": c["source_doc"],
                "num_candidate_prompts": len(options),
                "prompt_options": options,
            }
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    n_prompts = sum(len(v) for v in prompts.values())
    print(f"Wrote {len(tasks)} claim tasks over {n_prompts} seed prompts to {OUT_DIR / 'tasks.json'}")
    print(testable["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
