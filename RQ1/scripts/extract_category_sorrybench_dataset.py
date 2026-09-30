import json
from pathlib import Path

# Configuration
CATEGORY_IDS = [15] # Examples: CATEGORY_IDS = [7], CATEGORY_IDS = [7, 8, 9], CATEGORY_IDS = range(1, 45)
DOMAIN = "harmful_violent_content"

input_path = Path(
    "RQ1/data/sorrybench_base.json"
)

with open(
    input_path,
    "r",
    encoding="utf-8"
) as f:
    dataset = json.load(f)

for category_id in CATEGORY_IDS:

    output_path = Path(
        f"RQ1/outputs/processed/category{category_id}_sorrybench.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    filtered_data = []

    for item in dataset:

        category = str(item.get("category"))

        if category != str(category_id):
            continue

        filtered_data.append({
            "question_id": item.get("question_id"),
            "category": category,
            "domain": DOMAIN,
            "prompt": item.get("prompt")
        })

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            filtered_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved {len(filtered_data)} prompts "
        f"for category {category_id} "
        f"to {output_path}"
    )