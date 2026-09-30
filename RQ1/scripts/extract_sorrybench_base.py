from datasets import load_dataset
import json
import os

# load dataset
ds = load_dataset("sorry-bench/sorry-bench-202503")

# keep only original prompts
base_prompts = []

for sample in ds["train"]:

    if sample["prompt_style"] == "base":

        base_prompts.append({
            "question_id": sample["question_id"],
            "category": sample["category"],
            "prompt": sample["turns"][0]
        })

# create folder
os.makedirs("RQ1/data", exist_ok=True)

# save
with open("RQ1/data/sorrybench_base.json", "w") as f:
    json.dump(base_prompts, f, indent=2)

print(f"saved {len(base_prompts)} base prompts")