import argparse
import csv
import glob
import json
import os

MUTATED_TIERS = "RQ2/data/evaluation/filtered_harmful/ensemble_compliance_tiers.json"
ORIGINAL_TIERS = "RQ2/data/evaluation/original_prompt_experiment/tiered/ensemble_compliance_tiers.json"
ORIGINAL_INPUTS = "RQ2/data/evaluation/original_prompt_experiment/inputs/*/*_original_prompts.json"
DEFAULT_OUTPUT_DIR = "RQ2/outputs/complied_prompts"
DEFAULT_MIN_VOTES = 2

FIELDS = [
    "prompt_id", "question_id", "model", "prompt_type", "dataset_category",
    "tier", "models_flagged", "original_prompt", "prompt", "op_path",
    "lexical_difference", "model_response",
]


def load_original_prompts():
    by_question = {}
    for path in glob.glob(ORIGINAL_INPUTS):
        with open(path, encoding="utf-8") as f:
            for entry in json.load(f):
                by_question[entry["question_id"]] = entry["original_prompt"]
    return by_question


def question_id(prompt_id):
    return int(prompt_id.split("-")[0].lstrip("q"))


def collect(tier_path, prompt_type, min_votes, originals):
    with open(tier_path, encoding="utf-8") as f:
        records = json.load(f)

    complied = []
    for r in records:
        # tier is "<n>/3": how many of the 3 voting judges called it compliance
        if int(r["tier"].split("/")[0]) < min_votes:
            continue
        qid = question_id(r["prompt_id"])
        complied.append({
            "prompt_id": r["prompt_id"],
            "question_id": qid,
            "model": r["model"],
            "prompt_type": prompt_type,
            "dataset_category": r["dataset_category"],
            "tier": r["tier"],
            "models_flagged": r["models_flagged"],
            "original_prompt": originals.get(qid),
            "prompt": r["mutated_prompt"],
            "op_path": r["op_path"],
            "lexical_difference": r["lexical_difference"],
            "model_response": r["model_response"],
        })

    complied.sort(key=lambda p: (p["model"], p["question_id"], p["prompt_id"]))
    return complied


def write_json(prompts, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def write_csv(prompts, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for p in prompts:
            row = dict(p)
            row["models_flagged"] = ";".join(p["models_flagged"] or [])
            row["op_path"] = ";".join(p["op_path"] or [])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Export the prompts each model complied with (tier 2/3 or 3/3), for the complied-prompt reruns"
    )
    parser.add_argument("-v", "--min-votes", type=int, default=DEFAULT_MIN_VOTES)
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    originals = load_original_prompts()

    everything = []
    for prompt_type, tier_path in (("original", ORIGINAL_TIERS), ("mutated", MUTATED_TIERS)):
        prompts = collect(tier_path, prompt_type, args.min_votes, originals)
        everything.extend(prompts)
        for model in sorted({p["model"] for p in prompts}):
            subset = [p for p in prompts if p["model"] == model]
            output_path = os.path.join(args.output_dir, f"{model}_{prompt_type}_complied.json")
            write_json(subset, output_path)
            print(f"wrote {len(subset)} {prompt_type} prompts to {output_path}")

    csv_path = os.path.join(args.output_dir, "complied_prompts.csv")
    write_csv(everything, csv_path)
    print(f"wrote {len(everything)} rows to {csv_path}")


if __name__ == "__main__":
    main()
