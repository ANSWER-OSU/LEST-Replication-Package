import argparse
import glob
import json
import os
import re
import sys

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_SRC)
sys.path.insert(0, os.path.join(PROJECT_SRC, "mutation_experiment"))

from mutation_experiment.approval import diversity_filter, THRESHOLD_LEXICAL_DIVERSITY

MUTATED_DIR = "RQ2/data/mutated"
SEED_RE = re.compile(r"_fuzzed_prompt_seed(\d+)\.json$")


def find_categories():
    categories = []
    for name in sorted(os.listdir(MUTATED_DIR)):
        if os.path.isdir(os.path.join(MUTATED_DIR, name)):
            categories.append(name)
    return categories


def find_seed_files(category):
    """(seed, path) for every fuzzed_prompt seed file in this category, sorted by seed."""
    pattern = os.path.join(MUTATED_DIR, category, f"{category}_fuzzed_prompt_seed*.json")
    files = []
    for path in glob.glob(pattern):
        match = SEED_RE.search(path)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files)


def load_approved_across_seeds(category):
    combined = []
    for seed, path in find_seed_files(category):
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
        for r in results:
            if not r.get("approved"):
                continue
            record = dict(r)
            record["prompt_id"] = f"{record['prompt_id']}-s{seed}"
            if record.get("parent_id"):
                record["parent_id"] = f"{record['parent_id']}-s{seed}"
            record["seed"] = seed
            combined.append(record)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-l", "--diversity-threshold", type=float, default=THRESHOLD_LEXICAL_DIVERSITY,
    )
    args = parser.parse_args()

    for category in find_categories():
        combined = load_approved_across_seeds(category)
        if not combined:
            continue

        stats = diversity_filter(combined, diversity_min=args.diversity_threshold)

        out_path = os.path.join(
            MUTATED_DIR, category, f"{category}_fuzzed_prompt_combined_seeds.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"{category}: {stats['approved']} of {stats['total']} approved across seeds -> {out_path}")
        print(f"  -> rejected for cross-seed diversity: {stats['rejected']}")


if __name__ == "__main__":
    main()
