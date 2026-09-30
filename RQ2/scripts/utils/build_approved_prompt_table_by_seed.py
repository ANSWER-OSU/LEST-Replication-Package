import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

MUTATED_DIR = "RQ2/data/mutated"
DEFAULT_OUTPUT = "RQ2/outputs/approved_prompts_by_seed.csv"

SEED_RE = re.compile(r"_fuzzed_prompt_seed(\d+)\.json$")


def find_categories():
    categories = []
    for name in sorted(os.listdir(MUTATED_DIR)):
        if os.path.isdir(os.path.join(MUTATED_DIR, name)):
            categories.append(name)
    return categories


def count_approved_by_seed_individual(category):
    counts = Counter()
    pattern = os.path.join(MUTATED_DIR, category, f"{category}_fuzzed_prompt_seed*.json")
    for path in sorted(glob.glob(pattern)):
        match = SEED_RE.search(path)
        if not match:
            continue
        seed = int(match.group(1))
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
        counts[seed] += sum(1 for r in results if r.get("approved"))
    return counts


def count_approved_by_seed_combined(category):
    path = os.path.join(MUTATED_DIR, category, f"{category}_fuzzed_prompt_combined_seeds.json")
    if not os.path.exists(path):
        return Counter()
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    counts = Counter()
    for r in results:
        if r.get("approved"):
            counts[r["seed"]] += 1
    return counts


def build_table(source):
    count_fn = count_approved_by_seed_combined if source == "combined" else count_approved_by_seed_individual
    categories = find_categories()
    counts_by_category = {category: count_fn(category) for category in categories}
    seeds = sorted({seed for counts in counts_by_category.values() for seed in counts})
    return categories, seeds, counts_by_category


def write_table(categories, seeds, counts_by_category, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category"] + [f"seed{s}" for s in seeds] + ["total"])

        seed_totals = Counter()
        for category in categories:
            counts = counts_by_category[category]
            row = [counts.get(s, 0) for s in seeds]
            for seed, value in zip(seeds, row):
                seed_totals[seed] += value
            writer.writerow([category] + row + [sum(row)])

        totals_row = [seed_totals.get(s, 0) for s in seeds]
        writer.writerow(["total"] + totals_row + [sum(totals_row)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "-s", "--source", choices=["combined", "individual"], default="combined",
        help="'combined' (default): cross-seed-deduped combined file. "
             "'individual': each seed's own fuzzed_prompt file, no cross-seed filtering.",
    )
    args = parser.parse_args()

    categories, seeds, counts_by_category = build_table(args.source)
    write_table(categories, seeds, counts_by_category, args.output)
    print(f"wrote {len(categories)} categories x {len(seeds)} seeds ({args.source}) to {args.output}")


if __name__ == "__main__":
    main()
