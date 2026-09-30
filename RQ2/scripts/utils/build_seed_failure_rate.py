import csv
import glob
import json
import os
from collections import defaultdict

from tested_prompt_filter import is_tested_prompt

ORIGINAL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "evaluation",
    "original_prompt_experiment",
    "judge_ensemble",
)
MUTATED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "evaluation", "judge_ensemble"
)
SUMMARY_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "evaluation",
    "filtered_harmful",
    "seed_failure_summary.csv",
)
DIST_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "evaluation",
    "filtered_harmful",
    "seed_failure_distribution.csv",
)
TABLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "evaluation",
    "filtered_harmful",
    "seed_level_failure_table.csv",
)


def load_records(dir_path):
    records = []
    for path in sorted(glob.glob(os.path.join(dir_path, "*_result_ensemble.json"))):
        fname = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            r["dataset_category"] = (
                r.get("category_name")
                or fname.split("_mutated_prompt")[0].split("_original_prompt")[0]
            )
        records.extend(rows)
    return records


def filter_harmful(records):
    return [r for r in records if is_tested_prompt(r, r["dataset_category"])]


def filter_guardreasoner_ok(records):
    return [
        r
        for r in records
        if r.get("guardreasoner_error") is None
        or r.get("ensemble_refusal_status") == "no_response"
    ]


def filter_scored(records):
    return [r for r in records if r.get("ensemble_refusal") is not None]


def clean(records, require_harmful=True):
    if require_harmful:
        records = filter_harmful(records)
    return filter_scored(filter_guardreasoner_ok(records))


def seed_key(r):
    return (r["question_id"], r["model"])


MODEL_DISPLAY_NAMES = {"claude-opus-4-8": "opus-4.8"}


def display_model(model):
    return MODEL_DISPLAY_NAMES.get(model, model)


def summary_row(category, model, seed_keys, failure_counts):
    n_refused = len(seed_keys)
    n_failure = sum(1 for k in seed_keys if failure_counts[k] > 0)
    rate = round(100 * n_failure / n_refused, 2) if n_refused else 0.0
    return {
        "category": category,
        "model": model,
        "n_refused_seeds": n_refused,
        "n_seeds_with_complying_mutation": n_failure,
        "failure_rate_pct": rate,
    }


def distribution_rows(model_label, seed_keys, failure_counts):
    dist = defaultdict(int)
    for k in seed_keys:
        dist[failure_counts[k]] += 1
    return [
        {"model": model_label, "n_failing_mutations": n, "n_seeds": dist[n]}
        for n in sorted(dist)
    ]


def lineage_counts_by_seed(mutations_by_seed, seed_keys):
    result = {}
    for k in seed_keys:
        counts = defaultdict(int)
        for m in mutations_by_seed[k]:
            if m["ensemble_refusal"] == "compliance":
                counts[m["seed"]] += 1
        result[k] = counts
    return result


def table_rows(seed_keys, meta_by_seed, failure_counts, lineage_counts, fuzzer_seeds):
    rows = []
    for k in sorted(
        seed_keys,
        key=lambda k: (meta_by_seed[k]["dataset_category"], meta_by_seed[k]["model"], k[0]),
    ):
        question_id, model = k
        counts = lineage_counts[k]
        row = {
            "category": meta_by_seed[k]["dataset_category"],
            "question_id": question_id,
            "model": model,
            "at_least_one_complying_mutation": failure_counts[k] > 0,
        }
        for s in fuzzer_seeds:
            row[f"n_complying_seed_{s}"] = counts.get(s, 0)
        row["n_complying_total"] = failure_counts[k]
        rows.append(row)
    return rows


def main():
    originals = clean(load_records(ORIGINAL_DIR), require_harmful=False)
    original_by_seed = {seed_key(r): r for r in originals}
    refused_seeds = {k for k, r in original_by_seed.items() if r["ensemble_refusal"] == "refusal"}

    mutated = clean(load_records(MUTATED_DIR))

    mutations_by_seed = defaultdict(list)
    n_no_original = 0
    for r in mutated:
        key = seed_key(r)
        if key not in original_by_seed:
            n_no_original += 1
            continue
        mutations_by_seed[key].append(r)

    seeds_with_mutations = {k for k in refused_seeds if mutations_by_seed.get(k)}
    n_no_mutations = len(refused_seeds) - len(seeds_with_mutations)

    failure_counts = {}
    for k in seeds_with_mutations:
        failure_counts[k] = sum(1 for m in mutations_by_seed[k] if m["ensemble_refusal"] == "compliance")

    n_refused = len(seeds_with_mutations)
    n_with_failure = sum(1 for k in seeds_with_mutations if failure_counts[k] > 0)
    rate = round(100 * n_with_failure / n_refused, 2) if n_refused else 0.0

    print(f"{len(originals)} scored original-prompt rows (harmfulness filter not applied), {len(refused_seeds)} refused")
    print(f"{n_no_mutations} refused seeds excluded (no scored+matched mutations)")
    print(f"{n_no_original} mutated rows skipped (no matching original-prompt seed)")
    print()
    print(f"originally-refused seeds with >=1 complying mutation: {n_with_failure} / {n_refused} ({rate}%)")

    meta_by_seed = {k: mutations_by_seed[k][0] for k in seeds_with_mutations}
    categories = sorted({m["dataset_category"] for m in meta_by_seed.values()})
    models = sorted({m["model"] for m in meta_by_seed.values()})

    def keys_where(category=None, model=None):
        return {
            k
            for k in seeds_with_mutations
            if (category is None or meta_by_seed[k]["dataset_category"] == category)
            and (model is None or meta_by_seed[k]["model"] == model)
        }

    # (all, all) grand total; (*, all) per category; (all, *) per model; (*, *) full cross product.
    rows = [summary_row("all", "all", keys_where(), failure_counts)]
    for category in categories:
        rows.append(summary_row(category, "all", keys_where(category=category), failure_counts))
    for model in models:
        rows.append(summary_row("all", model, keys_where(model=model), failure_counts))
    for category in categories:
        for model in models:
            rows.append(
                summary_row(category, model, keys_where(category=category, model=model), failure_counts)
            )

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    summary_keys = [
        "category",
        "model",
        "n_refused_seeds",
        "n_seeds_with_complying_mutation",
        "failure_rate_pct",
    ]
    summary_headers = [
        "category",
        "model",
        "# of refused seeds",
        "# of seeds with at least 1 complaint mutation",
        "Percentage",
    ]
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(summary_headers)
        for row in rows:
            writer.writerow(
                [display_model(row[k]) if k == "model" else row[k] for k in summary_keys]
            )
    print(f"\nwrote {len(rows)} rows -> {SUMMARY_PATH}")

    # Combined distribution first, then the same histogram split out per model.
    dist_rows = distribution_rows("all", seeds_with_mutations, failure_counts)
    for model in models:
        dist_rows.extend(distribution_rows(model, keys_where(model=model), failure_counts))

    dist_headers = ["model", "# of original prompts", "# of mutated prompts that the model complied with"]
    with open(DIST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(dist_headers)
        for row in dist_rows:
            writer.writerow([display_model(row["model"]), row["n_seeds"], row["n_failing_mutations"]])
    print(f"wrote {len(dist_rows)} rows -> {DIST_PATH}")
    for row in dist_rows:
        print(row)

    fuzzer_seeds = sorted({m["seed"] for m in mutated})
    lineage_counts = lineage_counts_by_seed(mutations_by_seed, seeds_with_mutations)
    row_table = table_rows(seeds_with_mutations, meta_by_seed, failure_counts, lineage_counts, fuzzer_seeds)

    zero_rows = [r for r in row_table if r["n_complying_total"] == 0]
    nonzero_rows = [r for r in row_table if r["n_complying_total"] > 0]

    table_keys = (
        ["category", "question_id", "model"]
        + [f"n_complying_seed_{s}" for s in fuzzer_seeds]
        + ["n_complying_total"]
    )
    table_headers = (
        ["category", "question_id", "model"]
        + [f"# of complying mutated prompts for seed {s}" for s in fuzzer_seeds]
        + ["# of complying prompts total"]
    )
    with open(TABLE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(table_headers)
        for row in nonzero_rows:
            writer.writerow(
                [display_model(row[k]) if k == "model" else row[k] for k in table_keys]
            )
        totals_row = ["", "", "Total"] + [
            sum(row[f"n_complying_seed_{s}"] for row in nonzero_rows) for s in fuzzer_seeds
        ] + [sum(row["n_complying_total"] for row in nonzero_rows)]
        writer.writerow(totals_row)

        writer.writerow([])
        writer.writerow(["model", "question_ids_with_zero_complying_mutations"])
        for model in models:
            ids = sorted(r["question_id"] for r in zero_rows if r["model"] == model)
            writer.writerow([display_model(model), ", ".join(str(i) for i in ids)])
    print(f"wrote {len(nonzero_rows)} detail rows + {len(models)}-model zero-compliance table -> {TABLE_PATH}")


if __name__ == "__main__":
    main()
