import csv
import glob
import json
import os
from collections import Counter, defaultdict

from tested_prompt_filter import dataset_category, is_tested_prompt

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
CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "evaluation",
    "filtered_harmful",
    "response_transitions.csv",
)
TRANSITIONS = ("R->R", "R->C", "C->R", "C->C")


def load_ensemble(dir_path, pattern):
    records = []
    for path in sorted(glob.glob(os.path.join(dir_path, pattern))):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            r["_dataset_category"] = dataset_category(r, path)
        records.extend(rows)
    return records


def short(label):
    return "R" if label == "refusal" else "C"


def count_transitions():
    baseline = {
        (r["_dataset_category"], r["model"], r["question_id"]): r.get("ensemble_refusal")
        for r in load_ensemble(ORIGINAL_DIR, "*_original_prompt_result_*_wildguard_result_ensemble.json")
    }

    counts = defaultdict(Counter)
    skipped = 0
    for r in load_ensemble(MUTATED_DIR, "*_mutated_prompt_result_*_wildguard_result_ensemble.json"):
        if not is_tested_prompt(r, r["_dataset_category"]):
            continue
        before = baseline.get((r["_dataset_category"], r["model"], r["question_id"]))
        after = r.get("ensemble_refusal")
        if before is None or after is None:
            skipped += 1
            continue
        counts[(r["model"], r["seed"])][f"{short(before)}->{short(after)}"] += 1
    return counts, skipped


def pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


def table_rows(counts):
    rows = []
    for model in sorted({m for m, _ in counts}):
        seeds = sorted(s for m, s in counts if m == model)
        total = Counter()
        for seed in seeds:
            total.update(counts[(model, seed)])
            rows.append((model, str(seed), counts[(model, seed)]))
        rows.append((model, "total", total))
    return rows


def csv_row(model, seed, c):
    refused = c["R->R"] + c["R->C"]
    complied = c["C->R"] + c["C->C"]
    return {
        "model": model,
        "seed": seed,
        "R->R": c["R->R"],
        "R->C": c["R->C"],
        "C->R": c["C->R"],
        "C->C": c["C->C"],
        "total": refused + complied,
        "R->R_pct": pct(c["R->R"], refused),
        "R->C_pct": pct(c["R->C"], refused),
        "C->R_pct": pct(c["C->R"], complied),
        "C->C_pct": pct(c["C->C"], complied),
    }


def main():
    counts, skipped = count_transitions()
    rows = table_rows(counts)

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(csv_row("", "", Counter()).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, seed, c in rows:
            writer.writerow(csv_row(model, seed, c))

    if skipped:
        print(f"{skipped} tested mutations skipped (missing original or mutated ensemble label)")
    for model, seed, c in rows:
        print(f"{model:<16} {seed:>5}  " + "  ".join(f"{t}={c[t]}" for t in TRANSITIONS))
    print(f"saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
