import csv
import sys
from collections import Counter

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
from fleiss_kappa import fleiss_kappa, report, result_row, selection_items, write_csv  # noqa: E402

CATEGORIES = ["Selected", "Not selected"]


def claim_items(row):
    """Every candidate seed prompt (same category as the claim) is one binary item."""
    return selection_items(row["prompt_votes"], int(row["num_annotators"]), int(row["num_candidate_prompts"]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "RQ2/outputs/claim_to_prompt.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "RQ2/outputs/claim_to_prompt_fleiss_kappa.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rater_counts = Counter(int(r["num_annotators"]) for r in rows)
    categories = list(dict.fromkeys(r["category"] for r in rows))

    print("=" * 64)
    print("FLEISS' KAPPA  -  claim to seed prompt mapping annotation")
    print("=" * 64)
    print(f"Total claims                : {len(rows)}")
    print(f"Raters per claim            : "
          + ", ".join(f"{n} raters x {c} claims" for n, c in sorted(rater_counts.items(), reverse=True)))
    print("Items                       : one per (claim, candidate prompt), Selected / Not selected")

    measures = [("1. prompt selection agreement (all claims)", rows)]
    measures += [(f"{i}. prompt selection agreement ({c})", [r for r in rows if r["category"] == c])
                 for i, c in enumerate(categories, 2)]

    out_rows = []
    for title, subset in measures:
        counts = [item for r in subset for item in claim_items(r)]
        res = fleiss_kappa(counts, CATEGORIES)
        report(title, res, CATEGORIES)
        out_rows.append(result_row(title, res, CATEGORIES))
    write_csv(output_path, out_rows)


if __name__ == "__main__":
    main()
