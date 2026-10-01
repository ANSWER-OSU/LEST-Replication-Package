import csv
import sys
from collections import Counter

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
from fleiss_kappa import fleiss_kappa, parse_votes, report  # noqa: E402

LABEL_FIELDS = ["grammatical", "preserves_category", "harmful"]
CATEGORIES = ["Yes", "No"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "annotated_prompts.csv"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    fields = [fld for fld in LABEL_FIELDS if f"{fld}_votes" in reader.fieldnames]

    rater_counts = Counter(sum(parse_votes(r.get(f"{fields[0]}_votes", "")).values()) for r in rows)

    print("=" * 64)
    print("FLEISS' KAPPA  -  prompt mutation annotation")
    print("=" * 64)
    print(f"Total items                 : {len(rows)}")
    print(f"Raters per item             : "
          + ", ".join(f"{n} raters x {c} items" for n, c in sorted(rater_counts.items(), reverse=True)))

    for i, field in enumerate(fields, 1):
        counts = [parse_votes(r.get(f"{field}_votes", "")) for r in rows]
        res = fleiss_kappa(counts, CATEGORIES)
        report(f"{i}. {field} agreement (binary, all items)", res, CATEGORIES)


if __name__ == "__main__":
    main()
