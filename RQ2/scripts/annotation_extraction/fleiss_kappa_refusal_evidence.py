import csv
import sys
from collections import Counter

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
from fleiss_kappa import fleiss_kappa, parse_votes, report, result_row, write_csv  # noqa: E402

FIELD = "refusal_evidence"
CATEGORIES = ["Yes", "No", "Unclear"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "RQ2/outputs/refusal_evidence.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "RQ2/outputs/refusal_evidence_fleiss_kappa.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts = [parse_votes(r[f"{FIELD}_votes"]) for r in rows]
    rater_counts = Counter(sum(c.values()) for c in counts)

    print("=" * 64)
    print("FLEISS' KAPPA  -  refusal oracle annotation")
    print("=" * 64)
    print(f"Total claims                : {len(rows)}")
    print(f"Raters per claim            : "
          + ", ".join(f"{n} raters x {c} claims" for n, c in sorted(rater_counts.items(), reverse=True)))

    title = f"1. {FIELD} agreement (3-class, all claims)"
    res = fleiss_kappa(counts, CATEGORIES)
    report(title, res, CATEGORIES)
    write_csv(output_path, [result_row(title, res, CATEGORIES)])


if __name__ == "__main__":
    main()
