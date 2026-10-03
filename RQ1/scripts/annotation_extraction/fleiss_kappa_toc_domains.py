import csv
import sys
from collections import Counter

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
from fleiss_kappa import fleiss_kappa, report, result_row, selection_items, write_csv  # noqa: E402

CATEGORIES = ["Selected", "Not selected"]


def task_items(row):
    """Every ToC section of the card is one binary item for this domain."""
    return selection_items(row["section_votes"], int(row["num_annotators"]), int(row["num_sections"]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "RQ1/outputs/toc_domains.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "RQ1/outputs/toc_domains_fleiss_kappa.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rater_counts = Counter(int(r["num_annotators"]) for r in rows)
    domains = list(dict.fromkeys(r["domain"] for r in rows))

    print("=" * 64)
    print("FLEISS' KAPPA  -  ToC section to safety domain annotation")
    print("=" * 64)
    print(f"Total tasks (domain x card) : {len(rows)}")
    print(f"Raters per task             : "
          + ", ".join(f"{n} raters x {c} tasks" for n, c in sorted(rater_counts.items(), reverse=True)))
    print("Items                       : one per (task, ToC section), Selected / Not selected")

    measures = [("1. section selection agreement (all domains)", rows)]
    measures += [(f"{i}. section selection agreement ({d})", [r for r in rows if r["domain"] == d])
                 for i, d in enumerate(domains, 2)]

    out_rows = []
    for title, subset in measures:
        counts = [item for r in subset for item in task_items(r)]
        res = fleiss_kappa(counts, CATEGORIES)
        report(title, res, CATEGORIES)
        out_rows.append(result_row(title, res, CATEGORIES))
    write_csv(output_path, out_rows)


if __name__ == "__main__":
    main()
