import csv
import sys
from collections import Counter

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
from fleiss_kappa import fleiss_kappa, parse_votes, report, result_row, write_csv  # noqa: E402

INTENT_FIELD = "preserves_intent"
COMPLIANT_FIELD = "actually_compliant"
CHOICES = ["Yes", "No", "Unclear"]

# actually_compliant is only asked when the annotator answered Yes or Unclear
# to preserves_intent, so each annotator gives exactly one collapsed label
NOT_PRESERVED = "Intent not preserved"
COLLAPSED = [NOT_PRESERVED] + [f"Compliant:{c}" for c in CHOICES]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "RQ2/outputs/annotated_refuse_comply.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "RQ2/outputs/refuse_comply_fleiss_kappa.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    intent_counts, compliant_counts, collapsed_counts = [], [], []
    for r in rows:
        intent = parse_votes(r[f"{INTENT_FIELD}_votes"])
        compliant = parse_votes(r[f"{COMPLIANT_FIELD}_votes"])
        if sum(compliant.values()) != sum(intent.values()) - intent.get("No", 0):
            raise SystemExit(f"case {r['case_id']}: {COMPLIANT_FIELD} votes don't match "
                             f"the annotators who said {INTENT_FIELD} was Yes/Unclear")
        intent_counts.append(intent)
        if compliant:
            compliant_counts.append(compliant)
        collapsed = {f"Compliant:{c}": n for c, n in compliant.items()}
        collapsed[NOT_PRESERVED] = intent.get("No", 0)
        collapsed_counts.append(collapsed)

    rater_counts = Counter(sum(c.values()) for c in intent_counts)

    print("=" * 64)
    print("FLEISS' KAPPA  -  refuse-to-comply annotation")
    print("=" * 64)
    print(f"Total cases                 : {len(rows)}")
    print(f"Raters per case             : "
          + ", ".join(f"{n} raters x {c} cases" for n, c in sorted(rater_counts.items(), reverse=True)))
    print(f"Compliance subset           : all cases with {COMPLIANT_FIELD} votes (ragged)")

    measures = [
        (f"1. {INTENT_FIELD} agreement (3-class, all cases)", intent_counts, CHOICES),
        (f"2. {COMPLIANT_FIELD} agreement (3-class, all cases with compliance votes)",
         compliant_counts, CHOICES),
        ("3. TOTAL agreement (collapsed 4-class)", collapsed_counts, COLLAPSED),
    ]
    out_rows = []
    for title, counts, categories in measures:
        res = fleiss_kappa(counts, categories)
        report(title, res, categories)
        out_rows.append(result_row(title, res, categories))
    write_csv(output_path, out_rows)


if __name__ == "__main__":
    main()
