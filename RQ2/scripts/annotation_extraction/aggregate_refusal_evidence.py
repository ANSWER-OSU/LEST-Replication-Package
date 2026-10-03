import argparse
import sys

import pandas as pd

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
sys.path.insert(0, "RQ2/scripts/annotation_extraction")
from aggregate_claims import NO_CONSENSUS, breakdown, majority  # noqa: E402
from aggregate_claim_to_prompt import RQ1_CLAIMS, claim_numbering, id_order  # noqa: E402

#field config
JOIN_KEY = "claim_id"
CLAIM_TEXT = "claim_text"
EVIDENCE_FIELD = "refusal_evidence"
REASON_FIELD = "reason"
CHOICES = ["Yes", "No", "Unclear"]

# claim-level metadata to carry through (constant per claim)
METADATA_COLS = ["category", "section", "source_doc"]


#per-claim collapse
def collapse_claim(group):
    evidence_votes = group[EVIDENCE_FIELD].tolist()
    label_evidence, _ = majority(evidence_votes)

    # one '<vote>: <reason>' entry per annotator, for auditing disagreements
    reasons = " || ".join(
        f"{v}: {str(r).strip()}"
        for v, r in zip(group[EVIDENCE_FIELD], group[REASON_FIELD]) if pd.notna(r)
    )

    record = {
        JOIN_KEY: group.name,
        CLAIM_TEXT: group[CLAIM_TEXT].iloc[0],
        f"label_{EVIDENCE_FIELD}": label_evidence,
        f"{EVIDENCE_FIELD}_agree": group[EVIDENCE_FIELD].dropna().nunique() <= 1,
        "num_annotators": len(group),
        f"{EVIDENCE_FIELD}_votes": breakdown(evidence_votes),
        "reasons": reasons,
    }
    for col in METADATA_COLS:
        if col in group.columns:
            record[col] = group[col].iloc[0]
    return pd.Series(record)


def aggregate(df):
    for col in METADATA_COLS:
        if col not in df.columns:
            continue
        bad = df.groupby(JOIN_KEY)[col].apply(lambda s: s.dropna().nunique() > 1)
        if bad.any():
            ids = bad[bad].index.tolist()
            print(f"  warning: '{col}' is not constant within claims {ids}; using first value")

    return (df.groupby(JOIN_KEY, sort=True)
              .apply(collapse_claim, include_groups=False)
              .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csvs", nargs="+", help="raw Label Studio export(s)")
    ap.add_argument("output_csv", help="path for the per-claim output")
    args = ap.parse_args()

    parts = []
    for path in args.input_csvs:
        df = pd.read_csv(path)
        required = {JOIN_KEY, CLAIM_TEXT, EVIDENCE_FIELD, REASON_FIELD}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{path}: missing required columns {sorted(missing)}")
        print(f"read {len(df)} annotations across {df[JOIN_KEY].nunique()} claims from {path}")
        parts.append(df)

    # several exports of the same claims would silently pool annotators, so flag it
    seen = set()
    for path, df in zip(args.input_csvs, parts):
        overlap = seen & set(df[JOIN_KEY])
        if overlap:
            print(f"  warning: {path} repeats claims {sorted(overlap)} from an earlier input; votes are pooled")
        seen |= set(df[JOIN_KEY])

    df = pd.concat(parts, ignore_index=True)
    if "annotation_id" in df.columns:
        dupes = df["annotation_id"].duplicated()
        if dupes.any():
            print(f"  warning: dropping {dupes.sum()} annotations exported more than once")
            df = df[~dupes]

    claims = aggregate(df)

    # same c1, c2, ... numbering as claim_to_prompt.csv; the RQ1 id is kept alongside
    numbering = claim_numbering()
    unknown = set(claims[JOIN_KEY]) - set(numbering)
    if unknown:
        raise SystemExit(f"claims {sorted(unknown)} are not testable claims in {RQ1_CLAIMS}")
    claims["rq1_claim_id"] = claims[JOIN_KEY]
    claims[JOIN_KEY] = claims[JOIN_KEY].map(numbering)
    claims = claims.sort_values(JOIN_KEY, key=lambda s: s.map(id_order)).reset_index(drop=True)
    missing = sorted(set(numbering.values()) - set(claims[JOIN_KEY]), key=id_order)
    if missing:
        print(f"  note: no annotations yet for {', '.join(missing)}")

    ordered = (
        [JOIN_KEY, "rq1_claim_id", CLAIM_TEXT, f"label_{EVIDENCE_FIELD}", f"{EVIDENCE_FIELD}_agree",
         "num_annotators", f"{EVIDENCE_FIELD}_votes"]
        + [c for c in METADATA_COLS if c in claims.columns]
        + ["reasons"]
    )
    claims = claims[ordered]
    claims.to_csv(args.output_csv, index=False)

    labels = claims[f"label_{EVIDENCE_FIELD}"]
    print(f"wrote {len(claims)} claims -> {args.output_csv}")
    print(f"  {EVIDENCE_FIELD}: agree {claims[f'{EVIDENCE_FIELD}_agree'].sum()}, "
          + ", ".join(f"{c} {(labels == c).sum()}" for c in CHOICES)
          + f", no consensus {(labels == NO_CONSENSUS).sum()}")


if __name__ == "__main__":
    main()
