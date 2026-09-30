import argparse
from collections import Counter

import pandas as pd

#field config
CLAIM_ID = "id"
CLAIM_TEXT = "claim_text"

# raw annotation fields -> output label names
SOURCE_LABEL = "is_claim"
TESTABLE_LABEL = "testable"

# claim-level metadata to carry through (constant per id)
METADATA_COLS = [
    "category",
    "section",
    "source_doc",
    "contains_numeric",
    "has_hedged_quantifier",
    "has_universal_quantifier",
    "model_is_subject",
    "modal_verb_type",
    "tense",
    "verb_type",
]

# the value of is_claim that means "this is a claim"
CLAIM_POSITIVE = "Claim"
NO_CONSENSUS = "NO_CONSENSUS"


#voting
def majority(votes):
    """Majority winner over non-null votes.

    Returns (label, is_tie). With no votes, returns (None, False).
    A tie between the top two distinct values returns (NO_CONSENSUS, True).
    """
    votes = [v for v in votes if pd.notna(v)]
    if not votes:
        return None, False
    ranked = Counter(votes).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return NO_CONSENSUS, True
    return ranked[0][0], False


def breakdown(votes):
    """Compact vote tally string, e.g. 'Claim:2|Not a claim:1'. Useful for audit."""
    counts = Counter(v for v in votes if pd.notna(v))
    return ",".join(f"{k}:{n}" for k, n in counts.most_common())


#per-claim collapse 
def collapse_claim(group):
    claim_votes = group[SOURCE_LABEL].tolist()
    label_claim, _ = majority(claim_votes)

    # testable only matters among annotators who said "Claim"
    testable_votes = group.loc[group[SOURCE_LABEL] == CLAIM_POSITIVE, TESTABLE_LABEL].tolist()
    if label_claim == CLAIM_POSITIVE:
        label_testable, _ = majority(testable_votes)
    else:
        label_testable = ""  # not applicable when the claim itself isn't a claim

    # full agreement: every annotator produced the identical (is_claim, testable) pair.
    # testable is null-coupled to is_claim, so this captures disagreement on either field.
    pairs = set(zip(group[SOURCE_LABEL], group[TESTABLE_LABEL].astype(object).where(group[TESTABLE_LABEL].notna(), None)))
    all_agree = len(pairs) == 1

    record = {
        CLAIM_ID: group.name,
        CLAIM_TEXT: group[CLAIM_TEXT].iloc[0],
        "label_claim": label_claim,
        "label_testable": label_testable,
        "all_agree": all_agree,
        "num_annotators": len(group),
        "claim_votes": breakdown(claim_votes),
        "testable_votes": breakdown(testable_votes),
    }
    for col in METADATA_COLS:
        if col in group.columns:
            record[col] = group[col].iloc[0]
    return pd.Series(record)


def aggregate(df):
    # sanity: metadata columns really should be constant per claim
    for col in METADATA_COLS:
        if col not in df.columns:
            continue
        bad = df.groupby(CLAIM_ID)[col].apply(lambda s: s.dropna().nunique() > 1)
        if bad.any():
            ids = bad[bad].index.tolist()
            print(f"  warning: '{col}' is not constant within claim ids {ids}; using first value")

    out = df.groupby(CLAIM_ID, sort=True).apply(collapse_claim).reset_index(drop=True)

    ordered = (
        [CLAIM_ID, CLAIM_TEXT, "label_claim", "label_testable", "all_agree",
         "num_annotators", "claim_votes", "testable_votes"]
        + [c for c in METADATA_COLS if c in out.columns]
    )
    return out[ordered]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="raw Label Studio export")
    ap.add_argument("output_csv", help="path for the per-claim output")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    required = {CLAIM_ID, CLAIM_TEXT, SOURCE_LABEL, TESTABLE_LABEL}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"input is missing required columns: {sorted(missing)}")

    print(f"read {len(df)} annotations across {df[CLAIM_ID].nunique()} claims")
    result = aggregate(df)
    result.to_csv(args.output_csv, index=False)

    disagreements = (~result["all_agree"]).sum()
    print(f"wrote {len(result)} claims -> {args.output_csv}")
    print(f"  full agreement : {result['all_agree'].sum()}")
    print(f"  disagreement   : {disagreements}")
    print(f"  no consensus on label_claim : {(result['label_claim'] == NO_CONSENSUS).sum()}")


if __name__ == "__main__":
    main()
