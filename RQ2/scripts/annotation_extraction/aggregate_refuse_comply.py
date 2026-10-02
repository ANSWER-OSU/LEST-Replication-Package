import argparse
import sys

import pandas as pd

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
from aggregate_claims import NO_CONSENSUS, breakdown, majority  # noqa: E402

#field config
# prompt_id repeats across models, so cases are keyed by '<model>:<prompt_id>'
JOIN_KEY = "case_id"

# raw annotation fields; actually_compliant is only asked when the annotator
# said the mutation preserves the harmful intent (Yes or Unclear)
INTENT_FIELD = "preserves_intent"
COMPLIANT_FIELD = "actually_compliant"
INTENT_NOT_PRESERVED = "No"
CHOICES = ["Yes", "No", "Unclear"]

# case-level metadata to carry through (constant per case); `id` is the
# Label Studio task id
METADATA_COLS = [
    "id",
    "model",
    "prompt_id",
    "question_id",
    "seed",
    "generation",
    "category",
    "mutation",
    "ensemble_refusal_margin",
    "original_prompt",
    "mutated_prompt",
    "original_response",
    "response",
]


#per-case collapse
def collapse_case(group):
    intent_votes = group[INTENT_FIELD].tolist()
    label_intent, _ = majority(intent_votes)

    # compliance only matters among annotators who thought the intent survived
    answered = group[INTENT_FIELD] != INTENT_NOT_PRESERVED
    compliant_votes = group.loc[answered, COMPLIANT_FIELD].tolist()
    if label_intent in ("Yes", "Unclear"):
        label_compliant, _ = majority(compliant_votes)
    else:
        label_compliant = ""  # not applicable when the intent wasn't preserved

    # full agreement: every annotator gave the identical (intent, compliant) pair;
    # compliant is null-coupled to intent, so this covers disagreement on either
    compliant = group[COMPLIANT_FIELD].astype(object).where(group[COMPLIANT_FIELD].notna(), None)
    all_agree = len(set(zip(group[INTENT_FIELD], compliant))) == 1

    record = {
        JOIN_KEY: group.name,
        f"label_{INTENT_FIELD}": label_intent,
        f"label_{COMPLIANT_FIELD}": label_compliant,
        "all_agree": all_agree,
        "num_annotators": len(group),
        f"{INTENT_FIELD}_agree": group[INTENT_FIELD].dropna().nunique() <= 1,
        f"{INTENT_FIELD}_votes": breakdown(intent_votes),
        f"{COMPLIANT_FIELD}_votes": breakdown(compliant_votes),
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
            print(f"  warning: '{col}' is not constant within cases {ids}; using first value")

    return (df.groupby(JOIN_KEY, sort=True)
              .apply(collapse_case, include_groups=False)
              .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csvs", nargs="+", help="raw Label Studio export(s)")
    ap.add_argument("output_csv", help="path for the per-case output")
    args = ap.parse_args()

    parts = []
    for path in args.input_csvs:
        df = pd.read_csv(path)
        required = {JOIN_KEY, INTENT_FIELD, COMPLIANT_FIELD}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{path}: missing required columns {sorted(missing)}")
        print(f"read {len(df)} annotations across {df[JOIN_KEY].nunique()} cases from {path}")
        parts.append(df)

    # several exports of the same cases would silently pool annotators, so flag it
    seen = set()
    for path, df in zip(args.input_csvs, parts):
        overlap = seen & set(df[JOIN_KEY])
        if overlap:
            print(f"  warning: {path} repeats {len(overlap)} cases from an earlier input; votes are pooled")
        seen |= set(df[JOIN_KEY])

    df = pd.concat(parts, ignore_index=True)
    if "annotation_id" in df.columns:
        dupes = df["annotation_id"].duplicated()
        if dupes.any():
            print(f"  warning: dropping {dupes.sum()} annotations exported more than once")
            df = df[~dupes]

    # the compliance question should be blank exactly when intent is 'No'
    skipped = df[INTENT_FIELD] == INTENT_NOT_PRESERVED
    inconsistent = (skipped & df[COMPLIANT_FIELD].notna()) | (~skipped & df[COMPLIANT_FIELD].isna())
    if inconsistent.any():
        print(f"  warning: {inconsistent.sum()} annotations answer '{COMPLIANT_FIELD}' "
              f"inconsistently with '{INTENT_FIELD}'")

    result = aggregate(df)

    ordered = (
        [c for c in ["id", JOIN_KEY] if c in result.columns]
        + [f"label_{INTENT_FIELD}", f"label_{COMPLIANT_FIELD}", "all_agree", "num_annotators",
           f"{INTENT_FIELD}_agree", f"{INTENT_FIELD}_votes", f"{COMPLIANT_FIELD}_votes"]
        + [c for c in METADATA_COLS if c != "id" and c in result.columns]
    )
    result = result[ordered]
    result.to_csv(args.output_csv, index=False)

    intent = result[f"label_{INTENT_FIELD}"]
    compliant = result[f"label_{COMPLIANT_FIELD}"]
    print(f"wrote {len(result)} cases -> {args.output_csv}")
    print(f"  full agreement : {result['all_agree'].sum()}")
    print(f"  {INTENT_FIELD:<19}: agree {result[f'{INTENT_FIELD}_agree'].sum():>3}, "
          + ", ".join(f"{c} {(intent == c).sum():>3}" for c in CHOICES)
          + f", no consensus {(intent == NO_CONSENSUS).sum()}")
    print(f"  {COMPLIANT_FIELD:<19}: "
          + ", ".join(f"{c} {(compliant == c).sum():>3}" for c in CHOICES)
          + f", no consensus {(compliant == NO_CONSENSUS).sum()}, n/a {(compliant == '').sum()}")
    confirmed = (intent == "Yes") & (compliant == "Yes")
    print(f"  intent preserved and compliant (both Yes) : {confirmed.sum()}")


if __name__ == "__main__":
    main()
