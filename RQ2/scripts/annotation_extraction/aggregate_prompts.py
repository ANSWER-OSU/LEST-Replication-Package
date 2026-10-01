import argparse
import sys

import pandas as pd

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
from aggregate_claims import NO_CONSENSUS, breakdown, majority  # noqa: E402

#field config 
JOIN_KEY = "prompt_id"

# raw annotation fields; each becomes label_<field> / <field>_votes / <field>_agree
LABEL_FIELDS = ["grammatical", "preserves_category", "harmful"]

# prompt-level metadata to carry through (constant per prompt); `id` is the
# Label Studio task id, which is only kept from the first export
METADATA_COLS = [
    "id",
    "question_id",
    "seed",
    "generation",
    "category",
    "mutation",
    "original_prompt",
    "mutated_prompt",
]


#per-prompt collapse
def collapse_prompt(group, fields):
    record = {JOIN_KEY: group.name}
    for field in fields:
        votes = group[field].tolist()
        label, _ = majority(votes)
        record[f"label_{field}"] = label
        record[f"{field}_agree"] = group[field].dropna().nunique() <= 1
        record[f"{field}_votes"] = breakdown(votes)
    record["num_annotators"] = len(group)
    for col in METADATA_COLS:
        if col in group.columns:
            record[col] = group[col].iloc[0]
    return pd.Series(record)


def aggregate(df, fields):
    for col in METADATA_COLS:
        if col not in df.columns:
            continue
        bad = df.groupby(JOIN_KEY)[col].apply(lambda s: s.dropna().nunique() > 1)
        if bad.any():
            ids = bad[bad].index.tolist()
            print(f"  warning: '{col}' is not constant within prompts {ids}; using first value")

    return (df.groupby(JOIN_KEY, sort=True)
              .apply(lambda g: collapse_prompt(g, fields), include_groups=False)
              .reset_index(drop=True))


def merge_parts(parts):
    out = parts[0]
    for i, part in enumerate(parts[1:], 2):
        missing = set(out[JOIN_KEY]) - set(part[JOIN_KEY])
        extra = set(part[JOIN_KEY]) - set(out[JOIN_KEY])
        if missing or extra:
            print(f"  warning: input {i} lacks {len(missing)} prompts from earlier inputs "
                  f"and adds {len(extra)} new ones")

        # metadata should agree across exports (task id excepted)
        shared = [c for c in METADATA_COLS if c in out.columns and c in part.columns]
        both = out[[JOIN_KEY, *shared]].merge(part[[JOIN_KEY, *shared]], on=JOIN_KEY)
        for col in shared:
            if col == "id":
                continue
            n = (both[f"{col}_x"].astype(str) != both[f"{col}_y"].astype(str)).sum()
            if n:
                print(f"  warning: '{col}' differs between inputs for {n} prompts; keeping the earlier value")

        part = part.rename(columns={"num_annotators": "num_annotators_new"})
        out = out.merge(part, on=JOIN_KEY, how="outer", suffixes=("", "_new"))
        for col in shared:
            out[col] = out[col].fillna(out.pop(f"{col}_new"))

        new = out.pop("num_annotators_new")
        if (out["num_annotators"].notna() & new.notna() & (out["num_annotators"] != new)).any():
            print("  warning: annotator counts differ between inputs; num_annotators is the max")
        out["num_annotators"] = pd.concat([out["num_annotators"], new], axis=1).max(axis=1).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csvs", nargs="+", help="raw Label Studio export(s)")
    ap.add_argument("output_csv", help="path for the per-prompt output")
    args = ap.parse_args()

    parts, seen = [], []
    for path in args.input_csvs:
        df = pd.read_csv(path)
        fields = [f for f in LABEL_FIELDS if f in df.columns]
        if JOIN_KEY not in df.columns or not fields:
            raise SystemExit(f"{path}: needs '{JOIN_KEY}' and at least one of {LABEL_FIELDS}")
        dupes = [f for f in fields if f in seen]
        if dupes:
            raise SystemExit(f"{path}: fields {dupes} already come from an earlier input")
        seen += fields
        print(f"read {len(df)} annotations across {df[JOIN_KEY].nunique()} prompts "
              f"from {path} (fields: {', '.join(fields)})")
        parts.append(aggregate(df, fields))

    result = merge_parts(parts)
    fields = [f for f in LABEL_FIELDS if f in seen]
    result["all_agree"] = result[[f"{f}_agree" for f in fields]].fillna(False).all(axis=1)

    ordered = (
        [c for c in ["id", JOIN_KEY] if c in result.columns]
        + [f"label_{f}" for f in fields]
        + ["all_agree", "num_annotators"]
        + [f"{f}_agree" for f in fields]
        + [f"{f}_votes" for f in fields]
        + [c for c in METADATA_COLS if c != "id" and c in result.columns]
    )
    result = result[ordered]
    result.to_csv(args.output_csv, index=False)

    print(f"wrote {len(result)} prompts -> {args.output_csv}")
    print(f"  full agreement (all fields) : {result['all_agree'].sum()}")
    for field in fields:
        labels = result[f"label_{field}"]
        print(f"  {field:<19}: agree {result[f'{field}_agree'].sum():>3}, "
              f"Yes {(labels == 'Yes').sum():>3}, No {(labels == 'No').sum():>3}, "
              f"no consensus {(labels == NO_CONSENSUS).sum()}")


if __name__ == "__main__":
    main()
