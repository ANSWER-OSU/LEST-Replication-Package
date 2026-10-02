import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
from aggregate_claims import NO_CONSENSUS, breakdown, majority  # noqa: E402

#field config
JOIN_KEY = "claim_id"
CLAIM_TEXT = "claim_text"
ORACLE_FIELD = "refusal_oracle"
PROMPTS_FIELD = "mapped_prompts"

# claim-level metadata to carry through (constant per claim)
METADATA_COLS = ["category", "section", "source_doc", "num_candidate_prompts"]

# the 97 seed prompts, listed in full in the per-prompt output
PROMPTS_DIR = Path("RQ2/data/evaluation/original_prompt_experiment/inputs")
CATEGORIES = ["fairness_bias", "harmful_violent_content", "mental_health_self_harm"]


#parsing
def parse_prompts(cell):
    """Question ids ('q51') selected in one annotation.

    Label Studio exports several selections as '{"choices": [...]}', a single
    selection as the bare choice string, and no selection as empty. Each
    choice is '<question id>: <prompt text>'.
    """
    if pd.isna(cell) or not str(cell).strip():
        return []
    cell = str(cell)
    choices = json.loads(cell)["choices"] if cell.startswith("{") else [cell]
    return [c.split(":", 1)[0].strip() for c in choices]


def qid_order(qid):
    return int(qid.lstrip("q"))


def tally(counter):
    """'q51:3,q52:1' ordered by votes, then question id."""
    items = sorted(counter.items(), key=lambda kv: (-kv[1], qid_order(kv[0]) if kv[0].startswith("q") else kv[0]))
    return ",".join(f"{k}:{n}" for k, n in items)


#per-claim collapse
def collapse_claim(group):
    oracle_votes = group[ORACLE_FIELD].tolist()
    label_oracle, _ = majority(oracle_votes)

    n = len(group)
    prompt_votes = Counter(q for cell in group[PROMPTS_FIELD] for q in parse_prompts(cell))
    majority_prompts = sorted((q for q, v in prompt_votes.items() if v > n / 2), key=qid_order)

    record = {
        JOIN_KEY: group.name,
        CLAIM_TEXT: group[CLAIM_TEXT].iloc[0],
        f"label_{ORACLE_FIELD}": label_oracle,
        f"{ORACLE_FIELD}_agree": group[ORACLE_FIELD].dropna().nunique() <= 1,
        "num_annotators": n,
        f"{ORACLE_FIELD}_votes": breakdown(oracle_votes),
        "prompt_votes": tally(prompt_votes),
        "num_prompts_voted": len(prompt_votes),
        "majority_prompts": ",".join(majority_prompts),
        "num_majority_prompts": len(majority_prompts),
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


#per-prompt view
def load_seed_prompts():
    rows = []
    for cat in CATEGORIES:
        with open(PROMPTS_DIR / cat / f"{cat}_original_prompts.json", encoding="utf-8") as f:
            for p in json.load(f):
                rows.append({
                    "question_id": f"q{p['question_id']}",
                    "category": cat,
                    "original_prompt": p["original_prompt"],
                })
    return pd.DataFrame(rows)


def prompts_to_claims(claims):
    """One row per seed prompt, listing the claims annotators linked it to."""
    seeds = load_seed_prompts()
    votes, majority_claims = {}, {}
    for _, c in claims.iterrows():
        for item in filter(None, c["prompt_votes"].split(",")):
            qid, v = item.split(":")
            votes.setdefault(qid, {})[c[JOIN_KEY]] = int(v)
        for qid in filter(None, c["majority_prompts"].split(",")):
            majority_claims.setdefault(qid, []).append(c[JOIN_KEY])

    unknown = set(votes) - set(seeds["question_id"])
    if unknown:
        print(f"  warning: annotations reference unknown prompts {sorted(unknown)}")

    def claim_tally(qid):
        items = sorted(votes.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        return ",".join(f"{k}:{n}" for k, n in items)

    seeds["claim_votes"] = seeds["question_id"].map(claim_tally)
    seeds["num_claims_voted"] = seeds["question_id"].map(lambda q: len(votes.get(q, {})))
    seeds["majority_claims"] = seeds["question_id"].map(
        lambda q: ",".join(str(c) for c in sorted(majority_claims.get(q, []))))
    seeds["num_majority_claims"] = seeds["question_id"].map(lambda q: len(majority_claims.get(q, [])))
    return seeds[["question_id", "category", "majority_claims", "num_majority_claims",
                  "claim_votes", "num_claims_voted", "original_prompt"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csvs", nargs="+", help="raw Label Studio export(s), each covering different claims")
    ap.add_argument("claims_csv", help="path for the per-claim output")
    ap.add_argument("prompts_csv", help="path for the per-prompt output")
    args = ap.parse_args()

    parts = []
    for path in args.input_csvs:
        df = pd.read_csv(path)
        required = {JOIN_KEY, CLAIM_TEXT, ORACLE_FIELD, PROMPTS_FIELD}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{path}: missing required columns {sorted(missing)}")
        print(f"read {len(df)} annotations across {df[JOIN_KEY].nunique()} claims from {path}")
        parts.append(df)

    # inputs are separate batches of claims; one claim spread over two
    # exports would silently pool annotators, so flag it
    seen = set()
    for path, df in zip(args.input_csvs, parts):
        overlap = seen & set(df[JOIN_KEY])
        if overlap:
            print(f"  warning: {path} repeats claims {sorted(overlap)} from an earlier input; votes are pooled")
        seen |= set(df[JOIN_KEY])

    claims = aggregate(pd.concat(parts, ignore_index=True))
    ordered = (
        [JOIN_KEY, CLAIM_TEXT, f"label_{ORACLE_FIELD}", f"{ORACLE_FIELD}_agree", "num_annotators",
         f"{ORACLE_FIELD}_votes", "majority_prompts", "num_majority_prompts",
         "prompt_votes", "num_prompts_voted"]
        + [c for c in METADATA_COLS if c in claims.columns]
    )
    claims = claims[ordered]
    claims.to_csv(args.claims_csv, index=False)

    prompts = prompts_to_claims(claims)
    prompts.to_csv(args.prompts_csv, index=False)

    labels = claims[f"label_{ORACLE_FIELD}"]
    print(f"wrote {len(claims)} claims -> {args.claims_csv}")
    print(f"  {ORACLE_FIELD}: agree {claims[f'{ORACLE_FIELD}_agree'].sum()}, "
          f"Yes {(labels == 'Yes').sum()}, No {(labels == 'No').sum()}, "
          f"no consensus {(labels == NO_CONSENSUS).sum()}")
    print(f"  claims with no prompt votes : {(claims['num_prompts_voted'] == 0).sum()}")
    print(f"  claims with no majority prompt : {(claims['num_majority_prompts'] == 0).sum()}")
    print(f"wrote {len(prompts)} prompts -> {args.prompts_csv}")
    print(f"  prompts with any claim vote : {(prompts['num_claims_voted'] > 0).sum()}")
    print(f"  prompts with a majority claim : {(prompts['num_majority_claims'] > 0).sum()}")


if __name__ == "__main__":
    main()
