import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

#field config
JOIN_KEY = "claim_id"
CLAIM_TEXT = "claim_text"
PROMPTS_FIELD = "mapped_prompts"

# claim-level metadata to carry through (constant per claim)
METADATA_COLS = ["category", "section", "source_doc", "num_candidate_prompts"]

# testable claims, numbered c1, c2, ... in RQ1 id order for the outputs (the
# same set the Label Studio tasks were built from); the RQ1 id is kept alongside
RQ1_CLAIMS = Path("RQ1/outputs/annotated_claims.csv")

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


def id_order(prefixed_id):
    """Numeric sort key for 'q51' / 'c7'."""
    return int(prefixed_id[1:])


def tally(counter):
    """'q51:3,q52:1' ordered by votes, then id."""
    items = sorted(counter.items(), key=lambda kv: (-kv[1], id_order(kv[0])))
    return ",".join(f"{k}:{n}" for k, n in items)


def claim_numbering():
    claims = pd.read_csv(RQ1_CLAIMS)
    ids = sorted(claims.loc[claims["label_testable"] == "Testable", "id"])
    return {cid: f"c{i}" for i, cid in enumerate(ids, 1)}


#per-claim collapse
def collapse_claim(group):
    n = len(group)
    prompt_votes = Counter(q for cell in group[PROMPTS_FIELD] for q in parse_prompts(cell))
    majority_prompts = sorted((q for q, v in prompt_votes.items() if v > n / 2), key=id_order)

    record = {
        JOIN_KEY: group.name,
        CLAIM_TEXT: group[CLAIM_TEXT].iloc[0],
        "num_annotators": n,
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

    seeds["claim_votes"] = seeds["question_id"].map(lambda q: tally(votes.get(q, {})))
    seeds["num_claims_voted"] = seeds["question_id"].map(lambda q: len(votes.get(q, {})))
    seeds["majority_claims"] = seeds["question_id"].map(
        lambda q: ",".join(sorted(majority_claims.get(q, []), key=id_order)))
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
        required = {JOIN_KEY, CLAIM_TEXT, PROMPTS_FIELD}
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
        [JOIN_KEY, "rq1_claim_id", CLAIM_TEXT, "num_annotators", "majority_prompts", "num_majority_prompts",
         "prompt_votes", "num_prompts_voted"]
        + [c for c in METADATA_COLS if c in claims.columns]
    )
    claims = claims[ordered]
    claims.to_csv(args.claims_csv, index=False)

    prompts = prompts_to_claims(claims)
    prompts.to_csv(args.prompts_csv, index=False)

    print(f"wrote {len(claims)} claims -> {args.claims_csv}")
    print(f"  claims with no prompt votes : {(claims['num_prompts_voted'] == 0).sum()}")
    print(f"  claims with no majority prompt : {(claims['num_majority_prompts'] == 0).sum()}")
    print(f"wrote {len(prompts)} prompts -> {args.prompts_csv}")
    print(f"  prompts with any claim vote : {(prompts['num_claims_voted'] > 0).sum()}")
    print(f"  prompts with a majority claim : {(prompts['num_majority_claims'] > 0).sum()}")


if __name__ == "__main__":
    main()
