import argparse

import pandas as pd

#field config
JOIN_KEY = "claim_id"
EVIDENCE_LABEL = "label_refusal_evidence"
PROMPTS_COL = "majority_prompts"

# a claim is only tested against the model its source doc describes
SOURCE_DOC_TO_MODEL = {
    "claude": "claude-opus-4-8",
    "gpt5": "gpt-5.5",
}


def split_prompts(cell):
    if pd.isna(cell) or not str(cell).strip():
        return []
    return [p.strip() for p in str(cell).split(",") if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refusal-evidence", default="RQ2/outputs/refusal_evidence.csv")
    ap.add_argument("--claim-to-prompt", default="RQ2/outputs/claim_to_prompt.csv")
    ap.add_argument("--refuse-comply", default="RQ2/outputs/annotated_refuse_comply.csv")
    ap.add_argument("--output", help="optional path for the per-claim breakdown")
    args = ap.parse_args()

    evidence = pd.read_csv(args.refusal_evidence)
    mapping = pd.read_csv(args.claim_to_prompt)
    cases = pd.read_csv(args.refuse_comply)

    unknown = set(evidence["source_doc"]) - set(SOURCE_DOC_TO_MODEL)
    if unknown:
        raise SystemExit(f"no model configured for source_doc {sorted(unknown)}")

    # a case counts as complied when the mutation kept the harmful intent and the
    # model actually complied (same definition as aggregate_refuse_comply.py)
    cases["question"] = "q" + cases["question_id"].astype(str)
    complied = cases[(cases["label_preserves_intent"] == "Yes")
                     & (cases["label_actually_compliant"] == "Yes")]

    testable = evidence[evidence[EVIDENCE_LABEL] == "Yes"]
    claims = testable.merge(mapping[[JOIN_KEY, PROMPTS_COL]], on=JOIN_KEY, how="left")

    rows = []
    for _, claim in claims.iterrows():
        model = SOURCE_DOC_TO_MODEL[claim["source_doc"]]
        prompts = split_prompts(claim[PROMPTS_COL])
        hits = complied[(complied["model"] == model) & complied["question"].isin(prompts)]
        rows.append({
            JOIN_KEY: claim[JOIN_KEY],
            "claim_text": claim["claim_text"],
            "source_doc": claim["source_doc"],
            "model": model,
            "num_mapped_prompts": len(prompts),
            "num_annotated_cases": int(((cases["model"] == model) & cases["question"].isin(prompts)).sum()),
            "num_complied_cases": len(hits),
            "complied_case_ids": ",".join(hits["case_id"]),
            "has_complied": len(hits) > 0,
        })
    result = pd.DataFrame(rows)

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"wrote {len(result)} claims -> {args.output}")

    mapped = result[result["num_mapped_prompts"] > 0]
    print(result.drop(columns=["claim_text", "complied_case_ids"]).to_string(index=False))
    print()
    print(f"refusal-testable claims          : {len(result)}")
    print(f"  with mapped prompts            : {len(mapped)}")
    print(f"  with a same-model complied case: {mapped['has_complied'].sum()}")
    for doc, grp in mapped.groupby("source_doc"):
        print(f"    {doc:<6}: {grp['has_complied'].sum()} / {len(grp)}")


if __name__ == "__main__":
    main()
