import argparse

import pandas as pd

#field config
JOIN_KEY = "claim_id"
EVIDENCE_LABEL = "label_refusal_evidence"
PROMPTS_COL = "majority_prompts"

# majority label that means a claim was decided refusal-testable
TESTABLE_LABELS = ("Yes",)
# votes whose reasons explain why a claim is not refusal-testable
NOT_TESTABLE_VOTES = ("No", "Unclear")

CLAIM_COLS = ["claim_id", "rq1_claim_id", "claim_text", "category", "section", "source_doc",
              EVIDENCE_LABEL, "refusal_evidence_votes"]


def split_prompts(cell):
    if pd.isna(cell) or not str(cell).strip():
        return []
    return [p.strip() for p in str(cell).split(",") if p.strip()]


def not_testable_reasons(row):
    # only claims decided not testable get reasons, and only the No/Unclear ones;
    # reasons are stored as '<vote>: <reason> || ...', one per annotator
    if row[EVIDENCE_LABEL] in TESTABLE_LABELS or pd.isna(row["reasons"]):
        return ""
    entries = [r.strip() for r in str(row["reasons"]).split(" || ")]
    return " || ".join(r for r in entries if r.split(":", 1)[0] in NOT_TESTABLE_VOTES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refusal-evidence", default="RQ2/outputs/refusal_evidence.csv")
    ap.add_argument("--claim-to-prompt", default="RQ2/outputs/claim_to_prompt.csv")
    ap.add_argument("--prompt-to-claims", default="RQ2/outputs/prompt_to_claims.csv")
    ap.add_argument("--output", default="RQ2/outputs/claims_overview.csv")
    args = ap.parse_args()

    evidence = pd.read_csv(args.refusal_evidence)
    mapping = pd.read_csv(args.claim_to_prompt)
    prompt_text = pd.read_csv(args.prompt_to_claims).set_index("question_id")["original_prompt"]

    claims = evidence.merge(mapping[[JOIN_KEY, PROMPTS_COL]], on=JOIN_KEY, how="left")
    if len(claims) != len(evidence):
        raise SystemExit("claim_to_prompt has duplicate claim ids")

    prompts = claims[PROMPTS_COL].map(split_prompts)
    unknown = sorted({p for ps in prompts for p in ps} - set(prompt_text.index))
    if unknown:
        print(f"  warning: no prompt text for {', '.join(unknown)}")

    result = claims[CLAIM_COLS].copy()
    result["mapped_prompts"] = prompts.map(",".join)
    result["num_mapped_prompts"] = prompts.map(len)
    result["mapped_prompt_texts"] = prompts.map(
        lambda ps: " || ".join(f"{p}: {prompt_text.get(p, '')}" for p in ps))
    result["not_testable_reasons"] = claims.apply(not_testable_reasons, axis=1)

    result.to_csv(args.output, index=False)

    testable = result[EVIDENCE_LABEL] == "Yes"
    mapped = result["num_mapped_prompts"] > 0
    print(f"wrote {len(result)} claims -> {args.output}")
    print(f"  refusal-testable    : {testable.sum()} ({(testable & mapped).sum()} with mapped prompts)")
    print(f"  not refusal-testable: {(~testable).sum()} ({(~testable & mapped).sum()} with mapped prompts)")


if __name__ == "__main__":
    main()
