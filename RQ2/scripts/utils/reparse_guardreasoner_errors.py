import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "judge_ensemble"))
from ensemble_io import load_json, save_json  # noqa: E402
from guardreasoner_schema import JUDGE_DIR  # noqa: E402
from run_guardreasoner import parse_guardreasoner  # noqa: E402


LABEL_KEYS = (
    "guardreasoner_refusal",
    "guardreasoner_prompt_harmfulness",
    "guardreasoner_response_harmfulness",
)


def reparse_file(path, reparse_all=False):
    records = load_json(path)
    eligible_errors = (None, "parse_failed") if reparse_all else ("parse_failed",)

    n_failed = 0
    n_recovered = 0
    n_changed = 0
    for r in records:
        if r.get("guardreasoner_error") not in eligible_errors or not r.get("guardreasoner_reasoning"):
            continue
        n_failed += r.get("guardreasoner_error") == "parse_failed"
        before = tuple(r.get(k) for k in LABEL_KEYS)
        refusal, prompt_harm, response_harm = parse_guardreasoner(r["guardreasoner_reasoning"])
        r["guardreasoner_refusal"] = refusal
        r["guardreasoner_prompt_harmfulness"] = prompt_harm
        r["guardreasoner_response_harmfulness"] = response_harm
        if r.get("guardreasoner_error") == "parse_failed" and refusal is not None:
            n_recovered += 1
        r["guardreasoner_error"] = None if refusal is not None else "parse_failed"
        n_changed += before != (refusal, prompt_harm, response_harm)

    if n_changed:
        save_json(path, records)

    return n_failed, n_recovered, n_changed


def main():
    parser = argparse.ArgumentParser(description="Re-parse GuardReasoner's stored reasoning text for rows that failed to parse")
    parser.add_argument(
        "--all",
        action="store_true",
        help="re-derive labels for every row with a stored generation, not just parse_failed ones",
    )
    parser.add_argument(
        "--judge-dir",
        default=JUDGE_DIR,
        help="directory of *_guardreasoner.json files (default: RQ2/data/evaluation/judge_ensemble)",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.judge_dir, "*_guardreasoner.json")))
    total_failed = total_recovered = total_changed = 0
    for path in paths:
        n_failed, n_recovered, n_changed = reparse_file(path, reparse_all=args.all)
        total_failed += n_failed
        total_recovered += n_recovered
        total_changed += n_changed
        print(
            f"{os.path.basename(path)}: {n_recovered}/{n_failed} parse_failed recovered, "
            f"{n_changed} rows with changed labels"
        )
    print(
        f"\nTOTAL: {total_recovered}/{total_failed} parse_failed rows recovered, "
        f"{total_changed} rows with changed labels"
    )


if __name__ == "__main__":
    main()
