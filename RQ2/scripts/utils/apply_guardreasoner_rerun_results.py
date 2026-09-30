import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "judge_ensemble"))
from ensemble_io import load_json, save_json  # noqa: E402
from guardreasoner_schema import GUARDREASONER_KEYS, JUDGE_DIR  # noqa: E402

RERUN_RESULT_PATH = os.path.join(JUDGE_DIR, "guardreasoner_rerun_queue_guardreasoner.json")


def group_by_source(rerun_rows):
    grouped = defaultdict(list)
    for r in rerun_rows:
        grouped[r["_source_file"]].append(r)
    return grouped


def apply_group(source_file, rerun_rows):
    original_path = os.path.join(JUDGE_DIR, source_file)
    if not os.path.isfile(original_path):
        print(f"skip {source_file}: no matching original file at {original_path}")
        return 0, 0, len(rerun_rows), 0

    rerun_by_id = {r["prompt_id"]: r for r in rerun_rows}
    original_rows = load_json(original_path)

    n_matched = 0
    n_recovered = 0
    n_downgrade_skipped = 0
    for r in original_rows:
        rerun_row = rerun_by_id.pop(r["prompt_id"], None)
        if rerun_row is None:
            continue
        n_matched += 1

        if rerun_row.get("guardreasoner_error") is not None:
            n_downgrade_skipped += 1
            continue

        for key in GUARDREASONER_KEYS:
            r[key] = rerun_row.get(key)
        n_recovered += 1

    n_unmatched = len(rerun_by_id)  # rerun rows never popped -- no matching prompt_id

    if n_matched:
        save_json(original_path, original_rows)

    return n_matched, n_recovered, n_unmatched, n_downgrade_skipped


def main():
    if not os.path.isfile(RERUN_RESULT_PATH):
        print(f"no rerun-result file found at {RERUN_RESULT_PATH}")
        return

    rerun_rows = load_json(RERUN_RESULT_PATH)
    grouped = group_by_source(rerun_rows)

    total_matched = total_recovered = total_unmatched = total_downgrade_skipped = 0
    for source_file in sorted(grouped):
        n_matched, n_recovered, n_unmatched, n_downgrade_skipped = apply_group(
            source_file, grouped[source_file]
        )
        total_matched += n_matched
        total_recovered += n_recovered
        total_unmatched += n_unmatched
        total_downgrade_skipped += n_downgrade_skipped
        print(
            f"{source_file}: {n_recovered}/{n_matched} recovered, "
            f"{n_unmatched} unmatched, {n_downgrade_skipped} downgrade-skipped"
        )

    print(
        f"\nTOTAL: {total_recovered}/{total_matched} recovered, "
        f"{total_unmatched} unmatched, {total_downgrade_skipped} downgrade-skipped"
    )
    if total_unmatched:
        print(f"WARNING: {total_unmatched} rerun row(s) had no matching prompt_id in their source file.")
    print(
        "Reminder: re-run aggregate_ensemble.py for each affected base input to fold "
        "these into *_result_ensemble.json before regenerating the evaluation tables."
    )


if __name__ == "__main__":
    main()
