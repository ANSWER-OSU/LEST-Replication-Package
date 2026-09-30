import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "judge_ensemble"))
from ensemble_io import load_json, save_json  # noqa: E402
from guardreasoner_schema import GUARDREASONER_KEYS  # noqa: E402

JUDGE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "data", "evaluation", "original_prompt_experiment", "judge_ensemble",
)
QUEUE_DIR = os.path.join(JUDGE_DIR, "rerun_queue")
QUEUE_PATH = os.path.join(QUEUE_DIR, "guardreasoner_rerun_queue.json")

NEEDS_RERUN_EXCLUDED_ERROR = "no_response"


def strip_guardreasoner_fields(record):
    return {k: v for k, v in record.items() if k not in GUARDREASONER_KEYS}


def main():
    combined = []

    for path in sorted(glob.glob(os.path.join(JUDGE_DIR, "*_guardreasoner.json"))):
        source_file = os.path.basename(path)
        records = load_json(path)

        needs_rerun = [
            {**strip_guardreasoner_fields(r), "_source_file": source_file}
            for r in records
            if r.get("guardreasoner_error") is not None
            and r.get("guardreasoner_error") != NEEDS_RERUN_EXCLUDED_ERROR
        ]
        print(f"{source_file}: {len(needs_rerun)} rows need rerun")
        combined.extend(needs_rerun)

    if not combined:
        print("\nnothing to queue")
        return

    save_json(QUEUE_PATH, combined)

    print(f"\nTOTAL queued for rerun: {len(combined)} -> {QUEUE_PATH}")
    print(
        f"On the GPU machine, run: python RQ2/scripts/judge_ensemble/run_guardreasoner.py {QUEUE_PATH}, "
        "then bring the resulting "
        "guardreasoner_rerun_queue_guardreasoner.json file back to this repo."
    )


if __name__ == "__main__":
    main()
