import argparse
import glob
import json
import os
import re
from collections import defaultdict

MODEL_TAG = "gpt"
RESPONSE_FIELD = "gpt_response"
EXPORT_PATH = "RQ2/outputs/complied_prompts/gpt-5.5_{prompt_type}_complied.json"
DEFAULT_INPUT_DIR = "RQ2/data/evaluation/complied_rerun"
DEFAULT_OUTPUT_PATH = "RQ2/data/evaluation/complied_rerun/merged/gpt_complied_reruns_merged.json"

RUN_FILE = re.compile(rf"_(original|mutated)_complied_run(\d+)_{MODEL_TAG}\.json$")
PROMPT_TYPE_ORDER = {"original": 0, "mutated": 1}


def expected_count(prompt_type):
    path = EXPORT_PATH.format(prompt_type=prompt_type)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return len(json.load(f))


def load_runs(input_dir):
    merged = []
    counts = defaultdict(lambda: {"records": 0, "null": 0})

    for path in sorted(glob.glob(os.path.join(input_dir, f"*_run*_{MODEL_TAG}.json"))):
        match = RUN_FILE.search(os.path.basename(path))
        if not match:
            continue
        prompt_type, run = match.group(1), int(match.group(2))

        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        for r in records:
            record = dict(r)
            record["prompt_id"] = f"{r['base_prompt_id']}-r{run}"
            record["run"] = run
            merged.append(record)

            counts[(prompt_type, run)]["records"] += 1
            if r.get(RESPONSE_FIELD) is None:
                counts[(prompt_type, run)]["null"] += 1

    return merged, counts


def report(counts):
    expected = {pt: expected_count(pt) for pt in PROMPT_TYPE_ORDER}
    for (prompt_type, run), c in sorted(counts.items(), key=lambda kv: (PROMPT_TYPE_ORDER[kv[0][0]], kv[0][1])):
        exp = expected[prompt_type]
        flags = []
        if exp is not None and c["records"] < exp:
            flags.append(f"INCOMPLETE ({exp - c['records']} prompts not run yet)")
        if c["null"]:
            flags.append(f"{c['null']} null responses -- rerun the script to retry them")
        suffix = f"  <- {'; '.join(flags)}" if flags else ""
        print(f"  {prompt_type:8} run {run}: {c['records']}/{exp if exp is not None else '?'}{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    merged, counts = load_runs(args.input_dir)
    if not merged:
        print(f"no {MODEL_TAG} rerun files found in {args.input_dir}")
        return

    ids = [r["prompt_id"] for r in merged]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise SystemExit(f"{duplicates} duplicate prompt_ids after re-tagging -- not writing output")

    merged.sort(key=lambda r: (
        PROMPT_TYPE_ORDER.get(r.get("prompt_type"), 99),
        r["question_id"], r["base_prompt_id"], r["run"],
    ))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    report(counts)
    print(f"wrote {len(merged)} records to {args.output}")


if __name__ == "__main__":
    main()
