import argparse
import glob
import json
import os
import re

PROMPT_ID_RE = re.compile(r"^(q\d+-g\d+)-s(\d+)$")
MUTATED_DIR = "RQ2/data/mutated"


def find_fuzzed_file(base_prompt_id, seed):
    pattern = os.path.join(MUTATED_DIR, "*", f"*_fuzzed_prompt_seed{seed}.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        by_id = {e["prompt_id"]: e for e in entries}
        if base_prompt_id in by_id:
            return path, by_id
    return None, None


def build_lineage(base_prompt_id, by_id):
    """Return [target, ..., generation-1 entry], nearest-first."""
    lineage = []
    current_id = base_prompt_id
    while current_id in by_id:
        entry = by_id[current_id]
        lineage.append(entry)
        current_id = entry["parent_id"]
    return lineage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--prompt-id", required=True,
                         help="seed-tagged prompt_id, e.g. q12-g1532-s1")
    args = parser.parse_args()

    match = PROMPT_ID_RE.match(args.prompt_id)
    if not match:
        raise ValueError(f"expected a seed-tagged prompt_id like q12-g1532-s1, got {args.prompt_id!r}")
    base_prompt_id, seed = match.group(1), int(match.group(2))

    path, by_id = find_fuzzed_file(base_prompt_id, seed)
    if by_id is None:
        raise ValueError(f"couldn't find {base_prompt_id} in any category's seed{seed} fuzzed_prompt file")

    lineage = build_lineage(base_prompt_id, by_id)
    lineage.reverse()  # generation 1 first, target last

    print(f"prompt_id: {args.prompt_id}  (source: {path})")
    print(f"gen0 [original]: {lineage[0]['original_prompt']}")
    for entry in lineage:
        print(f"gen{entry['generation']} [{entry['mutation']}]: {entry['mutated_prompt']}")


if __name__ == "__main__":
    main()
