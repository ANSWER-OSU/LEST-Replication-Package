import argparse
import json
import os
import random
from collections import Counter

from build_response_transitions import MUTATED_DIR, ORIGINAL_DIR, load_ensemble
from tested_prompt_filter import is_tested_prompt

DEFAULT_SEED = 0
DEFAULT_OUTPUT = "RQ2/annotations/refuse_comply/tasks.json"


def response_text(record):
    return record.get("claude_response") or record.get("gpt_response") or ""


def refuse_comply_cases():
    # same selection as build_response_transitions.py: tested mutated prompts
    # whose original was refused and whose mutation was complied with
    baseline = {
        (r["_dataset_category"], r["model"], r["question_id"]): r
        for r in load_ensemble(ORIGINAL_DIR, "*_original_prompt_result_*_wildguard_result_ensemble.json")
    }

    cases = []
    for r in load_ensemble(MUTATED_DIR, "*_mutated_prompt_result_*_wildguard_result_ensemble.json"):
        if not is_tested_prompt(r, r["_dataset_category"]):
            continue
        orig = baseline.get((r["_dataset_category"], r["model"], r["question_id"]))
        if orig is None:
            continue
        if orig.get("ensemble_refusal") == "refusal" and r.get("ensemble_refusal") == "compliance":
            cases.append((r, orig))
    return cases


def to_label_studio_task(record, original):
    return {
        "data": {
            "case_id": f"{record['model']}:{record['prompt_id']}",
            "model": record["model"],
            "prompt_id": record["prompt_id"],
            "question_id": record["question_id"],
            "category": record["_dataset_category"],
            "seed": record["seed"],
            "generation": record["generation"],
            "mutation": record["mutation"],
            "original_prompt": record["original_prompt"],
            "mutated_prompt": record["mutated_prompt"],
            "response": response_text(record),
            "original_response": response_text(original),
            "ensemble_refusal_margin": record.get("ensemble_refusal_margin"),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Every Refuse->Comply case (original refused, mutation complied) as Label Studio tasks"
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="shuffle seed (default 0)")
    args = parser.parse_args()

    cases = refuse_comply_cases()
    tasks = [to_label_studio_task(r, o) for r, o in cases]
    # interleave models and categories so annotators don't see them in blocks
    random.Random(args.seed).shuffle(tasks)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"{len(tasks)} Refuse->Comply cases")
    for (model, category), n in sorted(Counter((t["data"]["model"], t["data"]["category"]) for t in tasks).items()):
        print(f"  {model:<16} {category:<24} {n}")
    for model, n in sorted(Counter(t["data"]["model"] for t in tasks).items()):
        print(f"  {model:<16} total {n}")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
