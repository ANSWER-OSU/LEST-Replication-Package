import argparse
import glob
import json
import os
import random

from tested_prompt_filter import dataset_category, is_tested_prompt

JUDGE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "evaluation", "judge_ensemble"
)
DEFAULT_N = 150
DEFAULT_SMALL_OP_THRESHOLD = 20
DEFAULT_SEED = 0
DEFAULT_OUTPUT = "RQ2/outputs/label_studio/mutation_quality_sample.json"


def load_pool():
    # the tested mutated prompts, deduped across the claude/gpt files
    by_prompt_id = {}
    pattern = os.path.join(JUDGE_DIR, "*_mutated_prompt_result_*_wildguard_result_ensemble.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            category = dataset_category(r, path)
            if r["prompt_id"] in by_prompt_id or not is_tested_prompt(r, category):
                continue
            by_prompt_id[r["prompt_id"]] = {**r, "category_name": category}
    return list(by_prompt_id.values())


def allocate_proportional(counts, total):
    operators = list(counts.keys())
    total_capacity = sum(counts.values())
    budget = min(max(0, total), total_capacity)

    alloc = {op: 0 for op in operators}
    if total_capacity == 0 or budget == 0:
        return alloc

    raw_share = {op: budget * counts[op] / total_capacity for op in operators}
    alloc = {op: int(raw_share[op]) for op in operators}
    leftover = budget - sum(alloc.values())

    by_fraction_desc = sorted(operators, key=lambda op: raw_share[op] - alloc[op], reverse=True)
    for op in by_fraction_desc:
        if leftover <= 0:
            break
        if alloc[op] < counts[op]:
            alloc[op] += 1
            leftover -= 1

    return alloc


def sample_stratified(pool, n, small_op_threshold, rng):
    by_operator = {}
    for r in pool:
        by_operator.setdefault(r["mutation"], []).append(r)

    counts = {op: len(records) for op, records in by_operator.items()}

    # operators too rare to split proportionally are taken whole
    small_ops = {op for op, count in counts.items() if count < small_op_threshold}
    quotas = {op: counts[op] for op in small_ops}

    remaining_budget = n - sum(quotas.values())
    remaining_counts = {op: count for op, count in counts.items() if op not in small_ops}
    quotas.update(allocate_proportional(remaining_counts, remaining_budget))

    sample = []
    for op, records in by_operator.items():
        sample.extend(rng.sample(records, quotas[op]))
    return sample, quotas, counts, small_ops


def to_label_studio_task(record):
    return {
        "data": {
            "prompt_id": record["prompt_id"],
            "original_prompt": record["original_prompt"],
            "mutated_prompt": record["mutated_prompt"],
            "mutation": record["mutation"],
            "category": record["category_name"],
            "generation": record["generation"],
            "question_id": record["question_id"],
            "seed": record["seed"],
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sample of tested mutated prompts for Label Studio mutation-quality labeling"
    )
    parser.add_argument("-n", "--num-samples", type=int, default=DEFAULT_N)
    parser.add_argument("--small-op-threshold", type=int, default=DEFAULT_SMALL_OP_THRESHOLD,
                        help="operators with fewer available prompts than this are taken whole (default 20)")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default 0)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    pool = load_pool()
    print(f"pool: {len(pool)} tested mutated prompts")

    sample, quotas, counts, small_ops = sample_stratified(pool, args.num_samples, args.small_op_threshold, rng)

    print(f"mutation operator -> available / sampled (< {args.small_op_threshold} available = take all):")
    for op in sorted(counts, key=lambda o: -counts[o]):
        tag = " (take all)" if op in small_ops else ""
        print(f"  {op}: {counts[op]} / {quotas[op]}{tag}")

    tasks = [to_label_studio_task(r) for r in sample]
    rng.shuffle(tasks)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(tasks)} Label Studio tasks to {args.output}")


if __name__ == "__main__":
    main()
