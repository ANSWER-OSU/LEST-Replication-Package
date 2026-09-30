import json
import os
import random
from collections import defaultdict

# config  
INPUT_PATH = "data/evaluation/self_harm_wildguard_result.json"
TARGET_TOTAL = 80          # total sample size you want out the other end
SEED = 64

# Priority weighting.
SENTIMENT_WEIGHT = 0.7
SEMANTIC_WEIGHT  = 0.3
MAX_PER_ORIGINAL = None

# The strata. Order only matters for how leftover slots are handed out.
CATEGORIES = ["voice", "pglm", "modifier", "mixed"]

OUTPUT_PATH = (
    f"data/evaluation/"
    f"self_harm_stratified_sample_{TARGET_TOTAL}.json"
)

rng = random.Random(SEED)
os.makedirs("data/evaluation", exist_ok=True)


# scoring + categorisation
def priority_score(item):
    """Higher = more wanted. Sentiment-led, semantic distance as tie-shaper."""
    sentiment = item.get("sentiment_difference", 0.0)
    semantic  = item.get("semantic_similarity", 1.0)
    return SENTIMENT_WEIGHT * sentiment + SEMANTIC_WEIGHT * (1.0 - semantic)


def category_of(item):
    chain = item["mutation_chain"]
    if len(chain) > 1:
        return "mixed"
    return chain[0]  # "voice" | "pglm" | "modifier"


# the reusable primitive: even allocation with shortfall redistribution
def even_quota(group_sizes, total, rng):
    alloc = {k: 0 for k in group_sizes}
    remaining = total

    while remaining > 0:
        open_keys = [k for k in group_sizes if alloc[k] < group_sizes[k]]
        if not open_keys:
            break  # no capacity left anywhere -> can't reach `total`

        share = remaining // len(open_keys)

        if share == 0:
            # fewer slots left than open groups: 1 each, random order
            rng.shuffle(open_keys)
            for k in open_keys:
                if remaining == 0:
                    break
                alloc[k] += 1
                remaining -= 1
            continue

        for k in open_keys:
            give = min(share, group_sizes[k] - alloc[k])
            alloc[k] += give
            remaining -= give

    return alloc


# selection
def pick_from_category(items, quota, rng):
    """Pick `quota` items, spread evenly over original prompts, best-first."""
    if quota <= 0 or not items:
        return [], {}

    by_prompt = defaultdict(list)
    for it in items:
        by_prompt[it["original_prompt"]].append(it)

    for p in by_prompt:
        rng.shuffle(by_prompt[p])
        by_prompt[p].sort(key=priority_score, reverse=True)

    # capacity per prompt, optionally capped
    prompt_sizes = {}
    for p, muts in by_prompt.items():
        cap = len(muts) if MAX_PER_ORIGINAL is None else min(len(muts), MAX_PER_ORIGINAL)
        prompt_sizes[p] = cap

    prompt_quota = even_quota(prompt_sizes, quota, rng)

    picked = []
    for p, n in prompt_quota.items():
        picked.extend(by_prompt[p][:n])

    picked.sort(key=priority_score, reverse=True)  # high-sentiment first
    return picked, prompt_quota


def stratified_sample(items, target_total, rng):
    by_cat = defaultdict(list)
    for it in items:
        c = category_of(it)
        if c in CATEGORIES:
            by_cat[c].append(it)

    cat_sizes = {c: len(by_cat[c]) for c in CATEGORIES}
    cat_quota = even_quota(cat_sizes, target_total, rng)

    sample, report = [], {}
    for c in CATEGORIES:
        picked, prompt_quota = pick_from_category(by_cat[c], cat_quota[c], rng)
        sample.extend(picked)
        report[c] = {
            "available": cat_sizes[c],
            "quota": cat_quota[c],
            "filled": len(picked),
            "per_prompt": {p: n for p, n in prompt_quota.items() if n > 0},
        }
    return sample, report


# run
print("loading wildguard results...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

harmful_items = [
    it for it in data
    if it.get("wildguard_prompt_harmfulness") == "harmful"
]
print(f"total harmful prompts: {len(harmful_items)}")

sample, report = stratified_sample(harmful_items, TARGET_TOTAL, rng)
rng.shuffle(sample)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)

# report
print("\n" + "-" * 40)
print(f"STRATIFIED SAMPLE COMPLETE  (target={TARGET_TOTAL})")
print("-" * 40)
for c in CATEGORIES:
    r = report[c]
    print(f"\n[{c}]  available={r['available']}  "
          f"quota={r['quota']}  filled={r['filled']}")
    for p, n in sorted(r["per_prompt"].items(), key=lambda kv: -kv[1]):
        preview = (p[:50] + "...") if len(p) > 50 else p
        print(f"    {n:>3}  {preview}")

print(f"\ntotal saved: {len(sample)}")
print(f"saved to: {OUTPUT_PATH}")
