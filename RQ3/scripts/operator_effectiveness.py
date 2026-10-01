import glob
import json
import os
from collections import Counter

Z_95 = 1.959963984540054  # z-score for a 95% confidence interval


def wilson_ci(k, n, z=Z_95):
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfwidth = (z / denom) * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return max(0.0, center - halfwidth), min(1.0, center + halfwidth)

TABLE_GLOB = "RQ3/data/analysis/*_mutation_all_experiment_*.json"
BASELINE_GLOB = "RQ2/data/evaluation/original_prompt_experiment/judge_ensemble/*_wildguard_result_ensemble.json"
MODELS = ["claude", "gpt"]
CATEGORIES = ["fairness_bias", "harmful_violent_content", "mental_health_self_harm"]
OPERATORS = [
    "pcls",
    "modifier",
    "delete",
    "person_first_to_third",
    "person_third_to_first",
    "fairness",
    "paraphrase",
]

OPERATOR_ALIASES = {"pglm": "pcls"}


def load_tables():
    rows = []
    for path in sorted(glob.glob(TABLE_GLOB)):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    if not rows:
        raise SystemExit(f"no tables found at {TABLE_GLOB} -- run build_mutation_tables.py")
    for r in rows:
        r["last_operator"] = OPERATOR_ALIASES.get(r["last_operator"], r["last_operator"])
    return rows


def unique_mutations(rows):
    return {(r["category_name"], r["prompt_id"]): r["last_operator"] for r in rows}


def valid_mutations(rows):
    return Counter(unique_mutations(rows).values())


def valid_mutations_by_category(rows):
    counts = {category: Counter() for category in CATEGORIES}
    for (category, _), operator in unique_mutations(rows).items():
        counts[category][operator] += 1
    return counts


def print_valid_mutations(rows):
    counts = valid_mutations(rows)
    total = sum(counts.values())
    print("Valid mutations generated, by last operator (descending)")
    print(f"{'operator':24s}{'valid':>8s}{'share':>9s}")
    for op in sorted(OPERATORS, key=lambda op: counts[op], reverse=True):
        print(f"{op:24s}{counts[op]:8d}{counts[op] / total:9.1%}")
    print(f"{'total':24s}{total:8d}")


def print_valid_mutations_by_category(rows):
    by_category = valid_mutations_by_category(rows)
    totals = {category: sum(by_category[category].values()) for category in CATEGORIES}
    overall = Counter(unique_mutations(rows).values())
    grand_total = sum(totals.values())

    print("Valid mutations by last operator and category (share = within that category; descending by total)")
    print(f"{'operator':24s}" + "".join(f"{c:>26s}" for c in CATEGORIES) + f"{'total':>16s}")
    for op in sorted(OPERATORS, key=lambda op: overall[op], reverse=True):
        line = f"{op:24s}"
        for category in CATEGORIES:
            n = by_category[category][op]
            line += f"{n:>16d} ({n / totals[category]:5.1%})"
        line += f"{overall[op]:>8d} ({overall[op] / grand_total:5.1%})"
        print(line)
    print(f"{'total':24s}" + "".join(f"{totals[c]:>26d}" for c in CATEGORIES) + f"{grand_total:>16d}")


LABEL_NAMES = {"refusal": "refuse", "compliance": "comply", None: "no verdict"}


def label_counts(rows, model, key, order):
    counts = {group: Counter() for group in order}
    for r in rows:
        if r["model"] == model:
            counts[r[key]][LABEL_NAMES[r["label"]]] += 1
    return counts


def print_label_breakdown(rows):
    for key, order, title in (("category_name", CATEGORIES, "category"),):
        for model in MODELS:
            counts = label_counts(rows, model, key, order)
            print(f"Mutated prompts by {title}, {model}")
            print(f"{title:24s}{'mutations':>10s}{'refuse':>8s}{'comply':>8s}{'comply share':>14s}")
            total = Counter()
            for group in order:
                c = counts[group]
                total.update(c)
                decided = c["refuse"] + c["comply"]
                print(f"{group:24s}{sum(c.values()):10d}{c['refuse']:8d}{c['comply']:8d}{c['comply'] / decided:14.1%}")
            decided = total["refuse"] + total["comply"]
            print(f"{'total':24s}{sum(total.values()):10d}{total['refuse']:8d}{total['comply']:8d}{total['comply'] / decided:14.1%}")
            print()


def outcome_counts(rows, model, category=None):
    counts = {op: Counter() for op in OPERATORS}
    for r in rows:
        if r["model"] == model and (category is None or r["category_name"] == category):
            counts[r["last_operator"]][LABEL_NAMES[r["label"]]] += 1
    return counts


def _pct(n, total):
    return f"{n / total:.1%}" if total else "-"


def print_outcome_rates(rows):
    order = sorted(OPERATORS, key=lambda op: valid_mutations(rows)[op], reverse=True)
    for model in MODELS:
        counts = outcome_counts(rows, model)
        print(f"Outcome rates by last operator, {model} (share of the operator's valid mutations)")
        print(f"{'operator':24s}{'valid':>7s}{'refuse':>16s}{'comply':>16s}")
        total = Counter()
        for op in order:
            c = counts[op]; total.update(c); n = sum(c.values())
            print(f"{op:24s}{n:7d}" + "".join(f"{c[k]:>8d} ({_pct(c[k], n):>6s})" for k in ("refuse", "comply")))
        n = sum(total.values())
        print(f"{'total':24s}{n:7d}" + "".join(f"{total[k]:>8d} ({_pct(total[k], n):>6s})" for k in ("refuse", "comply")))
        print()


def print_outcome_rates_by_category(rows):
    order = sorted(OPERATORS, key=lambda op: valid_mutations(rows)[op], reverse=True)
    for model in MODELS:
        by_category = {c: outcome_counts(rows, model, category=c) for c in CATEGORIES}
        print(f"Outcome rates by last operator and category, {model} (refuse% / comply%, n)")
        print(f"{'operator':24s}" + "".join(f"{c:>34s}" for c in CATEGORIES))
        for op in order:
            line = f"{op:24s}"
            for c in CATEGORIES:
                cnt = by_category[c][op]; n = sum(cnt.values())
                cell = "-" if not n else f"{cnt['refuse']/n:.1%} / {cnt['comply']/n:.1%} (n={n})"
                line += f"{cell:>34s}"
            print(line)
        print()


def load_baselines():
    rows = []
    for path in sorted(glob.glob(BASELINE_GLOB)):
        model = "claude" if "_claude_" in path else "gpt"
        category = os.path.basename(path).split("_original_prompt_result_")[0]
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                rows.append({"category_name": category, "model": model, "label": r["ensemble_refusal"]})
    return rows


def print_baseline_outcomes():
    baselines = load_baselines()
    for model in MODELS:
        by_category = {c: Counter() for c in CATEGORIES}
        for r in baselines:
            if r["model"] == model:
                by_category[r["category_name"]][LABEL_NAMES[r["label"]]] += 1

        print(f"Original prompt outcomes, {model} (baseline)")
        print(f"{'category':26s}{'n':>5s}{'refuse':>8s}{'comply':>8s}")
        total = Counter()
        for category in CATEGORIES:
            c = by_category[category]
            total.update(c)
            print(f"{category:26s}{sum(c.values()):5d}{c['refuse']:8d}{c['comply']:8d}")
        print(f"{'total':26s}{sum(total.values()):5d}{total['refuse']:8d}{total['comply']:8d}")
        print()


def refuse_to_comply(rows, model, harmful_only=False, category=None):
    stats = {op: Counter() for op in OPERATORS}
    excluded = Counter()
    for r in rows:
        if r["model"] != model or (category and r["category_name"] != category):
            continue
        if r["baseline_label"] != "refusal":
            excluded["baseline complied" if r["baseline_label"] == "compliance" else "baseline no verdict"] += 1
        elif r["label"] is None:
            excluded["mutation no verdict"] += 1
        elif harmful_only and r["wildguard_prompt_harmfulness"] != "harmful":
            excluded["prompt rated unharmful"] += 1
        else:
            stats[r["last_operator"]]["eligible"] += 1
            stats[r["last_operator"]]["flips"] += r["label"] == "compliance"
    return stats, excluded


def print_refuse_to_comply_detailed(rows):
    baselines = load_baselines()
    order = sorted(OPERATORS, key=lambda op: valid_mutations(rows)[op], reverse=True)
    for model in MODELS:
        b = Counter(LABEL_NAMES[r["label"]] for r in baselines if r["model"] == model)
        print(f"Refuse -> Comply walkthrough, {model}")
        print(f"  Step 1-2: of 97 original prompts, {b['refuse']} refused, {b['comply']} complied with")
        print()
        stats, excluded = refuse_to_comply(rows, model)
        print(f"  excluded (mutations of the {b['comply']} already-complied originals): {excluded.get('baseline complied', 0)}")
        print(f"  {'operator':24s}{'eligible':>10s}{'still refusal':>15s}{'flips':>8s}{'rate':>9s}")
        tot_e = tot_f = 0
        for op in order:
            e, f = stats[op]["eligible"], stats[op]["flips"]
            tot_e += e; tot_f += f
            rate = f"{f / e:9.1%}" if e else f"{'-':>9s}"
            print(f"  {op:24s}{e:10d}{e - f:15d}{f:8d}{rate}")
        print(f"  {'total':24s}{tot_e:10d}{tot_e - tot_f:15d}{tot_f:8d}{tot_f / tot_e:9.1%}")
        print()


def seed_exposure(rows, model):
    by_op_seed = {op: set() for op in OPERATORS}
    flipped_op_seed = {op: set() for op in OPERATORS}
    for r in rows:
        if r["model"] != model or r["baseline_label"] != "refusal":
            continue
        seed = (r["category_name"], r["question_id"])
        by_op_seed[r["last_operator"]].add(seed)
        if r["label"] == "compliance":
            flipped_op_seed[r["last_operator"]].add(seed)
    return by_op_seed, flipped_op_seed


def print_seed_exposure(rows):
    order = sorted(OPERATORS, key=lambda op: valid_mutations(rows)[op], reverse=True)
    for model in MODELS:
        by_op_seed, flipped_op_seed = seed_exposure(rows, model)
        print(f"Seeds with >=1 flip, by last operator, {model} (seed count, not mutation count)")
        print(f"{'operator':24s}{'seeds tried':>12s}{'seeds flipped':>15s}{'rate':>9s}{'95% CI':>16s}")
        for op in order:
            n, k = len(by_op_seed[op]), len(flipped_op_seed[op])
            if n:
                lo, hi = wilson_ci(k, n)
                rate, ci = f"{k / n:9.1%}", f"{lo:.1%}-{hi:.1%}"
            else:
                rate, ci = f"{'-':>9s}", "-"
            print(f"{op:24s}{n:12d}{k:15d}{rate}{ci:>16s}")
        all_tried = set().union(*by_op_seed.values())
        all_flipped = set().union(*flipped_op_seed.values())
        n, k = len(all_tried), len(all_flipped)
        lo, hi = wilson_ci(k, n)
        print(f"{'any operator':24s}{n:12d}{k:15d}{k / n:9.1%}{f'{lo:.1%}-{hi:.1%}':>16s}")
        print()


def print_refuse_to_comply(rows, harmful_only=False):
    counts = valid_mutations(rows)
    order = sorted(OPERATORS, key=lambda op: counts[op], reverse=True)
    scope = "harmful prompts only" if harmful_only else "all valid mutations"
    for model in MODELS:
        stats, excluded = refuse_to_comply(rows, model, harmful_only)
        eligible = sum(stats[op]["eligible"] for op in OPERATORS)
        flips = sum(stats[op]["flips"] for op in OPERATORS)
        print(f"Refuse -> Comply, {model} ({scope})")
        print("  excluded: " + ", ".join(f"{reason} {n}" for reason, n in excluded.items()))
        print(f"{'operator':24s}{'eligible':>10s}{'flips':>8s}{'rate':>9s}")
        for op in order:
            e, f = stats[op]["eligible"], stats[op]["flips"]
            rate = f"{f / e:9.1%}" if e else f"{'-':>9s}"
            print(f"{op:24s}{e:10d}{f:8d}{rate}")
        print(f"{'total':24s}{eligible:10d}{flips:8d}{flips / eligible:9.1%}")
        print()


def print_refuse_to_comply_by_category(rows):
    counts = valid_mutations(rows)
    order = sorted(OPERATORS, key=lambda op: counts[op], reverse=True)
    for model in MODELS:
        by_category = {c: refuse_to_comply(rows, model, category=c)[0] for c in CATEGORIES}
        print(f"Refuse -> Comply by category, {model} (all valid mutations; flips/eligible)")
        print(f"{'operator':24s}" + "".join(f"{c:>28s}" for c in CATEGORIES))
        for op in order:
            line = f"{op:24s}"
            for c in CATEGORIES:
                e, f = by_category[c][op]["eligible"], by_category[c][op]["flips"]
                cell = f"{f}/{e} ({f / e:.1%})" if e else "-"
                line += f"{cell:>28s}"
            print(line)
        print()


def main():
    rows = load_tables()
    print_baseline_outcomes()
    print_valid_mutations(rows)
    print()
    print_valid_mutations_by_category(rows)
    print()
    print_label_breakdown(rows)
    print_outcome_rates(rows)
    print_outcome_rates_by_category(rows)
    print_refuse_to_comply_detailed(rows)
    print_seed_exposure(rows)
    print_refuse_to_comply(rows)
    print_refuse_to_comply(rows, harmful_only=True)
    print_refuse_to_comply_by_category(rows)


if __name__ == "__main__":
    main()
