import csv
import os
import sys
from collections import Counter

# Vote-cell parsing
def parse_votes(cell):
    """'Claim:3,Not a claim:1' -> {'Claim': 3, 'Not a claim': 1}. '' -> {}."""
    cell = (cell or "").strip()
    if not cell:
        return {}
    out = {}
    for part in cell.split(","):
        part = part.strip()
        if not part:
            continue
        label, _, count = part.rpartition(":")   # rpartition: category may hold spaces, never a colon
        out[label.strip()] = int(count)
    return out


def selection_items(vote_cell, n_raters, n_options):
    """Binary items for one multi-select task: one per option, Selected vs Not selected.

    vote_cell tallies only the options someone ticked ('q51:3,q53:1'); the
    remaining n_options - len(tally) options were left unticked by every rater.
    """
    votes = parse_votes(vote_cell)
    if len(votes) > n_options:
        raise ValueError(f"{len(votes)} options voted but only {n_options} available")
    items = [{"Selected": v, "Not selected": n_raters - v} for v in votes.values()]
    items += [{"Selected": 0, "Not selected": n_raters}] * (n_options - len(votes))
    return items


# Generalized Fleiss' kappa

def fleiss_kappa(item_counts, categories):
    cat_totals = Counter()
    total_ratings = 0
    p_i_values = []
    used, skipped = 0, 0

    for counts in item_counts:
        n_i = sum(counts.get(c, 0) for c in categories)
        if n_i < 2:
            skipped += 1
            continue
        sum_sq = sum(counts.get(c, 0) ** 2 for c in categories)
        p_i = (sum_sq - n_i) / (n_i * (n_i - 1))   # agreeing pairs / possible pairs
        p_i_values.append(p_i)
        for c in categories:
            cat_totals[c] += counts.get(c, 0)
        total_ratings += n_i
        used += 1

    if used == 0 or total_ratings == 0:
        raise ValueError("No scorable items (need at least 2 raters per item).")

    p_bar = sum(p_i_values) / used                          # mean observed agreement
    marginals = {c: cat_totals[c] / total_ratings for c in categories}
    p_e = sum(p ** 2 for p in marginals.values())           # expected by chance

    kappa = (p_bar - p_e) / (1 - p_e) if (1 - p_e) != 0 else float("nan")

    return {
        "kappa": kappa,
        "P_bar": p_bar,
        "P_e": p_e,
        "marginals": marginals,
        "n_items_used": used,
        "n_items_skipped": skipped,
        "total_ratings": total_ratings,
    }


# Build the three count matrices from the CSV rows                          
def build_matrices(rows, testability_unanimous_only=True):
    is_claim_counts = []      # categories: Claim / Not a claim
    testable_counts = []      # categories: Testable / Not testable
    collapsed_counts = []     # categories: Not a claim / Claim:Testable / Claim:Not testable

    n_unanimous_claim = 0

    for r in rows:
        claim = parse_votes(r.get("claim_votes", ""))
        testable = parse_votes(r.get("testable_votes", ""))

        n_claim_votes = sum(claim.values())
        n_say_claim = claim.get("Claim", 0)
        n_say_not = claim.get("Not a claim", 0)

        # 1. is_claim, every item
        is_claim_counts.append({"Claim": n_say_claim, "Not a claim": n_say_not})

        # 2. testability
        is_unanimous_claim = n_claim_votes > 0 and n_say_claim == n_claim_votes
        if is_unanimous_claim:
            n_unanimous_claim += 1

        include_for_testability = is_unanimous_claim if testability_unanimous_only else bool(testable)
        if include_for_testability:
            testable_counts.append({
                "Testable": testable.get("Testable", 0),
                "Not testable": testable.get("Not testable", 0),
            })

        # 3. collapsed 3-class: every annotator emits exactly one label
        collapsed_counts.append({
            "Not a claim": n_say_not,
            "Claim:Testable": testable.get("Testable", 0),
            "Claim:Not testable": testable.get("Not testable", 0),
        })

    return is_claim_counts, testable_counts, collapsed_counts, n_unanimous_claim


# Reporting                              
def report(title, res, categories):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  Fleiss' kappa     : {res['kappa']:.4f}")
    print(f"  Observed agreement: {res['P_bar']:.4f}")
    print(f"  Chance agreement  : {res['P_e']:.4f}")
    print(f"  Items scored      : {res['n_items_used']}"
          + (f"  (skipped {res['n_items_skipped']} with <2 raters)"
             if res['n_items_skipped'] else ""))
    print(f"  Total ratings     : {res['total_ratings']}")
    print("  Category marginals:")
    for c in categories:
        print(f"      {c:<22}: {res['marginals'][c]:.3f}")


def result_row(title, res, categories):
    return {
        "measure": title,
        "kappa": round(res["kappa"], 4),
        "observed_agreement": round(res["P_bar"], 4),
        "chance_agreement": round(res["P_e"], 4),
        "items_scored": res["n_items_used"],
        "items_skipped": res["n_items_skipped"],
        "total_ratings": res["total_ratings"],
        "category_marginals": "; ".join(f"{c}: {res['marginals'][c]:.3f}" for c in categories),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved to {path}")


def main():
    TESTABILITY_UNANIMOUS_CLAIMS_ONLY = False

    path = sys.argv[1] if len(sys.argv) > 1 else "labeled_claims.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "RQ1/outputs/claims_fleiss_kappa.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    is_claim_counts, testable_counts, collapsed_counts, n_unan = build_matrices(
        rows, testability_unanimous_only=TESTABILITY_UNANIMOUS_CLAIMS_ONLY)

    rater_counts = Counter(sum(parse_votes(r.get("claim_votes", "")).values()) for r in rows)

    print("=" * 64)
    print("FLEISS' KAPPA  -  nested claim / testability annotation")
    print("=" * 64)
    print(f"Total items                 : {len(rows)}")
    print(f"Raters per item             : "
          + ", ".join(f"{n} raters x {c} items" for n, c in sorted(rater_counts.items(), reverse=True)))
    print(f"Items with unanimous 'Claim': {n_unan}")
    print(f"Testability subset          : "
          + ("unanimous-claim items only" if TESTABILITY_UNANIMOUS_CLAIMS_ONLY
             else "all items with testability votes (ragged)"))

    testable_label = ("unanimous-claim items only" if TESTABILITY_UNANIMOUS_CLAIMS_ONLY
                      else "all items with testability votes")
    measures = [
        ("1. is_claim agreement (binary, all items)",
         is_claim_counts, ["Claim", "Not a claim"]),
        (f"2. testability agreement (binary, {testable_label})",
         testable_counts, ["Testable", "Not testable"]),
        ("3. TOTAL agreement (collapsed 3-class)",
         collapsed_counts, ["Not a claim", "Claim:Testable", "Claim:Not testable"]),
    ]
    out_rows = []
    for title, counts, categories in measures:
        res = fleiss_kappa(counts, categories)
        report(title, res, categories)
        out_rows.append(result_row(title, res, categories))
    write_csv(output_path, out_rows)


if __name__ == "__main__":
    main()
