import csv
import json
from collections import Counter
from pathlib import Path

# Configuration
BROAD_CATEGORIES = {
    "mental_health_self_harm": [6, 39],
    "harmful_violent_content": [3, 5, 7, 8, 9, 10, 11, 15, 22],
    "fairness_bias": [2, 30, 35, 36],
}

BASE_PATH   = Path("RQ1/data/sorrybench_base.json")
LEDGER_PATH = Path("RQ1/data/anchor/exclusion_ledger.json")
ANCHOR_PATH = Path("RQ1/data/sorrybench_anchors.json")
AUDIT_PATH  = Path("RQ1/data/anchor/anchor_audit.csv")
BROAD_ANCHOR_PATH = "RQ1/data/anchors_{broad}.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def category_to_broad(broad_categories):
    mapping = {}
    for broad, category_ids in broad_categories.items():
        for category_id in category_ids:
            if category_id in mapping:
                raise ValueError(
                    f"category {category_id} assigned to both "
                    f"'{mapping[category_id]}' and '{broad}'"
                )
            mapping[category_id] = broad
    return mapping


def validate_ledger(ledger, base_by_qid):
    errors = []

    seen = Counter(entry["question_id"] for entry in ledger)
    for qid, n in seen.items():
        if n > 1:
            errors.append(f"question_id {qid} appears {n} times in ledger")

    for entry in ledger:
        qid = entry["question_id"]

        if qid not in base_by_qid:
            errors.append(f"question_id {qid} not found in {BASE_PATH}")
            continue

        base_category = str(base_by_qid[qid]["category"])
        if str(entry["category"]) != base_category:
            errors.append(
                f"question_id {qid}: ledger says category "
                f"{entry['category']}, base says {base_category}"
            )

        if entry["prompt"].strip() != base_by_qid[qid]["prompt"].strip():
            errors.append(
                f"question_id {qid}: ledger prompt text does not match base"
            )

        if not entry.get("reason"):
            errors.append(f"question_id {qid}: missing reason code")

    return errors


def build():
    base   = load_json(BASE_PATH)
    ledger = load_json(LEDGER_PATH)

    base_by_qid = {int(item["question_id"]): item for item in base}

    errors = validate_ledger(ledger, base_by_qid)
    if errors:
        raise SystemExit(
            "ledger validation failed:\n  " + "\n  ".join(errors)
        )

    broad_of      = category_to_broad(BROAD_CATEGORIES)
    selected      = set(broad_of)
    excluded_by   = {entry["question_id"]: entry for entry in ledger}
    inactive = [
        qid for qid in excluded_by
        if int(base_by_qid[qid]["category"]) not in selected
    ]

    anchors    = []
    audit_rows = []

    for item in base:
        qid      = int(item["question_id"])
        category = int(item["category"])

        if category not in selected:
            continue

        broad = broad_of[category]
        entry = excluded_by.get(qid)

        if entry is None:
            anchors.append({
                "question_id":    qid,
                "category":       str(category),
                "broad_category": broad,
                "prompt":         item["prompt"],
            })
            audit_rows.append({
                "question_id":    qid,
                "category":       category,
                "broad_category": broad,
                "decision":       "keep",
                "reason":         "",
                "note":           "",
                "prompt":         item["prompt"],
            })
        else:
            audit_rows.append({
                "question_id":    qid,
                "category":       category,
                "broad_category": broad,
                "decision":       "exclude",
                "reason":         entry["reason"],
                "note":           entry.get("note", ""),
                "prompt":         item["prompt"],
            })

    considered = len(audit_rows)
    kept       = len(anchors)
    dropped    = considered - kept

    expected = sum(
        1 for item in base if int(item["category"]) in selected
    )
    if considered != expected:
        raise SystemExit(
            f"row count mismatch: considered {considered}, expected {expected}"
        )

    ANCHOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(ANCHOR_PATH, "w", encoding="utf-8") as f:
        json.dump(anchors, f, indent=2, ensure_ascii=False)

    broad_paths = {}
    for broad in BROAD_CATEGORIES:
        subset = [a for a in anchors if a["broad_category"] == broad]
        path = Path(BROAD_ANCHOR_PATH.format(broad=broad))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        broad_paths[broad] = (path, len(subset))

    split_total = sum(n for _, n in broad_paths.values())
    if split_total != len(anchors):
        raise SystemExit(
            f"split mismatch: per-broad files hold {split_total}, "
            f"combined holds {len(anchors)}"
        )

    with open(AUDIT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id", "category", "broad_category",
                "decision", "reason", "note", "prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    report(audit_rows, considered, kept, dropped, inactive, broad_paths)


def report(audit_rows, considered, kept, dropped, inactive, broad_paths):
    print("=" * 64)
    print("ANCHOR SET")
    print("=" * 64)

    for broad, category_ids in BROAD_CATEGORIES.items():
        broad_rows = [r for r in audit_rows if r["broad_category"] == broad]
        broad_kept = sum(1 for r in broad_rows if r["decision"] == "keep")
        print(f"\n{broad}  ({broad_kept}/{len(broad_rows)})")
        print(f"  {'cat':>5}  {'base':>5}  {'excl':>5}  {'kept':>5}")

        for category_id in sorted(category_ids):
            rows = [r for r in broad_rows if r["category"] == category_id]
            n_keep = sum(1 for r in rows if r["decision"] == "keep")
            print(
                f"  {category_id:>5}  {len(rows):>5}  "
                f"{len(rows) - n_keep:>5}  {n_keep:>5}"
            )

    print("\n" + "-" * 64)
    print("exclusions by reason code")
    reasons = Counter(
        r["reason"] for r in audit_rows if r["decision"] == "exclude"
    )
    for reason, n in reasons.most_common():
        print(f"  {reason:<26s} {n:>3}")

    print("\n" + "-" * 64)
    print(f"considered : {considered}")
    print(f"excluded   : {dropped}")
    print(f"anchors    : {kept}  ({kept / considered * 100:.1f}%)")

    if inactive:
        print(
            f"\nnote: {len(inactive)} ledger entries are for categories not "
            f"currently selected: {sorted(inactive)}"
        )

    print("\n" + "-" * 64)
    print("seed pools")
    for broad, (path, n) in broad_paths.items():
        print(f"  {n:>4}  {path}")
    print(f"  {kept:>4}  {ANCHOR_PATH}  (combined)")
    print(f"\naudit written to   : {AUDIT_PATH}")


if __name__ == "__main__":
    build()
