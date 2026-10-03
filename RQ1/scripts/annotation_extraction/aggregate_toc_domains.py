import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, "RQ1/scripts/annotation_extraction")
from aggregate_claims import breakdown  # noqa: E402

#field config
TASK_KEYS = ["domain", "source_doc"]
SECTIONS_FIELD = "toc_sections"

# task-level metadata to carry through (constant per domain x card)
METADATA_COLS = ["domain_name", "card_name", "num_sections"]

# the Label Studio tasks, whose toc_options list every ToC section in document order
TASKS_JSON = Path("RQ1/annotations/toc_domains/tasks.json")

# '<section id>: <'· ' per nesting level><title> (p<page>)'
OPTION_RE = re.compile(r"^(?P<id>[^:]+): (?P<dots>(?:· )*)(?P<title>.*) \(p(?P<page>\d+)\)$")


#parsing
def parse_sections(cell):
    """Section ids ('claude-t059') selected in one annotation.

    Label Studio exports several selections as '{"choices": [...]}', a single
    selection as the bare choice string, and no selection as empty.
    """
    if pd.isna(cell) or not str(cell).strip():
        return []
    cell = str(cell)
    choices = json.loads(cell)["choices"] if cell.startswith("{") else [cell]
    return [c.split(":", 1)[0].strip() for c in choices]


def load_toc():
    """Every ToC section of both cards, in document order, plus the domain list."""
    with open(TASKS_JSON, encoding="utf-8") as f:
        tasks = [t["data"] for t in json.load(f)]

    rows, seen = [], set()
    for t in tasks:
        if t["source_doc"] in seen:
            continue
        seen.add(t["source_doc"])
        for opt in t["toc_options"]:
            m = OPTION_RE.match(opt["value"])
            if not m:
                raise SystemExit(f"{TASKS_JSON}: cannot parse ToC option {opt['value']!r}")
            rows.append({
                "section_id": m["id"],
                "source_doc": t["source_doc"],
                "title": m["title"],
            })
    domains = list(dict.fromkeys(t["domain"] for t in tasks))
    return pd.DataFrame(rows), domains


def tally(counter, order):
    """'claude-t059:3,claude-t090:1' ordered by votes, then document order."""
    items = sorted(counter.items(), key=lambda kv: (-kv[1], order.get(kv[0], len(order))))
    return ",".join(f"{k}:{n}" for k, n in items)


#per-task collapse
def collapse_task(group, order):
    n = len(group)
    selections = [frozenset(parse_sections(cell)) for cell in group[SECTIONS_FIELD]]
    votes = Counter(s for sel in selections for s in sel)
    majority = sorted((s for s, v in votes.items() if v > n / 2), key=order.get)
    unanimous = sorted((s for s, v in votes.items() if v == n), key=order.get)

    record = dict(zip(TASK_KEYS, group.name))
    record.update({
        "num_annotators": n,
        "all_agree": len(set(selections)) == 1,
        "majority_sections": ",".join(majority),
        "num_majority_sections": len(majority),
        "unanimous_sections": ",".join(unanimous),
        "section_votes": tally(votes, order),
        "num_sections_voted": len(votes),
        "num_selected_per_annotator": breakdown([len(sel) for sel in selections]),
    })
    for col in METADATA_COLS:
        if col in group.columns:
            record[col] = group[col].iloc[0]
    return pd.Series(record)


def aggregate(df, order):
    for col in METADATA_COLS:
        if col not in df.columns:
            continue
        bad = df.groupby(TASK_KEYS)[col].apply(lambda s: s.dropna().nunique() > 1)
        if bad.any():
            ids = bad[bad].index.tolist()
            print(f"  warning: '{col}' is not constant within tasks {ids}; using first value")

    # an annotator submitting the same task twice would count twice, so flag it
    if "annotator" in df.columns:
        repeats = df.groupby(TASK_KEYS)["annotator"].apply(lambda s: s.duplicated().any())
        if repeats.any():
            print(f"  warning: an annotator labelled tasks {repeats[repeats].index.tolist()} more than once; "
                  "each submission counts as a vote")

    return (df.groupby(TASK_KEYS, sort=False)
              .apply(collapse_task, order=order, include_groups=False)
              .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csvs", nargs="+", help="raw Label Studio export(s)")
    ap.add_argument("output_csv", help="path for the per-domain-per-card output")
    args = ap.parse_args()

    parts = []
    for path in args.input_csvs:
        df = pd.read_csv(path)
        required = set(TASK_KEYS) | {SECTIONS_FIELD}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{path}: missing required columns {sorted(missing)}")
        print(f"read {len(df)} annotations across {df.groupby(TASK_KEYS).ngroups} tasks from {path}")
        parts.append(df)

    df = pd.concat(parts, ignore_index=True)
    if "annotation_id" in df.columns:
        dupes = df["annotation_id"].duplicated()
        if dupes.any():
            print(f"  warning: dropping {dupes.sum()} annotations exported more than once")
            df = df[~dupes]

    toc, domains = load_toc()
    order = {sid: i for i, sid in enumerate(toc["section_id"])}

    tasks = aggregate(df, order)
    unknown = set(tasks["domain"]) - set(domains)
    if unknown:
        raise SystemExit(f"domains {sorted(unknown)} are not in {TASKS_JSON}")

    # domain order as in the task file, then card order
    rank = {
        "domain": {d: i for i, d in enumerate(domains)},
        "source_doc": {d: i for i, d in enumerate(dict.fromkeys(toc["source_doc"]))},
    }
    tasks = tasks.sort_values(TASK_KEYS, key=lambda s: s.map(rank[s.name])).reset_index(drop=True)
    expected = {(d, doc) for d in domains for doc in rank["source_doc"]}
    missing = sorted(expected - set(zip(tasks["domain"], tasks["source_doc"])))
    if missing:
        print(f"  note: no annotations yet for {missing}")

    titles = dict(zip(toc["section_id"], toc["title"]))
    tasks["majority_section_titles"] = tasks["majority_sections"].map(
        lambda ids: " | ".join(titles.get(s, s) for s in filter(None, ids.split(","))))

    ordered = (
        TASK_KEYS + ["num_annotators", "all_agree", "majority_sections", "num_majority_sections",
                     "majority_section_titles", "unanimous_sections", "section_votes",
                     "num_sections_voted", "num_selected_per_annotator"]
        + [c for c in METADATA_COLS if c in tasks.columns]
    )
    tasks = tasks[ordered]
    tasks.to_csv(args.output_csv, index=False)

    print(f"wrote {len(tasks)} domain x card tasks -> {args.output_csv}")
    print(f"  full agreement : {tasks['all_agree'].sum()}")
    print(f"  tasks with no majority section : {(tasks['num_majority_sections'] == 0).sum()}")
    for _, t in tasks.iterrows():
        print(f"  {t['domain']:<25} {t['source_doc']:<7} majority {t['num_majority_sections']}, "
              f"unanimous {len(list(filter(None, t['unanimous_sections'].split(','))))}, "
              f"voted {t['num_sections_voted']}")


if __name__ == "__main__":
    main()
