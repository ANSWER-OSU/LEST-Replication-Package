"""
evaluation.py
-------------
Stratified k-fold evaluation of the three testability classifiers. Reports F1
(positive class = Testable) as the headline metric, as well as precision, recall and
accuracy for context. Run directly to evaluate all three on claims_labeled.csv. Additionally
one of the features, the term frequency vector is optional, and can be used on all the classifiers,
the linear classifiers, or none of the classifers using the --tf tag. 
Additionally: RIPPER is a propositional rule learner, so feeding it a
wide sparse bag-of-words tends to produce unstable, overfit rules. The default
mode (linear) therefore keeps TF out of RIPPER and lets it learn a readable
decision list over the structured features

    python evaluation.py --input claims_labeled.csv --folds 5 --repeats 5 --tf linear

"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from classifiers import (
    get_builders, build_ripper, tf_flags,
    TEXT_COL, BINARY_FEATURES, CATEGORICAL_FEATURES, TF_MODES,
)

FEATURE_COLUMNS = BINARY_FEATURES + CATEGORICAL_FEATURES

TARGET = "testable"

# RQ1/classification/evaluation.py -> RQ1/outputs
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "classifier_evaluation.csv"
)

# Columns that are pure annotation bookkeeping and must be dropped.
DROP_COLUMNS = [
    "label_claim",
    "all_agree",
    "num_annotators",
    "claim_votes",
    "testable_votes",
]

def _splitter(folds, repeats, random_state):
    if repeats and repeats > 1:
        return RepeatedStratifiedKFold(
            n_splits=folds, n_repeats=repeats, random_state=random_state
        )
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)


def evaluate_model(build_fn, X, y, folds=5, repeats=5, random_state=0):
    """Cross-validate one classifier. build_fn returns a fresh estimator each
    call so nothing leaks across folds."""
    splitter = _splitter(folds, repeats, random_state)
    scores = {"f1": [], "precision": [], "recall": [], "accuracy": []}

    for train_idx, test_idx in splitter.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = clone(build_fn())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

        scores["f1"].append(f1_score(y_te, y_pred, pos_label=1, zero_division=0))
        scores["precision"].append(precision_score(y_te, y_pred, pos_label=1, zero_division=0))
        scores["recall"].append(recall_score(y_te, y_pred, pos_label=1, zero_division=0))
        scores["accuracy"].append(accuracy_score(y_te, y_pred))

    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in scores.items()}


def print_report(results):
    name_w = max(len(n) for n in results)
    header = "%-*s   %-14s %-14s %-14s %-14s" % (
        name_w, "classifier", "F1", "precision", "recall", "accuracy"
    )
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        def cell(metric):
            m, s = res[metric]
            return "%.3f +/- %.3f" % (m, s)
        print("%-*s   %-14s %-14s %-14s %-14s" % (
            name_w, name, cell("f1"), cell("precision"), cell("recall"), cell("accuracy")
        ))


def write_report(results, path):
    rows = []
    for name, res in results.items():
        row = {"classifier": name}
        for metric in ("f1", "precision", "recall", "accuracy"):
            m, s = res[metric]
            row["%s_mean" % metric] = round(m, 3)
            row["%s_std" % metric] = round(s, 3)
        rows.append(row)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print("\nsaved to %s" % path)

def get_dataset(path):
    """Convenience: returns (df, X, y) where X has TEXT + feature columns."""
    df = load_and_clean(path)
    keep = [TEXT_COL] + FEATURE_COLUMNS
    X = df[keep].copy()
    y = df[TARGET].values
    return X, y

def load_and_clean(path):
    """Read the raw CSV, keep claims with a clean testability label, drop the
    bookkeeping columns, and build the binary target."""
    df = pd.read_csv(path)

    # Keep only rows the annotators agreed are claims.
    df = df[df["label_claim"] == "Claim"].copy()

    # Keep only rows with a clean testability label (drops NO_CONSENSUS / NaN).
    valid = {"Testable", "Not testable"}
    df = df[df["label_testable"].isin(valid)].copy()

    # Binary target: 1 = Testable, 0 = Not testable.
    df[TARGET] = (df["label_testable"] == "Testable").astype(int)

    # Drop bookkeeping columns and the now-redundant raw label.
    drop = [c for c in DROP_COLUMNS + ["label_testable"] if c in df.columns]
    df = df.drop(columns=drop)

    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Evaluate testability classifiers.")
    ap.add_argument("--input", default="claims_labeled.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5,
                    help="Repeated stratified k-fold repeats (1 = plain k-fold).")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="CSV path for the per-classifier metrics.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tf", choices=TF_MODES, default="linear",
                    help="Where to include the Term Frequency Vector: "
                         "'all' (logreg+svm+ripper), 'linear' (logreg+svm only), "
                         "or 'none'.")
    ap.add_argument("--show-rules", action="store_true",
                    help="Fit RIPPER on the full dataset and print the learned rules.")
    args = ap.parse_args()

    X, y = get_dataset(args.input)
    n_pos = int(y.sum())
    print("Loaded %d claims: %d testable, %d not testable" % (len(y), n_pos, len(y) - n_pos))
    print("TF vector mode: %s\n" % args.tf)

    builders = get_builders(args.tf)

    results = {}
    for name, fn in builders.items():
        results[name] = evaluate_model(
            fn, X, y, folds=args.folds, repeats=args.repeats, random_state=args.seed
        )

    print_report(results)
    write_report(results, args.output)

    if args.show_rules:
        _, ripper_tf = tf_flags(args.tf)
        print("\nRIPPER ruleset learned on the full dataset (use_tf=%s)" % ripper_tf)
        print("-" * 56)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rip = build_ripper(use_tf=ripper_tf)
            rip.fit(X, y)
        print(rip.rules_as_text())


if __name__ == "__main__":
    main()
