"""
classifiers.py
--------------
Three testability classifiers over the engineered feature set.

  build_logreg()      Logistic Regression
  build_linear_svm()  Linear SVM (LinearSVC)
  build_ripper()      RIPPER rule induction (wittgenstein)

"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import wittgenstein as lw

# Shared with evaluation.py, defined here (not there) so that module doesn't
# need to import anything back from here -- the two files used to import
# these in both directions, which is a circular import that fails whichever
# one is run as the entry point.
TEXT_COL = "claim_text"
BINARY_FEATURES = [
    "contains_numeric",
    "has_hedged_quantifier",
    "has_universal_quantifier",
    "model_is_subject",
]
CATEGORICAL_FEATURES = [
    "verb_type",        # other / internal_state / aggregate_stat / external_behavioral
    "modal_verb_type",  # none / possible / deontic / strong
    "tense",            # past / present / unknown  (kept as one feature)
]
TF_MODES = ("all", "linear", "none")


def _build_feature_union(use_tf=True, min_df=2):
    """ColumnTransformer over the structured features, optionally adding the
    Term Frequency Vector. Selects columns by name from the input DataFrame."""
    #This function is not used by the RIPPER classifier because it wants the column names whereas they can be dropped in this
    transformers = []
    if use_tf:
        transformers.append(
            ("tf", CountVectorizer(lowercase=True, min_df=min_df), TEXT_COL)
        )
    transformers.append(("binary", "passthrough", BINARY_FEATURES))
    transformers.append(
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
    )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_logreg(use_tf=True, min_df=2, C=1, class_weight="balanced", random_state=0):
    return Pipeline(
        steps=[
            ("features", _build_feature_union(use_tf=use_tf, min_df=min_df)),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_linear_svm(use_tf=True, min_df=2, C=1, class_weight="balanced", random_state=0):
    return Pipeline(
        steps=[
            ("features", _build_feature_union(use_tf=use_tf, min_df=min_df)),
            (
                "clf",
                LinearSVC(
                    C=C,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# RIPPER wrapper
# ---------------------------------------------------------------------------

class RipperClassifier(BaseEstimator, ClassifierMixin):
    """Thin sklearn-style wrapper around wittgenstein.RIPPER.

    Trains on the structured features (booleans + categorical taxonomies). If
    use_tf is True it also appends the Term Frequency Vector as numeric count
    columns, fit on the training fold only; wittgenstein bins those numerics
    into threshold conditions. Exposes fit / predict / ruleset_ so the same
    evaluation loop drives it and the rules can be printed for the paper.
    """

    def __init__(self, feature_cols=None, use_tf=False, tf_min_df=2,
                 k=2, prune_size=0.33, random_state=0):
        self.feature_cols = feature_cols
        self.use_tf = use_tf
        self.tf_min_df = tf_min_df
        self.k = k
        self.prune_size = prune_size
        self.random_state = random_state

    def _structured(self, X):
        cols = self.feature_cols or (BINARY_FEATURES + CATEGORICAL_FEATURES)
        # wittgenstein reads object columns as nominal, so cast the binary
        # flags to strings to get equality conditions rather than thresholds.
        return X[cols].astype(str)

    def _build_X(self, X, fit):
        Xs = self._structured(X)
        if not self.use_tf:
            return Xs
        if fit:
            self._vec = CountVectorizer(lowercase=True, min_df=self.tf_min_df)
            counts = self._vec.fit_transform(X[TEXT_COL])
        else:
            counts = self._vec.transform(X[TEXT_COL])
        names = ["tf_%s" % w for w in self._vec.get_feature_names_out()]
        count_df = pd.DataFrame(counts.toarray(), columns=names, index=Xs.index)
        return pd.concat([Xs, count_df], axis=1)

    def fit(self, X, y):
        self._model = lw.RIPPER(
            k=self.k,
            prune_size=self.prune_size,
            random_state=self.random_state,
        )
        Xs = self._build_X(X, fit=True)
        y = pd.Series(np.asarray(y), index=Xs.index, name="testable")
        self._model.fit(Xs, y, pos_class=1)
        self.ruleset_ = self._model.ruleset_
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        Xs = self._build_X(X, fit=False)
        preds = self._model.predict(Xs)
        return np.asarray(preds).astype(int)

    def rules_as_text(self):
        return str(self.ruleset_)


def build_ripper(use_tf=False, tf_min_df=2, k=2, prune_size=0.33, random_state=0):
    return RipperClassifier(
        use_tf=use_tf, tf_min_df=tf_min_df, k=k,
        prune_size=prune_size, random_state=random_state,
    )



def tf_flags(tf_mode):
    """Map a TF mode to (linear_use_tf, ripper_use_tf)."""
    if tf_mode not in TF_MODES:
        raise ValueError("tf_mode must be one of %s" % (TF_MODES,))
    linear = tf_mode in ("all", "linear")
    ripper = tf_mode == "all"
    return linear, ripper


def get_builders(tf_mode="linear"):
    """Return zero-arg builder callables for all three classifiers, configured
    for the requested TF inclusion mode."""
    linear_tf, ripper_tf = tf_flags(tf_mode)
    return {
        "logistic_regression": lambda: build_logreg(use_tf=linear_tf),
        "linear_svm": lambda: build_linear_svm(use_tf=linear_tf),
        "ripper": lambda: build_ripper(use_tf=ripper_tf),
    }
