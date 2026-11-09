"""
graduate-risk-analytics
src/robustness.py

Helpers for 07_robustness.ipynb: build every model (baselines +
Family A + Family B) for an arbitrary feature set, and evaluate them
leave-one-group-out (e.g. leave-one-cohort-out).
"""

import numpy as np
import pandas as pd

from src.classification_models import build_classification_models, build_baseline_models
from src.anomaly_models import build_anomaly_models
from src.evaluation import pr_auc_scorer, roc_auc_scorer
from src.feature_engineering import POS_LABEL


def build_study_models(numeric_features, nominal_features, y_train=None):
    """Baselines + Family A (A1-A4) + Family B (B1-B3), fresh and
    unfitted, for the given feature lists. y_train sets XGBoost's
    scale_pos_weight, as in 03_imbalance_classification.ipynb."""

    return {
        **build_baseline_models(numeric_features, nominal_features),
        **build_classification_models(numeric_features, nominal_features, y_train=y_train),
        **build_anomaly_models(numeric_features, nominal_features),
    }


def model_family(name):
    """'Baseline', 'A' or 'B' from a model name's prefix."""

    if name.startswith("BL"):
        return "Baseline"
    return name[0]


def leave_one_group_out(models, X, y, groups):
    """
    Fit each model on all groups but one and score the held-out
    group, for every group. Returns one row per (model, group) with
    the held-out group's size, positives, base rate, PR-AUC and
    ROC-AUC (NaN when the group has a single class).

    models: dict of name -> unfitted estimator (cloned per group).
    """

    from sklearn.base import clone

    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    groups = np.asarray(groups)

    rows = []

    for name, model in models.items():
        for g in np.unique(groups):

            test = groups == g
            fitted = clone(model).fit(X[~test], y[~test])
            y_test = y[test]
            n_pos = int((y_test == POS_LABEL).sum())
            has_both = 0 < n_pos < len(y_test)

            rows.append({
                "model": name,
                "family": model_family(name),
                "held_out_group": g,
                "n": int(test.sum()),
                "n_pos": n_pos,
                "base_rate": n_pos / len(y_test),
                "pr_auc": pr_auc_scorer(fitted, X[test], y_test) if n_pos else np.nan,
                "roc_auc": roc_auc_scorer(fitted, X[test], y_test) if has_both else np.nan,
            })

    return pd.DataFrame(rows)
