"""
graduate-risk-analytics
src/explainability.py

Feature importance / explainability for the winning model.

get_permutation_importance() uses sklearn's built-in permutation
importance — model-agnostic, verified against the real dataset.

get_shap_values() uses the `shap` library's TreeExplainer, per its
documented API, and returns SHAP values already selected for
POS_LABEL (the unemployed/minority class this study cares about) in
a version-robust way. Always cross-check SHAP's ranking against
get_permutation_importance()'s rather than trusting either alone;
they measure different things (SHAP: each prediction's attribution;
permutation importance: drop in overall score when a feature is
shuffled) and agreement between the two is a stronger signal than
either result individually.
"""

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

from src.evaluation import pr_auc_scorer


# ==========================================================
# PERMUTATION IMPORTANCE (VERIFIED)
# ==========================================================

def get_permutation_importance(fitted_model, X, y, n_repeats=20, random_state=42):
    """
    Permutation importance using the same PR-AUC-for-POS_LABEL
    scorer as the rest of this project's evaluation — so the
    importance ranking reflects the metric this study actually
    cares about, not sklearn's default scoring.

    Works for any fitted estimator/pipeline with predict_proba,
    including AnomalyDetectorWrapper-based Family B models — it
    only needs a scorer callable(estimator, X, y), which
    pr_auc_scorer already is.
    """

    result = permutation_importance(
        fitted_model, X, y,
        scoring=pr_auc_scorer,
        n_repeats=n_repeats,
        random_state=random_state
    )

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    return importance.reset_index(drop=True)


# ==========================================================
# SHAP (DOCUMENTED API — NOT EXECUTED HERE)
# ==========================================================

def get_shap_values(fitted_pipeline, X, max_samples=100, pos_label=None):
    """
    SHAP values for a fitted Pipeline's classifier step, via
    shap.TreeExplainer (tree-based models only — RandomForest,
    XGBoost). For non-tree models (e.g. the anomaly detectors,
    or LogisticRegression), use shap.KernelExplainer instead,
    which is model-agnostic but much slower — subsample X more
    aggressively if you go that route.

    Requires: pip install shap

    Returns (shap_values, X_transformed, feature_names):
        shap_values — 2D array (n_samples, n_features) of SHAP
            values for `pos_label` specifically (defaults to this
            project's POS_LABEL, the unemployed/minority class),
            already selected out of whatever shape this SHAP
            version returns. TreeExplainer's binary-classifier
            output shape has changed across SHAP versions — this
            handles both the old list-of-per-class-arrays format
            and the newer single (n_samples, n_features, n_classes)
            array format so callers don't have to know which one
            they got.
        X_transformed — the preprocessed X_sample SHAP was computed
            on (post-ColumnTransformer), matching feature_names.
        feature_names — matches X_transformed's columns.
    """

    import shap

    if pos_label is None:
        from src.feature_engineering import POS_LABEL
        pos_label = POS_LABEL

    preprocessor = fitted_pipeline.named_steps["preprocess"]
    classifier = fitted_pipeline.named_steps["classifier"]

    X_sample = X.sample(
        min(max_samples, len(X)), random_state=42
    ) if len(X) > max_samples else X

    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    raw_shap_values = explainer.shap_values(X_transformed)

    classes = list(classifier.classes_)
    pos_index = classes.index(pos_label)

    if isinstance(raw_shap_values, list):
        # older SHAP: list of (n_samples, n_features) arrays, one per class
        shap_values = raw_shap_values[pos_index]
    elif raw_shap_values.ndim == 3:
        # newer SHAP: single (n_samples, n_features, n_classes) array
        shap_values = raw_shap_values[:, :, pos_index]
    else:
        # already 2D (some binary-classifier configurations return
        # values for the positive class directly)
        shap_values = raw_shap_values

    return shap_values, X_transformed, feature_names
