"""
graduate-risk-analytics
src/explainability.py

Feature importance / explainability.

Primary (used in 06_explainability.ipynb):
    cv_permutation_importance() — permutation importance computed on
        every held-out fold of the same repeated stratified CV used
        everywhere else, so each importance is measured on data the
        model wasn't fitted on, with a corrected CI across folds.
    logreg_odds_ratios() — the logistic-regression baseline's
        coefficients as odds ratios for unemployment, with a
        bootstrap CI over respondents.

get_permutation_importance() (a single fitted model on a single
dataset) and get_shap_values() are kept for ad-hoc use.

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

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold

from src.evaluation import (
    pr_auc_scorer, corrected_ci, cv_split_sizes, N_SPLITS, N_REPEATS, RANDOM_STATE
)
from src.feature_engineering import POS_LABEL


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
# SHAP (optional, not used by the notebooks)
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


# ==========================================================
# CROSS-VALIDATED PERMUTATION IMPORTANCE
# ==========================================================

def cv_permutation_importance(model, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS,
                               perm_repeats=10, ci=0.95, random_state=RANDOM_STATE):
    """
    For each fold of a repeated stratified K-fold split: fit a clone of
    `model` on the training part, then compute permutation importance
    (drop in PR-AUC for POS_LABEL when a feature is shuffled) on the
    held-out part. Scoring importance on held-out folds rather than on
    the training data — or on one 32-row holdout — measures what each
    feature contributes to out-of-sample performance, and the many
    folds give a usable uncertainty estimate.

    Returns (summary, per_fold):
        summary — one row per feature: mean importance, Nadeau-Bengio
            corrected CI across folds, and share of folds where the
            importance was positive. Sorted by mean, descending.
        per_fold — long table (fold, feature, importance).
    """

    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y))

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):

        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            fitted, X.iloc[test_idx], y.iloc[test_idx],
            scoring=pr_auc_scorer, n_repeats=perm_repeats,
            random_state=random_state
        )

        for feature, value in zip(X.columns, result.importances_mean):
            rows.append({"fold": fold, "feature": feature, "importance": value})

    per_fold = pd.DataFrame(rows)
    n_train, n_test = cv_split_sizes(len(y), n_splits)

    summary_rows = []
    for feature, values in per_fold.groupby("feature")["importance"]:
        low, high = corrected_ci(values.values, n_train, n_test, ci=ci)
        summary_rows.append({
            "feature": feature,
            "importance_mean": values.mean(),
            "ci_low": low,
            "ci_high": high,
            "share_folds_positive": (values > 0).mean()
        })

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return summary, per_fold


# ==========================================================
# LOGISTIC REGRESSION ODDS RATIOS
# ==========================================================

def logreg_odds_ratios(model, X, y, n_boot=1000, ci=0.95, random_state=RANDOM_STATE):
    """
    Fit a preprocess -> LogisticRegression pipeline (e.g. the
    BL2_LogReg_Balanced baseline) on all of X, y and report each
    encoded feature's odds ratio for UNEMPLOYMENT (POS_LABEL), per one
    standard deviation for standardized numeric features, or for
    membership in a one-hot category.

    sklearn's coef_ refers to classes_[1]; the sign is flipped when
    that isn't POS_LABEL, so OR > 1 always means "higher value ->
    higher odds of being unemployed".

    CI: percentile bootstrap over respondents (stratified by y),
    refitting the whole pipeline each time. Rows ARE independent
    here, unlike CV fold scores, so a plain bootstrap is appropriate.
    The model is L2-regularized (sklearn's default C=1), so odds
    ratios are shrunk toward 1; with all program levels one-hot
    encoded (no reference level), program ORs are relative to the
    average program rather than to a baseline program.
    """

    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y))

    def fit_log_or(Xf, yf):
        fitted = clone(model).fit(Xf, yf)
        classifier = fitted.named_steps["classifier"]
        sign = 1 if classifier.classes_[1] == POS_LABEL else -1
        names = fitted.named_steps["preprocess"].get_feature_names_out()
        return pd.Series(sign * classifier.coef_[0], index=names)

    point = fit_log_or(X, y)

    rng = np.random.RandomState(random_state)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(np.where(y == label)[0], size=(y == label).sum(), replace=True)
            for label in np.unique(y)
        ])
        boot.append(fit_log_or(X.iloc[idx], y.iloc[idx]))
    boot = pd.DataFrame(boot)

    alpha = (1 - ci) / 2
    result = pd.DataFrame({
        "feature": point.index,
        "odds_ratio_unemployed": np.exp(point.values),
        "ci_low": np.exp(boot.quantile(alpha).values),
        "ci_high": np.exp(boot.quantile(1 - alpha).values),
        "share_boot_or_above_1": (boot > 0).mean().values
    })

    return result.sort_values("odds_ratio_unemployed", ascending=False).reset_index(drop=True)
