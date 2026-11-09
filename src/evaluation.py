"""
graduate-risk-analytics
src/evaluation.py

Evaluation framework shared by both model families:
repeated stratified k-fold CV, PR-AUC-primary scoring targeting
the minority (unemployed) class, model comparison tables, and
statistical significance testing (Friedman + Wilcoxon).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    make_scorer,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)

from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats import t as t_dist

from src.feature_engineering import POS_LABEL
from src.reporting import save_figure

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10  # repeated stratified CV, per the study's design
N_BOOT = 10000  # bootstrap resamples for PR-AUC confidence intervals


# ==========================================================
# CLASS-OF-INTEREST-AWARE PR-AUC SCORER
# ==========================================================

def pr_auc_scorer(estimator, X, y_true):
    """
    Average precision (PR-AUC) for POS_LABEL specifically. Uses
    estimator.classes_ to find the right probability column
    rather than assuming index 1 is "positive" — sklearn's
    default convention, which would score the majority
    (employed) class here, not the one this study is about.
    """

    y_proba = estimator.predict_proba(X)
    class_index = list(estimator.classes_).index(POS_LABEL)
    y_score = y_proba[:, class_index]

    return average_precision_score(y_true, y_score, pos_label=POS_LABEL)


def roc_auc_scorer(estimator, X, y_true):
    """
    ROC-AUC with POS_LABEL scored as the positive class. Reported
    alongside PR-AUC because PR-AUC's chance level is the base rate
    (~0.20 here), so a PR-AUC modestly above 0.20 can hide a model
    whose overall ranking is no better than chance (ROC-AUC ~0.5).
    """

    y_proba = estimator.predict_proba(X)
    class_index = list(estimator.classes_).index(POS_LABEL)
    y_score = y_proba[:, class_index]

    return roc_auc_score((np.asarray(y_true) == POS_LABEL).astype(int), y_score)


SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label=POS_LABEL, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=POS_LABEL, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=POS_LABEL, zero_division=0),
    "pr_auc": pr_auc_scorer,
    "roc_auc": roc_auc_scorer
}


# ==========================================================
# REPEATED STRATIFIED CROSS-VALIDATION
# ==========================================================

def cross_validate_model(model, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """
    Run repeated stratified k-fold CV for a single model and
    return the raw per-fold scores (not just the mean) — needed
    for the Friedman/Wilcoxon tests below, which compare
    distributions of fold scores across models, not single
    summary numbers.
    """

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )

    return cross_validate(model, X, y, cv=cv, scoring=SCORING)


def bootstrap_ci(scores, n_boot=N_BOOT, ci=0.95, random_state=RANDOM_STATE):
    """
    Percentile-bootstrap confidence interval for the mean of an
    array of per-fold scores (e.g. PR-AUC across repeated CV
    folds). Resamples the fold scores themselves with replacement
    n_boot times, takes the mean of each resample, and returns the
    (1-ci)/2 and 1-(1-ci)/2 percentiles of that distribution.

    This is a CI on the CV estimate itself (uncertainty from having
    only ~157 cases / a finite number of folds) — not a claim about
    a single point estimate's precision beyond what the data support.

    Caveat: resampling fold scores as if they were independent
    ignores the overlap between CV training sets, so this interval
    is too narrow. Prefer corrected_ci(); this is kept for
    comparison.
    """

    rng = np.random.RandomState(random_state)
    scores = np.asarray(scores)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        resample = rng.choice(scores, size=len(scores), replace=True)
        boot_means[i] = resample.mean()

    alpha = (1 - ci) / 2
    lower, upper = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])

    return lower, upper


def compare_models(models, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS,
                    n_boot=N_BOOT, ci=0.95):
    """
    Cross-validate every model in `models` (dict of name -> estimator)
    and return (summary_table, raw_scores):

        summary_table — one row per model, mean +/- std per metric,
            sorted by PR-AUC descending, plus pr_auc_ci_low/
            pr_auc_ci_high (bootstrap 95% CI on the mean PR-AUC —
            report this alongside the point estimate, not just
            mean +/- std, given the small sample here), the
            Nadeau-Bengio corrected CI, and base_rate (the share of
            POS_LABEL in y — PR-AUC's chance level).
        raw_scores — dict of {model_name: array of per-fold PR-AUC
            scores}, for feeding into the statistical tests below.
    """

    rows = {}
    raw_scores = {}
    n_train, n_test = cv_split_sizes(len(y), n_splits)

    for name, model in models.items():

        scores = cross_validate_model(model, X, y, n_splits, n_repeats)

        row = {}
        for metric in SCORING:
            row[metric] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()

        pr_auc_scores = scores["test_pr_auc"]
        ci_low, ci_high = bootstrap_ci(pr_auc_scores, n_boot=n_boot, ci=ci)
        row["pr_auc_ci_low"] = ci_low
        row["pr_auc_ci_high"] = ci_high

        corr_low, corr_high = corrected_ci(pr_auc_scores, n_train, n_test, ci=ci)
        row["pr_auc_corrected_ci_low"] = corr_low
        row["pr_auc_corrected_ci_high"] = corr_high
        row["base_rate"] = (np.asarray(y) == POS_LABEL).mean()

        rows[name] = row
        raw_scores[name] = pr_auc_scores

    results = pd.DataFrame(rows).T
    ordered_cols = (
        list(SCORING.keys()) + [f"{m}_std" for m in SCORING]
        + ["pr_auc_ci_low", "pr_auc_ci_high",
           "pr_auc_corrected_ci_low", "pr_auc_corrected_ci_high", "base_rate"]
    )

    summary_table = results[ordered_cols].sort_values("pr_auc", ascending=False)

    return summary_table, raw_scores


# ==========================================================
# STATISTICAL SIGNIFICANCE TESTING
# ==========================================================

def friedman_test(raw_scores, n_splits=None):
    """
    Friedman test across all models' per-fold PR-AUC scores —
    tests whether at least one model's performance distribution
    differs significantly from the others across the repeated CV
    folds. A non-significant result (p >= 0.05) means this study
    cannot claim any model is reliably better than the rest at
    the 95% confidence level.

    All score arrays must be the same length (same CV folds/
    repeats) for the test to be valid — compare_models() already
    guarantees this since every model runs through the same
    RepeatedStratifiedKFold split sequence.

    n_splits: if given, fold scores are first averaged within each
    CV repeat (folds arrive repeat by repeat), so the test's blocks
    are the r repeats rather than the k x r overlapping folds.
    Recommended — raw folds aren't independent blocks, which
    inflates the statistic (see corrected_resampled_ttest()).
    """

    arrays = [np.asarray(s, dtype=float) for s in raw_scores.values()]
    if n_splits is not None:
        arrays = [s.reshape(-1, n_splits).mean(axis=1) for s in arrays]

    statistic, p_value = friedmanchisquare(*arrays)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05
    }


def pairwise_wilcoxon(raw_scores, alpha=0.05, correction="bonferroni"):
    """
    Pairwise Wilcoxon signed-rank tests between every pair of
    models' per-fold PR-AUC scores — for pinpointing WHICH pairs
    differ significantly, once the Friedman test above says at
    least one difference exists somewhere.

    correction="bonferroni" divides alpha by the number of pairwise
    tests run, since testing many pairs increases the chance of a
    false positive by chance alone — without it, a "significant"
    pairwise result here would overstate how much confidence the
    number of comparisons actually earns.

    Caveat: treats the k x r fold scores as independent pairs,
    which they aren't (overlapping training sets), so p-values
    are optimistic. Kept as a sensitivity check;
    pairwise_corrected_ttest() is the primary test.
    """

    names = list(raw_scores.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    corrected_alpha = alpha / n_pairs if correction == "bonferroni" else alpha

    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            a, b = names[i], names[j]

            try:
                statistic, p_value = wilcoxon(raw_scores[a], raw_scores[b])
            except ValueError:
                # all differences are zero (identical score arrays)
                statistic, p_value = 0.0, 1.0

            rows.append({
                "model_a": a,
                "model_b": b,
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < corrected_alpha
            })

    return pd.DataFrame(rows).sort_values("p_value")


# ==========================================================
# CORRECTED RESAMPLED T-TEST (Nadeau & Bengio, 2003)
# ==========================================================
#
# Repeated k-fold CV scores are NOT independent samples: every
# repeat reuses the same rows, and any two folds' training sets
# overlap heavily. Tests and intervals that treat the k x r fold
# scores as i.i.d. (Wilcoxon, Friedman on raw folds, the
# percentile bootstrap above) therefore understate the variance
# and overstate significance. Nadeau & Bengio's correction
# inflates the variance term from 1/k to (1/k + n_test/n_train)
# to account for that overlap — the standard fix for comparing
# models on repeated CV (see also Bouckaert & Frank, 2004).

def _corrected_std_error(scores, n_train, n_test):
    scores = np.asarray(scores, dtype=float)
    k = len(scores)
    return np.sqrt((1 / k + n_test / n_train) * scores.var(ddof=1))


def cv_split_sizes(n_samples, n_splits=N_SPLITS):
    """(n_train, n_test) per fold for a k-fold split of n_samples."""

    n_test = n_samples / n_splits
    return n_samples - n_test, n_test


def corrected_ci(scores, n_train, n_test, ci=0.95):
    """
    t-based confidence interval for the mean of repeated-CV fold
    scores, using the Nadeau-Bengio corrected variance. Wider
    (and more honest) than bootstrap_ci(), which treats fold
    scores as independent.
    """

    scores = np.asarray(scores, dtype=float)
    se = _corrected_std_error(scores, n_train, n_test)
    t_crit = t_dist.ppf(1 - (1 - ci) / 2, df=len(scores) - 1)

    return scores.mean() - t_crit * se, scores.mean() + t_crit * se


def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
    """
    Two-sided corrected resampled paired t-test on two models'
    fold scores (same folds, same order). Returns (t, p_value).
    """

    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    se = _corrected_std_error(diff, n_train, n_test)

    if se < 1e-12:
        return 0.0, 1.0

    t_stat = diff.mean() / se
    p_value = 2 * t_dist.sf(abs(t_stat), df=len(diff) - 1)

    return t_stat, p_value


def pairwise_corrected_ttest(raw_scores, n_train, n_test, alpha=0.05,
                              correction="bonferroni"):
    """
    Pairwise corrected resampled t-tests between every pair of
    models' per-fold PR-AUC scores — the primary pairwise test
    for this study. Same output shape as pairwise_wilcoxon(), plus
    mean_diff (model_a minus model_b).
    """

    names = list(raw_scores.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    corrected_alpha = alpha / n_pairs if correction == "bonferroni" else alpha

    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            a, b = names[i], names[j]
            statistic, p_value = corrected_resampled_ttest(
                raw_scores[a], raw_scores[b], n_train, n_test
            )

            rows.append({
                "model_a": a,
                "model_b": b,
                "mean_diff": np.mean(raw_scores[a]) - np.mean(raw_scores[b]),
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < corrected_alpha
            })

    return pd.DataFrame(rows).sort_values("p_value")


# ==========================================================
# NESTED CROSS-VALIDATION (model selection/tuning + honest estimate)
# ==========================================================

def nested_cross_validate(build_model_fn, param_grid, X, y,
                           outer_splits=5, outer_repeats=5,
                           inner_splits=3, random_state=RANDOM_STATE,
                           return_best_params=False):
    """
    Nested CV: an outer RepeatedStratifiedKFold loop provides the
    honest generalization estimate (each outer test fold is never
    touched during that fold's model/hyperparameter selection);
    an inner StratifiedKFold + GridSearchCV, run only on that
    fold's training portion, picks hyperparameters by PR-AUC.

    This goes further than a single 80/20 holdout + CV-on-train
    (already used for model comparison/threshold tuning elsewhere
    in this module): it also tunes hyperparameters without letting
    that tuning leak into the performance estimate, and it does so
    across many different outer splits rather than just one, which
    matters given how few minority-class cases (~31) this dataset
    has to split.

    build_model_fn: zero-arg callable returning an unfitted
        Pipeline/estimator (e.g. lambda: build_a1_random_forest(...)).
    param_grid: sklearn-style param grid, keyed with the
        'classifier__' prefix matching build_model_fn's pipeline.

    Returns: array of per-outer-fold PR-AUC scores (POS_LABEL), or
    (scores, best_params) with return_best_params=True, where
    best_params lists the hyperparameters each outer fold's inner
    search picked — useful to check how stable tuning is.

    Every model run with the same outer_splits/outer_repeats/
    random_state sees the same outer folds, so per-fold scores can be
    compared with corrected_resampled_ttest().
    """

    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=random_state
    )

    X = X.reset_index(drop=True) if hasattr(X, "reset_index") else X
    y = np.asarray(y)

    outer_scores = []
    best_params = []

    for train_idx, test_idx in outer_cv.split(X, y):

        X_train_outer = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test_outer = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )

        search = GridSearchCV(
            build_model_fn(), param_grid, scoring=pr_auc_scorer,
            cv=inner_cv, n_jobs=None
        )
        search.fit(X_train_outer, y_train_outer)

        outer_scores.append(pr_auc_scorer(search.best_estimator_, X_test_outer, y_test_outer))
        best_params.append(search.best_params_)

    if return_best_params:
        return np.array(outer_scores), best_params

    return np.array(outer_scores)


# ==========================================================
# THRESHOLD SELECTION
# ==========================================================

def get_out_of_fold_probabilities(model, X, y, n_splits=N_SPLITS):
    """Out-of-fold predicted probabilities for POS_LABEL, for
    threshold selection without ever touching the holdout set."""

    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_proba_all = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

    classes = sorted(pd.Series(y).unique())
    pos_index = classes.index(POS_LABEL)

    return y_proba_all[:, pos_index]


def find_best_threshold(y_true, y_proba, metric="f1"):
    """Sweep thresholds against out-of-fold probabilities and
    return the threshold maximizing the chosen metric for
    POS_LABEL. metric: 'f1' (balanced), 'recall' (catch more,
    more false alarms), or 'precision' (fewer false alarms, miss
    more)."""

    precision, recall, thresholds = precision_recall_curve(
        y_true, y_proba, pos_label=POS_LABEL
    )

    precision_t, recall_t = precision[:-1], recall[:-1]

    f1_t = np.where(
        (precision_t + recall_t) > 0,
        2 * precision_t * recall_t / (precision_t + recall_t + 1e-12),
        0.0
    )

    scores = {"f1": f1_t, "recall": recall_t, "precision": precision_t}[metric]

    best_idx = int(np.argmax(scores))

    return thresholds[best_idx], precision_t[best_idx], recall_t[best_idx], f1_t[best_idx]


def predict_with_threshold(model, X, threshold=0.5):
    """Predict labels using a threshold on P(y == POS_LABEL),
    rather than the model's default .predict()."""

    classes = list(model.classes_)
    pos_index = classes.index(POS_LABEL)
    neg_label = [c for c in classes if c != POS_LABEL][0]

    y_score = model.predict_proba(X)[:, pos_index]

    return np.where(y_score >= threshold, POS_LABEL, neg_label)


def evaluate_on_holdout(model, X_test, y_test, threshold=0.5):
    """Final holdout metrics at a given threshold — use only for
    reporting the chosen model/threshold, never for selecting
    between candidates (that's what CV is for)."""

    classes = list(model.classes_)
    pos_index = classes.index(POS_LABEL)
    y_score = model.predict_proba(X_test)[:, pos_index]
    y_pred = predict_with_threshold(model, X_test, threshold)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_score, pos_label=POS_LABEL),
        "roc_auc": roc_auc_score((np.asarray(y_test) == POS_LABEL).astype(int), y_score),
        "base_rate": (np.asarray(y_test) == POS_LABEL).mean(),
        "threshold_used": threshold
    }


# ==========================================================
# PLOTS
# ==========================================================

def plot_precision_recall_curves(models, X_test, y_test, save_as=None):
    """PR curves for POS_LABEL across all fitted models.

    save_as: optional file name (no extension) to also save the
    figure to reports/figures/.

    Uses from_predictions with manually-extracted scores rather
    than from_estimator — sklearn's from_estimator runs an
    internal "is this a standard binary classifier" check that
    AnomalyDetectorWrapper fails (despite predict_proba working
    correctly), since it isn't a check sklearn's own estimators
    are subject to in the same way.
    """

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():

        if not hasattr(model, "predict_proba"):
            continue

        classes = list(model.classes_)
        pos_index = classes.index(POS_LABEL)
        y_score = model.predict_proba(X_test)[:, pos_index]

        PrecisionRecallDisplay.from_predictions(
            y_test, y_score, pos_label=POS_LABEL, name=name, ax=ax
        )

    baseline = (y_test == POS_LABEL).mean()
    ax.axhline(baseline, linestyle="--", color="gray",
               label=f"Chance (baseline={baseline:.2f})")

    ax.set_title(f"Precision-Recall Curves — Class {POS_LABEL} (Unemployed)")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    if save_as:
        save_figure(fig, save_as)
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, threshold=None,
                           labels=("Unemployed", "Employed"), title="Confusion Matrix",
                           save_as=None):
    """threshold=None uses the model's own .predict(); pass a
    threshold to apply it via predict_with_threshold instead.
    save_as: optional file name (no extension) to also save the
    figure to reports/figures/."""

    fig, ax = plt.subplots(figsize=(5, 4))

    if threshold is None:
        y_pred = model.predict(X_test)
        classes = sorted(model.classes_)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, labels=classes, display_labels=labels, cmap="Blues", ax=ax
        )
    else:
        y_pred = predict_with_threshold(model, X_test, threshold)
        classes = sorted(model.classes_)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, labels=classes, display_labels=labels, cmap="Blues", ax=ax
        )

    ax.set_title(title)
    plt.tight_layout()
    if save_as:
        save_figure(fig, save_as)
    plt.show()
