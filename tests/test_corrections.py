"""
Tests for the anomaly-score scaling fix and the corrected
resampled-CV statistics. Synthetic data only — no raw survey
export needed.

Run with: pytest tests/test_corrections.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_models import AnomalyDetectorWrapper
from src.evaluation import (
    corrected_ci, corrected_resampled_ttest, bootstrap_ci, friedman_test,
    pairwise_corrected_ttest, cv_split_sizes
)


@pytest.fixture
def anomaly_data():
    rng = np.random.RandomState(0)
    X_normal = rng.normal(0, 1, size=(80, 3))
    X_anom = rng.normal(4, 1, size=(20, 3))
    X = np.vstack([X_normal, X_anom])
    y = np.array([1] * 80 + [0] * 20)
    return X, y


@pytest.mark.parametrize("detector", [
    IsolationForest(random_state=0),
    OneClassSVM(nu=0.2),
    LocalOutlierFactor(novelty=True),
])
def test_anomaly_scores_do_not_depend_on_batch(anomaly_data, detector):
    """A row's probability must be the same whether it is scored
    alone or alongside other rows — per-batch min-max scaling
    broke this, making OOF-tuned thresholds meaningless on the
    holdout set."""

    X, y = anomaly_data
    model = AnomalyDetectorWrapper(detector).fit(X, y)

    full = model.predict_proba(X)
    one_by_one = np.vstack([model.predict_proba(X[i:i + 1]) for i in range(len(X))])
    subset = model.predict_proba(X[75:85])

    np.testing.assert_allclose(full, one_by_one)
    np.testing.assert_allclose(full[75:85], subset)


def test_anomaly_proba_ranks_anomalies_higher(anomaly_data):
    X, y = anomaly_data
    model = AnomalyDetectorWrapper(IsolationForest(random_state=0)).fit(X, y)

    p_anom = model.predict_proba(X)[:, list(model.classes_).index(0)]

    assert p_anom[y == 0].mean() > p_anom[y == 1].mean()
    assert np.all((p_anom > 0) & (p_anom < 1))


def test_anomaly_predict_matches_detector_boundary(anomaly_data):
    """P = 0.5 sits at decision_function == 0, so predict() agrees
    with the detector's own inlier/outlier call."""

    X, y = anomaly_data
    model = AnomalyDetectorWrapper(OneClassSVM(nu=0.2)).fit(X, y)

    detector_says_outlier = model.fitted_detector_.predict(X) == -1
    wrapper_says_pos = model.predict(X) == 0

    np.testing.assert_array_equal(detector_says_outlier, wrapper_says_pos)


def test_corrected_ci_is_wider_than_bootstrap():
    rng = np.random.RandomState(0)
    scores = rng.normal(0.4, 0.1, size=50)
    n_train, n_test = cv_split_sizes(125, 5)

    boot_low, boot_high = bootstrap_ci(scores, n_boot=2000)
    corr_low, corr_high = corrected_ci(scores, n_train, n_test)

    assert corr_high - corr_low > boot_high - boot_low
    assert corr_low < scores.mean() < corr_high


def test_corrected_ttest_matches_formula():
    rng = np.random.RandomState(1)
    a = rng.normal(0.45, 0.1, size=50)
    b = a - rng.normal(0.05, 0.05, size=50)
    n_train, n_test = 100, 25

    t_stat, p_value = corrected_resampled_ttest(a, b, n_train, n_test)

    d = a - b
    expected_t = d.mean() / np.sqrt((1 / 50 + n_test / n_train) * d.var(ddof=1))
    assert t_stat == pytest.approx(expected_t)
    assert 0 <= p_value <= 1


def test_corrected_ttest_identical_scores():
    a = np.full(10, 0.3)
    assert corrected_resampled_ttest(a, a, 100, 25) == (0.0, 1.0)


def test_pairwise_corrected_ttest_shape():
    rng = np.random.RandomState(2)
    raw = {name: rng.normal(0.3, 0.1, size=50) for name in "ABC"}

    result = pairwise_corrected_ttest(raw, 100, 25)

    assert len(result) == 3
    assert {"model_a", "model_b", "mean_diff", "p_value", "significant"} <= set(result.columns)


def test_friedman_per_repeat_blocks():
    rng = np.random.RandomState(3)
    raw = {name: rng.normal(0.3, 0.1, size=50) for name in "ABC"}

    per_fold = friedman_test(raw)
    per_repeat = friedman_test(raw, n_splits=5)

    assert per_fold["p_value"] != per_repeat["p_value"]
