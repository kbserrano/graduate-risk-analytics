"""
graduate-risk-analytics
src/anomaly_models.py

Family B: Anomaly Detection.
These models learn what "normal" (employed) graduates look like
and flag deviations from that pattern — rather than directly
learning an employed/unemployed boundary the way Family A does.

    B1 — Isolation Forest
    B2 — One-Class SVM
    B3 — Local Outlier Factor (novelty=True)

All three are fit ONLY on majority-class ("employed") training
rows, per the standard anomaly-detection convention: the model
learns what normal looks like, then anything that doesn't fit that
pattern at test time — including, hopefully, unemployed
graduates — scores as anomalous.

AnomalyDetectorWrapper adapts each detector to expose predict_proba/
classes_, so the same PR-AUC-based evaluation machinery used for
Family A (evaluation.py) works unmodified for Family B too.

All three are sklearn built-ins and have been verified end-to-end
against the real dataset.
"""

import numpy as np
from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

RANDOM_STATE = 42


# ==========================================================
# SHARED PREPROCESSOR
# ==========================================================

def build_preprocessor(numeric_features, nominal_features):
    """
    One-hot encode nominal features, standardize numeric ones.
    StandardScaler is used here (not MinMaxScaler) because
    One-Class SVM's RBF kernel and LOF's distance metric both
    assume roughly zero-centered, unit-variance inputs; swap in
    MinMaxScaler if you want bounded [0, 1] features instead —
    both are defensible choices for this project.
    """

    return ColumnTransformer(transformers=[
        ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_features),
        ("numeric", StandardScaler(), numeric_features)
    ])


# ==========================================================
# ANOMALY DETECTOR -> CLASSIFIER-LIKE ADAPTER
# ==========================================================

class AnomalyDetectorWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps an unsupervised anomaly detector (IsolationForest,
    OneClassSVM, or LocalOutlierFactor with novelty=True) so it
    can be scored by the same PR-AUC/precision/recall machinery
    used for Family A classifiers.

    Fitting: trains the detector on majority-class ("employed",
    label != pos_label) rows ONLY — the labels are used just to
    select that subset, not as a supervised training signal. This
    is the standard anomaly-detection setup: learn what normal
    looks like, flag what doesn't fit at test time.

    Scoring: detector.decision_function() returns higher values
    for more "normal" points, with 0 at the detector's own
    inlier/outlier boundary. This is inverted and passed through
    a logistic function, scaled by the spread (std) of the
    training rows' decision scores, into a pseudo-probability for
    pos_label (the anomalous/minority class):

        P(pos_label) = sigmoid(-decision_function(X) / scale_)

    The transform is fixed at fit time, so a row's score never
    depends on which other rows are scored alongside it — unlike
    per-batch min-max normalization, which made thresholds tuned
    on one batch (e.g. out-of-fold predictions) meaningless on
    another (e.g. the holdout set). P = 0.5 corresponds to the
    detector's native boundary, so predict() matches the
    detector's own inlier/outlier call. It's monotonic, so PR-AUC
    (which only depends on score ordering) is unaffected.
    """

    def __init__(self, detector, pos_label=0):
        self.detector = detector
        self.pos_label = pos_label

    def fit(self, X, y):

        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))

        normal_mask = y != self.pos_label

        self.fitted_detector_ = clone(self.detector)
        self.fitted_detector_.fit(X[normal_mask])

        train_scores = self.fitted_detector_.decision_function(X[normal_mask])
        scale = np.std(train_scores)
        self.scale_ = scale if scale > 1e-12 else 1.0

        return self

    def decision_function(self, X):
        return self.fitted_detector_.decision_function(X)

    def predict_proba(self, X):

        raw_scores = self.decision_function(X)

        normalized = expit(-raw_scores / self.scale_)

        pos_index = list(self.classes_).index(self.pos_label)

        proba = np.zeros((len(X), 2))
        proba[:, pos_index] = normalized
        proba[:, 1 - pos_index] = 1 - normalized

        return proba

    def predict(self, X):

        proba = self.predict_proba(X)

        pos_index = list(self.classes_).index(self.pos_label)

        neg_label = [c for c in self.classes_ if c != self.pos_label][0]

        return np.where(
            proba[:, pos_index] >= 0.5, self.pos_label, neg_label
        )


# ==========================================================
# B1 — ISOLATION FOREST
# ==========================================================

def build_b1_isolation_forest(numeric_features, nominal_features, contamination="auto"):
    """B1: Isolation Forest. contamination='auto' lets sklearn
    estimate the anomaly threshold; pass a float (e.g. the known
    minority-class proportion) to set it explicitly instead."""

    detector = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE
    )

    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, nominal_features)),
        ("classifier", AnomalyDetectorWrapper(detector))
    ])


# ==========================================================
# B2 — ONE-CLASS SVM
# ==========================================================

def build_b2_one_class_svm(numeric_features, nominal_features, nu=0.2):
    """B2: One-Class SVM (RBF kernel). nu is an upper bound on
    the training-set outlier fraction — defaults to 0.2, close
    to this dataset's actual minority-class rate (~20%)."""

    detector = OneClassSVM(kernel="rbf", nu=nu)

    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, nominal_features)),
        ("classifier", AnomalyDetectorWrapper(detector))
    ])


# ==========================================================
# B3 — LOCAL OUTLIER FACTOR
# ==========================================================

def build_b3_lof(numeric_features, nominal_features, n_neighbors=20, contamination="auto"):
    """B3: Local Outlier Factor. novelty=True is required so the
    fitted detector can score new (test-fold) data — the default
    novelty=False only supports fit_predict on the training data
    itself, which doesn't fit this project's train/test CV
    structure."""

    detector = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True
    )

    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, nominal_features)),
        ("classifier", AnomalyDetectorWrapper(detector))
    ])


# ==========================================================
# BUILD ALL FAMILY B MODELS
# ==========================================================

def build_anomaly_models(numeric_features, nominal_features):
    """Build the full Family B model dict."""

    return {
        "B1_IsolationForest": build_b1_isolation_forest(
            numeric_features, nominal_features
        ),
        "B2_OneClassSVM": build_b2_one_class_svm(
            numeric_features, nominal_features
        ),
        "B3_LocalOutlierFactor": build_b3_lof(
            numeric_features, nominal_features
        )
    }
