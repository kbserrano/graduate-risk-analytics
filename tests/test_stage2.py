"""
Tests for the Stage 2 changes: SMOTE-NC (A3), raw-data checksum,
logistic-regression odds ratios, CV permutation importance, and
nested CV parameter reporting. Synthetic data only.

Run with: pytest tests/test_stage2.py -v
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import POS_LABEL, CANDIDATE_FEATURES
from src.classification_models import build_a3_smote_rf, build_baseline_models
from src.preprocessing import verify_raw_data
from src.explainability import cv_permutation_importance, logreg_odds_ratios
from src.evaluation import nested_cross_validate


@pytest.fixture
def survey_like():
    """Small synthetic table shaped like the modeling data: one
    nominal column, two 0/1 columns, two continuous columns, and an
    unemployment signal driven by `score` and `working_student`."""

    rng = np.random.RandomState(0)
    n = 150
    X = pd.DataFrame({
        "program": rng.choice(["P1", "P2", "P3"], size=n),
        "working_student": rng.randint(0, 2, size=n),
        "sex": rng.randint(0, 2, size=n),
        "score": rng.normal(size=n),
        "noise": rng.normal(size=n),
    })
    risk = 1.5 * X["score"] - 1.5 * X["working_student"] + rng.normal(scale=0.7, size=n)
    y = pd.Series(np.where(risk > 1.0, POS_LABEL, 1 - POS_LABEL))
    return X, y


NUMERIC = ["working_student", "sex", "score", "noise"]
NOMINAL = ["program"]


def test_internship_dropped_from_features():
    assert "internship" not in CANDIDATE_FEATURES


def test_smotenc_synthetic_rows_are_valid(survey_like):
    X, y = survey_like
    model = build_a3_smote_rf(NUMERIC, NOMINAL)

    scaled = model.named_steps["scale"].fit_transform(X)
    X_res, _ = model.named_steps["resample"].fit_resample(scaled, y)
    synthetic = X_res.iloc[len(X):]

    assert len(synthetic) > 0
    assert synthetic["program"].isin(X["program"].unique()).all()
    assert synthetic[["working_student", "sex"]].isin([0, 1]).all().all()


def test_smotenc_pipeline_fits_and_predicts(survey_like):
    X, y = survey_like
    model = build_a3_smote_rf(NUMERIC, NOMINAL).fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_verify_raw_data(tmp_path):
    f = tmp_path / "raw.xlsx"
    f.write_bytes(b"abc")
    sha_abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    assert verify_raw_data(f, expected=sha_abc) is True
    with pytest.warns(UserWarning):
        assert verify_raw_data(f, expected="0" * 64) is False


def test_logreg_odds_ratios_direction(survey_like):
    X, y = survey_like
    logreg = build_baseline_models(NUMERIC, NOMINAL)["BL2_LogReg_Balanced"]

    odds = logreg_odds_ratios(logreg, X, y, n_boot=50).set_index("feature")

    # higher score -> more unemployment; working student -> less
    assert odds.loc["numeric__score", "odds_ratio_unemployed"] > 1
    assert odds.loc["numeric__working_student", "odds_ratio_unemployed"] < 1
    assert (odds["ci_low"] <= odds["odds_ratio_unemployed"]).all()
    assert (odds["odds_ratio_unemployed"] <= odds["ci_high"]).all()


def test_cv_permutation_importance_ranks_signal(survey_like):
    X, y = survey_like
    logreg = build_baseline_models(NUMERIC, NOMINAL)["BL2_LogReg_Balanced"]

    summary, per_fold = cv_permutation_importance(
        logreg, X, y, n_splits=3, n_repeats=2, perm_repeats=3
    )

    assert set(summary["feature"]) == set(X.columns)
    assert len(per_fold) == 3 * 2 * len(X.columns)
    assert summary.iloc[0]["feature"] == "score"
    assert summary.set_index("feature").loc["score", "ci_low"] > 0


def test_nested_cv_returns_best_params(survey_like):
    X, y = survey_like
    logreg = build_baseline_models(NUMERIC, NOMINAL)["BL2_LogReg_Balanced"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, params = nested_cross_validate(
            lambda: logreg, {"classifier__C": [0.1, 1]}, X, y,
            outer_splits=3, outer_repeats=1, return_best_params=True
        )

    assert len(scores) == len(params) == 3
    assert all(p["classifier__C"] in (0.1, 1) for p in params)
