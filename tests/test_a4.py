"""
Tests for A4's ValidCategoryADASYN: synthetic rows must have exactly
one program and 0/1 binary values, original rows must be untouched,
and the sampler must work with clone/set_params (used by nested CV).
Synthetic data only.

Run with: pytest tests/test_a4.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import POS_LABEL
from src.classification_models import build_a4_adasyn_rf
from src.resampling import ValidCategoryADASYN

NUMERIC = ["working_student", "sex", "score", "noise"]
NOMINAL = ["program"]


@pytest.fixture
def survey_like():
    rng = np.random.RandomState(1)
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


def _resample(model, X, y):
    Z = model.named_steps["preprocess"].fit_transform(X)
    Z_res, y_res = model.named_steps["resample"].fit_resample(Z, y)
    return Z, Z_res, y_res


def test_synthetic_rows_have_valid_categories(survey_like):
    X, y = survey_like
    Z, Z_res, _ = _resample(build_a4_adasyn_rf(NUMERIC, NOMINAL), X, y)

    synthetic = Z_res.iloc[len(Z):]
    onehot = [c for c in Z_res.columns if c.startswith("program_")]

    assert len(synthetic) > 0
    assert (synthetic[onehot].sum(axis=1) == 1).all()
    assert synthetic[onehot + ["working_student", "sex"]].isin([0, 1]).all().all()


def test_original_rows_untouched(survey_like):
    X, y = survey_like
    Z, Z_res, y_res = _resample(build_a4_adasyn_rf(NUMERIC, NOMINAL), X, y)

    np.testing.assert_allclose(Z_res.iloc[:len(Z)].to_numpy(), Z.to_numpy())
    assert (y_res[:len(y)] == y.to_numpy()).all()


def test_continuous_columns_still_interpolated(survey_like):
    X, y = survey_like
    Z, Z_res, _ = _resample(build_a4_adasyn_rf(NUMERIC, NOMINAL), X, y)

    synthetic_scores = Z_res.iloc[len(Z):]["score"]
    assert not synthetic_scores.isin(Z["score"]).all()


def test_clone_and_set_params(survey_like):
    model = clone(build_a4_adasyn_rf(NUMERIC, NOMINAL))
    model.set_params(resample__n_neighbors=3)

    assert model.named_steps["resample"].n_neighbors == 3
    assert model.named_steps["resample"].onehot_prefixes == ("program_",)


def test_requires_dataframe():
    sampler = ValidCategoryADASYN(random_state=0)
    with pytest.raises(TypeError):
        sampler.fit_resample(np.zeros((10, 2)), np.array([0] * 3 + [1] * 7))
