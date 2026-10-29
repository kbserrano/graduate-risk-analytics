"""
Tests for the robustness additions: outcome-definition variants,
ROC-AUC scoring, baselines, and leave-one-group-out evaluation.
Synthetic data only — no raw survey export needed.

Run with: pytest tests/test_stage1.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    create_target, create_target_variants, TARGET_COLUMN, POS_LABEL
)
from src.evaluation import roc_auc_scorer, pr_auc_scorer
from src.classification_models import build_baseline_models
from src.robustness import leave_one_group_out, model_family


def test_target_variants():
    df = pd.DataFrame({"current_situation": [
        "Employment",
        "Not employed but searching for a job",
        "Not employed and not searching for a job",
        "Occasional job (just to earn money)",
        "Occasional job (just to earn money), Not employed but searching for a job",
        "House wife/husband, family work",
        "Self-employed/freelance work, Further academic education",
    ]})

    out = create_target_variants(create_target(df))

    assert out[TARGET_COLUMN].tolist() == [1, 0, 0, 0, 0, 0, 1]
    lf = out["employment_status_labour_force"]
    assert lf.iloc[[0, 6]].tolist() == [1, 1]
    assert lf.iloc[[1, 4]].tolist() == [0, 0]          # searching -> at-risk
    assert lf.iloc[[2, 3, 5]].isna().all()              # not searching -> excluded
    assert out["employment_status_occasional_employed"].tolist() == [1, 0, 0, 1, 1, 0, 1]


class _ScoreModel:
    """Returns a fixed P(POS_LABEL) per row; X is the score itself."""
    classes_ = np.array([POS_LABEL, 1 - POS_LABEL])

    def predict_proba(self, X):
        p = np.asarray(X, dtype=float)
        return np.column_stack([p, 1 - p])


def test_roc_auc_scorer_targets_pos_label():
    y = np.array([POS_LABEL, POS_LABEL, 1 - POS_LABEL, 1 - POS_LABEL])
    scores_pos_high = np.array([0.9, 0.8, 0.2, 0.1])

    assert roc_auc_scorer(_ScoreModel(), scores_pos_high, y) == pytest.approx(1.0)
    assert roc_auc_scorer(_ScoreModel(), 1 - scores_pos_high, y) == pytest.approx(0.0)


@pytest.fixture
def toy_data():
    rng = np.random.RandomState(0)
    n = 120
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                      "grp": rng.choice(["a", "b", "c"], size=n)})
    y = pd.Series(np.where(X["x1"] + rng.normal(scale=0.5, size=n) > 0.8, 0, 1))
    return X, y


def test_dummy_baseline_pr_auc_equals_base_rate(toy_data):
    X, y = toy_data
    dummy = build_baseline_models(["x1", "x2"], [])["BL1_Dummy_Prior"].fit(X, y)

    assert pr_auc_scorer(dummy, X, y) == pytest.approx((y == POS_LABEL).mean())
    assert roc_auc_scorer(dummy, X, y) == pytest.approx(0.5)


def test_logreg_baseline_learns_signal(toy_data):
    X, y = toy_data
    lr = build_baseline_models(["x1", "x2"], [])["BL2_LogReg_Balanced"].fit(X, y)

    assert roc_auc_scorer(lr, X, y) > 0.8


def test_leave_one_group_out(toy_data):
    X, y = toy_data
    models = build_baseline_models(["x1", "x2"], [])

    out = leave_one_group_out(models, X, y, X["grp"])

    assert len(out) == 2 * 3
    assert set(out["held_out_group"]) == {"a", "b", "c"}
    assert out.groupby("model")["n"].sum().eq(len(y)).all()
    assert set(out["family"]) == {"Baseline"}


def test_model_family():
    assert model_family("BL2_LogReg_Balanced") == "Baseline"
    assert model_family("A4_ADASYN_RandomForest") == "A"
    assert model_family("B3_LocalOutlierFactor") == "B"
