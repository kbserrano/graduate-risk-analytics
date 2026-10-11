"""
Smoke tests for the graduate-risk-analytics pipeline.

Run with: pytest tests/test_pipeline.py -v

These check the pipeline produces a valid, complete dataset and that
the evaluation framework's core scorer behaves correctly — they are
not a substitute for inspecting the notebooks' actual results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    rename_columns, treat_skip_logic_missing_values, harmonize_categories,
    remove_duplicates, encode
)
from src.feature_engineering import (
    engineer_features, ANALYSIS_VARIABLES, FINAL_COLUMNS, TARGET_COLUMN,
    CANDIDATE_FEATURES, POS_LABEL
)
from src.evaluation import pr_auc_scorer


RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tracer_study_raw.xlsx"


@pytest.fixture(scope="module")
def processed_df():

    if not RAW_DATA_PATH.exists():
        pytest.skip(f"Raw data not found at {RAW_DATA_PATH} — "
                    f"place the survey export there to run this test.")

    df = pd.read_excel(RAW_DATA_PATH)
    df = rename_columns(df)
    df = df[ANALYSIS_VARIABLES].copy()
    df = treat_skip_logic_missing_values(df)
    df = harmonize_categories(df)
    df = remove_duplicates(df)
    df = encode(df)
    df = engineer_features(df)

    return df[FINAL_COLUMNS]


def test_pipeline_produces_no_missing_values(processed_df):
    assert processed_df.isnull().sum().sum() == 0


def test_pipeline_produces_expected_columns(processed_df):
    assert list(processed_df.columns) == FINAL_COLUMNS
    assert len(CANDIDATE_FEATURES) == processed_df.shape[1] - 1


def test_target_is_binary(processed_df):
    assert set(processed_df[TARGET_COLUMN].unique()) <= {0, 1}


def test_age_is_plausible(processed_df):
    # catches the kind of birth-year data-entry error found during
    # this project's development (e.g. 2009 typed instead of 2001)
    assert processed_df["age"].min() >= 18
    assert processed_df["age"].max() <= 80


def test_pr_auc_scorer_targets_pos_label():
    """A dummy 'model' that assigns high P(class=POS_LABEL) to every
    true POS_LABEL row should score close to 1.0 — confirms the
    scorer reads the correct probability column rather than
    assuming index 1 is always positive."""

    class DummyModel:
        classes_ = np.array([POS_LABEL, 1 - POS_LABEL])

        def predict_proba(self, X):
            # X here is just the true labels, for this synthetic check
            proba = np.zeros((len(X), 2))
            proba[:, 0] = (X == POS_LABEL).astype(float)
            proba[:, 1] = 1 - proba[:, 0]
            return proba

    y_true = np.array([POS_LABEL, POS_LABEL, 1 - POS_LABEL, 1 - POS_LABEL])
    score = pr_auc_scorer(DummyModel(), y_true, y_true)

    assert score > 0.99
