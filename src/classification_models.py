"""
graduate-risk-analytics
src/classification_models.py

Family A: Imbalance-Aware Classification.
Models that directly learn the employed/unemployed boundary,
each handling the ~80/20 imbalance a different way.

    A1 — Random Forest, class_weight="balanced"
    A2 — XGBoost, scale_pos_weight            (requires xgboost)
    A3 — SMOTE oversampling + Random Forest    (requires imbalanced-learn)
    A4 — ADASYN oversampling + Random Forest   (requires imbalanced-learn)

Plus reference baselines (build_baseline_models): a prior-only
dummy and a class-weighted logistic regression.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42


# ==========================================================
# SHARED PREPROCESSOR
# ==========================================================

def build_preprocessor(numeric_features, nominal_features):
    """One-hot encode nominal features, standardize numeric ones.
    Built fresh per model/fold so no state is shared across
    pipelines."""

    return ColumnTransformer(transformers=[
        ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_features),
        ("numeric", StandardScaler(), numeric_features)
    ])


# ==========================================================
# A1 — RANDOM FOREST, CLASS-WEIGHTED
# ==========================================================

def build_a1_random_forest(numeric_features, nominal_features):
    """A1: Random Forest with class_weight='balanced'. Verified
    against the real dataset."""

    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, nominal_features)),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])


# ==========================================================
# A2 — XGBOOST, SCALE_POS_WEIGHT
# ==========================================================

def build_a2_xgboost(numeric_features, nominal_features, scale_pos_weight=None):
    """
    A2: XGBoost with scale_pos_weight for imbalance.

    scale_pos_weight: if None, computed as
    (majority_count / minority_count) at fit time isn't possible
    inside a plain Pipeline (XGBoost has no built-in "balanced"
    option like sklearn's class_weight) — pass it explicitly,
    typically len(y[y==1]) / len(y[y==0]) computed on your
    training data before building this pipeline.

    Requires: pip install xgboost
    NOT EXECUTED in this project's build environment — verify
    before trusting its output.
    """

    from xgboost import XGBClassifier

    if scale_pos_weight is None:
        scale_pos_weight = 1.0

    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, nominal_features)),
        ("classifier", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE
        ))
    ])


# ==========================================================
# A3 — SMOTE + RANDOM FOREST
# ==========================================================

def build_a3_smote_rf(numeric_features, nominal_features):
    """
    A3: SMOTE-NC oversampling (fit only on training folds, via
    imbalanced-learn's Pipeline) + Random Forest.

    Plain SMOTE after one-hot encoding interpolates the dummy columns,
    producing synthetic graduates who are e.g. 0.4 BEED / 0.6 BSEd
    Math. SMOTE-NC instead interpolates only the continuous features
    and gives each synthetic row a real category (the most common one
    among its neighbours) for `program` and the 0/1 BINARY_FEATURES.
    Order: scale continuous features (so the k-NN distances aren't
    dominated by e.g. age) -> SMOTE-NC -> one-hot encode -> RF.

    Requires: pip install imbalanced-learn
    """

    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTENC
    from src.feature_engineering import BINARY_FEATURES

    categorical = list(nominal_features) + [f for f in numeric_features if f in BINARY_FEATURES]
    continuous = [f for f in numeric_features if f not in BINARY_FEATURES]

    scale = ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), continuous),
            ("categorical", "passthrough", categorical)
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    encode = ColumnTransformer(
        transformers=[("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_features)],
        remainder="passthrough"
    )

    return ImbPipeline([
        ("scale", scale),
        ("resample", SMOTENC(categorical_features=categorical, random_state=RANDOM_STATE)),
        ("encode", encode),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
    ])


# ==========================================================
# A4 — ADASYN + RANDOM FOREST
# ==========================================================

def build_a4_adasyn_rf(numeric_features, nominal_features):
    """
    A4: ADASYN oversampling (fit only on training folds, via
    imbalanced-learn's Pipeline) + Random Forest. ADASYN
    concentrates synthetic examples near the decision boundary
    (harder-to-classify minority points get more synthetic
    neighbors than SMOTE's uniform interpolation) — worth
    comparing against A3 rather than assuming one dominates.

    Requires: pip install imbalanced-learn
    NOT EXECUTED in this project's build environment — verify
    before trusting its output.
    """

    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import ADASYN

    return ImbPipeline([
        ("preprocess", build_preprocessor(numeric_features, nominal_features)),
        ("resample", ADASYN(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
    ])


# ==========================================================
# BUILD ALL FAMILY A MODELS
# ==========================================================

def build_classification_models(
    numeric_features,
    nominal_features,
    y_train=None,
    include_xgboost=True,
    include_smote=True,
    include_adasyn=True
):
    """
    Build the full Family A model dict. XGBoost/SMOTE/ADASYN
    variants are included by default but will raise an
    ImportError with a clear message at build time if their
    library isn't installed — set the corresponding include_*
    flag to False to skip them instead.

    y_train: training labels, used only to compute XGBoost's
    scale_pos_weight (ratio of majority to minority count). Pass
    the training fold's y, not the full dataset's, to avoid
    leaking test-fold class balance into this hyperparameter.
    """

    models = {
        "A1_RandomForest_ClassWeighted": build_a1_random_forest(
            numeric_features, nominal_features
        )
    }

    if include_xgboost:
        scale_pos_weight = 1.0
        if y_train is not None:
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            if n_pos > 0:
                scale_pos_weight = n_neg / n_pos
        models["A2_XGBoost_ScalePosWeight"] = build_a2_xgboost(
            numeric_features, nominal_features, scale_pos_weight
        )

    if include_smote:
        models["A3_SMOTE_RandomForest"] = build_a3_smote_rf(
            numeric_features, nominal_features
        )

    if include_adasyn:
        models["A4_ADASYN_RandomForest"] = build_a4_adasyn_rf(
            numeric_features, nominal_features
        )

    return models


# ==========================================================
# REFERENCE BASELINES
# ==========================================================

def build_baseline_models(numeric_features, nominal_features):
    """
    Reference models every Family A/B result should be read against:

        BL1 — DummyClassifier(strategy="prior"): predicts the training
              base rate for everyone, so its PR-AUC equals the base
              rate (PR-AUC's chance level) and its ROC-AUC is 0.5.
        BL2 — class-weighted L2 logistic regression on the same
              features/preprocessing: the simple, interpretable model
              the Random Forest / resampling variants need to beat to
              justify their added complexity.

    Kept out of build_classification_models() so they never compete
    for "best Family A model" or enter the Family A vs. B tests.
    """

    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "BL1_Dummy_Prior": DummyClassifier(strategy="prior"),
        "BL2_LogReg_Balanced": Pipeline([
            ("preprocess", build_preprocessor(numeric_features, nominal_features)),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=2000
            ))
        ])
    }
