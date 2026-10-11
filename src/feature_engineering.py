"""
graduate-risk-analytics
src/feature_engineering.py

Target construction and composite index features.
"""

import re
import pandas as pd


# ==========================================================
# TARGET DEFINITION
# ==========================================================

TARGET_COLUMN = "employment_status"
TARGET_SOURCE_COLUMN = "current_situation"

# Any of these selected in the multi-select H1 question counts as
# "employed" (label 1). Everyone else -> 0 ("unemployed"), the
# minority / at-risk class this study is about identifying.
EMPLOYED_RESPONSES = [
    "Employment",
    "Self-employed/freelance work",
    "Military or other civil service"
]

# POS_LABEL is the class this whole study is built around: the
# "unemployed" minority. Every evaluation function downstream
# (evaluation.py) targets this label explicitly rather than
# sklearn's default of label 1, since the default would silently
# score how well a model finds the *majority* (employed) class.
POS_LABEL = 0


def create_target(df):
    """Derive the binary employment_status target (1 = employed,
    0 = unemployed) from the multi-select current_situation field."""

    df = df.copy()

    pattern = "|".join(re.escape(r) for r in EMPLOYED_RESPONSES)

    df[TARGET_COLUMN] = (
        df[TARGET_SOURCE_COLUMN]
        .astype(str)
        .str.contains(pattern, case=False, na=False)
        .astype(int)
    )

    return df


# ==========================================================
# AGE
# ==========================================================

SURVEY_YEAR_MAP = {2023: 2023, 2024: 2025, 2025: 2025}


def create_age(df):
    """Derive age from batch (survey year) and birth_year."""

    df = df.copy()

    batch_numeric = pd.to_numeric(df["batch"], errors="coerce")
    survey_year = batch_numeric.replace(SURVEY_YEAR_MAP)

    df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
    df["age"] = survey_year - df["birth_year"]

    return df


# ==========================================================
# COMPOSITE INDICES
# ==========================================================

SCHOOL_CHOICE_VARS = [
    "school_reputation", "quality_education", "preferred_program",
    "free_tuition", "family_advice"
]

LEARNING_EXPERIENCE_VARS = [
    "group_work", "lectures", "practical_learning", "project_based_learning",
    "written_work", "self_study", "teacher_centered", "course_evaluation",
    "student_choice", "motivation_enhancement"
]

STUDY_CONDITIONS_VARS = [
    "classroom_learning", "campus_facilities", "learning_materials",
    "consultation_opportunities", "teaching_quality", "student_relationships",
    "technical_equipment", "campus_buildings", "library_resources"
]

GUIDANCE_SUPPORT_VARS = [
    "professional_advice", "exam_discussion", "occupational_advice",
    "study_advice"
]

EMPLOYABILITY_VARS = [
    "preparation_for_work", "theory_practice", "practice_oriented_teaching",
    "internship_search_support", "job_search_support"
]

COMPETENCY_VARS = [
    "applied_knowledge", "innovation", "authority", "adaptability",
    "leadership", "analytical_thinking", "goal_orientation",
    "work_organization", "teamwork", "stress_management"
]

RETROSPECTIVE_CHOICE_VARS = ["choose_same_program", "choose_ccsfp_schooling"]

CONSTRUCTS = {
    "school_choice_index": SCHOOL_CHOICE_VARS,
    "learning_experience_index": LEARNING_EXPERIENCE_VARS,
    "study_conditions_index": STUDY_CONDITIONS_VARS,
    "guidance_support_index": GUIDANCE_SUPPORT_VARS,
    "employability_index": EMPLOYABILITY_VARS,
    "competency_index": COMPETENCY_VARS,
    "retrospective_choice_index": RETROSPECTIVE_CHOICE_VARS
}

COMPOSITE_VARS = list(CONSTRUCTS.keys())


def create_composite_indices(df, constructs=CONSTRUCTS):
    """Average each construct's underlying Likert items into a
    single composite index (coercing to numeric defensively)."""

    df = df.copy()

    for index_name, columns in constructs.items():

        valid_columns = [c for c in columns if c in df.columns]

        if not valid_columns:
            continue

        for col in valid_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df[index_name] = df[valid_columns].mean(axis=1)

    return df


# ==========================================================
# FEATURE GROUPS FOR MODELING
# ==========================================================

# Composite indices excluded from modeling because they are
# post-outcome / leakage-prone: both are rated retrospectively in
# the same cross-sectional tracer survey where employment status
# is also reported, so an employed graduate rating "employability"
# or "would I choose this program again" after already knowing
# their own outcome is confounded with the target rather than a
# predictor of it. Still computed (CONSTRUCTS/COMPOSITE_VARS above)
# for descriptive/EDA use, just excluded from CANDIDATE_FEATURES.
LEAKAGE_RISK_VARS = ["employability_index", "retrospective_choice_index"]

MODELING_COMPOSITE_VARS = [v for v in COMPOSITE_VARS if v not in LEAKAGE_RISK_VARS]

NOMINAL_FEATURES = ["program"]

# Reduced, theory-driven feature set (~9 predictors instead of the
# original ~20) to keep the events-per-predictor ratio for the
# minority class (31 unemployed cases) closer to the 10-20:1
# guideline for binary prediction, and to keep only variables that
# could realistically be known before the employment outcome.
CANDIDATE_FEATURES = [
    "program", "study_hours", "internship", "working_student",
    "sex", "age"
] + MODELING_COMPOSITE_VARS

NUMERIC_FEATURES = [f for f in CANDIDATE_FEATURES if f not in NOMINAL_FEATURES]

ANALYSIS_VARIABLES = (
    [TARGET_SOURCE_COLUMN]
    + ["program", "batch", "study_hours", "considered_dropping_out",
       "internship", "mandatory_internship_count",
       "mandatory_internship_duration", "voluntary_internship_count",
       "voluntary_internship_duration", "working_student", "sex",
       "birth_year"]
    + SCHOOL_CHOICE_VARS + LEARNING_EXPERIENCE_VARS + STUDY_CONDITIONS_VARS
    + GUIDANCE_SUPPORT_VARS + EMPLOYABILITY_VARS + COMPETENCY_VARS
    + RETROSPECTIVE_CHOICE_VARS + ["study_satisfaction"]
)

FINAL_COLUMNS = CANDIDATE_FEATURES + [TARGET_COLUMN]


# ==========================================================
# PUBLIC ENTRY POINT
# ==========================================================

def engineer_features(df, constructs=CONSTRUCTS):
    """Full feature engineering pipeline: target, age, composites."""

    df = create_target(df)
    df = create_age(df)
    df = create_composite_indices(df, constructs)

    return df
