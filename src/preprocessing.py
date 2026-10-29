"""
graduate-risk-analytics
src/preprocessing.py

Raw survey ingestion, cleaning, and encoding for the CCSFP
Institute of Education Tracer Study (2023-2025 graduates).

Public entry point: preprocess(df_raw) -> pd.DataFrame
"""

import hashlib
import warnings
from pathlib import Path

import pandas as pd


# ==========================================================
# RAW DATA PROVENANCE
# ==========================================================

# SHA-256 of the raw export every committed result was produced from.
# This file is NOT the untouched survey export: one respondent's birth
# year was corrected by hand in the spreadsheet (2009 -> 2001, a data
# entry error giving an implausible age of ~14-16) before this
# checksum was taken. The original export isn't available in this
# project, so the correction can't be re-applied in code; if it turns
# up, keep it untouched, apply the fix in create_age(), and update
# this checksum.
RAW_DATA_SHA256 = "1f67cf25ffed56bc31f0e53bbd6f4e489837517266beafcf05c664b5fe361291"


def verify_raw_data(path, expected=RAW_DATA_SHA256):
    """Warn if the raw file differs from the one the committed results
    were produced from (e.g. a different export or further hand edits).
    Returns True when it matches."""

    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    if digest != expected:
        warnings.warn(
            f"{path} does not match RAW_DATA_SHA256 — results may not "
            f"reproduce the committed outputs (got {digest[:12]}...).",
            stacklevel=2
        )
        return False

    return True


# ==========================================================
# RAW COLUMN -> ANALYSIS NAME MAPPING
# ==========================================================

COLUMN_MAPPINGS = {

    "H1.  What applied to your current situation? Multiple answers possible.":
        "current_situation",

    "A0. What program did you complete?":
        "program",

    "A1.    In what year did you graduate from the CCSFP? ":
        "batch",

    "A4.   On the average, how many hours per week did you spend on study activities OUTSIDE of classes during the course of your study?  (This includes doing class requirements, participating in co- and extra-curricular activities.)":
        "study_hours",

    "A5.   During the courses of your studies, have you ever seriously considered dropping from the program you are enrolled in?":
        "considered_dropping_out",

    "A2.   To what extent were the following reasons important for your decision to study at the CCSFP? Please respond to each factor on the five-point scale. [Reputation of the school]":
        "school_reputation",

    "A2.   To what extent were the following reasons important for your decision to study at the CCSFP? Please respond to each factor on the five-point scale. [Quality education]":
        "quality_education",

    "A2.   To what extent were the following reasons important for your decision to study at the CCSFP? Please respond to each factor on the five-point scale. [Offers my preferred program]":
        "preferred_program",

    "A2.   To what extent were the following reasons important for your decision to study at the CCSFP? Please respond to each factor on the five-point scale. [Free tuition and other school fees]":
        "free_tuition",

    "A2.   To what extent were the following reasons important for your decision to study at the CCSFP? Please respond to each factor on the five-point scale. [Advice from parents/relatives/significant others]":
        "family_advice",

    "B1.  Did you do any internship (e.g., student teacher, etc.) during your course of studies (this does not refer to team projects, practical courses, etc.)?":
        "internship",

    "B2.  How many MANDATORY internships did you do in total  during your course of studies?":
        "mandatory_internship_count",

    "B3.  How many total number of weeks did this(ese) MANDATORY internship(s) last? ":
        "mandatory_internship_duration",

    "B4.  How many VOLUNTARY internships did you do in total during your course of studies?":
        "voluntary_internship_count",

    "B4.  How many total number of weeks did this(ese) VOLUNTARY internship(s) last? ":
        "voluntary_internship_duration",

    "C1.  Were you employed during your study? (Please include full-time as well as part-time work; do NOT include internships.)":
        "working_student",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Group work]":
        "group_work",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Lectures]":
        "lectures",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Internship and practical learning]":
        "practical_learning",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Project and/or problem -oriented learning]":
        "project_based_learning",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Written work]":
        "written_work",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Self-study activities]":
        "self_study",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Teacher as the main source of information]":
        "teacher_centered",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Evaluation of courses by students]":
        "course_evaluation",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Choice of course by students]":
        "student_choice",

    "D1.  To what extent were the following aspects of teaching and learning emphasized in your studies? [Enhancement of student motivation]":
        "motivation_enhancement",

    "D2.  How would you rate the study conditions and provisions you experienced at the CCSFP? [Quality of classroom learning]":
        "classroom_learning",

    "D2.  How would you rate the study conditions and provisions you experienced at the CCSFP? [Student recreational facilities on campus]":
        "campus_facilities",

    "D2.  How would you rate the study conditions and provisions you experienced at the CCSFP? [Supply of learning materials (e.g., books, internet access)]":
        "learning_materials",

    "D2.  How would you rate the study conditions and provisions you experienced at the CCSFP? [Opportunity for consultation with teaching staff]":
        "consultation_opportunities",

    "D2.  How would you rate the study conditions and provisions you experienced at the CCSFP? [Teaching quality of lectures]":
        "teaching_quality",

    "D2.  How would you rate the study conditions and provisions you experienced at the CCSFP? [Relationship  with fellow students]":
        "student_relationships",

    "D2.  How would you rate the study conditions and provisions you experienced at the CCSFP? [Availability of technical equipment in the laboratories (computer/science/speech labs]":
        "technical_equipment",

    "D2.  How would you rate the study conditions and provisions you experienced at the CCSFP? [Quality of building]":
        "campus_buildings",

    "D2.  How would you rate the study conditions and provisions you experienced at the CCSFP? [Books in the library]":
        "library_resources",

    "D3.   How do you rate the following advisory and guidance elements in your study course? [Professional advice and guidance provided by teaching staff]":
        "professional_advice",

    "D3.   How do you rate the following advisory and guidance elements in your study course? [Discussion of written examinations, assignments etc.]":
        "exam_discussion",

    "D3.   How do you rate the following advisory and guidance elements in your study course? [Individual occupational advice in your field]":
        "occupational_advice",

    "D3.   How do you rate the following advisory and guidance elements in your study course? [Individual study advice in your field]":
        "study_advice",

    "D4.   How do you rate the following elements related to employment and work in your study course? [Preparation for work]":
        "preparation_for_work",

    "D4.   How do you rate the following elements related to employment and work in your study course? [Relationship between theory and practice]":
        "theory_practice",

    "D4.   How do you rate the following elements related to employment and work in your study course? [Practice-oriented teaching contents]":
        "practice_oriented_teaching",

    "D4.   How do you rate the following elements related to employment and work in your study course? [Support with internship search]":
        "internship_search_support",

    "D4.   How do you rate the following elements related to employment and work in your study course? [Support for employment/job search]":
        "job_search_support",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Mastery of my field/subject-specific knowledge]":
        "applied_knowledge",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Ability to develop new ideas and solutions]":
        "innovation",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Ability to assert my authority]":
        "authority",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Ability to adapt to changing conditions]":
        "adaptability",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Ability to mobilize the capacities of others]":
        "leadership",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Analytical thinking]":
        "analytical_thinking",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Ability to work efficiently towards a goal]":
        "goal_orientation",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Ability to organize my work processes efficiently ]":
        "work_organization",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Ability to work productively with others ]":
        "teamwork",

    "E1.  To what extent did you acquire the following skills/competencies upon graduation? [Ability to perform well under pressure]":
        "stress_management",

    "E2.   Looking back, if you were free to choose again, to what extent would you probably choose the:   [same program]":
        "choose_same_program",

    "E2.   Looking back, if you were free to choose again, to what extent would you probably choose the:   [CCSFP for schooling]":
        "choose_ccsfp_schooling",

    "E3.   In retrospect, to what extent are you satisfied with your studies in general?":
        "study_satisfaction",

    "M1.  What is your gender?":
        "sex",

    "M2.  In which year were you born?":
        "birth_year"
}


# ==========================================================
# ENCODING MAPS
# ==========================================================

BINARY_MAPS = {

    "sex": {"Male": 0, "Female": 1},

    "internship": {"No": 0, "Yes": 1},

    "working_student": {"No": 0, "Yes": 1},

    "considered_dropping_out": {"No": 0, "Yes": 1}
}

ORDINAL_MAPS = {

    "study_hours": {
        "10 hours or less": 1,
        "11 to 19 hours": 2,
        "20 to 29 hours": 3,
        "30 to 39 hours": 4,
        "40 hours or more": 5
    },

    "mandatory_internship_count": {
        "0. Zero": 0, "1. One": 1, "2. Two": 2,
        "3. Three": 3, "4. Four or more": 4
    },

    "mandatory_internship_duration": {
        "0. Zero": 0,
        "1. One to two weeks": 1,
        "2. Three to four weeks": 2,
        "3. Five to six weeks": 3,
        "4. Six to seven weeks": 4,
        "5. More than seven weeks": 5
    },

    "voluntary_internship_count": {
        "0. Zero": 0, "1. One": 1, "2. Two": 2,
        "3. Three": 3, "4. Four or more": 4
    },

    "voluntary_internship_duration": {
        "0. Zero": 0,
        "1. Less than a week": 1,
        "2. One to two weeks": 2,
        "3. Three to four weeks": 3,
        "4. Five to six weeks": 4,
        "5. Six to seven weeks": 5,
        "6. More than seven weeks": 6
    }
}

LIKERT_MAP = {

    "Not at all important": 1, "Slightly important": 2,
    "Moderately important": 3, "Important": 4, "Very important": 5,

    "To no extent": 1, "To a little extent": 2, "To some extent": 3,
    "To a large extent": 4, "To a very large extent": 5,

    "Poor": 1, "Fair": 2, "Good": 3, "Very good": 4, "Excellent": 5,

    "Very dissatisfied": 1, "Dissatisfied": 2, "Neutral": 3,
    "Satisfied": 4, "Very satisfied": 5
}

LIKERT_VARS = [
    "school_reputation", "quality_education", "preferred_program",
    "free_tuition", "family_advice",
    "group_work", "lectures", "practical_learning", "project_based_learning",
    "written_work", "self_study", "teacher_centered", "course_evaluation",
    "student_choice", "motivation_enhancement",
    "classroom_learning", "campus_facilities", "learning_materials",
    "consultation_opportunities", "teaching_quality", "student_relationships",
    "technical_equipment", "campus_buildings", "library_resources",
    "professional_advice", "exam_discussion", "occupational_advice",
    "study_advice",
    "preparation_for_work", "theory_practice", "practice_oriented_teaching",
    "internship_search_support", "job_search_support",
    "applied_knowledge", "innovation", "authority", "adaptability",
    "leadership", "analytical_thinking", "goal_orientation",
    "work_organization", "teamwork", "stress_management",
    "study_satisfaction",
    "choose_same_program", "choose_ccsfp_schooling"
]

# Fills for skip-logic missing values (respondents routed around a
# question by the survey's branching logic, e.g. zero internships)
SKIP_LOGIC_REPLACEMENTS = {
    "mandatory_internship_count": "0. Zero",
    "mandatory_internship_duration": "0. Zero",
    "voluntary_internship_count": "0. Zero",
    "voluntary_internship_duration": "0. Zero"
}

# Raw-value typo/casing/numbering fixes, applied before encoding
HARMONIZATION_MAPS = {

    "mandatory_internship_count": {
        "1.  One": "1. One", "2.  Two": "2. Two",
        "3.  Three": "3. Three", "4.  Four or more": "4. Four or more"
    },

    "mandatory_internship_duration": {
        "3.  Five to six weeks": "3. Five to six weeks"
    },

    "voluntary_internship_count": {
        "0.  None": "0. Zero", "1.  One": "1. One", "2.  Two": "2. Two",
        "3.  Three": "3. Three", "4.  Four or more": "4. Four or more"
    },

    "voluntary_internship_duration": {
        "1.  Less than a week": "1. Less than a week",
        "1. One to two weeks": "2. One to two weeks",
        "2. One to two weeks": "2. One to two weeks",
        "2. Three to four weeks": "3. Three to four weeks",
        "3.  Five to six weeks": "4. Five to six weeks",
        "4.  Five to six weeks": "4. Five to six weeks",
        "4. Six to seven weeks": "5. Six to seven weeks",
        "5. Six to seven weeks": "5. Six to seven weeks",
        "5. More than seven weeks": "6. More than seven weeks"
    }
}


# ==========================================================
# CLEANING
# ==========================================================

def rename_columns(df):
    """Map raw Google Forms headers to analysis-friendly names."""
    return df.rename(columns=COLUMN_MAPPINGS)


def treat_skip_logic_missing_values(df):
    """Fill survey-branching-induced missing values with their
    semantic default (e.g. no internships -> '0. Zero')."""

    df = df.copy()

    for column, replacement in SKIP_LOGIC_REPLACEMENTS.items():
        if column in df.columns:
            df[column] = df[column].fillna(replacement)

    return df


def harmonize_categories(df):
    """Standardize inconsistent raw category labels."""

    df = df.copy()

    for col, mapping in HARMONIZATION_MAPS.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    return df


def remove_duplicates(df):
    """Drop exact duplicate rows."""
    return df.drop_duplicates().copy()


def check_mapping_coverage(df, column_to_mapping=None):
    """
    Compare each column's raw unique values against its encoding
    mapping's keys, and report any raw values that won't be
    matched (and would silently become NaN during encoding).

    Run AFTER treat_skip_logic_missing_values()/harmonize_categories(),
    BEFORE encode(), to catch typos/casing/numbering mismatches
    before they turn into unexplained missing data.
    """

    if column_to_mapping is None:

        column_to_mapping = {c: m for c, m in BINARY_MAPS.items()}
        column_to_mapping.update(ORDINAL_MAPS)

        for col in LIKERT_VARS:
            if col in df.columns:
                column_to_mapping[col] = LIKERT_MAP

    any_issues = False

    for col, mapping in column_to_mapping.items():

        if col not in df.columns:
            continue

        raw_values = set(df[col].dropna().unique())
        unmapped = raw_values - set(mapping.keys())

        if unmapped:
            any_issues = True
            print(
                f"'{col}': {len(unmapped)} raw value(s) not covered "
                f"by its mapping — {sorted(unmapped)}"
            )

    if not any_issues:
        print("All checked columns fully covered by their mappings.")


# ==========================================================
# ENCODING
# ==========================================================

def _encode_column(df, col, mapping):
    """Encode one column, warning if any raw value falls through
    (i.e. wasn't covered by the mapping/harmonization dict)."""

    before_na = df[col].isna().sum()

    encoded = pd.to_numeric(df[col].replace(mapping), errors="coerce")

    new_na = encoded.isna().sum() - before_na

    if new_na > 0:

        unmapped = sorted(set(
            df.loc[encoded.isna() & df[col].notna(), col].unique()
        ))

        print(
            f"Warning: encoding '{col}' introduced {new_na} new "
            f"NaN value(s) — raw values not in the mapping: {unmapped}"
        )

    return encoded


def encode(df, likert_columns=None):
    """Encode binary, ordinal, and Likert-scale columns to numeric."""

    df = df.copy()

    if likert_columns is None:
        likert_columns = LIKERT_VARS

    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = _encode_column(df, col, mapping)

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = _encode_column(df, col, mapping)

    for col in likert_columns:
        if col in df.columns:
            df[col] = _encode_column(df, col, LIKERT_MAP)

    return df


def validate(df):
    """Print a quick data-quality summary (shape, missingness,
    duplicates)."""

    print(f"Shape: {df.shape}")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print(f"Missing (total): {df.isnull().sum().sum()}")

    if len(missing):
        print("Missing (by column):")
        print(missing)
    else:
        print("Missing (by column): none")

    print(f"Duplicate rows: {df.duplicated().sum()}")

    return df


# ==========================================================
# PUBLIC ENTRY POINT
# ==========================================================

def preprocess(df_raw, likert_columns=None):
    """
    Full cleaning + encoding pipeline: rename -> skip-logic fill ->
    harmonize -> dedupe -> encode. Feature engineering (composite
    indices, target construction) happens separately in
    feature_engineering.py.
    """

    df = rename_columns(df_raw)
    df = treat_skip_logic_missing_values(df)
    df = harmonize_categories(df)
    df = remove_duplicates(df)
    df = encode(df, likert_columns=likert_columns)

    return df
