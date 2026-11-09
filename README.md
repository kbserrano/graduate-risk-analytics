# Graduate Risk Analytics

**Comparative Evaluation of Imbalance-Aware Classification and Anomaly
Detection Methods for Identifying At-Risk Graduates in Imbalanced Tracer
Study Data Using PR-AUC**

A benchmarking framework comparing two methodological families for
identifying unemployed graduates in a highly imbalanced (~80/20) tracer
study dataset:

- **Family A — Imbalance-Aware Classification**: directly learns the
  employed/unemployed boundary (class weighting, oversampling).
- **Family B — Anomaly Detection**: learns what "normal" (employed)
  graduates look like and flags deviations from that pattern.

Built around **PR-AUC** as the primary metric, since accuracy can look
misleadingly strong on a rare-event problem like this one. PR-AUC is
always read against its chance level (the base rate, ~0.20 here) and
reported alongside ROC-AUC, which shows whether a model ranks
unemployed graduates above employed ones any better than chance (0.5).

## Research questions this project answers

1. How well do imbalance-aware classification methods identify unemployed graduates?
2. How well do anomaly detection methods identify unemployed graduates?
3. Which approach achieves the highest PR-AUC score?
4. Which graduate characteristics are most important for identifying at-risk graduates?
5. Are unemployed graduates better characterized as a minority class or as anomalous observations?

## Headline result

**With 157 graduates (31 unemployed), unemployment is only weakly
predictable from these survey variables. A plain class-weighted logistic
regression is the most accurate and most stable model, and no other model
significantly beats it. The imbalance-aware classifiers (Family A) score
above the anomaly detectors (Family B) in most comparisons, but the gap is
not statistically reliable. The one robust predictor is working while
studying: graduates who worked during their studies were much less likely
to be unemployed.** An earlier version of this README reported Family A as
clearly best; that result came from one favourable train/test split
(details below).

All numbers: repeated stratified 5-fold CV, 95% CIs using the
Nadeau–Bengio corrected variance (see "Methodology notes").

### Main comparison: nested cross-validation (`notebooks/08_nested_cv.ipynb`)

Hyperparameters tuned inside each outer fold (outer 5-fold × 3 repeats on
all 157 rows, inner 3-fold grid search), so tuning never sees the test
data. This is the most conservative estimate in the project.

| Model | Family | PR-AUC | 95% corrected CI |
|---|---|---|---|
| Logistic regression, class-weighted | Baseline | **0.331** | [0.201, 0.460] |
| A1 — Random Forest, class-weighted | Classification | 0.321 | [0.208, 0.435] |
| A3 — SMOTE-NC + Random Forest | Classification | 0.318 | [0.186, 0.451] |
| A4 — ADASYN + Random Forest | Classification | 0.317 | [0.211, 0.424] |
| A2 — XGBoost, scale_pos_weight | Classification | 0.292 | [0.186, 0.398] |
| B1 — Isolation Forest | Anomaly detection | 0.270 | [0.179, 0.362] |
| B3 — Local Outlier Factor | Anomaly detection | 0.247 | [0.162, 0.332] |
| B2 — One-Class SVM | Anomaly detection | 0.247 | [0.150, 0.344] |
| Chance level (base rate) | — | 0.197 | — |

- **No model is significantly different from logistic regression**
  (corrected resampled t-test on the 15 outer folds, Bonferroni across 7
  comparisons). The anomaly detectors come closest to being significantly
  *worse* (B2, B3: p = 0.06–0.08).
- **Family A vs. Family B:** all 12 cross-family pairs favour Family A
  (+0.02 to +0.07), but none is significant (smallest p = 0.22).
- Tuning is unstable at this sample size: for example, logistic
  regression's chosen regularization strength flips between C = 0.01 and
  C = 10 across outer folds (`reports/tables/nested_cv_best_params.csv`).

### Untuned models on all 157 rows (`notebooks/07_robustness.ipynb`)

| Model | Family | PR-AUC | 95% corrected CI | ROC-AUC |
|---|---|---|---|---|
| Logistic regression, class-weighted | Baseline | **0.361** | [0.238, 0.483] | **0.638** |
| A3 — SMOTE-NC + Random Forest | Classification | 0.309 | [0.196, 0.421] | 0.557 |
| A4 — ADASYN + Random Forest | Classification | 0.304 | [0.191, 0.418] | 0.554 |
| A1 — Random Forest, class-weighted | Classification | 0.302 | [0.207, 0.397] | 0.552 |
| B1 — Isolation Forest | Anomaly detection | 0.275 | [0.202, 0.347] | 0.532 |
| A2 — XGBoost, scale_pos_weight | Classification | 0.266 | [0.182, 0.351] | 0.487 |
| B3 — Local Outlier Factor | Anomaly detection | 0.253 | [0.173, 0.332] | 0.497 |
| B2 — One-Class SVM | Anomaly detection | 0.228 | [0.159, 0.296] | 0.447 |
| Prior-only dummy (chance level) | Baseline | 0.197 | — | 0.500 |

- **Logistic regression beats chance** (vs. the dummy: +0.16 PR-AUC,
  p = 0.010), so there is a real but modest signal.
- **No study model is significantly better than logistic regression**, and
  every one has a lower mean PR-AUC. The study models' ROC-AUCs
  (0.45–0.56) are close to chance.
- **Family A vs. Family B:** 11 of the 12 cross-family pairs favour Family
  A, none significantly (corrected resampled t-test, Bonferroni across 21
  pairs; smallest p = 0.21).

### Why this differs from the earlier headline

Notebooks 03–05 follow the original design: cross-validate on one
125-row training split (`random_state=42`) and score the remaining 32
rows once. On that split Family A looks best (A1 PR-AUC 0.416 vs. 0.343
for logistic regression and 0.207–0.282 for Family B). Repeating the
same protocol with four other split seeds (`07_robustness.ipynb`,
"Split sensitivity") shows that split is the exception:

| Training split seed | 42 (original) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| Best model (CV PR-AUC) | A1 | LogReg | LogReg | LogReg | LogReg |
| Family A mean | 0.402 | 0.357 | 0.307 | 0.283 | 0.253 |
| Family B mean | 0.243 | 0.290 | 0.262 | 0.293 | 0.276 |
| Logistic regression | 0.343 | 0.408 | 0.345 | 0.343 | 0.299 |

Seed 42 is the only split on which any Family A model beats logistic
regression. Even there, no study model is significantly better than
logistic regression (`reports/tables/vs_baseline_tests.csv`) and no pair
of the seven study models differs significantly (smallest p = 0.006, A4
vs. B2, against Bonferroni α = 0.00238; `pairwise_corrected_ttest.csv`).
The 32-row holdout (6 unemployed graduates) is too small to settle
anything: A1's holdout PR-AUC ranges from 0.15 to 0.63 across the five
splits, and on seed 42 it is 0.149, below the 0.19 base rate.

### Robustness checks (`notebooks/07_robustness.ipynb`)

| Scenario | n (unemployed) | Base rate | Logistic regression | Best Family A | Best Family B |
|---|---|---|---|---|---|
| Primary target, all features | 157 (31) | 0.197 | **0.361** | 0.309 (A3) | 0.275 (B1) |
| Labour-force target¹ | 145 (19) | 0.131 | **0.315** | 0.273 (A1) | 0.241 (B1) |
| Occasional work = employed | 157 (26) | 0.166 | **0.296** | 0.241 (A4) | 0.250 (B1) |
| Six-month outcome² | 142 (19) | 0.134 | **0.281** | 0.254 (A3) | 0.231 (B1) |
| Pre-outcome features only³ | 157 (31) | 0.197 | **0.387** | 0.272 (A3) | 0.260 (B2) |

¹ Unemployed = not employed *and* searching; respondents not looking for
work (homemakers, further study, not searching) are excluded instead of
being counted as unemployed. ² Not employed within the first six months
after graduation (survey question F2), the same window for every
respondent; the 15 who left it blank are excluded, 9 of whom are
currently unemployed. ³ Program, study hours, working student, sex,
age — no retrospective survey indices.

- No scenario produces a significant Family A vs. Family B difference;
  Family A is ahead in 7–11 of the 12 pairs depending on the scenario.
- Logistic regression has the highest PR-AUC in every scenario and beats
  the dummy in every scenario (p = 0.003–0.034).
- Logistic regression does *best* without the retrospective indices
  (0.387), so they add nothing measurable.

**Graduation cohort.** Unemployment differs sharply by cohort: 26% (2023,
16/61), 4% (2024, 2/54), 31% (2025, 13/42). A model that knows only the
cohort reaches PR-AUC 0.278, as good as most study models. The survey
export has no completion dates, but the same ordering appears for the
first six months after graduation, a window that doesn't depend on when
each cohort was surveyed: 15% (2023), 4% (2024), 24% (2025) not
employed. So survey timing alone doesn't explain the 2024 cohort's low
unemployment; a genuine cohort difference (labour market, hiring cycle,
or who responded) is more likely. Training on two cohorts and testing on
the third, logistic regression reaches ROC-AUC 0.60 / 0.69 / 0.69 (2023 /
2024 / 2025); the Random Forests reach about 0.50 on 2023 and 0.56–0.76 on
the others, and the anomaly detectors fall to 0.28–0.38 on 2024.

### Which characteristics matter (`notebooks/06_explainability.ipynb`)

- **Working while studying** is the only robust predictor. In logistic
  regression its odds ratio for unemployment is **0.38** (95% bootstrap
  CI 0.14–0.56), and it is the only feature whose held-out permutation
  importance is reliably positive, for both logistic regression (positive
  in 86% of CV folds) and Random Forest (96%). Raw rates: 5% unemployment
  among graduates who worked while studying vs. 28% among those who
  didn't.
- **BSEd Math** graduates show higher odds of unemployment (odds ratio
  2.1), but the CI crosses 1 (0.95–5.06), so this is only suggestive.
- **Every other feature**, including all five survey indices, has a CI
  that includes no effect.

These are associations in one cross-sectional survey, not causal
effects: working students may differ in other ways (e.g. already having
an employer, or a different socioeconomic background).

### Answers to the research questions

- **RQ1 (Family A):** the imbalance-aware classifiers detect unemployed
  graduates modestly better than chance (nested-CV PR-AUC 0.29–0.32 vs.
  0.20), but don't improve on logistic regression.
- **RQ2 (Family B):** the anomaly detectors are weaker (nested-CV PR-AUC
  0.25–0.27) and near or below chance by ROC-AUC (untuned, 0.45–0.53).
- **RQ3 (highest PR-AUC):** logistic regression, in nested CV and in every
  robustness scenario. Among the seven study models, the Random Forest
  variants (A1, A3, A4) are usually highest, but their ranking changes
  with the data split and no pair differs reliably.
- **RQ4 (important characteristics):** working while studying (strongly
  protective); weaker, uncertain signals for program (BSEd Math). No
  survey index has a reliable effect.
- **RQ5 (minority class or anomalies?):** the evidence leans toward the
  minority-class framing — supervised models (Family A and logistic
  regression) are consistently ahead of anomaly detection — but the
  Family A vs. B difference is not statistically reliable at this sample
  size, and the simplest supervised model does best.

### Limitations

- **Small sample:** 157 respondents, 31 unemployed (19 under the
  labour-force definition); about 2 unemployed cases per encoded
  predictor in each training fold, against a 10–20 guideline.
- **Outcome definition:** current status at survey time, not time to
  first job; the primary target mixes job-seekers with people not in the
  labour force (sensitivity results above).
- **Cohort differences:** unemployment varies sharply by cohort, and
  the survey dates per cohort aren't recorded (see above).
- **Single institution and programs; response rate unknown**, so results
  may not generalize beyond these respondents.
- **Raw data:** one birth year was corrected by hand in the raw export
  before analysis; the original export isn't available, so the correction
  is documented (`RAW_DATA_SHA256` in `src/preprocessing.py`) but can't be
  re-applied in code.

## Feature set (reduced and leakage-checked)

The original candidate set had ~20 predictors against only 31
minority-class (unemployed) cases — roughly 1.5 events per predictor,
well below the 10–20:1 guideline typically recommended for binary
prediction. It also included composite indices built from survey items
that graduates rated *after* the same survey asked about their current
employment status (`employability_index`, `retrospective_choice_index`),
which risks target leakage: an employed graduate rating their own
"employability" or whether they'd choose the program again in hindsight
is confounded with the outcome, not a genuine pre-outcome predictor of it.

Modeling now uses a 10-variable, theory-driven set:
`program`, `study_hours`, `working_student`, `sex`, `age`,
`school_choice_index`, `learning_experience_index`,
`study_conditions_index`, `guidance_support_index`, `competency_index`
(see `CANDIDATE_FEATURES` in `src/feature_engineering.py`). The two
leakage-risk indices are still computed (`CONSTRUCTS`/`COMPOSITE_VARS`)
for descriptive/EDA use, just excluded from the modeling feature set —
`LEAKAGE_RISK_VARS` in the same file documents which and why.
`internship` was dropped as well: 154 of 157 graduates did one (the
education programs require it), so it carried almost no information.

The five remaining composite indices are also rated retrospectively in
the same survey, so they carry the same risk in weaker form.
`PRE_OUTCOME_FEATURES` (program, study hours, working student, sex,
age) drops them all; `07_robustness.ipynb` shows that logistic
regression does slightly *better* on that set (PR-AUC 0.387 vs. 0.361),
so the indices add no measurable predictive value.

Even 10 variables (13 encoded columns) leave only 2.4 unemployed cases
per predictor on the full data and 1.9 per training fold
(`02_preprocessing.ipynb`), well short of the 10–20:1 guideline —
another reason to prefer a simple, regularized model.

## Project structure

```
graduate-risk-analytics/
├── data/
│   ├── raw/              # original survey export (gitignored — see below)
│   ├── processed/        # cleaned/encoded output of 02_preprocessing.ipynb (gitignored)
│   └── external/         # reserved for any future auxiliary data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_imbalance_classification.ipynb
│   ├── 04_anomaly_detection.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_explainability.ipynb
│   ├── 07_robustness.ipynb
│   └── 08_nested_cv.ipynb
├── src/
│   ├── preprocessing.py        # column mapping, cleaning, encoding
│   ├── feature_engineering.py  # target construction, composite indices
│   ├── classification_models.py # Family A (A1-A4) + reference baselines
│   ├── anomaly_models.py        # Family B (B1-B3) + AnomalyDetectorWrapper
│   ├── evaluation.py            # CV framework, PR-AUC scoring, stats tests
│   ├── explainability.py        # permutation importance, SHAP
│   ├── robustness.py            # model builders + leave-one-group-out for 07
│   └── reporting.py             # saves figures/tables to reports/ (cwd-independent)
├── reports/
│   ├── figures/          # PNG figures saved by the notebooks
│   ├── tables/           # result tables (.csv/.json) + raw fold scores (.pkl, gitignored)
│   └── manuscript/        # thesis writing lives here
├── tests/
│   ├── test_pipeline.py
│   ├── test_corrections.py
│   ├── test_stage1.py
│   ├── test_stage2.py
│   └── test_reporting.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

`requirements.txt` pins the exact versions the committed results were
produced with. Newer versions may shift results slightly (e.g. different
random-number streams in scikit-learn or imbalanced-learn). The tests run
on every push via GitHub Actions (`.github/workflows/tests.yml`); the
ones that need the raw survey export skip there.

## Reproducing the results

Rerunning requires the raw survey export in `data/raw/`, which is not
published (see "Data and privacy" below); the committed notebook outputs,
figures, and tables show the results as run.

Run the notebooks in order. Each is self-contained given the previous
notebook's output (`data/processed/tracer_study_processed.csv` after
`02_preprocessing.ipynb`; `reports/tables/*.csv`/`.pkl` after `03`/`04`):

1. `01_eda.ipynb` — exploratory analysis; saves EDA figures and summary tables.
2. `02_preprocessing.ipynb` — produces `data/processed/tracer_study_processed.csv`.
3. `03_imbalance_classification.ipynb` — produces `reports/tables/family_a_*` and `reports/figures/family_a_*`.
4. `04_anomaly_detection.ipynb` — produces `reports/tables/family_b_*` and `reports/figures/family_b_*`.
5. `05_evaluation.ipynb` — combines both with the baselines, runs statistical tests.
6. `06_explainability.ipynb` — logistic-regression odds ratios and cross-validated permutation importance (RQ4).
7. `07_robustness.ipynb` — all models on the full dataset, alternative outcome definitions, pre-outcome features, cohort checks, and split sensitivity.
8. `08_nested_cv.ipynb` — nested cross-validation for all models (main comparison).

`02_preprocessing.ipynb` first checks the raw file against
`RAW_DATA_SHA256` (`src/preprocessing.py`) and warns if it differs from
the file the committed results were produced from. Every figure and table
under `reports/` is written by these notebooks via `src/reporting.py`.

## Data and privacy

The raw survey export (`data/raw/`) contains graduate demographic and
employment information and is **gitignored** — it is not published in
this repository. Researchers with a legitimate need for the underlying
data should contact the institution directly. `data/processed/` (also
gitignored by default) contains only encoded/aggregated features, no
free-text or directly identifying fields, but is still excluded unless
you deliberately choose to version a fixed snapshot for reproducibility.

## Methodology notes worth knowing before extending this project

- **Every model is read against two baselines** (`build_baseline_models()`
  in `src/classification_models.py`): a prior-only dummy (PR-AUC = base
  rate, ROC-AUC = 0.5) and a class-weighted logistic regression. They are
  kept out of Family A so they never become "best Family A model" or enter
  the Family A vs. B tests.
- **Results should come from the full dataset, not one split.** With
  n=157 the 80/20 split in notebooks 03–05 changes the ranking depending on
  the seed (see "Why this differs from the earlier headline"); notebook
  07's full-data CV is the more reliable comparison.
- **PR-AUC targets the minority class explicitly** (`POS_LABEL = 0`,
  "Unemployed") throughout `src/evaluation.py` — sklearn's default
  scoring conventions target label `1`, which would silently measure how
  well a model finds the *majority* (employed) class instead.
- **Family B models are fit only on majority-class training rows**
  (`AnomalyDetectorWrapper`), per the standard anomaly-detection
  convention — they never see labeled minority-class examples during
  fitting, unlike Family A.
- **Anomaly scores are mapped to probabilities with a transform fixed at
  fit time** — `sigmoid(-decision_function / scale_)`, where `scale_` is
  the spread of the training rows' scores. A row's score never depends on
  which other rows are scored with it, so a threshold tuned on
  out-of-fold predictions means the same thing on the holdout set, and
  P=0.5 sits at the detector's own inlier/outlier boundary. (An earlier
  version min-max normalized each batch separately, which left PR-AUC
  unchanged but made Family B's precision/recall/F1 and tuned thresholds
  depend on batch composition.)
- **Decision thresholds are tuned on out-of-fold CV predictions**, never
  on the holdout test set — using holdout performance to pick a threshold
  (or a model) would make the "held-out" set no longer truly held out.
- **Statistical significance uses the corrected resampled t-test**
  (Nadeau & Bengio, 2003; `pairwise_corrected_ttest()` in
  `src/evaluation.py`), Bonferroni-corrected across all 21 pairs.
  Repeated-CV folds share most of their training rows, so tests that
  treat the 50 fold scores as independent (Wilcoxon, Friedman on raw
  folds) understate variance and overstate significance; the correction
  inflates the variance term from `1/k` to `1/k + n_test/n_train`. The
  Friedman test is run on per-repeat mean scores for the same reason.
- **Sample size is small** (n=157, ~31 minority-class examples) —
  cross-validated evidence should always be weighted over any single
  holdout split's numbers, which can be noisy enough to look worse (or
  better) than the CV-level pattern for reasons that are pure small-sample
  variance, not a real effect.
- **PR-AUC is reported with a 95% corrected CI alongside mean ± std**
  (`corrected_ci()` in `src/evaluation.py`), using the same Nadeau–Bengio
  variance correction. The percentile bootstrap over fold scores
  (`bootstrap_ci()`) is still saved for comparison, but it treats folds
  as independent and comes out roughly 4× too narrow here.
- **Nested cross-validation** (`nested_cross_validate()` in
  `src/evaluation.py`, run for every model in `08_nested_cv.ipynb`) gives
  the most honest generalization estimate: an outer
  repeated-stratified-K-fold loop provides the test folds, and
  hyperparameter selection happens only inside each outer fold's own
  inner CV, so tuning never leaks into the reported score. All models use
  the same outer folds, so their scores can be compared pairwise.
- **Explainability is measured out of sample.** Permutation importance is
  computed on every held-out CV fold (`cv_permutation_importance()` in
  `src/explainability.py`) rather than on the training data or one small
  holdout, and logistic-regression odds ratios get a bootstrap CI over
  respondents (`logreg_odds_ratios()`). SHAP (`get_shap_values()`) is kept
  for ad-hoc use but not used in the notebooks.
- **Sex is a predictor; that needs a justification in the thesis.** It is
  in the feature set as a demographic control, not as something the
  institution could act on, and it has no reliable association with
  unemployment here (logistic-regression odds ratio 1.31, 95% CI
  0.82–2.45). If any model from this project were used to flag
  individual graduates for support, sex should be dropped or performance
  checked separately for each group, so the model doesn't steer support
  by sex.
- **Synthetic rows keep valid categories.** Plain SMOTE or ADASYN after
  one-hot encoding interpolates the dummy columns (a synthetic graduate
  0.4 BEED / 0.6 BSEd Math). A3 resamples before encoding with SMOTE-NC.
  ADASYN has no categorical-aware variant, so A4 uses
  `ValidCategoryADASYN` (`src/resampling.py`): plain ADASYN, then each
  synthetic row is snapped to its nearest program and to 0/1 for the
  binary features. Continuous features stay interpolated.

## License

MIT — see `LICENSE`.
