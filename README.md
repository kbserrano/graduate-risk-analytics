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
regression is the most accurate and most stable model. Neither the
imbalance-aware classifiers (Family A) nor the anomaly detectors
(Family B) reliably beat it, and the two families can't be separated
from each other.** An earlier version of this README reported Family A as
clearly ahead; that result came from one favourable train/test split
(details below).

All numbers: repeated stratified 5-fold CV (10 repeats), 95% CIs using the
Nadeau–Bengio corrected variance (see "Methodology notes").

### Main comparison (all 157 rows — `notebooks/07_robustness.ipynb`)

| Model | Family | PR-AUC | 95% corrected CI | ROC-AUC |
|---|---|---|---|---|
| Logistic regression, class-weighted | Baseline | **0.367** | [0.242, 0.492] | **0.646** |
| B3 — Local Outlier Factor | Anomaly detection | 0.301 | [0.185, 0.418] | 0.512 |
| A1 — Random Forest, class-weighted | Classification | 0.301 | [0.192, 0.409] | 0.547 |
| A4 — ADASYN + Random Forest | Classification | 0.298 | [0.179, 0.417] | 0.542 |
| B1 — Isolation Forest | Anomaly detection | 0.297 | [0.203, 0.392] | 0.548 |
| A3 — SMOTE + Random Forest | Classification | 0.295 | [0.190, 0.400] | 0.539 |
| B2 — One-Class SVM | Anomaly detection | 0.267 | [0.165, 0.370] | 0.460 |
| A2 — XGBoost, scale_pos_weight | Classification | 0.266 | [0.182, 0.351] | 0.487 |
| Prior-only dummy (chance level) | Baseline | 0.197 | — | 0.500 |

- **Logistic regression beats chance** (vs. the dummy: +0.17 PR-AUC,
  p=0.009), so there is a real but modest signal.
- **No study model is significantly better than logistic regression**, and
  every one has a lower mean PR-AUC. The seven study models' ROC-AUCs
  (0.46–0.55) are close to chance.
- **Family A vs. Family B: no difference.** Only 5 of the 12 cross-family
  pairs favour Family A, and none is significant (corrected resampled
  t-test, Bonferroni across 21 pairs; smallest p = 0.60).

### Why this differs from the earlier headline

Notebooks 03–05 follow the original design: cross-validate on one
125-row training split (`random_state=42`) and score the remaining 32
rows once. On that split Family A looks best (A4 PR-AUC 0.440 vs. 0.385
for logistic regression and 0.264–0.302 for Family B). Repeating the
same protocol with four other split seeds (`07_robustness.ipynb`,
"Split sensitivity") shows that split is the exception:

| Training split seed | 42 (original) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| Best model (CV PR-AUC) | A4 | LogReg | LogReg | LogReg | B3 |
| Family A mean | 0.421 | 0.344 | 0.306 | 0.273 | 0.253 |
| Family B mean | 0.288 | 0.319 | 0.265 | 0.284 | 0.304 |
| Logistic regression | 0.385 | 0.410 | 0.345 | 0.339 | 0.334 |

Seed 42 is the only split on which any Family A model beats logistic
regression. Even there no difference is significant: no pair of the seven
study models (smallest p = 0.034, Bonferroni α = 0.00238) and no study
model vs. logistic regression (`reports/tables/vs_baseline_tests.csv`).
The 32-row holdout (6 unemployed graduates) is too small to settle
anything: A4's holdout PR-AUC ranges from 0.17 to 0.37 across the five
splits, and on seed 42 it is 0.171, below the 0.19 base rate.

### Robustness checks (`notebooks/07_robustness.ipynb`)

| Scenario | n (unemployed) | Base rate | Logistic regression | Best Family A | Best Family B |
|---|---|---|---|---|---|
| Primary target, all features | 157 (31) | 0.197 | **0.367** | 0.301 (A1) | 0.301 (B3) |
| Labour-force target¹ | 145 (19) | 0.131 | **0.291** | 0.288 (A3) | 0.263 (B3) |
| Occasional work = employed | 157 (26) | 0.166 | **0.295** | 0.255 (A3) | 0.281 (B1) |
| Pre-outcome features only² | 157 (31) | 0.197 | **0.395** | 0.267 (A4) | 0.298 (B2) |

¹ Unemployed = not employed *and* searching; respondents not looking for
work (homemakers, further study, not searching) are excluded instead of
being counted as unemployed. ² Program, study hours, internship,
working student, sex, age — no retrospective survey indices.

- No scenario produces a significant Family A vs. Family B difference.
- Logistic regression has the highest PR-AUC in every scenario and beats
  the dummy in every scenario (p = 0.004–0.023).
- Logistic regression does *best* without the retrospective indices
  (0.395), so they add nothing measurable.

**Graduation cohort.** Unemployment differs sharply by cohort: 26% (2023,
16/61), 4% (2024, 2/54), 31% (2025, 13/42). A model that knows only the
cohort reaches PR-AUC 0.278, as good as most study models. Training on
two cohorts and testing on the third, logistic regression reaches ROC-AUC
0.60 / 0.69 / 0.75 (2023 / 2024 / 2025), while the Random Forests are
near 0.5 on 2023 and 2024. Cohorts may have been surveyed at different
times after graduation; until that is documented, cohort differences are
a possible confound (see "Limitations").

### Answers to the research questions

- **RQ1 (Family A):** the imbalance-aware classifiers detect unemployed
  graduates only slightly better than chance (PR-AUC 0.27–0.30 vs. 0.20;
  ROC-AUC 0.49–0.55) and do not improve on logistic regression.
- **RQ2 (Family B):** the anomaly detectors perform about the same as
  Family A by PR-AUC (0.27–0.30) and near or below chance by ROC-AUC
  (0.46–0.55).
- **RQ3 (highest PR-AUC):** logistic regression, in every scenario. Among
  the seven study models, the ranking changes with the data split, and no
  pair differs significantly.
- **RQ4 (important characteristics):** not yet answerable reliably — see
  `06_explainability.ipynb` caveats. Working while studying is the one
  feature with a large raw difference (5% unemployment among working
  students vs. 28% among others).
- **RQ5 (minority class or anomalies?):** the data don't distinguish the
  two framings. Anomaly detection is not better, but neither is
  imbalance-aware classification; the simplest supervised model does best.

### Limitations

- **Small sample:** 157 respondents, 31 unemployed (19 under the
  labour-force definition); about 2 unemployed cases per predictor in
  each training fold.
- **Outcome definition:** current status at survey time, not time to
  first job; the primary target mixes job-seekers with people not in the
  labour force (sensitivity results above).
- **Cohort and survey timing:** see above.
- **Single institution and programs; response rate unknown**, so results
  may not generalize beyond these respondents.
- **Nested cross-validation** has only been run once for A1 (PR-AUC
  0.315 ± 0.117, `reports/tables/family_a_nested_cv_summary.csv`); that
  run is not captured in a notebook and its CI is an uncorrected
  bootstrap.

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

Modeling now uses an 11-variable, theory-driven set:
`program`, `study_hours`, `internship`, `working_student`, `sex`, `age`,
`school_choice_index`, `learning_experience_index`,
`study_conditions_index`, `guidance_support_index`, `competency_index`
(see `CANDIDATE_FEATURES` in `src/feature_engineering.py`). The two
leakage-risk indices are still computed (`CONSTRUCTS`/`COMPOSITE_VARS`)
for descriptive/EDA use, just excluded from the modeling feature set —
`LEAKAGE_RISK_VARS` in the same file documents which and why.

The five remaining composite indices are also rated retrospectively in
the same survey, so they carry the same risk in weaker form.
`PRE_OUTCOME_FEATURES` (program, study hours, internship, working
student, sex, age) drops them all; `07_robustness.ipynb` shows that
logistic regression does slightly *better* on that set (PR-AUC 0.395
vs. 0.367), so the indices add no measurable predictive value.

Even 11 variables leave only about 2 unemployed cases per predictor in
each training fold (31 cases, 14 encoded columns), well short of the
10–20:1 guideline — another reason to prefer a simple, regularized model.

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
│   └── 07_robustness.ipynb
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
│   └── test_reporting.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`xgboost`, `imbalanced-learn`, and `shap` are listed as optional in
`requirements.txt` — install them to run the full model set (A2-A4) and
the SHAP cross-check in `06_explainability.ipynb`.

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
6. `06_explainability.ipynb` — permutation importance / SHAP for A1 (not yet updated to the findings above).
7. `07_robustness.ipynb` — all models on the full dataset, alternative outcome definitions, pre-outcome features, cohort checks, and split sensitivity. Source of the headline numbers.

Every figure and table under `reports/` is written by these notebooks via
`src/reporting.py`, except `reports/tables/family_a_nested_cv_summary.csv`,
which comes from a one-off `nested_cross_validate()` run not yet captured in
a notebook.

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
  `src/evaluation.py`) gives the honest generalization estimate for the
  winning model: an outer repeated-stratified-K-fold loop provides the
  test folds, and hyperparameter selection happens only inside each outer
  fold's own inner CV, so tuning never leaks into the reported score. This
  is separate from (and stricter than) the single 80/20 holdout +
  CV-on-train setup used elsewhere for model comparison and threshold
  tuning.
- **Explainability is permutation-importance-first.** For n=157,
  permutation importance (`get_permutation_importance()`, model-agnostic,
  scored on the same PR-AUC-for-minority-class metric as everything else)
  is the primary method; SHAP (`get_shap_values()`) is an optional
  validation appendix, not a co-equal method — a more elaborate
  explainability stack than the data can really support would overstate
  this study's precision.

## License

MIT — see `LICENSE`.
