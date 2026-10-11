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

Built around **PR-AUC** as the primary metric, since ROC-AUC and accuracy
can both look misleadingly strong on a rare-event problem like this one —
neither is sensitive to how well a model finds the minority class
specifically, which is the entire point here.

## Research questions this project answers

1. How well do imbalance-aware classification methods identify unemployed graduates?
2. How well do anomaly detection methods identify unemployed graduates?
3. Which approach achieves the highest PR-AUC score?
4. Which graduate characteristics are most important for identifying at-risk graduates?
5. Are unemployed graduates better characterized as a minority class or as anomalous observations?

## Headline result (see `notebooks/05_evaluation.ipynb` for full detail)

Based on a reduced, leakage-checked feature set — see
["Feature set" below](#feature-set-reduced-and-leakage-checked) for why.
All four Family A models (A1–A4) have now been executed against the real
dataset with `xgboost`/`imbalanced-learn` installed.

| Model | Family | PR-AUC (mean ± std) | 95% bootstrap CI |
|---|---|---|---|
| A4 — ADASYN + Random Forest | Classification | **0.440 ± 0.114** | [0.408, 0.471] |
| A3 — SMOTE + Random Forest | Classification | 0.432 ± 0.114 | [0.401, 0.464] |
| A1 — Random Forest, class-weighted | Classification | 0.426 ± 0.126 | [0.392, 0.461] |
| A2 — XGBoost, scale_pos_weight | Classification | 0.386 ± 0.136 | [0.350, 0.426] |
| B3 — Local Outlier Factor | Anomaly Detection | 0.302 ± 0.117 | [0.271, 0.335] |
| B1 — Isolation Forest | Anomaly Detection | 0.287 ± 0.103 | [0.259, 0.316] |
| B2 — One-Class SVM | Anomaly Detection | 0.264 ± 0.100 | [0.237, 0.292] |

Repeated Stratified 5-Fold CV (10 repeats). A Friedman test across all
**seven** models' fold-level PR-AUC scores is significant (χ²=94.96,
p=2.8×10⁻¹⁸). Pairwise Wilcoxon signed-rank tests (Bonferroni-corrected
across all 21 pairs, α=0.05/21=0.00238) show:
- **A1, A3, and A4 each significantly beat all three Family B models**
  (p≤0.00004 in every case).
- **A2 significantly beats B1 and B2** (p≤0.00085), but **not B3**
  (p=0.00967 — below the uncorrected 0.05 but above the Bonferroni-corrected
  threshold, so not significant after correction for 21 comparisons).
- No pair *within* either family differs significantly (A1 vs A2 vs A3 vs
  A4 are statistically indistinguishable from each other, and likewise for
  B1 vs B2 vs B3).

**Nested cross-validation (A1 only, honest estimate with hyperparameter
tuning folded in):** outer 5-fold × 3-repeat CV (15 outer folds), inner
3-fold grid search over `n_estimators`/`max_depth`, gives **PR-AUC = 0.315
± 0.117 (95% CI: [0.259, 0.376])** — lower than A1's 0.426 simple-CV figure
above, because that number still benefits from picking one fixed
hyperparameter set ahead of time using knowledge of the whole training
set. The gap between 0.426 and 0.315 is itself a useful result: a concrete
illustration of how much a single train/CV setup can overstate performance
on a dataset this size. 0.315 is the more defensible number to lead with
for A1 specifically; the same nested procedure hasn't yet been run for
A2–A4, so their 0.386–0.440 figures likely overstate their honest
generalization performance by a similar amount — extending nested CV to
all four is the natural next step (see below). Either way, A1's honest
estimate (0.315) still clears every Family B model's simple-CV estimate
(0.264–0.302).

**Answering RQ3 and RQ5**: imbalance-aware classification outperforms
anomaly detection on this dataset — three of four classification models
(A1, A3, A4) significantly beat every anomaly-detection model, and the
fourth (A2) significantly beats two of three and directionally beats the
third. Unemployed graduates are better characterized as a minority class
to be directly classified than as anomalous observations to be detected
against a learned "normal" pattern. This conclusion is now based on the
full seven-model comparison, not a provisional A1-only subset.

**What's still open**: nested cross-validation (the honest,
tuning-leakage-free estimate) has only been run for A1. Extending it to
A2–A4 would give a fully honest ranking within Family A — plausible given
A1's own 0.426→0.315 shrinkage that A4's simple-CV lead over A1 (0.440 vs
0.426) narrows or reverses under nested CV, though this cannot change the
Family A vs. Family B conclusion above, since even A1's most conservative
(nested-CV) estimate already clears every Family B model.

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

Modeling now uses an 11-variable, theory-driven, pre-outcome set:
`program`, `study_hours`, `internship`, `working_student`, `sex`, `age`,
`school_choice_index`, `learning_experience_index`,
`study_conditions_index`, `guidance_support_index`, `competency_index`
(see `CANDIDATE_FEATURES` in `src/feature_engineering.py`). The two
leakage-risk indices are still computed (`CONSTRUCTS`/`COMPOSITE_VARS`)
for descriptive/EDA use, just excluded from the modeling feature set —
`LEAKAGE_RISK_VARS` in the same file documents which and why.

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
│   └── 06_explainability.ipynb
├── src/
│   ├── preprocessing.py        # column mapping, cleaning, encoding
│   ├── feature_engineering.py  # target construction, composite indices
│   ├── classification_models.py # Family A (A1-A4)
│   ├── anomaly_models.py        # Family B (B1-B3) + AnomalyDetectorWrapper
│   ├── evaluation.py            # CV framework, PR-AUC scoring, stats tests
│   ├── explainability.py        # permutation importance, SHAP
│   └── reporting.py             # saves figures/tables to reports/ (cwd-independent)
├── reports/
│   ├── figures/          # PNG figures saved by the notebooks
│   ├── tables/           # result tables (.csv/.json) + raw fold scores (.pkl, gitignored)
│   └── manuscript/        # thesis writing lives here
├── tests/
│   ├── test_pipeline.py
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
5. `05_evaluation.ipynb` — combines both, runs statistical tests.
6. `06_explainability.ipynb` — permutation importance / SHAP on the winning model.

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

- **PR-AUC targets the minority class explicitly** (`POS_LABEL = 0`,
  "Unemployed") throughout `src/evaluation.py` — sklearn's default
  scoring conventions target label `1`, which would silently measure how
  well a model finds the *majority* (employed) class instead.
- **Family B models are fit only on majority-class training rows**
  (`AnomalyDetectorWrapper`), per the standard anomaly-detection
  convention — they never see labeled minority-class examples during
  fitting, unlike Family A.
- **Decision thresholds are tuned on out-of-fold CV predictions**, never
  on the holdout test set — using holdout performance to pick a threshold
  (or a model) would make the "held-out" set no longer truly held out.
- **Statistical significance uses Bonferroni-corrected pairwise Wilcoxon
  tests**, not raw p-values — with 6 pairwise comparisons among 4 models,
  an uncorrected 0.05 threshold would overstate confidence.
- **Sample size is small** (n=157, ~31 minority-class examples) —
  cross-validated evidence should always be weighted over any single
  holdout split's numbers, which can be noisy enough to look worse (or
  better) than the CV-level pattern for reasons that are pure small-sample
  variance, not a real effect.
- **PR-AUC is reported with a 95% bootstrap CI alongside mean ± std**
  (`bootstrap_ci()` in `src/evaluation.py`), not as a bare point estimate —
  with this few folds/cases, the point estimate alone understates how much
  uncertainty there really is.
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
