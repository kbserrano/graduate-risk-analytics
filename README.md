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

| Model | Family | PR-AUC (mean ± std) | 95% corrected CI |
|---|---|---|---|
| A4 — ADASYN + Random Forest | Classification | **0.440 ± 0.114** | [0.319, 0.560] |
| A3 — SMOTE + Random Forest | Classification | 0.432 ± 0.114 | [0.312, 0.552] |
| A1 — Random Forest, class-weighted | Classification | 0.426 ± 0.126 | [0.293, 0.558] |
| A2 — XGBoost, scale_pos_weight | Classification | 0.386 ± 0.136 | [0.243, 0.529] |
| B3 — Local Outlier Factor | Anomaly Detection | 0.302 ± 0.117 | [0.179, 0.426] |
| B1 — Isolation Forest | Anomaly Detection | 0.287 ± 0.103 | [0.178, 0.395] |
| B2 — One-Class SVM | Anomaly Detection | 0.264 ± 0.100 | [0.158, 0.370] |

Repeated Stratified 5-Fold CV (10 repeats) on the 125-row training
split. CIs use the Nadeau–Bengio corrected variance, because the 50
fold scores come from heavily overlapping training sets and are not
independent (see "Methodology notes").

- **Every Family A model has a higher mean PR-AUC than every Family B
  model**, and all 12 cross-family differences favor Family A (+0.08 to
  +0.18).
- **None of those differences is statistically significant.** Pairwise
  corrected resampled t-tests (Bonferroni across 21 pairs,
  α=0.05/21=0.00238) find no significant pair; the smallest p-value is
  0.034 (A4 vs. B2), and most cross-family pairs have p > 0.05 even
  before correction. The Friedman test is significant (per-repeat
  blocks: χ²=51.69, p=2.2×10⁻⁹), but it can't say which models differ.
- An earlier version of this README reported A1, A3 and A4 as
  significantly beating every Family B model (Wilcoxon, p≤0.00004).
  That test treated the 50 overlapping folds as independent samples and
  overstated significance; it is still saved
  (`reports/tables/pairwise_wilcoxon.csv`) as a sensitivity check only.
- **The 32-row holdout disagrees with CV.** On it (6 unemployed), A4
  scores PR-AUC 0.171 — below the 0.1875 base rate — while B3 scores
  0.289. With 6 positives this is mostly noise, but it's another sign the
  ranking is not settled.

**Nested cross-validation (A1 only):** outer 5-fold × 3-repeat CV, inner
3-fold grid search over `n_estimators`/`max_depth`, gave **PR-AUC = 0.315
± 0.117**, vs. 0.426 for A1 above. A1 in the simple setup uses Random
Forest defaults (no tuning), so the gap isn't tuning leakage; it more
likely reflects the different data (all 157 rows vs. the 125-row training
split), fold layout, and noisy inner-fold tuning with ~4 minority cases
per inner fold. That run isn't captured in a notebook, and its saved CI
([0.259, 0.376]) is a percentile bootstrap, so it is too narrow for the
same reason as above.

**Answering RQ3 and RQ5**: imbalance-aware classification has the higher
PR-AUC on this dataset, consistently across all model pairs, so the
evidence *leans toward* treating unemployed graduates as a minority class
to classify rather than as anomalies. At n=157 with 31 unemployed cases,
though, the difference is not statistically reliable once the dependence
between CV folds is accounted for — the study can report a consistent
direction, not a demonstrated advantage.

**What's still open**: a nested, notebook-captured evaluation of all
seven models, ideally on the full dataset rather than the 125-row
training split, would give the most defensible comparison. More data
(e.g. additional tracer-study batches) is the only thing likely to make
the Family A vs. Family B difference significant.

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
│   ├── test_corrections.py
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
