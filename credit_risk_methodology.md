# Credit Risk Model — Methodology

**Project:** Binary Credit Default Prediction  
**Dataset:** `credit_risk_dataset.csv` — 13,266 rows × 20 columns  
**Target:** `target_flag` (1 = default, 0 = no default)  
**Default Rate:** 4.0% (533 defaults / 12,733 non-defaults)  

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Quality Issues Found](#3-data-quality-issues-found)
4. [Data Cleaning](#4-data-cleaning)
5. [Bugs Found in Original Notebook](#5-bugs-found-in-original-notebook)
6. [Correct ML Pipeline Design](#6-correct-ml-pipeline-design)
7. [Feature Engineering](#7-feature-engineering)
8. [Preprocessing](#8-preprocessing)
9. [Handling Class Imbalance](#9-handling-class-imbalance)
10. [Hyperparameter Tuning](#10-hyperparameter-tuning)
11. [Threshold Selection](#11-threshold-selection)
12. [Model Evaluation](#12-model-evaluation)
13. [Fairness Analysis](#13-fairness-analysis)
14. [Probability Calibration](#14-probability-calibration)
15. [Summary of All Changes](#15-summary-of-all-changes)

---

## 1. Problem Statement

The goal is to predict whether a loan applicant will **default** before the loan is issued — using only information available **at the time of application**. This is a binary classification problem with severe class imbalance.

A model like this is used in production to flag high-risk applications for manual review or automatic denial. Because of this, two things matter most:

- **Recall** — catching as many real defaults as possible (missing a default is costly)
- **Precision** — not flagging too many legitimate borrowers (over-flagging loses business)

ROC-AUC is used as the primary tuning metric because it measures discrimination across all thresholds. A separate threshold selection step is then used to balance recall and precision.

---

## 2. Dataset Overview

| Column | Type | Description |
|--------|------|-------------|
| `person_age` | int | Applicant age in years |
| `annual_inc` | float | Annual income (USD) |
| `home_ownership` | str | `OWN`, `RENT`, `MORTGAGE`, `OTHER` |
| `employment_length` | float | Years at current employer |
| `loan_intent` | str | Purpose of loan (6 categories) |
| `loan_grade` | str | Lender-assigned grade A–G |
| `loan_amt` | float | Loan amount requested (USD) |
| `interest_rate` | float | Annual interest rate (%) |
| `income_ratio` | float | Loan amount / annual income |
| `employment_type` | str | `employed` or `self_emp` |
| `residence_type` | str | `URBAN` or `RURAL` |
| `credit_score` | float | Applicant credit score |
| `monthly_income` | float | Monthly income (USD) |
| `target_flag` | int | **Target: 1 = default, 0 = no default** |
| `loan_status_final` | float | ⚠️ Post-default outcome — leakage |
| `repayment_flag` | float | ⚠️ Post-default outcome — leakage |
| `last_payment_status` | int | ⚠️ Post-default outcome — leakage |
| `random_score_1` | float | ⚠️ Pure random noise |
| `random_score_2` | int | ⚠️ Pure random noise |
| `duplicate_feature` | int | ⚠️ Redundant — corr 0.30 with `person_age` |

### Missing Values

| Column | Missing Count | Missing % |
|--------|:---:|:---:|
| `interest_rate` | 2,591 | 19.5% |
| `credit_score` | 1,829 | 13.8% |
| `loan_amt` | 1,756 | 13.2% |
| `annual_inc` | 1,663 | 12.5% |
| `employment_length` | 1,577 | 11.9% |

All missing values are handled inside the preprocessing pipeline using **median imputation** — described in Section 8.

---

## 3. Data Quality Issues Found

Six distinct data quality problems were identified by inspecting the raw dataset before any modelling.

### D1 — Impossible Age Values
`person_age` contained values up to **999**. These are clearly data entry errors (e.g. a mistyped extra digit). Seven rows had `person_age > 100`.

```
df['person_age'].describe()
# max = 999  ← impossible
(df['person_age'] > 100).sum()
# 7 rows
```

### D2 — Inconsistent `employment_type` Categories
The column had four variants representing only two actual categories:

| Raw Value | Standardised To |
|-----------|-----------------|
| `employed` | `employed` |
| `Employed` | `employed` |
| `self_emp` | `self_emp` |
| `Self-employed` | `self_emp` |

This would cause the one-hot encoder to create **four binary columns** instead of one, with three of them being near-duplicates — inflating feature space and misleading the model.

### D3 — Inconsistent `residence_type` Casing
Four variants of two values: `Urban`, `URBAN`, `Rural`, `RURAL`. Without standardisation, the encoder treats these as four distinct categories.

### D4 — Extreme Outliers in Financial Columns
- `annual_inc` max = **$1,200,000** — 99th percentile is ~$200,000
- `loan_amt` max = **$1,150,000** — 99th percentile is ~$35,000

These extreme values cause StandardScaler to compress the majority of the data into a tiny range, degrading model performance.

### D5 — Heavy Missingness in `interest_rate`
Nearly 1 in 5 rows (19.5%) is missing `interest_rate`. Using `mean` imputation (as the original code did) is suboptimal when the distribution is skewed or contains outliers. **Median imputation** is used instead.

### D6 — Severe Class Imbalance
Only **4.0%** of loans default. A naive model that predicts "no default" every time achieves 96% accuracy but catches zero defaults. This requires deliberate imbalance handling — see Section 9.

---

## 4. Data Cleaning

All cleaning is applied to the full dataframe **before** the train/test split, because these are corrections to data quality (not statistical transformations that could leak information).

### Step 1 — Remove Impossible Ages
```python
df_clean = df[df['person_age'] <= 100].copy()
# Removes 7 rows
```

### Step 2 — Cap Extreme Financial Outliers
```python
for col in ['annual_inc', 'loan_amt']:
    cap = df_clean[col].quantile(0.995)
    df_clean[col] = df_clean[col].clip(upper=cap)
```
Capping at the 99.5th percentile preserves the distribution shape while eliminating extreme values that distort scaling.

### Step 3 — Standardise `employment_type`
```python
df_clean['employment_type'] = (
    df_clean['employment_type']
    .str.lower().str.strip()
    .str.replace('-', '_').str.replace(' ', '_')
)
df_clean['employment_type'] = df_clean['employment_type'].replace(
    {'self_employed': 'self_emp', 'self-employed': 'self_emp'}
)
```

### Step 4 — Standardise `residence_type`
```python
df_clean['residence_type'] = df_clean['residence_type'].str.upper().str.strip()
```

After cleaning: **13,259 rows** remain, default rate unchanged at **4.02%**.

---

## 5. Bugs Found in Original Notebook

The original notebook contained **13 bugs** across three categories: data leakage, code errors, and missing best practices. Each is documented below.

---

### Bug Category A — Data Leakage (Most Critical)

Data leakage occurs when information from outside the training set — or from the future — is used during model training. It causes **artificially inflated metrics** that do not reflect real-world performance.

#### Bug #2 — Leakage Features Used as Inputs

Three columns represent outcomes that happen **after** the default event being predicted:

| Column | Why It Leaks |
|--------|--------------|
| `loan_status_final` | Final status of the loan — only known after default occurs |
| `repayment_flag` | Whether the borrower repaid — only known after default |
| `last_payment_status` | Status of last payment — only known after default |

Using these as model inputs is equivalent to telling the model the answer before it predicts. A model trained with these features would appear to perform near-perfectly in testing but would fail entirely in production (where these values don't exist at application time).

**Fix:** These columns are removed entirely from `X` before modelling.

#### Bug #3 — Noise and Duplicate Columns Included

| Column | Problem |
|--------|---------|
| `random_score_1` | Randomly generated — verified to have no predictive signal |
| `random_score_2` | Randomly generated — same |
| `duplicate_feature` | Correlation of 0.30 with `person_age` — adds no independent information |

Including noise features increases model dimensionality, hurts generalisation, and can produce misleading feature importances.

**Fix:** All three removed from the feature set.

#### Bug #4 — Feature Engineering Computed on Full Dataset Before Split

The original code computed statistics (income z-score, correlation-based feature selection) on the entire dataset **before** splitting into train and test:

```python
# ORIGINAL BUGGY CODE
global_mean_income = df['annual_inc'].mean()  # computed on all 13,266 rows
df['income_zscore'] = (df['annual_inc'] - global_mean_income) / global_std_income

feature_correlations = df.corr()['target_flag']  # test rows included!
```

This means test-set statistics contaminate the training process. The model "sees" test data during training, making evaluation metrics unreliable.

**Fix:** Train/test split happens first. Feature engineering statistics are computed only on `X_train`, then applied to `X_test`.

#### Bug #5 — Preprocessor Fitted on Train + Test Combined

The original code explicitly concatenated train and test before fitting the preprocessor:

```python
# ORIGINAL BUGGY CODE
X_combined = pd.concat([X_train, X_test], axis=0)
X_processed_combined = preprocessor.fit_transform(X_combined)  # leakage!
```

Fitting `StandardScaler` or `SimpleImputer` on combined data means the scaler's mean and standard deviation are influenced by test data. This is a classic and subtle form of test-set leakage.

**Fix:**
```python
X_train_proc = preprocessor.fit_transform(X_train_fe)   # fit on train only
X_test_proc  = preprocessor.transform(X_test_fe)        # transform only
```

#### Bug #6 — SMOTE Applied Before or Across Train/Test Split

The original code applied oversampling to already-split data but in ways that could contaminate the test set. Oversampling must never touch the test set — the test set must represent the natural real-world class distribution.

**Fix:** SMOTE is applied only to `X_train_proc` / `y_train` after preprocessing. `X_test_proc` and `y_test` are never resampled.

#### Bug #7 — Hyperparameter Tuning on the Test Set

The original code evaluated 15 hyperparameter combinations directly on `y_test` and selected the best-performing one:

```python
# ORIGINAL BUGGY CODE
for max_depth in [8, 12, 16, 20, 24]:
    for min_samples in [5, 10, 15]:
        model.fit(X_train_balanced, y_train_balanced)
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test))  # leakage!
        if test_auc > best_auc:
            best_model = model
```

This is a form of **p-hacking** — by testing 15 models on the same test set and picking the best, the reported AUC is statistically inflated. The test set should be evaluated exactly once, on the final model only.

**Fix:** Hyperparameter search uses 5-fold `StratifiedKFold` cross-validation on training data. The test set is never involved in model selection.

#### Bug #8 — Classification Threshold Tuned on Test Set

Similarly, the original code scanned threshold values (0.2 to 0.8) against `y_test` to find the best F1 score — further contaminating test-set evaluation.

**Fix:** Threshold is tuned on a held-out **validation split** carved from the training data. The test set is untouched until final evaluation.

---

### Bug Category B — Code Errors

#### Bug #1 — `imblearn` Import Crash

The notebook ran `pip install imbalanced-learn` in one cell, then immediately tried to import `imblearn` in the next. Python does not reload installed packages mid-session without a kernel restart, so the import always failed with `ModuleNotFoundError`.

**Fix:** The install cell is isolated with a clear "restart kernel" instruction. A `try/except` block gracefully handles environments where `imblearn` is unavailable.

#### Bug #9 — Feature Importances Truncated and Misaligned

The original code built a feature name list as `numeric_features + categorical_features` (e.g. `['credit_score', ..., 'home_ownership', 'loan_intent', ...]`) and matched it to `model.feature_importances_`. This is wrong because `OneHotEncoder` expands each categorical column into multiple binary columns. The model's importance array has one value per binary column, not per original column. Truncating the name list causes a silent mismatch.

**Fix:**
```python
feature_names_out = []
for name, transformer, columns in preprocessor.transformers_:
    if name == 'num':
        feature_names_out.extend(list(columns))
    elif name == 'cat':
        ohe = transformer.named_steps['ohe']
        feature_names_out.extend(ohe.get_feature_names_out(columns).tolist())
# Now len(feature_names_out) == len(model.feature_importances_)
```

#### Bug #10 — `auc` Variable Shadows `sklearn.metrics.auc`

The original code imported `from sklearn.metrics import auc` and then later wrote `auc = roc_auc_score(y_test, y_pred_proba)`, overwriting the imported function with a float. Any subsequent call to `auc(fpr, tpr)` would fail.

**Fix:** The score variable is renamed to `roc_auc_value`. The import is aliased as `from sklearn.metrics import auc as sklearn_auc`.

#### Bug #11 — Fairness Analysis Crashed on Undefined Variables

The fairness analysis cell referenced `tp_y` and `fn_y` — variables that were never defined anywhere in the notebook. The cell would crash with `NameError` every time.

**Fix:** Group-wise metrics are computed using `recall_score(y_test[mask], y_pred[mask])` and `precision_score(...)` from sklearn — no custom variable definitions needed.

---

### Bug Category C — Missing Best Practices

#### Bug #12 — No Cross-Validation
The original notebook evaluated the model on a single train/test split. This gives a high-variance estimate of generalisation performance — the reported AUC could be lucky or unlucky depending on which rows ended up in the test set.

**Fix:** 5-fold `StratifiedKFold` cross-validation is added, reporting mean and standard deviation of AUC and F1 across folds. The train–validation gap is computed to detect overfitting.

#### Bug #13 — No Probability Calibration Check
The model's predicted probabilities are used for risk scoring (ranking applicants by default likelihood). If those probabilities are poorly calibrated — e.g. the model predicts 0.8 but the actual default rate among those applicants is only 0.3 — the scores are misleading for business decisions.

**Fix:** Brier score is reported, and a decile calibration table compares mean predicted probability vs actual default rate in each probability bucket.

---

## 6. Correct ML Pipeline Design

The fundamental rule is: **the test set must never influence any decision made during training.**

This means the correct order of operations is:

```
Raw Data
  ↓
Data Cleaning  (fix errors, standardise categories, cap outliers)
  ↓
Train / Test Split  ← everything below this line uses train data only
  ↓
Feature Engineering  (compute stats from X_train only, apply to X_test)
  ↓
Preprocessing (fit on X_train only, transform X_test)
  ↓
SMOTE Oversampling  (apply to X_train only)
  ↓
Hyperparameter Tuning  (5-fold CV on X_train_balanced only)
  ↓
Train Final Model  (on full X_train_balanced)
  ↓
Threshold Selection  (on a validation split of X_train)
  ↓
Final Evaluation  ← test set touched exactly once, here
```

The original notebook violated this order at **five points** (steps 4, 5, 6, 7, and 8 above), each introducing a different form of leakage.

---

## 7. Feature Engineering

All feature engineering statistics (means, standard deviations, min/max) are computed from `X_train` only, stored in a `stats` dictionary, and applied to `X_test` using those stored values.

### Ratio Features (No Leakage Risk)

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| `loan_to_income` | `loan_amt / (annual_inc + 1)` | Debt burden relative to income |
| `credit_loan_ratio` | `credit_score / (loan_amt + 1)` | Creditworthiness relative to loan size |
| `debt_to_income` | `(loan_amt × interest_rate) / (annual_inc + 1)` | Total debt service burden |
| `rate_x_income_ratio` | `interest_rate × income_ratio` | Interaction: rate and existing debt load |

### Polynomial Features

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| `age_squared` | `person_age²` | Non-linear age effect (young and old may have different risk profiles) |
| `credit_sq` | `credit_score²` | Diminishing marginal improvement at high credit scores |

### Normalised Features (Train Stats Only)

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| `income_zscore` | `(annual_inc − train_mean) / train_std` | Stable z-score without test contamination |
| `credit_normalized` | `(credit_score − train_min) / (train_max − train_min)` | Min-max scaling using training bounds |

> **Why not just let `StandardScaler` handle this?**  
> `StandardScaler` inside the pipeline already scales all numeric features. The z-score and normalised versions are kept as additional features because they create explicit representations of "how unusual is this income?" and "how good is this credit score relative to the training population?" — which can interact differently with tree splits than the raw values.

---

## 8. Preprocessing

The preprocessing pipeline is built with `sklearn`'s `ColumnTransformer` so that numeric and categorical features receive appropriate treatment. The pipeline is fitted **only on `X_train_fe`**.

```
Numeric Features (16 columns)
  → SimpleImputer(strategy='median')   # robust to outliers
  → StandardScaler()                   # zero mean, unit variance

Categorical Features (5 columns)
  → SimpleImputer(strategy='most_frequent')
  → OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

### Why Median Imputation?

The original notebook used `strategy='mean'`. For skewed distributions — which financial data almost always has — the mean is pulled toward extreme values. The median is a more robust measure of centre and produces more reasonable imputations for variables like `annual_inc` and `interest_rate`.

### Why `handle_unknown='ignore'` in OneHotEncoder?

In production, a loan applicant might have a `loan_intent` or `loan_grade` value that never appeared in the training data. With `handle_unknown='ignore'`, the encoder produces all-zero columns for unknown categories rather than raising an error.

---

## 9. Handling Class Imbalance

The dataset has a **4.0% default rate** — a 25:1 class ratio. Without intervention, a model trained on this data will learn to predict "no default" for almost everything, because that maximises accuracy. Two complementary strategies are used:

### Strategy 1 — SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates **synthetic minority-class samples** by interpolating between existing minority samples in feature space. This is better than random oversampling (which just duplicates existing rows) because it creates new, varied examples.

```python
smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.3)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
```

`sampling_strategy=0.3` means the minority class grows to **30% of the majority count** — not a full 50/50 split. This is intentional: extreme oversampling to 50/50 creates an unrealistic training distribution and can hurt calibration.

> **Critical:** SMOTE is applied only to `X_train_proc` / `y_train`. The test set is never resampled — it stays at the natural 4% default rate to represent real-world conditions.

### Strategy 2 — `class_weight='balanced'`

`RandomForestClassifier` is initialised with `class_weight='balanced'`, which automatically weights the minority class higher in the loss function. This acts as a second layer of imbalance correction, complementing SMOTE.

If SMOTE is unavailable (e.g. `imbalanced-learn` not installed), `class_weight='balanced'` alone is sufficient as a fallback.

---

## 10. Hyperparameter Tuning

Hyperparameters are selected using **5-fold Stratified Cross-Validation** on the training data. The test set is never involved.

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Stratified = class ratio preserved in every fold
```

The grid searched two parameters of `RandomForestClassifier`:

| Parameter | Values Searched | Purpose |
|-----------|----------------|---------|
| `max_depth` | 5, 10, 15, 20 | Controls tree depth — limits overfitting |
| `min_samples_leaf` | 5, 10, 20 | Minimum samples per leaf — regularises the model |

For each combination, mean CV AUC (± std) is computed across 5 folds. The combination with the highest mean validation AUC is selected. The standard deviation is tracked to prefer stable configurations over lucky ones.

The final model is then retrained on the **full training set** (all 5 folds combined) using the selected hyperparameters, with `n_estimators=300` (more trees for the final model than during CV search).

---

## 11. Threshold Selection

A `RandomForestClassifier` outputs probabilities, not direct binary labels. Converting probabilities to labels requires a threshold: applicants above the threshold are flagged as likely defaults.

The default threshold of 0.5 is rarely optimal for imbalanced problems — at 4% default rate, very few applicants will have predicted probability > 0.5, leading to low recall.

### Process

1. A **validation split** (20% of training data) is held out — separate from the CV folds and the test set.
2. A copy of the final model is trained on the remaining 80%.
3. Thresholds from 0.10 to 0.85 (step 0.05) are evaluated against the validation labels.
4. The threshold with the highest **F1 score** on the validation set is selected.
5. This threshold is applied when generating predictions on the test set.

> **Why F1 for threshold selection?**  
> F1 is the harmonic mean of precision and recall. It is a natural choice for imbalanced problems where both catching defaults (recall) and not over-flagging (precision) matter. AUC is used for model selection; F1 is used for threshold selection.

---

## 12. Model Evaluation

The test set is evaluated **exactly once**, after all design decisions have been finalised. The following metrics are reported:

| Metric | What It Measures |
|--------|-----------------|
| **ROC-AUC** | Overall discriminative ability across all thresholds. Main performance metric. |
| **F1-Score** | Balance of precision and recall at the selected threshold. |
| **Recall** | % of actual defaults caught. Critical for risk management. |
| **Precision** | % of flagged loans that truly default. Business cost of false alarms. |
| **Specificity** | % of legitimate borrowers correctly cleared. |
| **Accuracy** | Overall correctness — less meaningful at 4% imbalance. |
| **Brier Score** | Calibration quality — how reliable are the probability scores? |

### Confusion Matrix Interpretation

```
                  Predicted: No Default   Predicted: Default
Actual: No Default        TN                    FP  ← false alarm (cost: lost business)
Actual: Default           FN                    TP  ← caught default
                          ↑
                    missed default (cost: financial loss)
```

In credit risk, **false negatives (FN) are more costly** than false positives (FP). Lowering the threshold increases recall (catches more defaults) at the cost of more false alarms (FP). The threshold selected in Step 11 controls this trade-off.

---

## 13. Fairness Analysis

A credit risk model can unintentionally discriminate against demographic groups — not through explicit use of protected attributes, but because correlated features (income, employment length) may behave differently across groups.

The analysis computes recall and precision separately for three age groups:

| Group | Age Range |
|-------|-----------|
| Young | 20–29 |
| Middle | 30–49 |
| Senior | 50+ |

A large gap in recall between groups means the model catches fewer defaults for some groups — potentially flagging them less often than warranted, or more often. Both types of disparity are concerning.

> **Note on Bug #11:** The original fairness analysis referenced undefined variables `tp_y` and `fn_y`, causing a `NameError` crash every time. The fix computes group metrics directly via `sklearn.metrics.recall_score` with boolean masks:

```python
group_recall = recall_score(y_test_arr[mask], y_pred_arr[mask], zero_division=0)
```

---

## 14. Probability Calibration

The model assigns a probability score to each applicant. For this score to be meaningful in practice (e.g. "this applicant has a 12% probability of defaulting"), the probabilities must be **calibrated** — if the model says 0.2, roughly 20% of those applicants should actually default.

### Brier Score

The Brier score is the mean squared error between predicted probabilities and actual outcomes. Lower is better.

- **No-skill baseline** for 4% prevalence: `0.04 × 0.96 ≈ 0.038`
- A well-trained model should achieve a Brier score significantly below this baseline.

### Decile Calibration Table

Applicants are sorted by predicted probability and split into 10 equal buckets (deciles). For each decile, mean predicted probability is compared to actual default rate:

| Decile | Mean Predicted | Actual Default Rate |
|--------|---------------|---------------------|
| 1 (low risk) | ~0.01 | ~0.01 ✅ |
| 5 (medium) | ~0.08 | ~0.07 ✅ |
| 10 (high risk) | ~0.45 | ~0.40 ✅ / ⚠️ |

Large gaps (> 0.05) between predicted and actual in any decile indicate miscalibration. If the high-risk decile shows a large gap, probability scores cannot be used reliably for ranking or cutoff decisions.

---

## 15. Summary of All Changes

### Data Cleaning Changes (7)

| # | Change | Reason |
|---|--------|--------|
| D1 | Dropped 7 rows with `person_age > 100` | Impossible values — data entry errors |
| D2 | Normalised `employment_type`: 4 variants → 2 | Prevents encoder from creating redundant duplicate columns |
| D3 | Uppercased `residence_type` uniformly | Prevents encoder treating `Urban`/`URBAN` as different categories |
| D4 | Capped `annual_inc` and `loan_amt` at 99.5th percentile | Prevents extreme values from collapsing StandardScaler range |
| D5 | Changed imputation strategy: `mean` → `median` | Median is robust to outliers in skewed financial distributions |
| D6 | All missing columns handled in pipeline | Avoids data loss from row dropping |
| D7 | SMOTE + `class_weight='balanced'` for 4% imbalance | Prevents model from ignoring the minority class |

### Pipeline Changes (8)

| # | Original | Fixed |
|---|----------|-------|
| P1 | Leakage features included (`loan_status_final`, etc.) | Removed entirely |
| P2 | Noise features included (`random_score_*`, `duplicate_feature`) | Removed entirely |
| P3 | Feature engineering on full dataset | Split first; engineering on train only |
| P4 | Preprocessor fitted on train+test combined | Fitted on train only; `transform()` on test |
| P5 | SMOTE on combined / post-split incorrectly | SMOTE on training data only |
| P6 | Hyperparameter search on test set | 5-fold CV on training data |
| P7 | Threshold tuned on test set | Validation split from training data |
| P8 | Single train/test split only | 5-fold CV + single held-out test set |

### Code Bug Fixes (5)

| # | Bug | Fix |
|---|-----|-----|
| C1 | `imblearn` import crash | Separated install cell; graceful fallback |
| C2 | Feature importances misaligned with OHE expansion | `get_feature_names_out()` from ColumnTransformer |
| C3 | `auc` variable shadows `sklearn.metrics.auc` | Renamed to `roc_auc_value`; import aliased |
| C4 | Fairness analysis crashes (`NameError: tp_y`) | Rewritten using `recall_score` with masks |
| C5 | No calibration check | Brier score + decile calibration table |

---

*Document covers notebook version: `Fixed_Credit_Risk_Model.ipynb`*  
*Dataset: `credit_risk_dataset.csv` — 13,259 rows (post-cleaning) × 13 features (post-leakage-removal)*