# Credit Risk Default Prediction — Project Methodology

**Dataset:** credit_risk_dataset.csv
**Task:** Binary classification — predicting loan default before disbursement
**Author:** [Your Name]
**Date:** March 2026

---

## 1. Introduction

Lending institutions lose significant revenue every year to loan defaults that could have been anticipated with better screening. The purpose of this project was to construct a machine learning model capable of estimating the probability that a given applicant will default on their loan, using only information that realistically exists at the time of application.

The dataset comprised 13,266 borrower records collected across multiple loan categories. Each record included personal financial indicators such as income, credit score, and employment details, along with the outcome label indicating whether the borrower eventually defaulted. The severe imbalance in that outcome — roughly 4 defaults for every 96 non-defaults — made this a particularly challenging problem requiring deliberate methodological choices at nearly every stage.

This document walks through the decisions made during data cleaning, pipeline construction, model selection, and evaluation, including a thorough account of the errors discovered in the original notebook and how each was corrected.

---

## 2. Understanding the Dataset

Before writing a single line of modelling code, the dataset was studied in its raw form. This step is often rushed, but it proved essential here because several columns were not what they appeared to be on the surface.

The 20 columns divided into three groups once examined carefully. The first group consisted of legitimate applicant characteristics gathered before loan approval: age, income, employment tenure, loan amount, interest rate, loan purpose, home ownership, credit score, and a few derived ratios. These are the features a bank would reasonably have access to when deciding whether to approve an application.

The second group contained three columns — `loan_status_final`, `repayment_flag`, and `last_payment_status` — that described what happened *after* the loan was issued. These columns exist because the dataset was assembled from historical records where both the application details and the eventual outcomes were stored together. Including them in a predictive model creates a situation where the model is essentially told the answer before it makes a prediction. This type of contamination, known as target leakage, produces models that appear highly accurate in testing but are completely useless in deployment.

The third group was more subtle. Three columns — `random_score_1`, `random_score_2`, and `duplicate_feature` — contributed nothing meaningful. The random scores showed no correlation with the target across any slice of the data, while `duplicate_feature` turned out to correlate at 0.30 with `person_age`, making it an imprecise copy of a feature already in the dataset. Keeping noisy or redundant inputs degrades model generalisation and muddies feature importance analysis.

Missing values were spread across five columns, with `interest_rate` being the most severely affected at 19.5% missing. None of the missingness patterns appeared to be related to the target, so imputation was considered appropriate rather than deletion.

---

## 3. Data Cleaning

### 3.1 Removing Invalid Records

The `person_age` column contained values ranging up to 999 years. Seven records fell above the realistic human maximum of 100 and were dropped outright. These were not outliers in any statistical sense — they were entry errors, and retaining them would introduce a small but entirely fictitious signal into the model.

### 3.2 Managing Extreme Financial Values

Income and loan amounts showed a pronounced right skew. While most applicants earned between forty and ninety thousand dollars annually, a small number had recorded incomes above one million dollars. Similarly, the vast majority of loans were under fifty thousand dollars, but a handful exceeded one million. At these extremes, individual records can disproportionately influence the scaling parameters used in preprocessing, compressing the bulk of the distribution into a narrow band where the model loses the ability to distinguish between cases.

Rather than dropping these records, both columns were capped at their 99.5th percentile values. This approach retains the observation while eliminating the distorting effect on scaling — a sensible compromise when the underlying transaction might be legitimate even if the recorded value looks unusual.

### 3.3 Standardising Categorical Labels

Two categorical columns had inconsistent label formatting that went unnoticed in the original code. The `employment_type` column stored the same two categories under four different spellings: `employed`, `Employed`, `self_emp`, and `Self-employed`. Without correction, a one-hot encoder treats each of these as a separate category and creates four binary columns where only two are needed. The resulting redundancy inflates the feature space and, more problematically, splits observations from the same true category across different model inputs.

The fix was straightforward: lowercase all values, replace hyphens and spaces with underscores, and map variant spellings to a single canonical form. The same logic was applied to `residence_type`, where `Urban`, `URBAN`, `Rural`, and `RURAL` were unified into `URBAN` and `RURAL`.

---

## 4. Errors in the Original Notebook

The original notebook contained thirteen distinct errors. Some were immediate runtime failures; others were logical mistakes that allowed the code to run while producing silently invalid results. Both types were identified and corrected.

### 4.1 The Install-Import Sequencing Problem

The notebook installed `imbalanced-learn` using pip in one cell and then attempted to import it in the very next cell. Python does not make newly installed packages available in the current session without restarting the kernel. The import failed with `ModuleNotFoundError` every time the notebook was run in a fresh environment. The correction was to place the installation step in a clearly labelled cell with an explicit instruction to restart before continuing, and to wrap the subsequent import in a `try/except` block so the notebook degrades gracefully if the library is unavailable.

### 4.2 Data Leakage Through Post-Outcome Features

As described in Section 2, three columns represent events that occur after the default decision being predicted. Including these in the feature matrix is the most damaging error in the original notebook. A model trained with these inputs learns to recognise the outcome rather than the risk factors, and will fail completely when deployed against new applications where these post-outcome values do not yet exist. All three columns were excluded from the feature set entirely.

### 4.3 Feature Engineering Before the Train-Test Split

The original code computed the mean and standard deviation of `annual_inc` from the full dataset and then used those values to create an income z-score column. It also ran a correlation analysis between all features and the target on the full dataset to select which features to keep. Both operations incorporate information from the test set into the training process. The test mean pulls the z-score values toward it; the correlation analysis selects features based partly on their relationships within test rows.

In the corrected version, the train-test split occurs before any of this work. Income statistics and credit score bounds are calculated using only the training rows, stored in a dictionary, and then applied to the test set using those stored values. The test set provides no information at any stage of this process.

### 4.4 Preprocessing Fitted on Combined Data

The original code concatenated the training and test sets, fitted the `ColumnTransformer` on the combined data, and then split the transformed output back apart. This approach passes test-set observations through the `SimpleImputer` and `StandardScaler` fitting step, meaning the imputed medians and scaling parameters reflect test data. The fix is to call `fit_transform()` only on the training set and `transform()` only on the test set.

### 4.5 Oversampling Applied Across the Split

The SMOTE step in the original code operated on data that still included test observations. Generating synthetic samples from the combined distribution contaminates the test set with synthetic points that share characteristics with training rows. In the corrected version, SMOTE runs only after preprocessing, only on the training data, and the test set is never touched by the resampler.

### 4.6 Hyperparameter Search Using the Test Set

The original notebook looped through fifteen combinations of `max_depth` and `min_samples_leaf`, evaluated each model's AUC against the test labels, and kept the model with the highest test-set score. This is a well-documented methodological error sometimes called "peeking." By effectively running fifteen statistical tests on the same test set, the apparent best performance is inflated by random chance. Whatever AUC is reported for that best model is optimistic by construction.

The corrected approach runs this search entirely within the training set using five-fold stratified cross-validation. Each fold uses approximately 80% of training data to fit and 20% to validate, and the process repeats five times so every training observation appears in a validation fold exactly once. The final model is selected on the basis of mean cross-validation AUC, which is a stable and unbiased estimate.

### 4.7 Threshold Tuning on the Test Set

After selecting a model, the original code scanned threshold values from 0.2 to 0.8 and chose the one producing the best F1 score on `y_test`. This is the same logical error as the hyperparameter search: the threshold is a model decision, and optimising it directly against the test labels renders the reported F1 invalid as a measure of generalisation.

In the fixed notebook, a separate validation split is carved from the training data before the final model is trained. This validation set is used only for threshold selection. The test set remains completely untouched until the single final evaluation step.

### 4.8 Feature Importances Misaligned with Model Dimensions

The original code built a feature name list by concatenating the numeric column names with the categorical column names, then matched this list to `model.feature_importances_`. The problem is that a `OneHotEncoder` expands each categorical column into several binary columns — one per category level. The model operates in this expanded space and produces one importance value per binary column. The name list, however, contained only the original column names and was shorter than the importance array, causing a silent alignment mismatch where importances were attributed to the wrong features.

The correct approach extracts the full expanded name list from the fitted `ColumnTransformer` using `get_feature_names_out()`, which returns one name per transformed output column including all one-hot encoded variants.

### 4.9 Variable Name Shadowing

The line `auc = roc_auc_score(y_test, y_pred_proba)` overwrote the previously imported `auc` function from `sklearn.metrics`. Any later call to `auc(fpr, tpr)` to compute the area under the ROC curve would then fail because `auc` was now a float, not a function. The score variable was renamed to `roc_auc_value` and the import was aliased as `sklearn_auc` to prevent future conflicts.

### 4.10 Fairness Analysis Crashing on Undefined Variables

The fairness analysis cell attempted to calculate group-level recall using variables `tp_y` and `fn_y` that were never assigned anywhere in the notebook. The cell raised a `NameError` on every execution. The corrected version computes group metrics by applying boolean index masks to the test arrays and passing them directly to `sklearn.metrics.recall_score` and `precision_score`, which handle zero-division edge cases through the `zero_division` parameter.

### 4.11 Missing Cross-Validation

Evaluating a model on a single train-test split yields an estimate of performance that depends heavily on which particular rows ended up in each partition. A fortunate split can make a mediocre model look good; an unfortunate one can obscure a genuinely strong model. Five-fold stratified cross-validation on the training set was added to estimate performance variance and confirm that results are consistent across different data subsets.

### 4.12 Missing Probability Calibration

A random forest produces probability estimates for each prediction. These probabilities are used in risk scoring — ranking applicants by their estimated likelihood of default. If those probabilities are systematically too high or too low, the scores become unreliable for any threshold-based decision. The corrected notebook reports the Brier score and a decile-level calibration table to assess whether the model's probability outputs are trustworthy.

---

## 5. Pipeline Construction

The complete modelling pipeline was structured so that no information from the test set could influence any training decision. The ordering was:

Raw data cleaning ran first across the full dataset, because correcting entry errors and standardising labels is a data repair operation rather than a statistical one. The train-test split followed immediately after. From that point forward, every transformation, every statistic, and every model decision was derived exclusively from training rows.

Feature engineering came next. The new features were computed using training-set statistics, stored, and applied consistently to both sets without recalculation. Preprocessing — imputation and scaling — was fitted on the training set and applied as a fixed transformation to the test set. SMOTE oversampling followed on the training data only.

Hyperparameter selection used five-fold cross-validation within the training set. Once the optimal configuration was identified, the final model was trained on all available training data using those parameters. A separate validation split from the training set was used for threshold selection. The held-out test set was evaluated exactly once at the end.

---

## 6. Feature Engineering

Beyond the original eight numeric columns, six additional features were constructed to capture relationships that the raw values do not express individually.

The loan-to-income ratio divides the requested loan amount by annual income, directly measuring how much of a borrower's yearly earnings the loan represents. The debt-to-income feature takes this further by incorporating the interest rate — a borrower asking for fifty thousand dollars at twenty percent interest creates a very different obligation than the same loan at six percent. The credit-to-loan ratio inverts this perspective, expressing how much creditworthiness the borrower brings relative to the size of the ask.

Two polynomial terms were included: `age_squared` and `credit_sq`. These allow the model to capture non-linear relationships — for instance, that very young and elderly borrowers might share elevated risk profiles that a linear age term alone would not represent correctly. The interaction between interest rate and income ratio captures cases where high rates compound an already stretched income, a pattern that neither variable captures independently.

All normalised features — income z-score and credit score min-max normalisation — were computed from training statistics exclusively, then applied to the test set using those stored values.

---

## 7. Handling Class Imbalance

With only 4% of observations in the positive class, standard model training tends to optimise toward predicting non-default on nearly every case, because doing so produces a very high accuracy despite catching no actual defaults at all. Two measures were taken to address this.

SMOTE was applied to the preprocessed training data. Unlike random oversampling, which duplicates existing minority-class rows, SMOTE generates synthetic samples by interpolating between real minority-class observations in feature space. For each minority sample, the algorithm selects one of its five nearest neighbours and places a new synthetic point somewhere along the line connecting them. This produces a more varied representation of the minority class rather than repeatedly presenting identical observations during training.

The sampling strategy was set to 0.3, meaning the minority class grew to approximately 30% of the majority class count rather than achieving full parity. Oversampling to 50/50 creates a training distribution that differs substantially from the real-world distribution, which can impair the model's probability calibration. A moderate ratio was chosen as a pragmatic compromise between giving the model sufficient positive examples and maintaining some connection to the actual class distribution.

`class_weight='balanced'` was additionally specified in the `RandomForestClassifier`. This adjusts the weight of each training sample inversely proportional to its class frequency, so errors on minority-class samples are penalised more heavily during tree construction. This operates at a different level from SMOTE — rather than changing the data, it changes how the model responds to errors in that data — and the two approaches complement each other.

---

## 8. Model Selection and Tuning

`RandomForestClassifier` was selected as the primary algorithm. Decision tree ensembles are well suited to tabular financial data: they handle non-linear relationships naturally, are relatively robust to feature scale differences, require minimal preprocessing assumptions, and produce feature importance estimates that can be used for model interpretation and compliance reporting.

Hyperparameter search covered `max_depth` values of 5, 10, 15, and 20, combined with `min_samples_leaf` values of 5, 10, and 20. These two parameters primarily control model complexity: deeper trees with fewer samples per leaf will fit the training data more closely, while shallower trees with larger minimum leaf sizes are more regularised and generalise better. The grid covers a range from quite conservative to potentially overfit configurations, allowing cross-validation to identify where the model performs best on unseen data.

Each combination was evaluated using five-fold stratified cross-validation on the training set, with ROC-AUC as the scoring metric. Stratification ensures that each fold contains the same proportion of defaults as the full training set, which matters when the class ratio is as extreme as it is here. The combination with the highest mean validation AUC across the five folds was selected, and the final model was trained on the full training set using 300 estimators rather than the 100 used during search.

---

## 9. Classification Threshold Selection

Because the default rate in the test set reflects the natural 4% real-world rate, the model's raw predicted probabilities are concentrated near zero for almost all applicants. At the conventional 0.5 threshold, very few cases would be flagged and recall would be poor.

Threshold selection was treated as a separate decision from model selection and was carried out on a dedicated validation split drawn from the training data. The validation split was never used for any other purpose — not for cross-validation, not for model comparison, only for finding the threshold value that maximises F1 score on held-out training observations.

F1 was chosen as the selection criterion because it balances precision and recall without assuming that one is more important than the other. In practice, the relative cost of false positives and false negatives is a business decision — missing a default is typically more expensive than incorrectly flagging a good borrower — but F1 provides a reasonable default objective in the absence of a specific cost matrix.

---

## 10. Evaluation

The test set was evaluated once, after all modelling decisions had been finalised. The primary reported metric is ROC-AUC, which measures the model's ability to rank defaulters above non-defaulters across all possible thresholds. A value of 0.5 indicates no discriminative ability; a value of 1.0 indicates perfect separation.

F1 score, recall, precision, and specificity are additionally reported at the selected threshold. Recall is particularly important for credit risk: a low recall means the bank is approving loans that will default. Precision reflects the cost of unnecessary rejections. Reporting both alongside F1 gives a complete picture of the trade-off the threshold embodies.

The Brier score assesses calibration. For a dataset with 4% prevalence, the no-skill baseline Brier score is approximately 0.038. A model that produces well-calibrated probabilities should comfortably beat this baseline and have a Brier score well below it. The decile calibration table provides more granular insight: applicants are grouped into ten probability buckets and the average predicted probability in each bucket is compared to the actual default rate. Close agreement across all deciles confirms that the probability scores can be trusted for risk ranking purposes.

---

## 11. Fairness Considerations

Even when protected attributes such as race or gender are excluded from the model, demographic disparities can emerge indirectly through correlated features. Income and employment length, for example, may differ systematically across age groups in ways that the model interprets purely as risk signals.

To partially address this, recall and precision were computed separately for three age groups: applicants under 30, those between 30 and 49, and those aged 50 and above. A substantial gap in recall between groups — for instance, the model catching 70% of defaults among middle-aged borrowers but only 40% among younger ones — would indicate that the model provides unequal protection against financial loss depending on the borrower's demographic profile.

This analysis is reported transparently rather than used to adjust the model, on the grounds that any post-hoc adjustment to close demographic gaps should be a deliberate decision made with stakeholders rather than an automatic technical intervention.

---

## 12. Reproducibility

All random operations in the pipeline — the train-test split, SMOTE sampling, cross-validation fold assignment, and model training — were seeded with `random_state=42`. The preprocessing pipeline is encapsulated in a `ColumnTransformer` that can be serialised alongside the trained model. Feature engineering uses explicitly stored training statistics rather than recomputed values, ensuring that the same transformation is applied in production as was applied during training.

The full list of features, their types, and the order in which they are expected by the preprocessor are documented within the notebook. Any new data passed to the model must undergo the same cleaning steps described in Section 3 before entering the feature engineering and preprocessing steps.

---

*This document describes the methodology applied in `Fixed_Credit_Risk_Model.ipynb`. All analysis was performed on `credit_risk_dataset.csv` containing 13,259 usable records after cleaning.*