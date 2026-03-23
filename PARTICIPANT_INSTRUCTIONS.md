AI MODEL FIX: CREDIT RISK DEBUGGING CHALLENGE
===============================================================================

## CHALLENGE OBJECTIVE

Your mission: **Fix a broken machine learning model for credit risk prediction**

This is a scenario where a financial institution deployed a credit risk model that:
- May contain bugs
- May use wrong approaches
- May leak confidential information (data leakage)
- May not properly handle imbalanced data
- May use incorrect performance metrics

**Your job:** Find and fix ALL issues, then optimize the model for production.

---

## COMPETITION FORMAT

### Time: 1 hour
### Team Size: 2-4 people
### Difficulty: Moderate to Hard
### Required Skills:
- Python, Pandas, Scikit-learn
- Understanding of ML workflows
- Statistical knowledge (imbalance, metrics, etc.)
- Domain understanding (finance/credit risk basics)

---

## WHAT YOU'LL RECEIVE

1. **Corrupted Dataset:** `credit_risk_dataset.csv`
   - 13,266 loan records
   - 20 features (some problematic)
   - Binary target: `target_flag` (1=default, 0=no default)

2. **Broken Notebook:** `Broken_Credit_Risk_Model.ipynb`
   - A Python Jupyter notebook with an ML pipeline
   - Contains 10+ intentional errors
   - Some errors obvious, some subtle
   - Code may not run, or may run but produce wrong results

3. **Reference Solution:** `CORRECTED_Credit_Risk_Model.ipynb`
   - BONUS: Available after submission to review
   - Shows the correct approach

---

## YOUR DELIVERABLES

### Fixed Notebook 
- Complete, working Python notebook
- All errors corrected
- Comments explaining fixes
- Proper preprocessing, modeling, and evaluation

**What judges look for:**
- ✓ Code runs without errors
- ✓ All major bugs fixed
- ✓ Follows ML best practices
- ✓ Well-documented code




**Minimum expectations:**
- Identify at least 8 errors out of 14 hidden errors
- Top teams should find 12+



###  Final Results & Analysis (20 points)
Present:

**A. Model Performance Metrics**
```python
Accuracy:      ____
Precision:     ____
Recall:        ____
F1-Score:      ____
ROC-AUC:       ____ 
```

**B. Confusion Matrix**
```
              Predicted
              0    1
Actual   0   TN   FP
         1   FN   TP
```

**C. Analysis Questions:**
- [ ] What was the worst error and why?
- [ ] How much did each fix improve performance?
- [ ] What would you change to improve further?
- [ ] How does your model compare to baselines?

---

## DATASET DESCRIPTION

### Location
`credit_risk_dataset.csv` (13,266 rows × 20 columns)

### Target Variable
**target_flag** (0=No Default, 1=Default)
- Class distribution: 96% negatives, 4% positives (IMBALANCED!)
- This is realistic - defaults are rare

### Features

#### Numeric Features
| Feature | Description | Issues |
|---------|-------------|--------|
| `person_age` | Age of applicant |
| `annual_inc` | Annual income |
| `employment_length` | Years employed |
| `loan_amt` | Loan amount |
| `interest_rate` | Interest rate % |
| `credit_score` | Credit score |
| `monthly_income` | Monthly income |
| `income_ratio` | Income to loan ratio |

#### Categorical Features
| Feature | Description | Issues |
|---------|-------------|--------|
| `home_ownership` | Rent/Own/Mortgage |
| `loan_intent` | Purpose of loan |
| `loan_grade` | Loan grade A-G |
| `employment_type` | Type of employment |
| `residence_type` | Urban/Rural |


---

## TECHNICAL REQUIREMENTS

### Environment
```
Python 3.8+
pandas >= 1.0
scikit-learn >= 0.24
numpy >= 1.19
imbalanced-learn >= 0.8 (for SMOTE)
matplotlib (for visualization)
seaborn (for visualization)
```

### Installation
```bash
pip install pandas scikit-learn numpy imbalanced-learn matplotlib seaborn
pip install imblearn
```

### Deliverable Format
- **File name:** `FIXED_[TeamName]_Credit_Risk.ipynb`
- **Type:** Jupyter Notebook (.ipynb)
- **Language:** Python
- **Must include:** Markdown cells explaining each fix




## LEARNING OUTCOMES

By completing this challenge, you'll learn:

1. **Data Leakage:** How to identify and prevent information leakage
2. **Preprocessing:** Proper techniques for handling missing values and categorical data
3. **Class Imbalance:** How to handle rare event prediction
4. **Evaluation:** Why accuracy is misleading for imbalanced data
5. **ML Pipelines:** Building production-ready ML workflows
6. **Feature Engineering:** Identifying useful vs. noisy features
7. **Threshold Tuning:** Optimizing decision boundaries for business goals

---

## EXPECTED PERFORMANCE PROGRESSION

### As You Fix Errors (Typical Progression)
```
Starting Point (All Errors):
  - Code: Won't run or crashes
  - ROC-AUC: ~0.50 (won't work)
  - Accuracy: ~96% (misleading!)

After Syntax Fixes:
  - Code: Runs but slow
  - ROC-AUC: ~0.65
  - Accuracy: ~94%

After Removing Leakage:
  - Code: Runs smoothly
  - ROC-AUC: ~0.70
  - Accuracy: ~91%

After Proper Preprocessing:
  - Code: Clean
  - ROC-AUC: ~0.72
  - Accuracy: ~90%

After Imbalance Handling + Threshold Tuning:
  - Code: Production-ready
  - ROC-AUC: ~0.75
  - Accuracy: ~88% (lower but honest!)
  - Recall: ~65% (catching most defaults)
```

---

## SCORING RUBRIC

### Total Max Points: 10

| Component | Points | Criteria |
|-----------|--------|----------|
| **Model Performance** | ROC-AUC: >0.85 (3pts), 0.70-0.85 (2pts), <0.70 (1pts) |
| **Models / Algorithms** | 2 point for each model or algorithm used max 6 points |
| **Code Quality** | Clean, documented, follows best practices |
| **Explanation** | Clear reasoning for each fix and impact |
| **Bonus** | 1 for identifying the correct evaluation parameter as the best metric |

### Winning Teams Typically:
- ✓ Fix 12-14 out of 14 errors
- ✓ Achieve ROC-AUC > 0.73
- ✓ Write clear explanation of each bug
- ✓ Show understanding of finance/credit risk context
- ✓ Go beyond basic fixes with optimizations

---

##  SUBMISSION CHECKLIST

Before submitting, verify:

- [ ] Notebook runs without errors
- [ ] All cells execute successfully
- [ ] Markdown cells document each fix
- [ ] Error documentation report is comprehensive
- [ ] Final metrics are clearly displayed
- [ ] Confusion matrix are shown
- [ ] Code is clean and commented

---

##  FAQ

**Q: Can we use external libraries?**
A: Yes! Common ML libraries (scikit-learn, imbalanced-learn, XGBoost, LightGBM) are allowed. Avoid reinventing the wheel.

**Q: Do we need to improve the model beyond fixes?**
A: No, fixing all bugs is the requirement. Optimizations are bonus points.

**Q: Can we use online resources?**
A: Yes! Stack Overflow, scikit-learn docs, Medium articles - all permitted.

**Q: Can we change the dataset?**
A: No - use the provided dataset as-is but you can make changes in the given dataset.


---

This is a **debugging challenge** - expect it to be difficult!

The best teams will:
1. Think critically about ML practices
2. Test hypotheses systematically
3. Document findings clearly
4. Explain the "why" not just the "what"

**Good luck!**

---
