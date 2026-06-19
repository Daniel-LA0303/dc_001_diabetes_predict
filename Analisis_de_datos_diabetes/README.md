# Diabetes Prediction — ML Project

Predicting diabetes diagnosis from patient health indicators using conventional machine learning.

## Dataset

`diabetes_prediction_dataset.csv`

**Features:**
- `gender`, `age`
- `hypertension`, `heart_disease`
- `smoking_history`
- `bmi`, `HbA1c_level`, `blood_glucose_level`

**Target:** `diabetes` (0 = no, 1 = yes)

## Tech Stack

- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn, missingno
- **Statistics:** statsmodels
- **ML:** scikit-learn (RandomForestClassifier)

## Project Status

This project follows a standard Data Science pipeline. Progress so far:

### ✅ Data Analysis (EDA)
- [x] Data loading and initial exploration
- [x] Missing values check
- [x] Distribution analysis (numeric features)
- [x] Outcome variable distribution (class balance check)
- [x] Data visualization (histograms, heatmap, pairplot)
- [x] Correlation analysis
- [ ] Duplicate analysis
- [ ] Outlier detection
- [ ] Multicollinearity analysis
- [ ] Statistical analysis (hypothesis testing)
- [ ] Feature selection
- [ ] Data leakage detection
- [ ] Domain analysis

### 🟡 Data Preparation
- [x] Encoding (categorical → numeric) — *currently uses OrdinalEncoder, should switch to one-hot for nominal features*
- [ ] Cleaning (duplicates, outliers)
- [ ] Feature scaling

### 🟡 Machine Learning
- [x] Train/test split — *not yet stratified*
- [x] Model training (Random Forest baseline)
- [x] Basic evaluation (accuracy, confusion matrix, classification report)
- [ ] Model comparison (Logistic Regression, KNN, XGBoost, etc.)
- [ ] Cross-validation
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] ROC curve / AUC
- [ ] Feature importance analysis
- [ ] Model comparison table
- [ ] Model persistence (save trained model)

### Deployment
- [ ] Export model
- [ ] API endpoint
- [ ] Monitoring

## Next Steps

1. Remove duplicate rows and handle outliers
2. Fix encoding (one-hot for `gender`, `smoking_history`)
3. Use stratified train/test split (`stratify=y`)
4. Address class imbalance (`class_weight='balanced'` or SMOTE)
5. Compare multiple models (Logistic Regression, KNN, Random Forest, XGBoost)
6. Add cross-validation and hyperparameter tuning
7. Plot ROC/AUC curve and feature importances
8. Save the best model with `joblib`
9. (Optional) Build a simple API to serve predictions

## Known Issues

- `OrdinalEncoder` is used on nominal categorical variables (`gender`, `smoking_history`), which incorrectly implies an order between categories. Should be replaced with one-hot encoding.
- Train/test split is not stratified, so class proportions may differ between train and test sets.
- Class imbalance in the target variable is observed but not addressed in modeling.
- Accuracy is reported as the primary metric, which can be misleading on an imbalanced dataset — precision/recall/F1/AUC for the positive class should be prioritized.

## How to Run

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels missingno
jupyter notebook
```

---
*This README reflects the current state of the project and will be updated as new steps are implemented.*
