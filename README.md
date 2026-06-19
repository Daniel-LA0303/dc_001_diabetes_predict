# Diabetes Prediction Project

A machine learning project that predicts the likelihood of diabetes from patient health indicators, including exploratory data analysis, a trained classification model, and a Streamlit web app for interactive predictions.

---

## Project Structure

```
diabetes-prediction/
├── diabetes_prediction_dataset.csv   # Dataset (100,000 rows)
├── notebook.ipynb                    # EDA + model training (Jupyter)
├── app.py                            # Streamlit web app
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

---

## Dataset

#### THIS DATASET IS FROM KAGGLE
**File:** `diabetes_prediction_dataset.csv`

| Feature | Description |
|---|---|
| `gender` | Male / Female / Other |
| `age` | Patient age |
| `hypertension` | 0 = No, 1 = Yes |
| `heart_disease` | 0 = No, 1 = Yes |
| `smoking_history` | Never, Current, Former, Ever, Not Current, No Info |
| `bmi` | Body Mass Index |
| `HbA1c_level` | Average blood sugar over ~3 months |
| `blood_glucose_level` | Blood glucose level |
| `diabetes` | **Target** — 0 = No, 1 = Yes |

---

## Part 1 — Notebook (EDA + Model Training)

`notebook.ipynb` covers the data science pipeline:

- Data loading and inspection (`head`, `info`, `describe`, `shape`)
- Missing values check
- Outcome variable distribution (class balance)
- Visualizations: histograms, pie chart, countplot, correlation heatmap, pairplot
- Categorical encoding (`gender`, `smoking_history`)
- Train/test split
- Model training: `RandomForestClassifier`
- Evaluation: accuracy, confusion matrix, classification report

### Current limitations (to be improved)
- Encoding uses `OrdinalEncoder` on nominal variables — should be one-hot encoded
- Train/test split is not stratified
- Class imbalance is observed but not addressed in modeling
- Only one model is trained — no comparison (e.g. Logistic Regression, KNN, XGBoost)
- No cross-validation or hyperparameter tuning yet
- No ROC/AUC curve or feature importance plot
- Accuracy is the main reported metric, which can be misleading on imbalanced data

### Run the notebook
```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels missingno jupyter
jupyter notebook
```

---

## Part 2 — Streamlit App (Interactive Prediction)

`app.py` is a web interface that lets a user enter their health information and get a diabetes risk prediction from the trained model.

### Features
- Two-column form: gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level
- "Submit" button triggers prediction
- Result shown as **Negative** or **Positive (consult a doctor)** 

### Known limitations (to be improved)
- Model currently **retrains on every app run** instead of loading a saved model — slow and non-deterministic (no `random_state` set)
- Numeric fields use `st.text_input` (returns strings) instead of `st.number_input` (returns proper numeric types)
- Bare `except:` block hides real errors instead of validating input properly
- Only shows binary result, not prediction probability/confidence
- No medical disclaimer shown to the user
- No logging of predictions (needed for future retraining / monitoring)

### Run the app
```bash
pip install streamlit
streamlit run app.py
```
App will open at `http://localhost:8501`

---

## Model Lifecycle (Current vs Target)

**Current state:**
```
notebook.ipynb  →  trains model inline
app.py          →  retrains model inline on every run (duplicated, inefficient)
```

**Target state:**
```
notebook.ipynb  →  trains + evaluates model  →  saves model.pkl (joblib)
app.py          →  loads model.pkl  →  takes user input  →  predicts  →  logs prediction
retrain.py      →  (future) merges newly labeled data  →  retrains  →  saves updated model.pkl
```

> **Note on retraining with user data:** logging user inputs is straightforward, but retraining requires *verified outcomes* (confirmed diabetes diagnoses), not just the model's own predictions. Using unverified predictions as training labels risks reinforcing the model's existing errors (label leakage).

---

## 🚀 Roadmap

### Data Science (notebook)
- [ ] Duplicate analysis
- [ ] Outlier detection
- [ ] One-hot encoding for nominal features
- [ ] Stratified train/test split
- [ ] Class imbalance handling (`class_weight='balanced'` or SMOTE)
- [ ] Model comparison (Logistic Regression, KNN, Random Forest, XGBoost)
- [ ] Cross-validation
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] ROC curve / AUC
- [ ] Feature importance plot
- [ ] Save trained model with `joblib`

### App (Streamlit)
- [ ] Load saved model instead of retraining on every run
- [ ] Switch text inputs to `number_input` with valid ranges
- [ ] Show prediction probability, not just binary result
- [ ] Add medical disclaimer
- [ ] Log predictions to CSV/database
- [ ] Add input tooltips (normal ranges for BMI, HbA1c, glucose)

### Deployment
- [ ] Model persistence
- [ ] API endpoint (optional)
- [ ] Monitoring / data drift checks

---

## Disclaimer

This project is for educational purposes only. Predictions are **not a medical diagnosis**. Always consult a healthcare professional for medical advice.

---

## Tech Stack

- **Language:** Python
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, missingno
- **Statistics:** statsmodels
- **ML:** scikit-learn (RandomForestClassifier)
- **Web App:** Streamlit
