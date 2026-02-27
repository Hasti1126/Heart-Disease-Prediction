#  Cardiovascular Risk Assessment — Clinical Decision Support Tool

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/Hasti-26/Cardiovascular-Risk-Assessment)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-3.0+-red)](https://gradio.app)

> A clinical decision support tool that predicts cardiovascular risk from 11 patient parameters — and explains *why* each factor contributes to the prediction, not just whether disease is likely.

---

## 🎯 Overview

Most heart disease prediction models give you a number and stop there. This tool goes further — it tells a clinician which findings are driving the risk, why they matter clinically, and what to do next.

Built on **918 patient records** from the UCI Heart Disease dataset, the tool combines a machine learning pipeline with structured clinical reasoning to produce outputs that mirror how a cardiologist actually thinks.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 88.6% |
| F1 Score | 0.900 |
| **Recall** | **93.1%** |
| Precision | 87.2% |
| AUC-ROC | 0.900 |

> Recall is the headline metric — in clinical screening, missing a sick patient is far more dangerous than a false alarm. 93.1% means the model catches 95 out of 102 actual heart disease cases.

---

## 🖥️ Demo

![Demo Screenshot](screenshot.png)

🔗 **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/Hasti-26/Cardiovascular-Risk-Assessment)

---

## 🔬 How It Works

### Input
The user enters 11 clinical parameters:

| Parameter | Description |
|---|---|
| Age | Patient age in years |
| Sex | Biological sex (M/F) |
| ChestPainType | TA / ATA / NAP / ASY |
| RestingBP | Resting blood pressure (mmHg) |
| Cholesterol | Total cholesterol (mg/dL) |
| FastingBS | Fasting blood sugar > 120 mg/dL |
| RestingECG | Normal / ST abnormality / LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y/N) |
| Oldpeak | ST depression (mm) |
| ST_Slope | Up / Flat / Down |

### Output
1. **Risk Probability** — exact percentage from the model
2. **Risk Tier** — Low / Moderate / High / Very High
3. **Feature Contribution Analysis** — top 6 features ranked by influence
4. **Per-feature Clinical Reasoning** — value-aware medical explanation
5. **Clinical Recommendation** — tier-specific next steps

### Risk Tiers

| Tier | Probability | Recommendation |
|---|---|---|
| Low | 0–30% | Routine follow-up. Rescreen in 3–5 years |
| Moderate | 30–60% | Consider stress testing. Follow-up in 6–12 months |
| High | 60–80% | Cardiology referral. Consider CT angiography |
| Very High | 80–100% | Urgent cardiology referral |

---

## ⚙️ Technical Details

### Pipeline
```
Raw Input → StandardScaler → Logistic Regression → Probability
```

### Key Technical Decision — Feature Attribution Fix

The naive approach multiplies model weights by raw input values:
```python
# WRONG — Age=52 dominates FastingBS=1 purely due to magnitude
contributions[col] = abs(weights[i] * raw_value)
```

The correct approach uses scaled values — what the model actually received:
```python
# CORRECT — all features on same z-score scale
transformed = input_df.copy()
for step_name, step in pipeline.steps[:-1]:
    transformed = step.transform(transformed)

contributions[col] = abs(weights[i] * transformed[0][i])
```

### Why Logistic Regression over XGBoost

Both models were trained and compared:

| Metric | LR | XGBoost |
|---|---|---|
| Recall | **0.931** | 0.892 |
| AUC-ROC | 0.900 | **0.929** |
| F1 | **0.900** | 0.892 |

XGBoost achieved higher AUC-ROC but lower Recall. Since missing a sick patient is the primary risk in clinical screening, **LR was chosen** based on superior Recall.

### Encoding — Matches App Exactly
```python
df['Sex']            = df['Sex'].map({"M": 1, "F": 0})
df['ExerciseAngina'] = df['ExerciseAngina'].map({"Y": 1, "N": 0})
df['ChestPainType']  = df['ChestPainType'].map({"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3})
df['RestingECG']     = df['RestingECG'].map({"Normal": 0, "ST": 1, "LVH": 2})
df['ST_Slope']       = df['ST_Slope'].map({"Up": 2, "Flat": 1, "Down": 0})
```

> ⚠️ LabelEncoder was intentionally avoided — it assigns labels alphabetically which creates a mismatch between training encoding and app encoding.

---

## 🔍 Error Analysis

Out of 184 test patients, 21 were misclassified:
- **7 False Negatives** — missed sick patients
- **14 False Positives** — healthy patients flagged as sick

**Pattern identified in false negatives:**
```
ExerciseAngina = 0 in ALL 7 missed cases
ST_Slope = Upsloping in 5 out of 7
```
The model struggles with **silent ischemia** — patients who have heart disease but show no exercise angina and normal ST slope. This mirrors a known clinical blind spot in diabetic and elderly populations.

**Subgroup fairness audit:**
| Group | Accuracy |
|---|---|
| Male (n=146) | 88.4% |
| Female (n=38) | 89.5% |
| Age < 45 | 89.1% |
| Age 45–55 | 84.5% ⚠️ |
| Age 55–65 | 90.6% |
| Age 65+ | 100% |

> 45–55 age group is the weakest — middle-aged patients are most at risk of being undertriaged.

---

## 📁 Project Structure

```
├── heart_disease_app.py          # Gradio application
├── Heart_Disease_Prediction.ipynb # Training notebook
├── heart_disease_pipeline.joblib  # Saved pipeline
├── heart.csv                      # Dataset
└── README.md
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/cardiovascular-risk-assessment

# Install dependencies
pip install gradio scikit-learn pandas numpy joblib

# Run the app
python heart_disease_app.py
```

---

## 📦 Dataset

[UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- 918 patient records
- 11 features
- Binary target: HeartDisease (0/1)
- 55.3% positive class, 44.7% negative class

---

## ⚠️ Disclaimer

This tool is for **clinical decision support only** and does not replace physician judgement, complete history, physical examination, or validated diagnostic pathways.

---

## 🛠️ Future Improvements

- [ ] XGBoost with SHAP interaction values for non-linear pattern detection
- [ ] Subgroup-specific models for 45–55 age group
- [ ] Confidence intervals on probability estimates
- [ ] Silent ischemia detection module

---

