# 🧠 Stroke Prediction ML API

> An end-to-end production-grade machine learning system for stroke risk prediction — built with XGBoost, FastAPI, and Streamlit.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://strokepredictionsmodel.streamlit.app/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 🌐 Live Demo

**👉 [strokepredictionsmodel.streamlit.app](https://strokepredictionsmodel.streamlit.app/)**

---

## 📌 Problem Statement

Stroke is one of the leading causes of death and disability worldwide. This dataset has a severe **class imbalance — only ~5% positive (stroke) cases**. A naive model that predicts "No Stroke" for everyone achieves 95% accuracy but is clinically useless.

**The goal:** Maximize **recall** on stroke class — because missing a real stroke (false negative) is far more dangerous than a false alarm (false positive).

---

## 🏗️ Project Structure

```
stroke_project/
├── app/
│   ├── main.py               ← FastAPI endpoints
│   ├── schemas.py            ← Pydantic input validation
│   ├── preprocess.py         ← Preprocessing + prediction logic
│   └── streamlit_app.py      ← Streamlit dashboard
├── models/
│   ├── stroke_model.pkl      ← Trained XGBoost model
│   ├── feature_columns.pkl   ← Training column order
│   ├── threshold.pkl         ← Custom decision threshold (0.61)
│   └── bmi_median.pkl        ← BMI imputation value (train only)
├── notebooks/
│   └── eda.ipynb             ← Full ML pipeline notebook
├── data/raw/
│   └── healthcare-dataset-stroke-data.csv
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ ML Pipeline

### 1. Data Preprocessing
- **Split first, transform after** — prevents data leakage
- BMI imputed using **training median only** (saved as artifact)
- Missing rate: 201 rows (~4%) with missing BMI

### 2. Feature Engineering
| Feature | Description |
|---------|-------------|
| `age_group` | Age bucketed into 0–30, 30–60, 60+ |
| `age_hypertension` | Age × Hypertension interaction |
| `glucose_bmi` | Glucose × BMI interaction |

### 3. Class Imbalance — SMOTE
- Applied **only on training data** — never on test
- Before: ~4,700 no-stroke vs ~250 stroke
- After: balanced 50/50 distribution

### 4. Model Comparison

| Model | Recall | Precision | F1 | ROC-AUC |
|-------|--------|-----------|-----|---------|
| Logistic Regression | 0.50 | 0.19 | 0.28 | 0.78 |
| Random Forest | 0.26 | 0.11 | 0.15 | 0.76 |
| **XGBoost (Tuned)** | **0.80** | **0.13** | **0.22** | **0.80** |

### 5. Hyperparameter Tuning — Optuna
- 50 trials of Bayesian optimization
- Objective: maximize recall on stroke class
- Best params saved and used for final model

### 6. Custom Threshold
- Default threshold: 0.5
- Tuned threshold: **0.61**
- Rationale: reduce false negatives (missed strokes) at the cost of more false positives — medically justified

---

## 🔌 API — FastAPI

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Stroke risk prediction |

### Sample Request
```json
{
  "gender": "Male",
  "age": 67,
  "hypertension": 1,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

### Sample Response
```json
{
  "stroke_probability": 0.9697,
  "prediction": 1,
  "risk": "High"
}
```

### Input Validation
- `Literal` types reject invalid categories instantly (422 error)
- `Field` constraints reject out-of-range numbers
- BMI is optional — imputed automatically if missing

---

## 📊 Streamlit Dashboard

4-tab interactive dashboard:

| Tab | Content |
|-----|---------|
| 🔮 Predict | Patient form + risk gauge + result cards |
| 📊 EDA | Distribution plots, stroke rate by category, correlation heatmap |
| 🏆 Model Comparison | ROC curves, recall bar chart, metric table |
| 🔍 SHAP | Beeswarm + bar importance plots |

Features:
- Dark / Light theme toggle
- Custom CSS injection
- Calls FastAPI `/predict` endpoint

---

## 🐳 Docker

```bash
# Build
docker build -t stroke-api .

# Run
docker run -p 8000:8000 stroke-api
```

API available at: `http://localhost:8000/docs`

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/mehvishsheikh31/stroke-prediction.git
cd stroke-prediction

# Install dependencies
pip install -r requirements.txt

# Start FastAPI (Terminal 1)
cd app
python -m uvicorn main:app --reload

# Start Streamlit (Terminal 2)
streamlit run app/streamlit_app.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | XGBoost + Optuna |
| Imbalance | SMOTE (imblearn) |
| API | FastAPI + Pydantic |
| Dashboard | Streamlit |
| Explainability | SHAP |
| Containerization | Docker |
| Visualization | Matplotlib + Seaborn |

---

## 📁 Key Design Decisions

- **No SMOTE at inference** — synthetic data is training-only
- **Artifacts loaded once at startup** — not on every request
- **4 separate artifacts** — model, feature columns, threshold, BMI median
- **Pydantic Literal validation** — blocks invalid inputs before preprocessing
- **Low precision accepted** — in medical screening, catching every stroke matters more than avoiding false alarms

---

## 👩‍💻 Author

**Mehvish Sheikh**
- GitHub: [@mehvishsheikh31](https://github.com/mehvishsheikh31)
- LinkedIn: [mehvishsheikh31](https://linkedin.com/in/mehvishsheikh31)
- B.Tech CSE (Data Science) — IIST Indore, 2027
