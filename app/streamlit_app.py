import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os

st.set_page_config(page_title="Stroke Prediction Dashboard", page_icon="🧠", layout="wide")

# ─── SESSION STATE ─────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# ─── HEADER ────────────────────────────
col_title, col_toggle = st.columns([8, 1])
with col_title:
    st.markdown("## 🧠 Stroke Prediction Dashboard")
with col_toggle:
    if st.button("🌙" if st.session_state.dark_mode else "☀️"):
        st.session_state.dark_mode = not st.session_state.dark_mode

# ─── COLORS ────────────────────────────
accent  = "#e74c3c"
accent2 = "#2e86ab"
card_bg = "#1a1d27"
border  = "#2a2d3a"
text    = "#e0e0e0"

# ─── LOAD DATA ─────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/healthcare-dataset-stroke-data.csv")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/stroke_model.pkl")
    cols  = joblib.load("models/feature_columns.pkl")
    thr   = joblib.load("models/threshold.pkl")
    bmi_m = joblib.load("models/bmi_median.pkl")
    return model, cols, thr, bmi_m

df = load_data()
model, feat_cols, threshold, bmi_median = load_artifacts()

# ─── TAB CONTROL (FIXES SHAKING) ───────
tabs = ["🔮 Predict", "📊 EDA", "🏆 Model Comparison", "🔍 SHAP Explainability"]
selected_tab = st.radio("", tabs, horizontal=True)

# ═══════════════════════════════════════
# 🔮 PREDICT
# ═══════════════════════════════════════
if selected_tab == "🔮 Predict":

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 1, 119, 45)
        hypert = st.selectbox("Hypertension", [0, 1])
        heart = st.selectbox("Heart Disease", [0, 1])

    with col2:
        married = st.selectbox("Ever Married", ["Yes", "No"])
        work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])

    with col3:
        glucose = st.number_input("Glucose", 1.0, 500.0, 100.0)
        bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
        smoking = st.selectbox("Smoking", ["never smoked", "formerly smoked", "smokes"])

    if st.button("🔮 Predict Stroke Risk"):
        payload = {
            "gender": gender,
            "age": age,
            "hypertension": hypert,
            "heart_disease": heart,
            "ever_married": married,
            "work_type": work,
            "Residence_type": residence,
            "avg_glucose_level": glucose,
            "bmi": bmi if bmi > 0 else None,
            "smoking_status": smoking
        }

        res = requests.post("http://127.0.0.1:8000/predict", json=payload)
        st.session_state.prediction_data = res.json()

    if st.session_state.prediction_data:
        data = st.session_state.prediction_data
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Stroke Probability", f"{data['stroke_probability']*100:.1f}%")
        with c2:
            st.metric("Prediction", data["prediction"])
        with c3:
            st.metric("Risk", data["risk"])

# ═══════════════════════════════════════
# 📊 EDA (FULL)
# ═══════════════════════════════════════
elif selected_tab == "📊 EDA":

    st.subheader("Dataset Overview")

    fig, ax = plt.subplots()
    sns.histplot(df["age"], kde=True, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x="stroke", y="avg_glucose_level", data=df, ax=ax)
    st.pyplot(fig)

# ═══════════════════════════════════════
# 🏆 MODEL COMPARISON
# ═══════════════════════════════════════
elif selected_tab == "🏆 Model Comparison":

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Recall": [0.50, 0.26, 0.80],
        "Precision": [0.19, 0.11, 0.13],
    })

    st.dataframe(results)

    fig, ax = plt.subplots()
    ax.bar(results["Model"], results["Recall"])
    st.pyplot(fig)

# ═══════════════════════════════════════
# 🔍 SHAP
# ═══════════════════════════════════════
elif selected_tab == "🔍 SHAP Explainability":

    if st.button("Generate SHAP"):
        explainer = shap.Explainer(model)
        shap_values = explainer(df.head(100))

        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(plt.gcf())