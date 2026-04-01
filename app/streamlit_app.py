import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Prediction Dashboard",
    page_icon="🧠",
    layout="wide"
)
# Default colors defined before anything renders
accent  = "#e74c3c"
accent2 = "#2e86ab"
card_bg = "#1a1d27"
border  = "#2a2d3a"
text    = "#e0e0e0"
subtext = "#9e9e9e"
bg      = "#0f1117"
input_bg = "#1e2130"

# ─── THEME TOGGLE ───────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

col_title, col_toggle = st.columns([8, 1])
with col_title:
    st.markdown("## 🧠 Stroke Prediction Dashboard")
with col_toggle:
    if st.button("🌙" if st.session_state.dark_mode else "☀️"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

dark = st.session_state.dark_mode

# ─── CSS INJECTION ──────────────────────────────────────────────────────────
if dark:
    bg        = "#0f1117"
    card_bg   = "#1a1d27"
    text      = "#e0e0e0"
    subtext   = "#9e9e9e"
    accent    = "#e74c3c"
    accent2   = "#2e86ab"
    border    = "#2a2d3a"
    input_bg  = "#1e2130"
else:
    bg        = "#f5f7fa"
    card_bg   = "#ffffff"
    text      = "#1a1a2e"
    subtext   = "#555555"
    accent    = "#e74c3c"
    accent2   = "#2e86ab"
    border    = "#e0e0e0"
    input_bg  = "#f0f2f6"

st.markdown(f"""
<style>
    /* Main background */
    .stApp {{ background-color: {bg}; color: {text}; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {card_bg};
        border-right: 1px solid {border};
    }}

    /* Cards */
    .card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }}

    /* Metric cards */
    .metric-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {accent};
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: {subtext};
        margin-top: 4px;
    }}

    /* Risk badge */
    .risk-high {{
        background: #e74c3c22;
        border: 1px solid #e74c3c;
        color: #e74c3c;
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }}
    .risk-low {{
        background: #2ecc7122;
        border: 1px solid #2ecc71;
        color: #2ecc71;
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        background: {card_bg};
        border-radius: 10px;
        padding: 6px 8px;
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: {subtext};
        font-weight: 500;
        padding: 8px 20px;
        margin: 0 2px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {accent} !important;
        color: white !important;
    }}

    /* Input labels */
    .stSelectbox label, .stNumberInput label {{
        color: {text} !important;
    }}

    /* Input backgrounds and text */
    .stSelectbox > div > div {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}
    input[type="number"] {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}

    /* Dropdown options */
    [data-baseweb="select"] * {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}
    [data-baseweb="popover"] * {{
        background-color: {card_bg} !important;
        color: {text} !important;
    }}

    /* Button */
    .stButton > button {{
        background: {accent};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }}
    .stButton > button:hover {{
        opacity: 0.85;
        color: white;
    }}

    /* Section headers */
    .section-header {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {accent2};
        border-left: 3px solid {accent2};
        padding-left: 10px;
        margin: 16px 0 12px 0;
    }}

    /* Hide streamlit branding */
    #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA & ARTIFACTS ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, "data", "raw", "healthcare-dataset-stroke-data.csv"))

@st.cache_resource
def load_artifacts():
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base, "models", "stroke_model.pkl"))
    cols  = joblib.load(os.path.join(base, "models", "feature_columns.pkl"))
    thr   = joblib.load(os.path.join(base, "models", "threshold.pkl"))
    bmi_m = joblib.load(os.path.join(base, "models", "bmi_median.pkl"))
    return model, cols, thr, bmi_m

df               = load_data()
model, feat_cols, threshold, bmi_median = load_artifacts()

COLORS = [accent2, accent]
plt.rcParams.update({
    "figure.facecolor": card_bg,
    "axes.facecolor":   card_bg,
    "axes.edgecolor":   border,
    "axes.labelcolor":  text,
    "xtick.color":      subtext,
    "ytick.color":      subtext,
    "text.color":       text,
    "grid.color":       border,
})

# ─── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict", "📊 EDA", "🏆 Model Comparison", "🔍 SHAP Explainability"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        gender     = st.selectbox("Gender", ["Male", "Female", "Other"])
        age        = st.number_input("Age", min_value=1, max_value=119, value=45)
        hypert     = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
        heart      = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col2:
        married    = st.selectbox("Ever Married", ["Yes", "No"])
        work       = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence  = st.selectbox("Residence Type", ["Urban", "Rural"])

    with col3:
        glucose    = st.number_input("Avg Glucose Level", min_value=1.0, max_value=499.0, value=100.0)
        bmi_input  = st.number_input("BMI (0 = unknown)", min_value=0.0, max_value=99.0, value=25.0)
        smoking    = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.markdown("")
    predict_btn = st.button("🔮 Predict Stroke Risk")

    if predict_btn:
        payload = {
            "gender":            gender,
            "age":               age,
            "hypertension":      hypert,
            "heart_disease":     heart,
            "ever_married":      married,
            "work_type":         work,
            "Residence_type":    residence,
            "avg_glucose_level": glucose,
            "bmi":               bmi_input if bmi_input > 0 else None,
            "smoking_status":    smoking
        }

        try:
            res  = requests.post("http://127.0.0.1:7860/predict", json=payload, timeout=5)
            data = res.json()

            st.markdown("---")
            st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{data['stroke_probability']*100:.1f}%</div>
                    <div class="metric-label">Stroke Probability</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{'1' if data['prediction'] else '0'}</div>
                    <div class="metric-label">Prediction (1 = Stroke)</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                risk_class = "risk-high" if data["risk"] == "High" else "risk-low"
                st.markdown(f"""
                <div class="{risk_class}">
                    {'⚠️ HIGH RISK' if data['risk'] == 'High' else '✅ LOW RISK'}
                </div>""", unsafe_allow_html=True)

            # Probability gauge
            st.markdown("")
            prob = data["stroke_probability"]
            fig, ax = plt.subplots(figsize=(8, 1.2))
            ax.barh(0, 1, color=border, height=0.4)
            ax.barh(0, prob, color=accent if prob >= threshold else "#2ecc71", height=0.4)
            ax.axvline(threshold, color="white", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Stroke Probability")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("Risk Gauge", fontsize=10)
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"API Error: {e}. Make sure FastAPI is running on port 8000.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in zip(
        [m1, m2, m3, m4],
        ["Total Patients", "Stroke Cases", "Stroke Rate", "Missing BMI"],
        [len(df), df["stroke"].sum(),
         f"{df['stroke'].mean()*100:.1f}%",
         df["bmi"].isnull().sum()]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Row 1 — Class distribution + Age
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sizes   = df["stroke"].value_counts()
        ax.pie(sizes, labels=["No Stroke", "Stroke"], colors=COLORS,
               autopct="%1.1f%%", startangle=90, explode=(0, 0.1))
        ax.set_title("Stroke Distribution")
        st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.kdeplot(data=df, x="age", hue="stroke", fill=True, palette=COLORS, ax=ax)
        ax.set_title("Age Distribution by Stroke")
        st.pyplot(fig); plt.close()

    # Row 2 — Glucose + BMI
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(x="stroke", y="avg_glucose_level", hue="stroke",
                    data=df, palette=COLORS, legend=False, ax=ax)
        ax.set_xticklabels(["No Stroke", "Stroke"])
        ax.set_title("Glucose Level vs Stroke")
        st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(x="stroke", y="bmi", hue="stroke",
                    data=df, palette=COLORS, legend=False, ax=ax)
        ax.set_xticklabels(["No Stroke", "Stroke"])
        ax.set_title("BMI vs Stroke")
        st.pyplot(fig); plt.close()

    # Row 3 — Categorical stroke rates
    st.markdown('<div class="section-header">Stroke Rate by Category</div>', unsafe_allow_html=True)
    cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "smoking_status"]

    c1, c2, c3 = st.columns(3)
    for i, col in enumerate(cat_cols):
        rate = df.groupby(col)["stroke"].mean().reset_index()
        rate["stroke"] = rate["stroke"] * 100
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x=col, y="stroke", data=rate, palette="Reds", ax=ax)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width()/2, p.get_height() + 0.1),
                        ha="center", fontsize=8)
        ax.set_title(f"Stroke % by {col}", fontsize=10)
        ax.set_ylabel("Stroke %")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        target = [c1, c2, c3][i % 3]
        with target:
            st.pyplot(fig)
        plt.close()

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    st.info("These metrics are from notebook evaluation on the held-out test set.")

    results = pd.DataFrame({
    "Model":     ["Logistic Regression", "Random Forest", "XGBoost (Tuned)"],
    "Recall":    [0.50, 0.26, 0.80],
    "Precision": [0.19, 0.11, 0.13],
    "F1 Score":  [0.28, 0.15, 0.22],
    "ROC-AUC":   [0.78, 0.76, 0.80],
})

    # Table
    st.dataframe(results.set_index("Model"), use_container_width=True)

    # Charts
    c1, c2 = st.columns(2)
    metrics = ["Recall", "Precision", "F1 Score", "ROC-AUC"]
    bar_colors = [accent2, "#2ecc71", accent]

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(metrics))
        w = 0.25
        for i, (_, row) in enumerate(results.iterrows()):
            ax.bar(x + i*w, [row[m] for m in metrics], w,
                   label=row["Model"], color=bar_colors[i])
        ax.set_xticks(x + w)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.set_title("Metric Comparison")
        st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(results["Model"], results["Recall"],
                      color=bar_colors)
        for bar, val in zip(bars, results["Recall"]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_title("Recall Comparison (Primary Metric)")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig); plt.close()

    st.markdown("""
       <div class="card">
       <b>Why XGBoost was selected:</b><br><br>
     • Highest recall (0.80) — catches the most actual stroke cases<br>
     • Best ROC-AUC (0.80) — strongest overall discrimination<br>
     • Optimized with Optuna (50 trials) for maximum recall<br>
     • Custom threshold 0.61 applied to further reduce false negatives<br>
     • Trade-off accepted: lower precision (0.13) is acceptable in medical screening
       </div>
     """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">SHAP Feature Explainability</div>', unsafe_allow_html=True)
    st.markdown("SHAP shows which features drive the model's predictions and by how much.")

    if st.button("Generate SHAP Plot", key="shap_btn"):
        with st.spinner("Computing SHAP values..."):
            try:
                # Rebuild X_test equivalent from raw data for SHAP
                raw = df.copy()
                raw["bmi"] = raw["bmi"].fillna(bmi_median)

                from sklearn.model_selection import train_test_split
                X = raw.drop("stroke", axis=1)
                y = raw["stroke"]
                _, X_test, _, _ = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

                X_test["age_group"] = pd.cut(
                    X_test["age"], bins=[0, 30, 60, 100], labels=[0, 1, 2]
                ).astype(int)
                X_test["age_hypertension"] = X_test["age"] * X_test["hypertension"]
                X_test["glucose_bmi"]      = X_test["avg_glucose_level"] * X_test["bmi"]

                X_test = pd.get_dummies(X_test, drop_first=True)
                X_test = X_test.reindex(columns=feat_cols, fill_value=0)
                bool_c = X_test.select_dtypes("bool").columns
                X_test[bool_c] = X_test[bool_c].astype(int)

                explainer   = shap.Explainer(model)
                shap_values = explainer(X_test[:100])

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.beeswarm(shap_values, max_display=15, show=False)
                plt.title("SHAP Beeswarm — Top 15 Features", fontsize=13)
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.close()

                # Bar summary
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                shap.plots.bar(shap_values, max_display=15, show=False)
                plt.title("Mean |SHAP| — Feature Importance", fontsize=13)
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.close()

            except Exception as e:
                st.error(f"SHAP error: {e}")