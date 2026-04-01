import os
import pandas as pd
import joblib

# Build absolute path to models folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load all artifacts once at startup
model            = joblib.load(os.path.join(BASE_DIR, "models", "stroke_model.pkl"))
feature_columns  = joblib.load(os.path.join(BASE_DIR, "models", "feature_columns.pkl"))
threshold        = joblib.load(os.path.join(BASE_DIR, "models", "threshold.pkl"))
bmi_median       = joblib.load(os.path.join(BASE_DIR, "models", "bmi_median.pkl"))


def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Impute missing BMI with training median
    df["bmi"] = df["bmi"].fillna(bmi_median)

    # Feature engineering — must match notebook exactly
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 60, 100], labels=[0, 1, 2]
    ).astype(int)
    df["age_hypertension"] = df["age"] * df["hypertension"]
    df["glucose_bmi"]      = df["avg_glucose_level"] * df["bmi"]

    # Encode
    df = pd.get_dummies(df, drop_first=True)

    # Align to training feature columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def predict(data: dict) -> dict:
    df = preprocess(data)
    prob       = model.predict_proba(df)[0][1]
    prediction = int(prob >= threshold)

    return {
        "stroke_probability": round(float(prob), 4),
        "prediction": prediction,
        "risk": "High" if prediction == 1 else "Low"
    }