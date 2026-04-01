from fastapi import FastAPI
from app.schemas import StrokeInput
from app.preprocess import predict

app = FastAPI(title="Stroke Prediction API")


@app.get("/")
def health_check():
    return {"status": "ok", "model": "XGBoost", "version": "1.0"}


@app.post("/predict")
def predict_stroke(input: StrokeInput):
    result = predict(input.model_dump())
    return result