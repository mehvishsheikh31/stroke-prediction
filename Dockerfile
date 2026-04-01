FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pandas numpy scikit-learn xgboost==3.2.0 imbalanced-learn joblib shap matplotlib seaborn optuna requests pydantic

COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]