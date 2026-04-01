
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pandas numpy scikit-learn xgboost==3.2.0 imbalanced-learn joblib shap matplotlib seaborn optuna requests pydantic

COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]