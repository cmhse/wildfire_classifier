FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install flask joblib numpy pandas xgboost scikit-learn
EXPOSE 5000
CMD ["python", "app.py"]