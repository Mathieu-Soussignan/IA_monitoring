from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from pydantic import BaseModel
from loguru import logger
import mlflow
import mlflow.sklearn
from datetime import datetime, timezone
import numpy as np
import sys
import os

from database import DatasetManager
from ml_models import MLModelManager
from prometheus_fastapi_instrumentator import Instrumentator

# Logging configuration
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(
    f"{LOG_DIR}/ml_api.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time} | {level} | {message}",
)

# FastAPI app & Prometheus
app = FastAPI(title="ML API - Jour 2", version="2.0.0")

# instrument all routes, expose /metrics
Instrumentator().instrument(app).expose(app)

# Prometheus gauges
TRAIN_ACCURACY = Gauge("model_train_accuracy", "Training accuracy of the model")
TEST_ACCURACY  = Gauge("model_test_accuracy",  "Test accuracy of the model")
DATASET_SIZE  = Gauge("dataset_size",         "Number of samples in training dataset")

# Security: Bearer token on /retrain
bearer_scheme = HTTPBearer()

def token_auth(
    cred: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    token = cred.credentials
    expected = os.getenv("API_TOKEN", "")
    if not expected or token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing bearer token"
        )

# Global singletons
dataset_manager = DatasetManager()
model_manager   = MLModelManager()
generation_counter = 0

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_used: str

class GenerateResponse(BaseModel):
    message: str
    generation_number: int
    samples_generated: int

class RetrainResponse(BaseModel):
    message: str
    train_accuracy: float
    test_accuracy: float
    mlflow_run_id: str

# Healthcheck
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Route de santé (UTC, timezone-aware)"""
    now_utc = datetime.now(timezone.utc)
    return HealthResponse(
        status="OK",
        timestamp=now_utc.isoformat()
    )

# Generate dataset
@app.post("/generate", response_model=GenerateResponse)
def generate_dataset(n_samples: int = 1000):
    global generation_counter
    generation_counter += 1

    X, y = model_manager.generate_dataset(n_samples)
    dataset_manager.save_dataset(X, y, generation_counter)

    return GenerateResponse(
        message="Dataset généré avec succès",
        generation_number=generation_counter,
        samples_generated=n_samples
    )

# Predict
@app.get("/predict", response_model=PredictionResponse)
def predict():
    X, y = dataset_manager.get_latest_dataset()
    if X is None:
        raise HTTPException(status_code=404, detail="Aucun dataset trouvé")

    try:
        pred = model_manager.predict(X.iloc[:1].values)[0]
        if hasattr(model_manager.model, "predict_proba"):
            proba = model_manager.model.predict_proba(X.iloc[:1].values)[0]
            confidence = float(max(proba))
        else:
            confidence = 0.5

        return PredictionResponse(
            prediction=int(pred),
            confidence=confidence,
            model_used=model_manager.model_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {e}")

# Retrain
@app.post(
    "/retrain",
    response_model=RetrainResponse,
    dependencies=[Depends(token_auth)]
)
def retrain_model():
    logger.info("🔁 Starting retraining process...")

    X, y = dataset_manager.get_latest_dataset()
    if X is None:
        logger.warning("⚠️ Aucun dataset trouvé pour l'entraînement.")
        raise HTTPException(status_code=404, detail="Aucun dataset pour l'entraînement")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5555")
    mlflow.set_tracking_uri(mlflow_uri)

    try:
        exp_name = "ml-api-retraining"

        try:
            mlflow.create_experiment(exp_name)
            logger.info(f"🧪 Expérience créée : {exp_name}")
        except Exception:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is not None:
                logger.info(f"📁 Expérience existante utilisée : {exp_name}")
            else:
                logger.error(f"❌ Expérience introuvable : {exp_name}")
                raise RuntimeError("Impossible de récupérer ou créer l'expérience MLflow")

        mlflow.set_experiment(exp_name)

        with mlflow.start_run() as run:
            logger.info("🛠️ Début du run MLflow")
            metrics = model_manager.train_model(X.values, y.values)

            # Prometheus
            TRAIN_ACCURACY.set(metrics["train_accuracy"])
            TEST_ACCURACY.set(metrics["test_accuracy"])
            DATASET_SIZE.set(len(X))

            # Log params & metrics
            mlflow.log_params({
                "n_samples": len(X),
                "n_features": X.shape[1],
                "algorithm": "LogisticRegression"
            })
            mlflow.log_metrics({
                "train_accuracy": metrics["train_accuracy"],
                "test_accuracy": metrics["test_accuracy"]
            })

            try:
                mlflow.sklearn.log_model(model_manager.model, "model")
                logger.success("✅ Modèle loggué dans MLflow")
            except Exception:
                logger.warning("⚠️ Modèle entraîné mais non loggué dans MLflow (modèle visible uniquement localement)")

            return RetrainResponse(
                message="Modèle réentraîné avec succès",
                train_accuracy=metrics["train_accuracy"],
                test_accuracy=metrics["test_accuracy"],
                mlflow_run_id=run.info.run_id
            )

    except Exception:
        logger.exception("❌ Erreur lors du tracking MLflow, fallback sans logging")

        metrics = model_manager.train_model(X.values, y.values)
        TRAIN_ACCURACY.set(metrics["train_accuracy"])
        TEST_ACCURACY.set(metrics["test_accuracy"])
        DATASET_SIZE.set(len(X))

        return RetrainResponse(
            message="Modèle réentraîné (sans MLflow)",
            train_accuracy=metrics["train_accuracy"],
            test_accuracy=metrics["test_accuracy"],
            mlflow_run_id="mlflow_error"
        )

@app.get("/drift")
def drift_check():
    score = model_manager.compute_drift()
    return {"drift_score": score}

# MLflow status
@app.get("/mlflow-status")
def check_mlflow_status():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5555")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        exps = mlflow.search_experiments()
        return {"mlflow_uri": mlflow_uri, "status": "connected", "experiments_count": len(exps)}
    except Exception as e:
        return {"mlflow_uri": mlflow_uri, "status": "error", "error": str(e)}

@app.get("/metrics")
def metrics():
    logger.debug("📊 Serving /metrics")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)