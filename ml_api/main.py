from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
from datetime import datetime
import numpy as np
import os

from database import DatasetManager
from ml_models import MLModelManager

app = FastAPI(title="ML API - Jour 2", version="2.0.0")

# Instances globales
dataset_manager = DatasetManager()
model_manager = MLModelManager()
generation_counter = 0

# Modèles Pydantic
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

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Route de santé"""
    return HealthResponse(
        status="OK",
        timestamp=datetime.now().isoformat()
    )

@app.post("/generate", response_model=GenerateResponse)
def generate_dataset(n_samples: int = 1000):
    """Génère un nouveau dataset"""
    global generation_counter
    generation_counter += 1
    
    # Générer les données
    X, y = model_manager.generate_dataset(n_samples)
    
    # Sauvegarder en BDD
    dataset_manager.save_dataset(X, y, generation_counter)
    
    return GenerateResponse(
        message="Dataset généré avec succès",
        generation_number=generation_counter,
        samples_generated=n_samples
    )

@app.get("/predict", response_model=PredictionResponse)
def predict():
    """Prédiction sur le dernier dataset"""
    # Récupérer le dernier dataset
    X, y = dataset_manager.get_latest_dataset()
    
    if X is None:
        raise HTTPException(status_code=404, detail="Aucun dataset trouvé")
    
    # Prédiction sur le premier échantillon
    try:
        prediction = model_manager.predict(X.iloc[:1].values)[0]
        
        # Calcul de "confiance" (probabilité)
        if hasattr(model_manager.model, 'predict_proba'):
            proba = model_manager.model.predict_proba(X.iloc[:1].values)[0]
            confidence = float(max(proba))
        else:
            confidence = 0.5
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence,
            model_used=model_manager.model_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@app.post("/retrain-debug")
def retrain_model_debug():
    """Version debug sans MLflow du tout"""
    try:
        # Récupérer le dernier dataset
        X, y = dataset_manager.get_latest_dataset()
        
        if X is None:
            return {"error": "Aucun dataset trouvé"}
        
        # Entraînement simple
        metrics = model_manager.train_model(X.values, y.values)
        
        return {
            "message": "Modèle réentraîné avec succès (debug)",
            "train_accuracy": float(metrics["train_accuracy"]),
            "test_accuracy": float(metrics["test_accuracy"]),
            "dataset_shape": f"{X.shape[0]}x{X.shape[1]}"
        }
    except Exception as e:
        return {"error": str(e), "error_type": str(type(e).__name__)}

@app.post("/retrain", response_model=RetrainResponse)
def retrain_model():
    """Réentraîne le modèle avec MLflow (version robuste)"""
    X, y = dataset_manager.get_latest_dataset()
    
    if X is None:
        raise HTTPException(status_code=404, detail="Aucun dataset pour l'entraînement")
    
    # Configuration MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5555")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        experiment_name = "ml-api-retraining"
        
        # Créer ou récupérer l'expériment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            # Entraînement
            metrics = model_manager.train_model(X.values, y.values)
            
            # Logging MLflow (paramètres et métriques)
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("algorithm", "LogisticRegression")
            
            mlflow.log_metric("train_accuracy", float(metrics["train_accuracy"]))
            mlflow.log_metric("test_accuracy", float(metrics["test_accuracy"]))
            
            # Log du modèle (version sécurisée avec try/catch)
            try:
                mlflow.sklearn.log_model(
                    model_manager.model, 
                    "model",
                    registered_model_name="logistic_regression_model"
                )
                model_logged = True
            except Exception as e:
                print(f"Erreur log model: {e}")
                model_logged = False
            
            return RetrainResponse(
                message="Modèle réentraîné avec succès",
                train_accuracy=float(metrics["train_accuracy"]),
                test_accuracy=float(metrics["test_accuracy"]),
                mlflow_run_id=run.info.run_id
            )
            
    except Exception as e:
        # Fallback sans MLflow si problème de connexion
        print(f"Erreur MLflow: {e}, fallback sans tracking")
        metrics = model_manager.train_model(X.values, y.values)
        
        return RetrainResponse(
            message="Modèle réentraîné (sans MLflow)",
            train_accuracy=float(metrics["train_accuracy"]),
            test_accuracy=float(metrics["test_accuracy"]),
            mlflow_run_id="mlflow_error"
        )

@app.get("/mlflow-status")
def check_mlflow_status():
    """Vérifier le statut de MLflow"""
    try:
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5555")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Test de connexion
        experiments = mlflow.search_experiments()
        
        return {
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "status": "connected",
            "experiments_count": len(experiments)
        }
    except Exception as e:
        return {
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)