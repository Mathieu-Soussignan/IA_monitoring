import pytest
from fastapi.testclient import TestClient
import json
import os
import tempfile
from ml_api.main import app
from ml_api.database import DatasetManager
from ml_api.ml_models import MLModelManager

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def temp_db():
    """Base de données temporaire pour les tests"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        yield tmp.name
    os.unlink(tmp.name)

def test_health_endpoint(client):
    """Test endpoint health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "timestamp" in data

def test_generate_dataset(client):
    """Test génération de dataset"""
    response = client.post("/generate?n_samples=100")
    assert response.status_code == 200
    data = response.json()
    assert data["samples_generated"] == 100
    assert "generation_number" in data

def test_predict_without_dataset(client):
    """Test prédiction sans dataset"""
    response = client.get("/predict")
    # Peut être 404 si aucun dataset ou 500 si aucun modèle
    assert response.status_code in [404, 500]

def test_full_workflow(client):
    """Test du workflow complet"""
    # 1. Générer un dataset
    generate_response = client.post("/generate?n_samples=200")
    assert generate_response.status_code == 200
    
    # 2. Entraîner le modèle
    retrain_response = client.post("/retrain")
    assert retrain_response.status_code == 200
    retrain_data = retrain_response.json()
    assert "train_accuracy" in retrain_data
    assert "test_accuracy" in retrain_data
    
    # 3. Faire une prédiction
    predict_response = client.get("/predict")
    assert predict_response.status_code == 200
    predict_data = predict_response.json()
    assert predict_data["prediction"] in [0, 1]
    assert 0 <= predict_data["confidence"] <= 1

def test_database_integration(temp_db):
    """Test intégration base de données"""
    db_manager = DatasetManager(temp_db)
    
    # Générer et sauvegarder
    import numpy as np
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    
    db_manager.save_dataset(X, y, 1)
    
    # Récupérer
    X_retrieved, y_retrieved = db_manager.get_latest_dataset()
    
    assert X_retrieved is not None
    assert y_retrieved is not None
    assert len(X_retrieved) == 10
    assert len(y_retrieved) == 10

def test_model_training():
    """Test entraînement du modèle"""
    model_manager = MLModelManager()
    
    # Générer des données
    X, y = model_manager.generate_dataset(100)
    
    # Entraîner
    metrics = model_manager.train_model(X, y)
    
    assert "train_accuracy" in metrics
    assert "test_accuracy" in metrics
    assert 0 <= metrics["train_accuracy"] <= 1
    assert 0 <= metrics["test_accuracy"] <= 1