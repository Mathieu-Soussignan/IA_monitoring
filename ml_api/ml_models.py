import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
from datetime import datetime

class MLModelManager:
    def __init__(self, model_path="models/current_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def generate_dataset(self, n_samples=1000):
        """Génère un dataset avec dérive temporelle"""
        # Feature 1: normale
        X1 = np.random.normal(0, 1, n_samples)
        
        # Feature 2: change de signe selon l'heure
        current_hour = datetime.now().hour
        sign_modifier = -0.5 if current_hour % 2 == 0 else 0.5
        X2 = np.random.normal(sign_modifier, 1, n_samples)
        
        # Combinaison linéaire pour la target
        X = np.column_stack([X1, X2])
        linear_combination = X1 + X2 * 0.5
        
        # Probabilité et classe
        probabilities = 1 / (1 + np.exp(-linear_combination))
        y = (probabilities > 0.5).astype(int)
        
        return X, y
    
    def train_model(self, X, y):
        """Entraîne une régression logistique"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train, y_train)
        
        # Métriques
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        
        # Sauvegarde
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "model_path": self.model_path
        }
    
    def load_model(self):
        """Charge le modèle existant"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = None
    
    def predict(self, X):
        """Fait une prédiction"""
        if self.model is None:
            raise ValueError("Aucun modèle entraîné")
        return self.model.predict(X)