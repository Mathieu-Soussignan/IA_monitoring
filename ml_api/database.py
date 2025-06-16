import sqlite3
import pandas as pd
from datetime import datetime
import json
import os

class DatasetManager:
    def __init__(self, db_path="/app/data/ml_data.db"):  # Chemin absolu
        self.db_path = db_path
        # Créer le dossier parent s'il n'existe pas
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialise la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_number INTEGER,
                features TEXT,
                target TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_dataset(self, X, y, generation_number):
        """Sauvegarde un dataset en BDD"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO datasets (generation_number, features, target)
            VALUES (?, ?, ?)
        ''', (generation_number, json.dumps(X.tolist()), json.dumps(y.tolist())))
        conn.commit()
        conn.close()
    
    def get_latest_dataset(self):
        """Récupère le dernier dataset"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT features, target FROM datasets 
            ORDER BY generation_number DESC LIMIT 1
        ''')
        result = cursor.fetchone()
        conn.close()
        
        if result:
            X = pd.DataFrame(json.loads(result[0]))
            y = pd.Series(json.loads(result[1]))
            return X, y
        return None, None