# 🧠 ML Drift Detection & Continuous Retraining API

Ce projet expose une API FastAPI pour entraîner, monitorer et réentraîner automatiquement un modèle de machine learning en cas de dérive de données. Il intègre un système complet de monitoring avec **MLflow**, **Prometheus**, **Discord Bot**, et un **flow Prefect** qui tourne en continu.

---

## Résumé des 5 jours de développement

### ✅ Jour 1 : Setup du projet

* Création de l’API `FastAPI` avec route `/predict`, `/train`, `/retrain`.
* Création du modèle de base `LogisticRegression` sur données synthétiques.
* Dockerisation du projet avec `docker-compose`.
* Premiers tests manuels des routes via Swagger UI.

### ✅ Jour 2 : Détection de dérive

* Ajout d'une fonction `compute_drift()` basée sur la chute d'accuracy sur un nouveau dataset généré dynamiquement.
* Ajout d’une route `/drift-check`.
* Seuil configurable (`DRIFT_THRESHOLD`) dans `.env`.
* Setup de Prometheus et Grafana pour exporter les métriques du modèle.

### ✅ Jour 3 : Intégration de MLflow

* Tracking des modèles et métriques dans `MLflow`.
* Gestion de fallback si le log du modèle échoue.
* Ajout de `mlflow.sklearn.log_model` (log automatique du modèle).
* Ajout d’un volume `./mlruns` pour persister les runs.

### ✅ Jour 4 : Automation & Notifications

* Flow Prefect `continuous_retrain_flow()` qui :

  * Vérifie la dérive toutes les 2 minutes.
  * Déclenche `/retrain` automatiquement si besoin.
  * Envoie un message Discord avec :

    * score de drift
    * accuracy après retrain
    * lien vers le run MLflow
* Setup du scheduler Prefect via `prefect.yaml` + cron `*/2 * * * *`.

---

## 🔧 Démarrage rapide

### 1. Cloner le repo

```bash
git clone https://github.com/ton-orga/ml-api.git
cd ml-api
```

### 2. Fichier `.env`

Créer un fichier `.env` à la racine avec :

```
DRIFT_THRESHOLD=0.25
DISCORD_WEBHOOK_URL=...
MLFLOW_TRACKING_URI=http://mlflow:5555
```

### 3. Lancer le projet (API + MLflow + Prometheus + Grafana)

```bash
docker-compose up --build
```

Accès :

* API : [http://localhost:8000/docs](http://localhost:8000/docs)
* MLflow : [http://localhost:5555](http://localhost:5555)
* Grafana : [http://localhost:3000](http://localhost:3000)
* Prefect UI (optionnel) : [http://localhost:4200](http://localhost:4200)

---

## ⚙️ Commandes utiles

### ▶️ Lancer le flow Prefect (auto-retrain)

```bash
prefect deployment run continuous-retrain/continuous-retrain
```

### ⚡ Lancer manuellement un retrain

```bash
curl -X POST http://localhost:8000/retrain \
  -H "Authorization: Bearer <ton_token>" \
  -H "Content-Type: application/json"
```

### 🔍 Vérifier le drift manuellement

```bash
curl http://localhost:8000/drift-check
```

---

## 📁 Arborescence (simplifiée)

```
.
├── app/
│   ├── main.py              # API FastAPI
│   ├── ml_models.py         # Modèle ML + drift
│   ├── routes.py            # Routes /predict /train /retrain
│   ├── tasks/               # Tâches Prefect
│   └── utils/               # DatasetManager, Discord, etc.
├── flows/
│   └── continuous_retrain.py
├── Dockerfile
├── docker-compose.yml
├── mlruns/                  # Logs MLflow
├── .env
└── README.md
```

---

## 📌 Notes

* Le drift est simulé par un changement de distribution horaire sur les features.
* Les logs Discord sont envoyés avec emoji, score, accuracy et lien MLflow.
* Le modèle est sauvegardé localement dans `models/current_model.pkl`.