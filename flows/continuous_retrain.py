import os
import httpx
from prefect import flow, task, get_run_logger

API_ML = os.getenv("ML_API_URL", "http://ml-api:8001")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))

@task(retries=2, retry_delay_seconds=60)
def check_drift() -> float:
    logger = get_run_logger()
    r = httpx.get(f"{API_ML}/drift")
    r.raise_for_status()
    score = r.json().get("drift_score", 0.0)
    logger.info(f"Drift score actuel : {score:.3f}")
    return score

@task
def trigger_retrain() -> dict:
    logger = get_run_logger()
    
    # Header avec le token Bearer si nécessaire
    headers = {"Authorization": f"Bearer {os.getenv('API_TOKEN', '')}"}

    r = httpx.post(f"{API_ML}/generate", params={"n_samples": 1500}, headers=headers)
    r.raise_for_status()
    logger.info("✅ Dataset régénéré")

    r2 = httpx.post(f"{API_ML}/retrain", headers=headers)
    r2.raise_for_status()
    logger.info("✅ Retrain déclenché")

    return r2.json()

@task
def notify_discord(message: str):
    if DISCORD_WEBHOOK_URL:
        httpx.post(DISCORD_WEBHOOK_URL, json={"content": message})

@flow(name="continuous-retrain", retries=1, retry_delay_seconds=300)
def continuous_retrain_flow():
    logger = get_run_logger()
    drift = check_drift()

    if drift > DRIFT_THRESHOLD:
        result = trigger_retrain()

        run_id = result.get("mlflow_run_id", "-")
        test_acc = result.get("test_accuracy", 0.0)
        mlflow_base_url = os.getenv("MLFLOW_BASE_URL", "http://mlflow-server:5555")

        msg = (
            f"🧠 **Drift détecté** (score = `{drift:.2f}`)\n"
            f"🔁 **Modèle réentraîné avec succès**\n"
            f"📊 Accuracy : test = `{test_acc:.2%}`\n"
            f"📂 Run MLflow : `{run_id}`\n"
            f"🔗 {mlflow_base_url}/#/experiments/1/runs/{run_id}"
        )
    else:
        msg = f"✅ Pas de drift détecté (score = `{drift:.2f}`), aucun réentraînement nécessaire."

    notify_discord(msg)
    logger.info("Fin du flow continuous-retrain")

if __name__ == "__main__":
    continuous_retrain_flow()