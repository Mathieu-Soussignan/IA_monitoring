"""
Flow Prefect de détection de dérive + retrain automatique
--------------------------------------------------------
  le rendra visible pour le serveur/worker Prefect.
"""

from datetime import datetime
import logging
import os
import random
import sys
import time

import requests
from prefect import flow, task, get_run_logger

# --------------------------------------------------------------------------- #
# 1. CONFIGURATION DU LOGGER
# --------------------------------------------------------------------------- #
ROOT_LOGGER = logging.getLogger("ml-monitor")
ROOT_LOGGER.setLevel(logging.INFO)                       # INFO local
handler = logging.StreamHandler(sys.stdout)              # -> stdout (docker logs)
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
ROOT_LOGGER.addHandler(handler)

# --------------------------------------------------------------------------- #
# 2. TÂCHES
# --------------------------------------------------------------------------- #
@task
def generate_random_number() -> float:
    """Génère un nombre aléatoire entre 0 et 1."""
    rng = random.random()
    get_run_logger().info(f"Nombre généré : {rng:.6f}")
    return rng


@task
def check_model_drift(random_number: float, threshold: float = 0.5) -> bool:
    """Retourne True si dérive détectée (nombre < threshold)."""
    drift = random_number < threshold
    get_run_logger().info(f"Dérive détectée : {drift}")
    return drift


@task
def send_discord_notification(message: str) -> None:
    """Envoie `message` sur le webhook Discord défini en variable env."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    logger = get_run_logger()

    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL manquant : notification ignorée.")
        return

    try:
        resp = requests.post(webhook_url, json={"content": message}, timeout=10)
        logger.warning(f"Notification Discord envoyée ({resp.status_code}).")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Erreur notification Discord : {exc}")


@task
def generate_ml_dataset(n_samples: int = 1000) -> dict | None:
    """POST /generate sur ml-api pour créer un dataset."""
    url = f"http://ml-api:8001/generate?n_samples={n_samples}"
    logger = get_run_logger()

    try:
        resp = requests.post(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"Dataset généré : {data}")
            return data
        logger.error(f"Erreur génération dataset : HTTP {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Erreur connexion API ML : {exc}")

    return None


@task
def trigger_real_retrain() -> dict:
    """POST /retrain sur ml-api ; retourne le JSON ou {'error': ...}."""
    url = "http://ml-api:8001/retrain"
    logger = get_run_logger()

    try:
        logger.warning("🔄 Déclenchement du retrain réel du modèle…")
        resp = requests.post(url, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            logger.warning(f"✅ Retrain terminé : {data}")
            return data

        error_msg = f"HTTP {resp.status_code}"
        logger.error(f"❌ Erreur retrain : {error_msg}")
        return {"error": error_msg}

    except Exception as exc:  # noqa: BLE001
        logger.error(f"❌ Erreur connexion retrain : {exc}")
        return {"error": str(exc)}


@task
def get_model_prediction() -> dict | None:
    """GET /predict sur ml-api pour une prédiction test."""
    url = "http://ml-api:8001/predict"
    logger = get_run_logger()

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"Prédiction obtenue : {data}")
            return data
        logger.error(f"Erreur prédiction : HTTP {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Erreur connexion prédiction : {exc}")

    return None


@task
def simulate_legacy_retrain() -> str:
    """Fallback simple quand l’API ML est indisponible."""
    logger = get_run_logger()
    logger.warning("🔄 Retrain simulé (mode legacy)…")
    time.sleep(2)
    logger.warning("✅ Retrain simulé terminé")
    return "Retrain simulated"


# --------------------------------------------------------------------------- #
# 3. FLOW PRINCIPAL
# --------------------------------------------------------------------------- #
@flow(name="random-check-flow")
def random_check_flow() -> None:
    """Vérifie la dérive et orchestre retrain + notifications Discord."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger = get_run_logger()

    # Étape 1 : dérive ?
    rnd = generate_random_number()
    drift = check_model_drift(rnd)

    if drift:
        logger.warning("🚨 Dérive détectée — actions correctives en cours…")

        dataset = generate_ml_dataset(1500)
        retrain = trigger_real_retrain()
        prediction = get_model_prediction()

        if retrain and "error" not in retrain:
            # Retrain réussi → notification SUCCESS
            msg = (
                "🚨 **DÉRIVE DÉTECTÉE – RETRAIN AUTOMATIQUE**\n"
                f"📅 {ts}\n"
                f"🎲 Dérive simulée : {rnd:.3f} < 0.5\n\n"
                "✅ **Retrain terminé avec succès :**\n"
                f"📊 Précision train : {retrain.get('train_accuracy', 0):.2%}\n"
                f"📈 Précision test : {retrain.get('test_accuracy', 0):.2%}\n"
                f"🔬 MLflow run : `{retrain.get('mlflow_run_id', '')[:8]}...`\n\n"
                f"📦 Dataset : {dataset.get('samples_generated') if dataset else 'N/A'} échantillons\n"
                f"🔮 Nouvelle prédiction : {prediction.get('prediction') if prediction else 'N/A'} "
                f"(confiance : {prediction.get('confidence', 0):.1%})"
            )
        else:
            # Échec API → retrain legacy + notification FAILOVER
            simulate_legacy_retrain()
            err = retrain.get("error", "API indisponible") if retrain else "API indisponible"

            msg = (
                "⚠️ **DÉRIVE DÉTECTÉE – RETRAIN EN MODE DÉGRADÉ**\n"
                f"📅 {ts}\n"
                f"🎲 Dérive simulée : {rnd:.3f} < 0.5\n\n"
                f"❌ Erreur API ML : {err}\n"
                "🔄 Retrain simulé effectué en fallback"
            )

        send_discord_notification(msg)

    else:
        # Pas de dérive – monitoring simple
        prediction = get_model_prediction()

        msg = (
            "✅ **Modèle stable**\n"
            f"📅 {ts}\n"
            f"🎲 Valeur : {rnd:.3f} ≥ 0.5\n\n"
            f"🔮 Prédiction : {prediction.get('prediction') if prediction else 'N/A'} "
            f"(confiance : {prediction.get('confidence', 0):.1%})\n"
            "📊 Surveillance continue active"
        )

        send_discord_notification(msg)


if __name__ == "__main__":
    random_check_flow()