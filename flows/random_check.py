from prefect import task, flow
import random
import requests
import time
from datetime import datetime
import os

@task
def generate_random_number():
    """Génère un nombre aléatoire entre 0 et 1"""
    number = random.random()
    print(f"Nombre généré: {number}")
    return number

@task
def check_model_drift(random_number):
    """Vérifie si le modèle a dérivé (simulation)"""
    drift_threshold = 0.5
    has_drifted = random_number < drift_threshold
    print(f"Dérive détectée: {has_drifted}")
    return has_drifted

@task
def send_discord_notification(message):
    """Envoie une notification Discord"""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook_url:
        payload = {"content": message}
        response = requests.post(webhook_url, json=payload)
        print(f"Notification envoyée: {response.status_code}")

@task
def trigger_retrain():
    """Simule le déclenchement d'un retrain"""
    print("🔄 Déclenchement du retrain du modèle...")
    time.sleep(2)  # Simulation du retrain
    print("✅ Retrain terminé avec succès")
    return "Retrain completed"

@flow(name="random-check-flow")
def random_check_flow():
    """Flow principal de vérification aléatoire"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Générer un nombre aléatoire
    random_num = generate_random_number()
    
    # Vérifier la dérive
    drift_detected = check_model_drift(random_num)
    
    if drift_detected:
        # Déclencher le retrain
        retrain_result = trigger_retrain()
        
        # Envoyer notification Discord
        message = f"🚨 [{timestamp}] Dérive détectée! Retrain automatique effectué. Nombre: {random_num:.3f}"
        send_discord_notification(message)
    else:
        # Tout va bien
        message = f"✅ [{timestamp}] Modèle stable. Nombre: {random_num:.3f}"
        send_discord_notification(message)

if __name__ == "__main__":
    random_check_flow()