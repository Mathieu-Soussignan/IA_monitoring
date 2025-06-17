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
        try:
            response = requests.post(webhook_url, json=payload)
            print(f"Notification envoyée: {response.status_code}")
        except Exception as e:
            print(f"Erreur notification Discord: {e}")

@task
def generate_ml_dataset(n_samples=1000):
    """Génère un nouveau dataset via l'API ML"""
    try:
        response = requests.post(f"http://ml-api:8001/generate?n_samples={n_samples}")
        if response.status_code == 200:
            result = response.json()
            print(f"Dataset généré: {result}")
            return result
        else:
            print(f"Erreur génération dataset: {response.status_code}")
            return None
    except Exception as e:
        print(f"Erreur connexion API ML: {e}")
        return None

@task
def trigger_real_retrain():
    """Déclenche un vrai retrain via l'API ML"""
    try:
        print("🔄 Déclenchement du retrain réel du modèle...")
        response = requests.post("http://ml-api:8001/retrain")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrain terminé: {result}")
            return result
        else:
            print(f"❌ Erreur retrain: {response.status_code}")
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        print(f"❌ Erreur connexion retrain: {e}")
        return {"error": str(e)}

@task
def get_model_prediction():
    """Obtient une prédiction du modèle"""
    try:
        response = requests.get("http://ml-api:8001/predict")
        if response.status_code == 200:
            result = response.json()
            print(f"Prédiction obtenue: {result}")
            return result
        else:
            print(f"Erreur prédiction: {response.status_code}")
            return None
    except Exception as e:
        print(f"Erreur connexion prédiction: {e}")
        return None

@task
def simulate_legacy_retrain():
    """Simule le retrain legacy (fallback)"""
    print("🔄 Retrain simulé (mode legacy)...")
    time.sleep(2)
    print("✅ Retrain simulé terminé")
    return "Retrain simulated"

@flow(name="random-check-flow")
def random_check_flow():
    """Flow principal de vérification avec vraie API ML"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Générer un nombre aléatoire pour simulation de dérive
    random_num = generate_random_number()
    
    # Vérifier la dérive
    drift_detected = check_model_drift(random_num)
    
    if drift_detected:
        # DÉRIVE DÉTECTÉE - Action corrective
        print("🚨 Dérive détectée - Actions correctives en cours...")
        
        # 1. Générer un nouveau dataset
        dataset_result = generate_ml_dataset(1500)  # Plus d'échantillons en cas de dérive
        
        # 2. Déclencher le retrain réel
        retrain_result = trigger_real_retrain()
        
        # 3. Obtenir une prédiction test
        prediction_result = get_model_prediction()
        
        # 4. Construire le message Discord enrichi
        if retrain_result and "error" not in retrain_result:
            train_acc = retrain_result.get("train_accuracy", 0)
            test_acc = retrain_result.get("test_accuracy", 0)
            mlflow_run = retrain_result.get("mlflow_run_id", "N/A")
            
            # Message de succès avec métriques
            message = f"""🚨 **DÉRIVE DÉTECTÉE - RETRAIN AUTOMATIQUE**
📅 {timestamp}
🎲 Dérive simulée: {random_num:.3f} < 0.5

✅ **Retrain terminé avec succès:**
📊 Précision train: {train_acc:.2%}
📈 Précision test: {test_acc:.2%}
🔬 MLflow run: `{mlflow_run[:8]}...`

📦 Dataset: {dataset_result.get('samples_generated', 'N/A')} échantillons
🔮 Nouvelle prédiction: {prediction_result.get('prediction', 'N/A')} (confiance: {prediction_result.get('confidence', 0):.1%})"""
            
        else:
            # Fallback en cas d'erreur API
            simulate_legacy_retrain()
            error_msg = retrain_result.get("error", "Erreur inconnue") if retrain_result else "API indisponible"
            message = f"""⚠️ **DÉRIVE DÉTECTÉE - RETRAIN EN MODE DÉGRADÉ**
📅 {timestamp}
🎲 Dérive simulée: {random_num:.3f} < 0.5

❌ Erreur API ML: {error_msg}
🔄 Retrain simulé effectué en fallback"""
        
        send_discord_notification(message)
        
    else:
        # MODÈLE STABLE - Surveillance continue
        
        # Obtenir quand même une prédiction pour monitoring
        prediction_result = get_model_prediction()
        
        # Message de stabilité
        if prediction_result:
            message = f"""✅ **Modèle stable**
📅 {timestamp}
🎲 Valeur: {random_num:.3f} ≥ 0.5

🔮 Prédiction: {prediction_result.get('prediction', 'N/A')} (confiance: {prediction_result.get('confidence', 0):.1%})
📊 Surveillance continue active"""
        else:
            message = f"""✅ **Modèle stable**
📅 {timestamp}
🎲 Valeur: {random_num:.3f} ≥ 0.5
📊 Surveillance continue active"""
        
        send_discord_notification(message)

if __name__ == "__main__":
    random_check_flow()