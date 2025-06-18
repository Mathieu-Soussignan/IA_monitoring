## Résumé du débogage de la stack MLOps

### 1. Health‑checks & redémarrages

* **mlflow** : `start_period` trop court → conteneur déclaré *unhealthy* avant la fin de l’installation.

  * *Fix* : start\_period passé à 180 s (ou image custom pré‑buildée).
* **prefect‑worker** : boucle de redémarrage provoquée par une commande d’entrée qui se terminait ;

  * *Fix* : commande simplifiée `prefect worker start --pool default-pool …`.

### 2. Variables d’environnement incohérentes

* `DISCORD_WEBHOOK_URL` manquait pour la fonction `send_discord_notification()`.
* `PREFECT_API_URL` pointait sur 127.0.0.1 à cause du .env.

  * *Fix* : ajout de `DISCORD_WEBHOOK_URL` dans .env et forçage de la bonne URL interne dans `environment:` du worker.

### 3. Bruit Discord

* Access‑logs Uvicorn (niveau INFO) polluaient le salon.

  * *Fix* : `LOG_LEVEL=WARNING` dans docker‑discord‑logger + `--no-access-log` et `--log-level warning` dans ml‑api.

### 4. Logs Prefect non captés

* Seul **ml-api** était labellisé.

  * *Fix* : ajout du label `discord-logger.enabled=true` sur le conteneur **prefect-worker**.

### 5. Crash en boucle de ml‑api

* Uvicorn rejetait `--access-log off` → « unexpected extra argument (off) ».

  * *Fix* : utilisation de `--no-access-log` et cible correcte `main:app` + rebuild de l’image.

### 6. Résolution DNS & appels HTTP

* Les appels `/retrain`, `/predict` échouaient tant que **ml-api** n’était pas stable.

  * *Fix* : correction du point 5 → conteneur stable, endpoints accessibles.

---

### État final

* Tous les conteneurs restent **Up** et *healthy*.
* Discord n’affiche que les logs `WARNING/ERROR` et les notifications formatées.
* Les flows Prefect exécutent correctement la génération de dataset, le retrain MLflow et la prédiction avec suivi dans MLflow et notifications Discord.
