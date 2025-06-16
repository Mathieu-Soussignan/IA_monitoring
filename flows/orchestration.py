from prefect import serve
from random_check import random_check_flow
from datetime import timedelta

if __name__ == "__main__":
    # Déployer le flow avec un intervalle de 30 secondes
    random_check_deployment = random_check_flow.to_deployment(
        name="random-check-deployment",
        interval=timedelta(seconds=30),
        tags=["monitoring", "drift-detection"]
    )
    
    serve(random_check_deployment)