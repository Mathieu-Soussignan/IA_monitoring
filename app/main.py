from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Projet Groupe API", version="1.0.0")
Instrumentator().instrument(app).expose(app)

@app.get("/")
def read_root():
    return {"message": "API Projet Groupe - Jour 1"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fastapi-app"}