from fastapi import FastAPI

app = FastAPI(title="Projet Groupe API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "API Projet Groupe - Jour 1"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fastapi-app"}