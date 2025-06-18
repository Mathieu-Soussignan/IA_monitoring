FROM python:3.11-slim
WORKDIR /app

# Copie les requirements et installe
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie TOUT le contenu du dossier ui/ dans /app/ui
COPY . /app/ui

EXPOSE 8501
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
