FROM python:3.11-slim

# — dépendances système —
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsqlite3-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

# — répertoires & app —
WORKDIR /app
RUN mkdir -p /app/data && chmod 755 /app/data

# — dépendances Python —
RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# — code source —
COPY . .

# — port exposé —
EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "warning", "--no-access-log"]