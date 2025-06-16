FROM python:3.11-slim
RUN apt-get update && apt-get install -y libsqlite3-dev build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN mkdir -p /app/data && chmod 755 /app/data
RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]