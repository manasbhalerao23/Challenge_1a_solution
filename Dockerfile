FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

RUN mkdir -p /app/input /app/output /app/summary

CMD ["python", "main.py", "input", "output", "summary"]