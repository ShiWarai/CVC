FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install setuptools wheel

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY commands_classifier/ ./commands_classifier/
COPY config.yaml .
COPY data/ ./data/

RUN mkdir -p models checkpoints

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 20001

CMD ["python", "-m", "commands_classifier.cli", "serve", "--host", "0.0.0.0", "--port", "20001"]

