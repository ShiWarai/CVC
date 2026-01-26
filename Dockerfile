FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --root-user-action=ignore --upgrade pip && \
    pip install --root-user-action=ignore setuptools wheel

COPY requirements-docker.txt .

RUN pip install --root-user-action=ignore -r requirements-docker.txt && \
    python -c "import uvicorn; import fastapi; print('✓ uvicorn и fastapi установлены')" || \
    (echo "✗ Ошибка: uvicorn или fastapi не установлены" && pip list | grep -E "(uvicorn|fastapi)" && exit 1)

COPY commands_classifier/ ./commands_classifier/
COPY config.yaml .
COPY data/ ./data/

RUN mkdir -p models checkpoints

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 20001

CMD ["python", "-m", "commands_classifier.cli", "serve", "--host", "0.0.0.0", "--port", "20001"]
