# Продакшен / инференс: CPU-only, компактный образ (публикация в GHCR и обычный docker compose).

FROM python:3.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# PyTorch inductor: без UID в /etc/passwd (docker user: GHA runner) getpass.getuser() падает при импорте.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch-inductor-cache

WORKDIR /app

# libgomp1 — OpenMP для numpy/torch wheels; минимальный runtime без CUDA.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --root-user-action=ignore --upgrade pip setuptools wheel

COPY requirements-docker.txt .
RUN pip install --root-user-action=ignore -r requirements-docker.txt

COPY app/ ./app/
COPY config.yaml .
COPY pytest.ini .
COPY data/ ./data/
COPY tests/ ./tests/

RUN mkdir -p models checkpoints

EXPOSE 20001

CMD ["python", "-m", "app.main", "--host", "0.0.0.0", "--port", "20001"]
