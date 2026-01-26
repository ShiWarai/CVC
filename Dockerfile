FROM python:3.11-slim

# Build argument для выбора между CPU и CUDA версией
ARG USE_CUDA=false

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --root-user-action=ignore --upgrade pip && \
    pip install --root-user-action=ignore setuptools wheel

# Копируем оба файла requirements
COPY requirements.txt .
COPY requirements-cuda.txt .

# Выбираем файл requirements в зависимости от USE_CUDA
RUN if [ "$USE_CUDA" = "true" ]; then \
        echo "Использование CUDA версии зависимостей..." && \
        pip install --root-user-action=ignore -r requirements-cuda.txt && \
        python -c "import uvicorn; import fastapi; import torch; print('✓ uvicorn, fastapi и torch (CUDA) установлены'); print('✓ PyTorch версия:', torch.__version__); print('ℹ️  CUDA доступность будет проверена при запуске контейнера')" || \
        (echo "✗ Ошибка установки CUDA зависимостей" && exit 1); \
    else \
        echo "Использование CPU версии зависимостей..." && \
        pip install --root-user-action=ignore -r requirements.txt && \
        python -c "import uvicorn; import fastapi; print('✓ uvicorn и fastapi установлены')" || \
        (echo "✗ Ошибка установки зависимостей" && exit 1); \
    fi

COPY commands_classifier/ ./commands_classifier/
COPY config.yaml .
COPY data/ ./data/

RUN mkdir -p models checkpoints

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 20001

CMD ["python", "-m", "commands_classifier.cli", "serve", "--host", "0.0.0.0", "--port", "20001"]

