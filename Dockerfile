FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Обновляем pip и устанавливаем зависимости
RUN pip install --root-user-action=ignore --upgrade pip setuptools wheel

COPY requirements-docker.txt .
RUN pip install --root-user-action=ignore -r requirements-docker.txt

COPY commands_classifier/ ./commands_classifier/
COPY config.yaml .
COPY data/ ./data/

RUN mkdir -p models checkpoints

# Создаем пользователя для запуска приложения (не root)
# Используем UID/GID 1000, который обычно соответствует первому пользователю на Linux
RUN groupadd -r appuser -g 1000 && \
    useradd -r -u 1000 -g appuser -d /app -s /bin/bash appuser && \
    chown -R appuser:appuser /app

EXPOSE 20001

# Переключаемся на пользователя appuser
USER appuser

CMD ["python", "-m", "commands_classifier.cli", "serve", "--host", "0.0.0.0", "--port", "20001"]
