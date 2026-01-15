FROM python:3.11-slim

# Build argument для выбора версии (cpu или cuda)
ARG PYTORCH_VERSION=cpu

# Обновляем список пакетов (build-essential и git не нужны для большинства современных Python пакетов)
RUN apt-get update && apt-get clean && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем соответствующий requirements файл и устанавливаем зависимости
COPY requirements*.txt .

# Обновляем pip и устанавливаем базовые пакеты
RUN pip install --upgrade pip && \
    pip install --no-cache-dir setuptools wheel

# Устанавливаем зависимости в зависимости от выбранной версии
RUN if [ "$PYTORCH_VERSION" = "cuda" ]; then \
        pip install --no-cache-dir -r requirements-cuda.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Копируем код приложения
COPY commands_classifier/ ./commands_classifier/
COPY config.yaml .
COPY data/ ./data/

# Создаем директории для моделей и базы данных
RUN mkdir -p models checkpoints

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Открываем порт
EXPOSE 8000

# Команда по умолчанию - запуск сервера
CMD ["python", "-m", "commands_classifier.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]

