FROM python:3.11-slim

# Обновляем список пакетов (build-essential и git не нужны для большинства современных Python пакетов)
RUN apt-get update && apt-get clean && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

