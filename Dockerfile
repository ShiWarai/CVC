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

COPY app/ ./app/
COPY config.yaml .
COPY pytest.ini .
COPY data/ ./data/
COPY tests/ ./tests/

RUN mkdir -p models checkpoints

EXPOSE 20001

CMD ["python", "-m", "app.cli", "serve", "--host", "0.0.0.0", "--port", "20001"]
