# CVC - Classification of Voice Commands

[![ML Pipeline](https://github.com/ShiWarai/CVC/actions/workflows/deploy.yml/badge.svg?branch=main)](https://github.com/ShiWarai/CVC/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/github/license/ShiWarai/CVC)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Docker Ready](https://img.shields.io/badge/docker-ready-blue?logo=docker)
[![CVC-Panda on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20CVC--Panda-Model-yellow)](https://huggingface.co/ShiWarai/CVC-Panda)

Мини-сервис для классификации голосовых команд (SetFit). Обучает модель на малом датасете и классифицирует текстовые команды. Создан для использования в проекте навыка для команд роботу-собаке.

## Стек технологий

| Категория | Технологии |
|-----------|------------|
| ML | SetFit, PyTorch, sentence-transformers, Hugging Face Hub, scikit-learn, datasets |
| API | FastAPI, Uvicorn |
| Данные | SQLite, pandas, PyYAML |
| Интерфейсы | REST API, CLI (Python) |
| Инфраструктура | Docker |
| Разработка | pytest, ruff, httpx |

## Оглавление

| Раздел | Содержание |
|--------|------------|
| [Стек технологий](#стек-технологий) | ML, API, данные, инфраструктура |
| [Быстрый старт](#быстрый-старт) | Запуск за 3 шага (Docker) |
| [Установка и запуск](#установка-и-запуск) | CPU / CUDA / ROCm, Docker, локально |
| [Использование](#использование) | CLI, Python-клиент, библиотека |
| [Конфигурация и API](#конфигурация-и-api) | config.yaml, эндпоинты |
| [Данные](#данные) | Формат датасета, параметры обучения |
| [Разработка](#разработка) | Тесты, линт, архитектура, структура проекта |
| [CI/CD](#cicd) | Пайплайн и ссылка на настройку |
| [Лицензия](#лицензия) | MIT |

## Быстрый старт

1. Создайте `.env` с `HF_TOKEN` и `HF_REPO_ID` ([токен](https://huggingface.co/settings/tokens), [модель](https://huggingface.co/google/embeddinggemma-300M) — принять условия).
2. Соберите и запустите контейнер:

   ```bash
   docker compose up --build -d
   ```

3. Сервер: **http://localhost:20001**. Документация API: http://localhost:20001/docs

Остановка: `docker-compose down`.

## Установка и запуск

### Требование: Hugging Face

Модель `google/embeddinggemma-300M` требует авторизации. В корне проекта создайте `.env`:

```bash
HF_TOKEN=your_token_here
HF_REPO_ID=your-username/model-name
```

### Варианты установки

| Вариант | Команда | Примечание |
|---------|---------|------------|
| **CPU** | `pip install -r requirements-docker.txt` | По умолчанию |
| **NVIDIA CUDA** | Сначала [PyTorch с CUDA](https://pytorch.org), затем `pip install -r requirements-cuda.txt` | CUDA 12.4+, PyTorch 2.6+ |
| **AMD ROCm** | Драйвер [AMD PyTorch Edition](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-1-1.html), Python 3.12, затем `pip install -r requirements-rocm.txt` | Только Windows |

### Docker

- **Инференс / прод** — образ из `Dockerfile`: `python:3.11-slim` + CPU PyTorch из `requirements-docker.txt` (лёгкий, публикация в GHCR).
- **Обучение с GPU** — `Dockerfile.cuda` на базе `pytorch/pytorch:…-cuda…-runtime` и `requirements-docker-cuda.txt` (без переустановки PyTorch CPU-колёсами). Запуск: `docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d` (нужен NVIDIA Container Toolkit). В CI job *Train and publish* используется тот же overlay.
- Для GPU **без** Docker — локально CUDA/ROCm по разделу «Варианты установки».
- Код, `config.yaml` и `data/` — в образе (`Dockerfile`); данные — в именованных томах Docker (`cvc_models`, `cvc_hf_cache`, `cvc_db`), без bind-mount из репозитория.
- Обычный запуск: `docker compose up --build -d` (после изменения кода — пересборка). Прод с GHCR: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`.

### Локальный запуск

```bash
python -m app.main
```

Опции: `--host`, `--port`, `--config`, `--reload`. БД создаётся при первом запуске, данные из `data/` или CSV из `config.yaml`.

## Использование

### CLI

После запуска сервера (Docker или локально):

```bash
python -m app.client predict --text "равняйся" [--show-confidence]
python -m app.client predict --file commands.txt
python -m app.client train [--batch-size 32 --iterations 30]
python -m app.client train-status
python -m app.client examples list
python -m app.client examples add --text "команда" --command "label"
python -m app.client examples delete --id 1
python -m app.client health
python -m app.client metrics
python -m app.client reset
python -m app.client load-from-hf [--repo-id "username/model-name"]
python -m app.client load-from-hf-status
python -m app.client command-feedback   # репорт «исправить команду» из RDS-2P-Salute
```

По умолчанию клиент подключается к `http://localhost:20001` (флаг `--url` для другого адреса).

### Python

- **API-клиент:** `CVCApiClient(base_url)` — методы `predict`, `predict_batch`, `embed`, `train`, `get_training_status`, `get_examples`, `add_example`, `delete_example`, `health`, `metrics`, `reset`, `load_from_hf`, `get_load_from_hf_status`, `get_command_feedback`.
- **Библиотека (без сервера):** `CommandsClassifier()` + `load_dataset(path)` → `train(texts, labels)`, `predict(text)`, `save(path)`, `load(path)`.

## Конфигурация и API

### config.yaml

Основные параметры:

```yaml
server:
  host: "0.0.0.0"
  port: 20001

model:
  path: "models/my_model"
  name: "google/embeddinggemma-300M"
  confidence_threshold: 0.5
  cache_dir: "models/.cache"

database:
  path: "db/training_data.db"
  csv_migration_path: "data"

# Опционально: URL репорта «исправить команду» из RDS-2P-Salute (по умолчанию: rds-2p-salute-app:8000)
# command_feedback:
#   url: "http://rds-2p-salute-app:8000/v1/admin/command-feedback"

training:
  iterations: 20
  epochs: 1
  batch_size: 32
  learning_rate: 2e-5
```

### Эндпоинты (API v1)

Все ручки версионированы префиксом `/v1`.

| Метод | Путь | Описание |
|-------|------|----------|
| POST | /v1/embed | Эмбеддинги (TEI) |
| GET | /v1/health | Проверка работоспособности |
| GET | /v1/metrics | Счётчики примеров и статус обучения |
| POST | /v1/predict | Классификация одного текста |
| POST | /v1/predict/batch | Batch классификация |
| POST | /v1/train | Запуск обучения (фоновый) |
| GET | /v1/train/status | Статус обучения |
| GET, POST, DELETE | /v1/examples, /v1/examples/{id} | Обучающие примеры |
| POST | /v1/reset | Сброс обучения (удаление модели, пометка примеров как необученных) |
| POST | /v1/load_from_hf | Загрузка модели с Hugging Face |
| GET | /v1/load_from_hf/status | Статус загрузки |
| GET | /v1/command-feedback | Репорт «исправить команду» из RDS-2P-Salute (прокси) |

Интерактивная документация: **http://localhost:20001/docs**. Устройство (CPU/CUDA/ROCm) определяется при запуске автоматически.

## Данные

### Формат датасета

Для миграции при старте — CSV или JSON в `data/` или путь в `config.yaml` (`database.csv_migration_path`).

**CSV:** колонки `text`, `command`.

**JSON:** список объектов `{"text": "...", "command": "..."}` или объект с массивами `{"text": [...], "command": [...]}`.

### Параметры обучения

`--iterations`, `--epochs`, `--batch-size`, `--learning-rate`. Значения по умолчанию — в `config.yaml` (секция `training`).

## Разработка

### Тесты и линт

Используется образ **cvc-dev** ([docker-compose.dev.yml](docker-compose.dev.yml)):

```bash
docker compose -f docker-compose.yml build cvc-api
docker compose -f docker-compose.yml -f docker-compose.dev.yml build cvc-dev

docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm cvc-dev ruff check .
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm cvc-dev pytest tests/ -v --tb=short --cov=app --cov-report=term-missing
```

### Архитектура

Проект построен по принципам чистой архитектуры (слои не зависят от деталей доставки и инфраструктуры).

| Слой | Назначение |
|------|------------|
| **domain** | Сущности (`Example`, `PredictionResult`, `TrainingStatus`), порты (`IClassifier`, `IExampleRepository`), утилиты (`text_utils`). Без внешних зависимостей. |
| **application** | Сценарии (use cases): предсказание (`PredictUseCase`), работа с примерами (`ExamplesUseCase`). Получают зависимости через конструктор. |
| **adapters** | Реализации портов: **persistence** — SQLite-репозиторий примеров; **ml** — SetFit-классификатор и retry для HF; **data_loading** — загрузка датасета из CSV/JSON. |
| **api** | FastAPI-приложение, роуты, глобальное состояние (state). В `init_app()` собираются use cases и адаптеры (composition root). |

Точка входа сервера: `main.py` → `app.api.server`; CLI к API: `client.py`.

### Структура проекта

```
CVC/
├── config.yaml
├── Dockerfile               # CPU slim: прод, GHCR, тест в CI
├── Dockerfile.cuda          # PyTorch CUDA: обучение (compose + job train-and-publish)
├── docker-compose.yml
├── docker-compose.cuda.yml  # GPU + Dockerfile.cuda
├── requirements-docker.txt | requirements-docker-cuda.txt | requirements-cuda.txt | requirements-rocm.txt
├── app/                     # Точка входа: python -m app.main
│   ├── main.py              # Запуск сервера
│   ├── domain/              # Сущности, порты, text_utils
│   ├── application/         # Use cases
│   ├── adapters/            # persistence (SQLite), ml (SetFit), data_loading
│   ├── api/                 # FastAPI, роуты, state
│   └── client.py            # HTTP-клиент и библиотека
├── data/                    # CSV/JSON для миграции
├── models/                  # Сохранённые модели
├── db/                      # SQLite (training_data.db)
├── tests/
└── docs/                    # cicd_setup.md и др.
```

## CI/CD

Пайплайн [.github/workflows/deploy.yml](.github/workflows/deploy.yml):

- При каждом push — job **test** на **ubuntu-latest**: линт + pytest в контейнере **`Dockerfile`** (CPU slim, как образ в GHCR).
- Job **Train and Publish** только на **self-hosted** с GPU (`docker-compose.cuda.yml` → **`Dockerfile.cuda`**): при метке `[retrain]` в сообщении коммита или при ручном запуске (Actions → Run workflow). Секреты: `HF_TOKEN`, `HF_REPO_ID`. При нескольких self-hosted раннерах задайте метку GPU (например `runs-on: [self-hosted, gpu]`).
- **Уведомления в Telegram** при успешной и неуспешной сборке (опционально: секреты `TELEGRAM_TOKEN`, `TELEGRAM_TO`). Подробнее: [docs/telegram_notifications.md](docs/telegram_notifications.md).

Подробная настройка (self-hosted runner, GPU, секреты): [docs/cicd_setup.md](docs/cicd_setup.md).

## Лицензия

MIT

*Проект создан с использованием нейросетей.*
