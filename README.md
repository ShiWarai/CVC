# CVC - Classification of Voice Commands

[![ML Pipeline](https://github.com/ShiWarai/CVC/actions/workflows/deploy.yml/badge.svg)](https://github.com/ShiWarai/CVC/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/github/license/ShiWarai/CVC)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Docker Ready](https://img.shields.io/badge/docker-ready-blue?logo=docker)
[![CVC-Panda on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20CVC--Panda-Model-yellow)](https://huggingface.co/ShiWarai/CVC-Panda)

Мини-сервис для классификации голосовых команд с использованием SetFit (few-shot learning). Позволяет обучать модель на малом датасете (8-16 примеров на класс) и классифицировать текстовые команды. Деплоимая модель: [CVC-Panda](https://huggingface.co/ShiWarai/CVC-Panda) на Hugging Face.

## Особенности

- **Few-shot learning**: Обучение на 5-50 примерах на класс
- **Поддержка русского языка**: Использует multilingual модели (embeddinggemma-300M)
- **Эффективное использование памяти**: Компактная модель (308M параметров) для работы на устройствах с ограниченными ресурсами
- **CPU-only в Docker**: Оптимизированная версия для контейнеров
- **Простой CLI интерфейс**: Легко использовать из командной строки
- **Гибкий формат датасета**: Поддержка CSV и JSON
- **REST API сервер**: FastAPI сервер с TEI-совместимыми эндпоинтами
- **База данных**: SQLite для хранения обучающих данных
- **Фоновое обучение**: Возможность дообучать модель через API без остановки сервера
- **Docker поддержка**: Готовая конфигурация для контейнеризации и развертывания

## Установка

Проект поддерживает три варианта установки в зависимости от вашего оборудования:

### CPU (по умолчанию)

Для работы на CPU без GPU ускорения:

```bash
pip install -r requirements-docker.txt
```

### NVIDIA CUDA

Для использования NVIDIA GPU с поддержкой CUDA:

```bash
# Сначала установите PyTorch с CUDA (см. подробные инструкции в разделе "Локальное обучение с GPU")
pip install -r requirements-cuda.txt
```

**Требования:** NVIDIA GPU с поддержкой CUDA 12.4+, PyTorch 2.6+

### AMD ROCm

Для использования AMD GPU через ROCm на Windows:

```bash
# Сначала установите драйвер AMD Software: PyTorch on Windows Edition 7.1.1
# Затем установите зависимости (см. подробные инструкции в разделе "Локальное обучение с GPU")
pip install -r requirements-rocm.txt
```

**Требования:** 
- Windows 11 (Windows 10 не поддерживается)
- Python 3.12
- Совместимое устройство AMD (RX 9070, RX 7900 XTX, Ryzen AI 9 365 и др.)
- Драйвер AMD Software: PyTorch on Windows Edition 7.1.1
- Минимум 32GB RAM (рекомендуется 64GB)

Подробные инструкции по установке для каждого варианта см. в разделе ["Локальное обучение с GPU"](#локальное-обучение-с-gpu).

**Важно:** Модель `google/embeddinggemma-300M` требует авторизации в Hugging Face:

1. Перейдите на [страницу модели](https://huggingface.co/google/embeddinggemma-300M) и примите условия использования
2. Получите токен доступа в [настройках аккаунта](https://huggingface.co/settings/tokens)
3. Создайте файл `.env` в корне проекта и добавьте токен и ID репозитория:

```bash
HF_TOKEN=your_token_here
HF_REPO_ID=your-username/model-name
```

Токен будет автоматически загружен при запуске приложения (через `python-dotenv`).

## Docker

Проект можно запустить в Docker контейнере для изоляции и удобства развертывания.

**Важно:** Docker контейнер использует CPU-only версию PyTorch. Для использования GPU (CUDA) запускайте приложение локально с установленным PyTorch CUDA (см. раздел "Локальное обучение с CUDA").

### Быстрый старт с Docker Compose

**Важно:** Модель `google/embeddinggemma-300M` требует авторизации в Hugging Face. Перед запуском создайте файл `.env`:

```bash
# Скопируйте пример файла
cp .env.example .env

# Отредактируйте .env и добавьте ваш Hugging Face токен и ID репозитория
# HF_TOKEN=your_token_here
# HF_REPO_ID=your-username/model-name
```

Получите токен на [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Затем запустите контейнер:

```bash
# Сборка и запуск контейнера
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка контейнера
docker-compose down
```

Сервер будет доступен по адресу `http://localhost:20001`.

**Примечание:** В Docker контейнере устройство для обучения всегда определяется как CPU (контейнер не содержит CUDA). Устройство определяется автоматически при запуске сервера.

### Использование Docker образа напрямую

**Важно:** Не забудьте передать токен Hugging Face через переменную окружения `-e HF_TOKEN=your_token_here`.

```bash
# Сборка образа (CPU-only версия)
docker build -t cvc-api .

# Запуск контейнера с токеном
docker run -d \
  --name cvc-api \
  -p 20001:20001 \
  -e HF_TOKEN=your_token_here \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/cache/huggingface:/app/.cache/huggingface \
  -v $(pwd)/db:/app/db \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/data:/app/data:ro \
  cvc-api
```

**Для Windows PowerShell:**
```powershell
# Сборка образа
docker build -t cvc-api .

# Запуск контейнера с токеном
docker run -d `
  --name cvc-api `
  -p 20001:20001 `
  -e HF_TOKEN=your_token_here `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -v ${PWD}/cache/huggingface:/app/.cache/huggingface `
  -v ${PWD}/db:/app/db `
  -v ${PWD}/config.yaml:/app/config.yaml:ro `
  -v ${PWD}/data:/app/data:ro `
  cvc-api
```

### Передача Hugging Face токена в Docker

**Рекомендуемый способ:** Используйте файл `.env` (см. раздел "Быстрый старт" выше). Docker Compose автоматически загрузит переменные из `.env` файла.

**Альтернативный способ:** Передача токена через переменную окружения при запуске:

```bash
# Для docker-compose
HF_TOKEN=your_token_here docker-compose up -d

# Для docker run напрямую
docker run -d \
  --name cvc-api \
  -p 20001:20001 \
  -e HF_TOKEN=your_token_here \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/cache/huggingface:/app/.cache/huggingface \
  -v $(pwd)/db:/app/db \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/data:/app/data:ro \
  cvc-api
```

**Для Windows PowerShell:**
```powershell
# Для docker-compose
$env:HF_TOKEN="your_token_here"; docker-compose up -d

# Для docker run
docker run -d `
  --name cvc-api `
  -p 20001:20001 `
  -e HF_TOKEN=your_token_here `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -v ${PWD}/cache/huggingface:/app/.cache/huggingface `
  -v ${PWD}/db:/app/db `
  -v ${PWD}/config.yaml:/app/config.yaml:ro `
  -v ${PWD}/data:/app/data:ro `
  cvc-api
```

### Volumes

Важно: Следующие директории монтируются как volumes для сохранения данных между перезапусками:
- `./models` - обученные модели (сохраняются после обучения)
- `./checkpoints` - промежуточные чекпоинты обучения
- `./cache/huggingface` - кэш Hugging Face (оригинальные модели, чтобы не скачивать их каждый раз)
- `./db` - директория с базой данных SQLite (файл `training_data.db` создается автоматически)

**Примечание:** При первом запуске контейнера директория `./cache/huggingface` будет создана автоматически, и в неё будет загружена оригинальная модель `google/embeddinggemma-300M` (требуется токен Hugging Face в `.env` файле). При последующих перезапусках модель будет загружаться из кэша, что значительно ускорит запуск.

**Примечание о кэшировании базовой модели:** Для локального использования можно настроить кэширование базовой модели в папку `models/.cache` через параметр `model.cache_dir` в `config.yaml`. Это позволит хранить базовую модель вместе с обученными моделями.

**Примечание о базе данных:** База данных монтируется как директория (`./db`), а не как файл, чтобы избежать проблемы Docker, когда несуществующий файл при монтировании создается как директория. Файл `training_data.db` будет автоматически создан внутри директории `./db` при первом запуске.

### Запуск тестов и линта

Для линта и тестов используется отдельный compose-файл [docker-compose.dev.yml](docker-compose.dev.yml) и образ **cvc-dev** (на базе cvc-api + pytest, ruff). Сначала соберите базовый образ cvc-api, затем cvc-dev:

```bash
docker compose -f docker-compose.yml build cvc-api
docker compose -f docker-compose.yml -f docker-compose.dev.yml build cvc-dev
```

Линт (ruff):

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm cvc-dev ruff check .
```

Тесты с покрытием (как в CI):

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm cvc-dev pytest tests/ -v --tb=short --cov=commands_classifier --cov-report=term-missing
```

Основной [docker-compose.yml](docker-compose.yml) — только для запуска API (cvc-api). **cvc-dev** — для разработки и CI (линт + тесты).

### Использование клиента с Docker контейнером

После запуска контейнера клиент можно использовать как обычно:

```bash
# Проверка работоспособности
python -m commands_classifier.client health

# Классификация
python -m commands_classifier.client predict --text "равняйся"

# Запуск обучения
python -m commands_classifier.client train
```

Клиент автоматически подключится к серверу на `http://localhost:20001`.

### Health Check

Контейнер включает health check, который проверяет доступность API каждые 30 секунд. Статус можно проверить:

```bash
docker ps  # Проверка статуса контейнера
docker-compose ps  # Или через compose
```

### Просмотр логов

```bash
# Все логи
docker-compose logs

# Следить за логами в реальном времени
docker-compose logs -f

# Логи конкретного сервиса
docker-compose logs cvc-api
```

## Быстрый старт

Вы можете запустить CVC двумя способами:
- **Локально** (требует установки зависимостей, см. раздел "Установка")
- **В Docker** (рекомендуется, см. раздел "Docker" выше)

### 1. Запуск API сервера

**Локальный запуск:**

```bash
python -m commands_classifier.cli serve
```

Или с кастомными параметрами:

```bash
python -m commands_classifier.cli serve \
  --host 0.0.0.0 \
  --port 20001 \
  --config config.yaml
```

Сервер автоматически создаст базу данных и выполнит миграцию данных из директории `data` (или указанного CSV файла, если указано в `config.yaml`). Если указана директория, загружаются все CSV файлы из неё.

### 2. Использование консольного клиента

После запуска сервера используйте консольный клиент для работы с API:

**Классификация текста:**
```bash
python -m commands_classifier.client predict --text "равняйся"
```

**Классификация с уверенностью:**
```bash
python -m commands_classifier.client predict --text "равняйся" --show-confidence
```

**Batch классификация:**
```bash
python -m commands_classifier.client predict --file commands.txt
```

**Запуск обучения модели:**
```bash
python -m commands_classifier.client train
```

**Запуск обучения с кастомными параметрами:**
```bash
python -m commands_classifier.client train --batch-size 32 --iterations 30
```

**Проверка статуса обучения:**
```bash
python -m commands_classifier.client train-status
```

**Работа с примерами:**
```bash
# Список всех примеров
python -m commands_classifier.client examples list

# Добавить пример
python -m commands_classifier.client examples add --text "новая команда" --command "new_command"

# Удалить пример
python -m commands_classifier.client examples delete --id 1
```

**Проверка работоспособности:**
```bash
python -m commands_classifier.client health
```

**Загрузка модели с Hugging Face Hub:**
```bash
# Использует HF_REPO_ID из конфигурации сервера (.env)
python -m commands_classifier.client load-from-hf

# Или с указанием конкретного репозитория
python -m commands_classifier.client load-from-hf --repo-id "username/model-name"
```

**Проверка статуса загрузки модели:**
```bash
python -m commands_classifier.client load-from-hf-status
```

## Использование Python клиента

```python
from commands_classifier.client import CVCApiClient

# Создание клиента
client = CVCApiClient("http://localhost:20001")

# Классификация текста
result = client.predict("равняйся", return_confidence=True)
print(f"Команда: {result['command']}, Уверенность: {result['confidence']}")

# Batch классификация
results = client.predict_batch(["равняйся", "отставить"])
print(results['commands'])

# Получение эмбеддингов
embeddings = client.embed(["равняйся", "отставить"])
print(embeddings['embeddings'])

# Запуск обучения
train_result = client.train(num_iterations=30, num_epochs=2, batch_size=32)
print(f"Обучение запущено: {train_result['training_id']}")

# Проверка статуса обучения
status = client.get_training_status()
print(f"Статус: {status['status']}, Прогресс: {status['progress']}")

# Работа с примерами
examples = client.get_examples()
new_example = client.add_example("новая команда", "new_command")
client.delete_example(new_example['id'])
```

## Использование как библиотеки (для разработки)

Если нужно работать с моделью напрямую без API:

```python
from commands_classifier.model import CommandsClassifier
from commands_classifier.dataset import load_dataset

# Загрузка датасета
texts, labels = load_dataset("data/commands_example.csv")

# Обучение модели
classifier = CommandsClassifier()
classifier.train(texts, labels)

# Классификация
command = classifier.predict("равняйся")
print(f"Команда: {command}")

# Сохранение модели
classifier.save("models/my_model")

# Загрузка сохраненной модели
classifier2 = CommandsClassifier()
classifier2.load("models/my_model")
result = classifier2.predict("отставить")
```

## API Сервер

CVC включает REST API сервер на FastAPI для работы с моделью через HTTP запросы.

### Запуск сервера

```bash
python -m commands_classifier.cli serve
```

Или с кастомными параметрами:

```bash
python -m commands_classifier.cli serve \
  --host 0.0.0.0 \
  --port 20001 \
  --config config.yaml
```

**Примечание:** Устройство для обучения определяется автоматически:
- Если установлен PyTorch с CUDA поддержкой и CUDA доступна → используется NVIDIA GPU
- Если установлен PyTorch с ROCm поддержкой и ROCm доступен → используется AMD GPU
- Иначе → используется CPU

После запуска сервер будет доступен по адресу `http://localhost:20001`. Документация API в интерактивном виде: [Swagger UI](http://localhost:20001/docs) и [ReDoc](http://localhost:20001/redoc); схема OpenAPI: `http://localhost:20001/openapi.json`.

### Локальное обучение с GPU

#### NVIDIA CUDA

Для использования NVIDIA CUDA при локальном запуске (не в Docker):

1. Создайте виртуальное окружение:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# или
source .venv/bin/activate  # Linux/Mac
```

2. Установите torch с CUDA поддержкой (принудительно, чтобы избежать кэша CPU версии):
```bash
# Сначала удалите CPU версию, если установлена
pip uninstall torch -y

# Очистите кэш pip (важно!)
pip cache purge

# Установите torch с CUDA версии 2.6+ (без кэша, чтобы гарантировать CUDA версию)
# Используем CUDA 12.4, так как torch 2.6 доступен для cu124, но не для cu121
pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# Проверьте установку
python -c "import torch; print('Version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
# Должно показать версию БЕЗ "+cpu" (например, "2.6.0+cu124") и CUDA available: True
# ВАЖНО: Модель google/embeddinggemma-300M требует torch>=2.6
```

3. Установите остальные зависимости:
```bash
pip install -r requirements-cuda.txt
```

**Примечание:** `requirements-cuda.txt` использует CUDA 12.4 (cu124), так как torch 2.6+ доступен только для cu124, cu118 или cu126, но не для cu121. RTX 2070 и другие современные GPU поддерживают CUDA 12.4.

4. Запустите сервер:
```bash
python -m commands_classifier.cli serve
```

Сервер автоматически определит доступность CUDA и будет использовать GPU, если он доступен.

#### AMD ROCm

Для использования AMD GPU через ROCm при локальном запуске на Windows (не в Docker):

**Важно:** AMD ROCm для PyTorch на Windows требует:
- Python 3.12
- Совместимую AMD видеокарту или процессор (см. [список совместимых устройств](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-1-1.html))
- Установленный драйвер AMD Software: PyTorch on Windows Edition 7.1.1

1. Установите драйвер AMD:
   - Скачайте драйвер с [официальной страницы AMD](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-1-1.html)
   - Установите драйвер (рекомендуется удалить старые драйверы перед установкой)

2. Создайте виртуальное окружение с Python 3.12:
```bash
python3.12 -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
# или
.venv\Scripts\activate.bat  # Windows CMD
```

3. Установите ROCm SDK (выполните перед установкой PyTorch):

**Для CMD:**
```cmd
pip install --no-cache-dir ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_devel-0.1.dev0-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm-0.1.dev0.tar.gz
```

**Для PowerShell:**
```powershell
pip install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_devel-0.1.dev0-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm-0.1.dev0.tar.gz
```

4. Установите PyTorch с поддержкой ROCm:

**Для CMD:**
```cmd
pip install --no-cache-dir ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torch-2.9.0+rocmsdk20251116-cp312-cp312-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchaudio-2.9.0+rocmsdk20251116-cp312-cp312-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchvision-0.24.0+rocmsdk20251116-cp312-cp312-win_amd64.whl
```

**Для PowerShell:**
```powershell
pip install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torch-2.9.0+rocmsdk20251116-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchaudio-2.9.0+rocmsdk20251116-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchvision-0.24.0+rocmsdk20251116-cp312-cp312-win_amd64.whl
```

5. Проверьте установку:
```bash
python -c "import torch; print('Version:', torch.__version__); print('CUDA available (ROCm):', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Должно показать версию с "+rocmsdk" и `CUDA available (ROCm): True`.

6. Установите остальные зависимости:
```bash
pip install -r requirements-rocm.txt
```

**Примечание:** `requirements-rocm.txt` содержит инструкции по установке, но PyTorch и ROCm SDK нужно устанавливать вручную (см. шаги 3-4 выше).

7. Запустите сервер:
```bash
python -m commands_classifier.cli serve
```

Сервер автоматически определит доступность AMD ROCm и будет использовать GPU, если он доступен.

**Дополнительная информация:**
- [Документация AMD ROCm для Radeon](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
- [Документация AMD ROCm для Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html)
- [Руководство по развертыванию LLM с AMD на Windows](https://gpuopen.com/learn/pytorch-windows-amd-llm-guide/)

**Примечание:** 
- В Docker контейнере всегда используется CPU (контейнер не содержит CUDA/ROCm).
- CUDA доступна только при локальном запуске с установленным PyTorch CUDA.
- AMD ROCm доступен только при локальном запуске с установленным PyTorch ROCm.
- Устройство определяется автоматически при запуске сервера.

### Конфигурация

Настройки сервера хранятся в файле `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 20001

model:
  path: "models/my_model"
  name: "google/embeddinggemma-300M"
  confidence_threshold: 0.5
  cache_dir: "models/.cache"  # Путь для кэширования базовой модели (опционально)

database:
  # Путь к базе данных. Для Docker используйте "db/training_data.db"
  # Для локального запуска можно использовать просто "training_data.db"
  path: "db/training_data.db"
  # Путь к CSV файлу или директории с CSV файлами для автоматической миграции
  # Если указана директория, загружаются все CSV файлы из неё
  csv_migration_path: "data"

training:
  iterations: 20
  epochs: 1
  batch_size: 32
  learning_rate: 2e-5
```

### Эндпоинты API

#### TEI-совместимые эндпоинты

**POST /embed** - Получение эмбеддингов (TEI совместимый)
```bash
curl -X POST "http://localhost:20001/embed" \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["равняйся", "отставить"]}'
```

**GET /health** - Проверка работоспособности
```bash
curl http://localhost:20001/health
```

**GET /metrics** - Метрики сервера
```bash
curl http://localhost:20001/metrics
```

#### Классификация команд

**POST /predict** - Классификация одного текста
```bash
curl -X POST "http://localhost:20001/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "равняйся", "return_confidence": true}'
```

**POST /predict/batch** - Batch классификация
```bash
curl -X POST "http://localhost:20001/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["равняйся", "отставить"], "return_confidence": true}'
```

#### Управление обучением

**POST /train** - Запуск обучения модели в фоновом режиме
```bash
curl -X POST "http://localhost:20001/train" \
  -H "Content-Type: application/json" \
  -d '{"num_iterations": 30, "num_epochs": 2, "batch_size": 32}'
```

**GET /train/status** - Статус обучения
```bash
curl http://localhost:20001/train/status
```

#### Управление обучающими данными

**GET /examples** - Получить все примеры
```bash
curl http://localhost:20001/examples
```

**POST /examples** - Добавить пример
```bash
curl -X POST "http://localhost:20001/examples" \
  -H "Content-Type: application/json" \
  -d '{"text": "новая команда", "command": "new_command"}'
```

**DELETE /examples/{id}** - Удалить пример
```bash
curl -X DELETE "http://localhost:20001/examples/1"
```

**GET /examples/{id}** - Получить пример по ID
```bash
curl http://localhost:20001/examples/1
```

#### Загрузка модели с Hugging Face Hub

**POST /load_from_hf** - Загрузка модели с Hugging Face Hub в фоновом режиме

Если `repo_id` не указан, сервер использует `HF_REPO_ID` из своей конфигурации (переменная окружения или `.env` файл).

```bash
# Использует HF_REPO_ID из конфигурации сервера
curl -X POST "http://localhost:20001/load_from_hf" \
  -H "Content-Type: application/json" \
  -d '{}'

# Или с указанием конкретного репозитория
curl -X POST "http://localhost:20001/load_from_hf" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "username/model-name", "local_dir": "models/my_model"}'
```

**GET /load_from_hf/status** - Статус загрузки модели
```bash
curl http://localhost:20001/load_from_hf/status
```

### База данных

Обучающие данные хранятся в SQLite базе данных (`db/training_data.db` по умолчанию в Docker, `training_data.db` для локального запуска). При первом запуске сервера, если база данных пустая, автоматически выполняется миграция данных из директории или CSV файла (указанного в `config.yaml`). Если указана директория, загружаются все CSV файлы из неё.

Вы можете управлять данными через API эндпоинты `/examples` или напрямую через SQLite.

### Фоновое обучение

Обучение модели выполняется в фоновом режиме, не блокируя работу API сервера. Одновременно может выполняться только одно обучение (блокировка предотвращает конфликты).

Статус обучения можно отслеживать через эндпоинт `/train/status`:

```json
{
  "training_id": "uuid",
  "status": "running|completed|failed",
  "progress": 0.75,
  "error": null,
  "started_at": "2025-01-15T10:30:00",
  "completed_at": null
}
```

После завершения обучения модель автоматически сохраняется и становится доступной для использования.

## Формат датасета

### CSV формат

Файл должен содержать две колонки:
- `text` - текст команды
- `command` - метка команды

Пример:
```csv
text,command
равняйся,align
стань прямо,align
отставить,dismiss
```

### JSON формат

Два варианта:

**Вариант 1: Список объектов**
```json
[
  {"text": "равняйся", "command": "align"},
  {"text": "отставить", "command": "dismiss"}
]
```

**Вариант 2: Объект с массивами**
```json
{
  "text": ["равняйся", "отставить"],
  "command": ["align", "dismiss"]
}
```

## Параметры обучения

- `--iterations` (по умолчанию: 20) - количество итераций контрастного обучения
- `--epochs` (по умолчанию: 1) - количество эпох fine-tuning
- `--batch-size` (по умолчанию: 32) - размер батча (больше = быстрее обучение, но требует больше памяти)
- `--learning-rate` (по умолчанию: 2e-5) - скорость обучения
**Примечание:** Для ускорения обучения увеличьте `batch_size` в `config.yaml` или через API. Больший batch_size ускорит процесс, но потребует больше оперативной памяти.

## Протестированные модели

- google/embeddinggemma-300M
- deepvk/USER-bge-m3
- RuBert-TinyV2

## Рекомендации

- **Размер датасета**: Для лучших результатов используйте 8-16 примеров на класс
- **Балансировка**: Старайтесь иметь примерно равное количество примеров для каждого класса
- **Качество данных**: Используйте разнообразные формулировки для каждой команды
- **Тестирование**: После обучения протестируйте модель на новых примерах

## CI/CD и автоматическое обучение

Проект включает CI/CD пайплайн ([.github/workflows/deploy.yml](.github/workflows/deploy.yml)) для автоматического обучения и деплоя моделей через GitHub Actions.

### Этапы пайплайна

1. **Job `test`** (ubuntu-latest):
   - Сборка dev-образа **cvc-dev** по [docker-compose.dev.yml](docker-compose.dev.yml) (на базе cvc-api + ruff)
   - Линтинг в контейнере: **ruff** `check . --output-format=github`
   - Тесты в контейнере: **pytest** с отчётом покрытия (`--cov=commands_classifier --cov-report=term-missing`)
   - При падении линта или тестов пайплайн останавливается

2. **Job `train-and-publish`** (self-hosted runner с GPU):
   - Запускается только после успешного прохождения `test`
   - Подготовка окружения, запуск контейнера с CUDA, обучение модели через API
   - Загрузка обученной модели на Hugging Face Hub

#### Когда запускается обучение

По умолчанию при каждом push выполняются только **тесты** (линт + pytest). Job «Train model and publish» запускается только в двух случаях:

- **Метка в коммите:** добавьте в сообщение коммита строку `[retrain]` — тогда после успешных тестов запустится обучение и публикация в Hugging Face.
- **Ручной запуск:** в GitHub → Actions → «ML Pipeline - Train and Publish» → «Run workflow». Появится опция «Запустить обучение и публикацию модели в HF» (по умолчанию включена; можно снять галочку, чтобы выполнить только тесты).

Подробная настройка: [docs/cicd_setup.md](docs/cicd_setup.md).

### Настройка self-hosted runner для GPU

Для использования GPU в CI/CD необходимо настроить self-hosted runner на машине с NVIDIA GPU.

#### 1. Установка GitHub Actions runner

```bash
# Перейдите в репозиторий на GitHub
# Settings → Actions → Runners → New self-hosted runner
# Следуйте инструкциям для Linux

# Пример команд (замените на актуальные из GitHub):
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
./config.sh --url https://github.com/YOUR_USERNAME/YOUR_REPO --token YOUR_TOKEN
sudo ./svc.sh install
sudo ./svc.sh start
```

#### 2. Настройка прав доступа к Docker

Добавьте пользователя runner в группу docker:

```bash
# Определите пользователя под которым работает runner (обычно это пользователь, который запустил config.sh)
# Если запускали через sudo, это может быть root
sudo usermod -aG docker $USER  # или имя пользователя runner
# Перезапустите runner после изменения групп
sudo ./svc.sh stop
sudo ./svc.sh start
```

#### 3. Установка NVIDIA Container Runtime

**Важно:** Для использования GPU в Docker контейнерах необходимо установить NVIDIA Container Runtime.

**Проверка текущего состояния:**

```bash
# 1. Проверка NVIDIA драйверов
nvidia-smi

# 2. Проверка поддержки NVIDIA в Docker
docker info | grep -i nvidia

# 3. Проверка что NVIDIA Container Runtime работает
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

Если команда из пункта 3 выполняется без ошибок и показывает информацию о GPU — NVIDIA Container Runtime установлен и работает.

**Установка NVIDIA Container Runtime:**

Если получили ошибку `could not select device driver "nvidia"`, выполните:

```bash
# Определяем дистрибутив
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# Добавляем репозиторий NVIDIA
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Устанавливаем
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Перезапускаем Docker
sudo systemctl restart docker

# Проверяем установку
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

#### 4. Настройка GitHub Secrets

В репозитории GitHub перейдите в:
**Settings → Secrets and variables → Actions → New repository secret**

Добавьте следующие секреты:

- **HF_TOKEN** - токен Hugging Face (обязательно)
  - Получите на https://huggingface.co/settings/tokens
  - Формат: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

- **HF_REPO_ID** - ID репозитория на Hugging Face для загрузки/выгрузки моделей (обязательно)
  - Формат: `username/model-name`
  - Пример: `your-username/cvc-commands-classifier`

- **NUM_ITERATIONS**, **NUM_EPOCHS**, **BATCH_SIZE**, **LEARNING_RATE** (опционально)
  - Для переопределения параметров обучения из config.yaml

#### 5. Проверка работы

После настройки workflow автоматически запускается при push в ветки `main`, `master` или `dev`. Также можно запустить вручную через **Actions → ML Pipeline - Train and Deploy → Run workflow**.

Подробная документация: [docs/cicd_setup.md](docs/cicd_setup.md)

## Структура проекта

```
CVC/
├── requirements-docker.txt     # Зависимости для Docker + тесты (CPU)
├── requirements-cuda.txt       # Зависимости для NVIDIA CUDA
├── requirements-rocm.txt       # Зависимости для AMD ROCm
├── README.md
├── config.yaml                 # Конфигурация сервера
├── commands_classifier/
│   ├── __init__.py
│   ├── model.py                 # Класс CommandsClassifier
│   ├── dataset.py               # Утилиты для загрузки датасетов
│   ├── db.py                    # Работа с SQLite базой данных
│   ├── cli.py                   # CLI для запуска сервера
│   ├── client.py                # Консольный клиент для API
│   ├── hf_retry.py              # Retry для вызовов Hugging Face Hub
│   └── api/
│       ├── __init__.py
│       ├── server.py            # FastAPI сервер
│       ├── state.py             # Глобальное состояние
│       ├── training.py         # Менеджер фонового обучения
│       ├── utils.py
│       └── routes/              # Эндпоинты API
├── data/                        # CSV датасеты для миграции
├── models/                     # Сохранённые модели (создаётся автоматически)
├── db/                          # SQLite (training_data.db создаётся автоматически)
├── tests/
└── docs/
```


## Лицензия

MIT

