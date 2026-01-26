# CVC - Classification of Voice Commands

Мини-сервис для классификации голосовых команд с использованием SetFit (few-shot learning). Позволяет обучать модель на малом датасете (8-16 примеров на класс) и классифицировать текстовые команды.

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
pip install -r requirements.txt
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
3. Создайте файл `.env` в корне проекта и добавьте токен:

```bash
HF_TOKEN=your_token_here
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

# Отредактируйте .env и добавьте ваш Hugging Face токен
# HF_TOKEN=your_token_here
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

После запуска сервер будет доступен по адресу `http://localhost:20001`. Документация API (Swagger UI) доступна по адресу `http://localhost:20001/docs`.

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

**POST /download** - Загрузка модели с Hugging Face Hub в фоновом режиме
```bash
curl -X POST "http://localhost:20001/download" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "username/model-name", "local_dir": "models/my_model"}'
```

**GET /download/status** - Статус загрузки модели
```bash
curl http://localhost:20001/download/status
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

## Структура проекта

```
CVC/
├── requirements.txt              # Зависимости
├── README.md                     # Документация
├── config.yaml                   # Конфигурация сервера
├── commands_classifier/
│   ├── __init__.py
│   ├── model.py                 # Класс CommandsClassifier
│   ├── dataset.py               # Утилиты для загрузки датасетов
│   ├── cli.py                   # CLI для запуска сервера
│   ├── client.py                # Консольный клиент для API
│   ├── db.py                    # Работа с SQLite базой данных
│   └── api/
│       ├── __init__.py
│       ├── server.py            # FastAPI сервер
│       └── training.py          # Менеджер фонового обучения
├── data/
│   └── commands_example.csv     # Пример датасета
├── models/                      # Сохраненные модели (создается автоматически)
└── db/
    └── training_data.db         # SQLite база данных (создается автоматически)
```


## Лицензия

MIT

