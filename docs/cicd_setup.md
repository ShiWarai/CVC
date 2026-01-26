# Настройка CI/CD для автоматического деплоя моделей

## Обзор

Этот документ описывает настройку CI/CD пайплайна для автоматического:
1. Pull нового датасета из Git
2. Запуска инференса на GPU-машине с Ubuntu и CUDA
3. Упаковки модели в tar.gz
4. Отправки модели на production-сервер

## Шаг 1: Создание SSH-ключа для CI/CD

Для генерации SSH-ключа используйте скрипт:

```bash
bash scripts/generate_ssh_key.sh
```

Скрипт создаст ключ в `~/.cvc_ssh_keys/deploy_key` (вне репозитория) и выведет публичный ключ для добавления на production-сервер.

**Важно:** SSH-ключи НЕ должны храниться в репозитории. Они должны находиться:
- На GPU-машине (где запускается CI/CD): `~/.cvc_ssh_keys/deploy_key`
- Или в GitHub Secrets (для использования в workflow)

Если ключ уже существует, для получения публичного ключа:
```bash
cat ~/.cvc_ssh_keys/deploy_key.pub
```

## Шаг 2: Настройка Hugging Face

Модели загружаются на Hugging Face Hub вместо прямого деплоя на сервер. Это упрощает процесс и обеспечивает версионирование.

**Подробная инструкция:** См. [docs/HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md)

Кратко:
1. Создайте репозиторий на https://huggingface.co/new
2. Получите токен с правами Write на https://huggingface.co/settings/tokens
3. Добавьте секреты в GitHub (см. Шаг 3)

## Шаг 3: Настройка GitHub Secrets

В репозитории GitHub перейдите в:
**Settings → Secrets and variables → Actions → New repository secret**

Добавьте следующие секреты:

1. **HF_TOKEN** - токен Hugging Face (обязательно)
   ```
   Получите на https://huggingface.co/settings/tokens
   Формат: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

2. **HF_REPO_ID** - ID репозитория на Hugging Face (обязательно)
   ```
   Формат: username/model-name
   Пример: your-username/cvc-commands-classifier
   ```

3. **NUM_ITERATIONS**, **NUM_EPOCHS**, **BATCH_SIZE**, **LEARNING_RATE** (опционально)
   ```
   Для переопределения параметров обучения из config.yaml
   ```

## Шаг 4: Настройка self-hosted runner на GPU-машине

Если вы используете GitHub Actions с self-hosted runner:

1. На GPU-машине (Ubuntu с CUDA) установите GitHub Actions runner:
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

2. Убедитесь, что runner имеет доступ к CUDA:
   ```bash
   # Проверьте доступность GPU
   nvidia-smi
   
   # Убедитесь, что Docker имеет доступ к GPU (если используете Docker)
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

3. Установите NVIDIA Container Toolkit (если используете Docker):
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## Шаг 5: Настройка скриптов для обучения и упаковки

Скрипты уже созданы в директории `scripts/`:
- `scripts/generate_ssh_key.sh` - генерация SSH-ключа (опционально, если нужен SSH для других задач)
- `scripts/train_via_api.py` - запуск обучения модели через API
- `scripts/upload_to_hf.py` - загрузка модели на Hugging Face Hub

## Шаг 6: Проверка workflow

Workflow файл находится в `.github/workflows/deploy.yml`. Он автоматически:
- Запускается при push в ветку `main`
- Pull'ит последние изменения из репозитория
- Запускает обучение модели на GPU через docker-compose
- Загружает модель на Hugging Face Hub
- Модель становится доступной по адресу: `https://huggingface.co/your-username/model-name`

## Тестирование

Для тестирования загрузки на Hugging Face:

```bash
# На GPU-машине (где будет запускаться CI/CD)
export HF_TOKEN="your_token_here"
export HF_REPO_ID="your-username/model-name"
python scripts/upload_to_hf.py
```

Если загрузка успешна, модель появится на https://huggingface.co/your-username/model-name

## Устранение неполадок

### Проблема: Модель не загружается на Hugging Face
- Проверьте, что `HF_TOKEN` установлен в GitHub Secrets
- Проверьте, что `HF_REPO_ID` указан правильно (формат: `username/model-name`)
- Убедитесь, что токен имеет права Write
- Проверьте, что репозиторий существует на Hugging Face

### Проблема: GPU не доступен в runner
- Проверьте `nvidia-smi` на GPU-машине
- Убедитесь, что пользователь runner имеет доступ к GPU
- Если используете Docker, проверьте установку NVIDIA Container Toolkit

### Проблема: Модель не скачивается на production-сервере
- Убедитесь, что на production-сервере установлен `huggingface-hub`
- Проверьте, что токен HF доступен на production-сервере (если репозиторий приватный)
- Проверьте правильность `HF_REPO_ID` при скачивании
