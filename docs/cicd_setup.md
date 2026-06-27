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
bash ci/generate_ssh_key.sh
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

4. **TELEGRAM_TOKEN**, **TELEGRAM_TO** (опционально)
   ```
   Для уведомлений в Telegram при падении пайплайна.
   Подробная настройка: см. docs/telegram_notifications.md
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

2. **Настройте права доступа к Docker (ВАЖНО!):**
   
   Вариант A: Добавьте пользователя runner в группу docker (рекомендуется):
   ```bash
   # Определите пользователя, под которым запущен runner
   # Если runner запущен вручную через ./run.sh, это пользователь, который запустил команду
   # Если runner запущен как сервис, проверьте:
   ps aux | grep Runner.Listener
   
   # Добавьте пользователя в группу docker
   # Если runner запущен от пользователя 'user':
   sudo usermod -aG docker user
   
   # Или если используете текущего пользователя:
   sudo usermod -aG docker $USER
   
   # ВАЖНО: После добавления в группу нужно:
   # 1. Выйти из сессии и войти заново (или выполнить newgrp docker)
   # 2. Если runner запущен как сервис - перезапустить:
   sudo systemctl restart actions.runner.*.service
   # 3. Если runner запущен вручную - перезапустить его после выхода/входа
   
   # Проверьте права:
   docker ps  # Должно работать без sudo
   ```
   
3. **Установите NVIDIA Container Runtime (для использования GPU в Docker):**
   
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
   
   Вариант B: Настройте sudo без пароля для команд docker (если вариант A не подходит):
   ```bash
   # Создайте файл sudoers для пользователя runner
   sudo visudo -f /etc/sudoers.d/docker-runner
   
   # Добавьте строку (замените 'user' на имя пользователя runner):
   user ALL=(ALL) NOPASSWD: /usr/bin/docker, /usr/bin/docker-compose, /usr/local/bin/docker-compose
   
   # Сохраните и проверьте:
   sudo docker ps
   ```
   
   **Примечание:** Если runner запущен вручную (через `./run.sh`), после добавления в группу docker:
   - Выйдите из текущей сессии терминала
   - Войдите заново (или выполните `newgrp docker` в текущей сессии)
   - Перезапустите runner: `./run.sh`

3. Убедитесь, что runner имеет доступ к CUDA:
   ```bash
   # Проверьте доступность GPU
   nvidia-smi
   
   # Убедитесь, что Docker имеет доступ к GPU (если используете Docker)
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   # Или с sudo, если права не настроены:
   sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

4. **Установите NVIDIA Container Runtime (если используете Docker):**
   
   **Важно:** Для использования GPU в Docker контейнерах необходимо установить NVIDIA Container Runtime (ранее назывался NVIDIA Container Toolkit).
   
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
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## Шаг 5: Настройка скриптов для обучения и упаковки

Скрипты уже созданы в директории `ci/`:
- `ci/generate_ssh_key.sh` - генерация SSH-ключа (опционально, если нужен SSH для других задач)
- `ci/train_via_api.py` - запуск обучения модели (напрямую, без вызова API)
- `ci/upload_to_hf.py` - загрузка модели на Hugging Face Hub

**API:** все эндпоинты CVC версионированы префиксом `/v1` (например, `/v1/health`, `/v1/train`). Полный список — в [README](../README.md#эндпоинты-api-v1).

## Шаг 6: Проверка workflow

Workflow файл находится в `.github/workflows/deploy.yml`. Кратко:
- При push в `main` / `dev` на **GitHub-hosted** (`ubuntu-latest`) выполняется job **test**: линт и pytest в Docker.
- После успешного pipeline [`.github/workflows/publish.yml`](.github/workflows/publish.yml):
  - **`main`** → prod-образ `ghcr.io/<owner>/cvc-robot-dog:main`;
  - **`dev`** → staging-образ `cvc-robot-dog:dev`.
- Job **Train and Publish** запускается только при `[retrain]` в сообщении коммита или вручную (workflow_dispatch) и выполняется на **self-hosted** с GPU: `docker-compose.cuda.yml` + **`Dockerfile.cuda`**, затем загрузка на Hugging Face Hub.
- Модель становится доступной по адресу: `https://huggingface.co/your-username/model-name`

## Тестирование

Для тестирования загрузки на Hugging Face:

```bash
# На GPU-машине (где будет запускаться CI/CD)
export HF_TOKEN="your_token_here"
export HF_REPO_ID="your-username/model-name"
python ci/upload_to_hf.py
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
- Если используете Docker, проверьте установку NVIDIA Container Runtime:
  ```bash
  docker info | grep -i nvidia
  docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
  ```
- Если получили ошибку `could not select device driver "nvidia"`, установите NVIDIA Container Runtime (см. Шаг 4, пункт 3)

### Проблема: Модель не скачивается на production-сервере
- Убедитесь, что на production-сервере установлен `huggingface-hub`
- Проверьте, что токен HF доступен на production-сервере (если репозиторий приватный)
- Проверьте правильность `HF_REPO_ID` при скачивании
