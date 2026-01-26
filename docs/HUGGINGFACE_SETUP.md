# Настройка Hugging Face для CI/CD

## Краткая инструкция

### 1. Создайте репозиторий на Hugging Face

1. Зайдите на [huggingface.co](https://huggingface.co) и авторизуйтесь
2. Перейдите в [New Model](https://huggingface.co/new)
3. Заполните форму:
   - **Model name**: например, `cvc-commands-classifier` или `your-username/cvc-model`
   - **Visibility**: выберите **Private** (для приватных моделей) или **Public** (для публичных)
   - **License**: выберите подходящую лицензию (например, MIT)
   - **Library**: выберите `sentence-transformers` или `other`
4. Нажмите **Create repository**

### 2. Получите токен доступа

1. Перейдите в [Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Нажмите **New token**
3. Заполните:
   - **Name**: например, `ci-cd-token`
   - **Type**: выберите **Write** (нужен для загрузки моделей)
4. Нажмите **Generate token**
5. **Скопируйте токен** (он показывается только один раз!)

### 3. Настройте GitHub Secrets

В репозитории GitHub перейдите в:
**Settings → Secrets and variables → Actions → New repository secret**

Добавьте следующие секреты:

1. **HF_TOKEN** - токен Hugging Face (из шага 2)
   ```
   hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

2. **HF_REPO_ID** - ID репозитория на Hugging Face
   ```
   username/model-name
   ```
   Например: `your-username/cvc-commands-classifier`

### 4. Проверка

После настройки пайплайн автоматически:
- Обучит модель
- Загрузит её на Hugging Face Hub
- Модель будет доступна по адресу: `https://huggingface.co/your-username/model-name`

## Использование модели на production-сервере

На production-сервере модель можно скачать так:

```python
from huggingface_hub import snapshot_download
import os

# Установите токен (если репозиторий приватный)
os.environ["HF_TOKEN"] = "your_token_here"

# Скачайте модель
snapshot_download(
    repo_id="your-username/model-name",
    local_dir="./models/model_name",
    token=os.getenv("HF_TOKEN")
)
```

Или через командную строку:

```bash
# Установите huggingface-hub
pip install huggingface-hub

# Скачайте модель (для приватных репозиториев нужен токен)
huggingface-cli download your-username/model-name --local-dir ./models/model_name --token $HF_TOKEN
```

## Преимущества использования Hugging Face

- ✅ Версионирование моделей (как Git для моделей)
- ✅ Автоматическое хранение и доступ
- ✅ Не нужен SSH/SCP для деплоя
- ✅ Метаданные и документация модели
- ✅ Интеграция с другими ML инструментами
- ✅ Бесплатно для базового использования

## Альтернатива: Локальное хранение

Если вы предпочитаете локальное хранение (SCP/FTP), можно вернуться к предыдущей версии workflow с SSH деплоем. Но для большинства случаев HF удобнее.
