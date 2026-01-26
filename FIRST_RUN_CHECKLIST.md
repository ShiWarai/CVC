# Чеклист первого запуска CI/CD

## ✅ Что уже должно быть сделано

- [x] Секреты в GitHub: `HF_TOKEN` и `HF_REPO_ID`
- [x] Self-hosted runner запущен на GPU-машине
- [x] Репозиторий создан на Hugging Face

## 🔍 Проверка перед первым запуском

### 1. Проверьте секреты в GitHub

Убедитесь, что в GitHub репозитории установлены секреты:

**Settings → Secrets and variables → Actions**

Должны быть:
- ✅ `HF_TOKEN` - токен Hugging Face с правами Write
- ✅ `HF_REPO_ID` - ID репозитория (например: `username/model-name`)

### 2. Проверьте runner

На GPU-машине проверьте статус runner:

```bash
# Проверьте, что runner запущен
sudo systemctl status actions.runner.*.service

# Или проверьте процессы
ps aux | grep Runner.Listener
```

Runner должен быть в статусе "running" и подключен к GitHub.

### 3. Проверьте Docker и зависимости

На GPU-машине (где запущен runner):

```bash
# Проверьте Docker
docker --version
docker-compose --version

# Проверьте Python
python --version  # Должен быть 3.10 или выше

# Проверьте GPU (если используется CUDA)
nvidia-smi
```

### 4. Проверьте сеть Docker

```bash
# Проверьте наличие сети
docker network ls | grep robot-services-network

# Если сети нет, создайте её
docker network create robot-services-network
```

### 5. Проверьте репозиторий на Hugging Face

1. Зайдите на https://huggingface.co/your-username/model-name
2. Убедитесь, что репозиторий существует
3. Проверьте, что токен имеет доступ (можно попробовать загрузить тестовый файл)

## 🚀 Первый запуск

### Вариант 1: Автоматический запуск

Пайплайн запустится автоматически при push в ветку `main` или `master`:

```bash
# Сделайте любой коммит и push
git add .
git commit -m "Trigger CI/CD pipeline"
git push origin main
```

### Вариант 2: Ручной запуск

1. Зайдите в GitHub репозиторий
2. Перейдите в **Actions**
3. Выберите workflow **ML Pipeline - Train and Deploy**
4. Нажмите **Run workflow**
5. Выберите ветку (обычно `main` или `master`)
6. Нажмите **Run workflow**

## 📊 Мониторинг выполнения

### В GitHub Actions

1. Зайдите в **Actions** в вашем репозитории
2. Выберите запущенный workflow
3. Следите за выполнением шагов:
   - ✅ Checkout code
   - ✅ Pull latest changes
   - ✅ Set up Python
   - ✅ Install Python dependencies
   - ✅ Check Docker and docker-compose
   - ✅ Run training via API
   - ✅ Upload model to Hugging Face

### На GPU-машине

Можно следить за логами runner:

```bash
# Логи runner (путь зависит от установки)
tail -f ~/actions-runner/_diag/Runner_*.log

# Или проверьте логи Docker контейнера
docker-compose logs -f
```

## ✅ Что должно произойти при успешном выполнении

1. **Обучение модели:**
   - Docker контейнер запустится через `docker-compose up -d`
   - Модель обучится через API endpoint `/train`
   - Модель сохранится в `models/panda_commands/` (или путь из config.yaml)

2. **Загрузка на Hugging Face:**
   - Скрипт `upload_to_hf.py` загрузит модель на HF Hub
   - Модель появится в вашем репозитории на HF
   - Будет доступна по адресу: `https://huggingface.co/your-username/model-name`

3. **Результат:**
   - В GitHub Actions будет зелёная галочка ✅
   - На Hugging Face появится новая версия модели
   - Все файлы модели будут загружены

## ❌ Возможные проблемы и решения

### Проблема: Runner не запускается

**Решение:**
```bash
# Перезапустите runner
sudo systemctl restart actions.runner.*.service

# Или переустановите runner (если нужно)
cd ~/actions-runner
./svc.sh stop
./svc.sh uninstall
./config.sh --url <URL> --token <TOKEN>
./svc.sh install
./svc.sh start
```

### Проблема: Ошибка "HF_TOKEN не установлен"

**Решение:**
- Проверьте, что секрет `HF_TOKEN` добавлен в GitHub Secrets
- Убедитесь, что имя секрета точно `HF_TOKEN` (регистр важен)

### Проблема: Ошибка "HF_REPO_ID не установлен"

**Решение:**
- Проверьте, что секрет `HF_REPO_ID` добавлен в GitHub Secrets
- Убедитесь, что формат правильный: `username/model-name` (без `https://huggingface.co/`)

### Проблема: Docker network не найдена

**Решение:**
```bash
docker network create robot-services-network
```

### Проблема: Ошибка при обучении

**Решение:**
- Проверьте логи Docker контейнера: `docker-compose logs`
- Убедитесь, что данные есть в базе данных или CSV файлах
- Проверьте, что `HF_TOKEN` доступен в контейнере (для загрузки базовой модели)

### Проблема: Ошибка при загрузке на HF

**Решение:**
- Проверьте правильность `HF_REPO_ID`
- Убедитесь, что токен имеет права Write
- Проверьте, что репозиторий существует на HF

## 🎯 Следующие шаги после успешного запуска

1. **Проверьте модель на Hugging Face:**
   - Зайдите на https://huggingface.co/your-username/model-name
   - Убедитесь, что все файлы загружены
   - Проверьте структуру репозитория

2. **Протестируйте загрузку модели:**
   - На production-сервере используйте API endpoint `/download`
   - Или скачайте модель вручную для проверки

3. **Настройте автоматические обновления:**
   - Пайплайн будет запускаться автоматически при push в `main`
   - Или настройте расписание (cron) в workflow

## 📝 Полезные команды для отладки

```bash
# Проверка статуса runner
sudo systemctl status actions.runner.*.service

# Логи runner
tail -f ~/actions-runner/_diag/Runner_*.log

# Проверка Docker
docker ps
docker-compose ps

# Логи контейнера
docker-compose logs -f cvc-api

# Проверка модели
ls -la models/panda_commands/

# Тест загрузки на HF (вручную)
export HF_TOKEN="your_token"
export HF_REPO_ID="username/model-name"
python scripts/upload_to_hf.py
```

---

**Готово к запуску!** Сделайте push в `main` или запустите workflow вручную через GitHub Actions.
