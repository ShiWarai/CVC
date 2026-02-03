# Как запустить тесты вручную для Pull Request

## Автоматический запуск тестов

После изменений в workflow файле, тесты теперь запускаются автоматически:

1. **При создании Pull Request** к веткам `main`, `master` или `dev`
2. **При каждом push** в ветку с открытым PR
3. **При push** напрямую в ветки `main`, `master` или `dev`

## Ручной запуск тестов (workflow_dispatch)

Если вам нужно запустить тесты вручную (например, для перезапуска после временного сбоя), выполните следующие шаги:

### Шаг 1: Перейдите в Actions

1. Откройте репозиторий на GitHub: https://github.com/ShiWarai/CVC
2. Перейдите на вкладку **Actions** (в верхнем меню репозитория)

### Шаг 2: Выберите workflow

1. В левой боковой панели найдите workflow **"ML Pipeline - Train and Publish"**
2. Кликните на него

### Шаг 3: Запустите workflow

1. Справа вверху найдите кнопку **"Run workflow"** (серая кнопка)
2. Кликните на неё - откроется выпадающее меню
3. Выберите ветку, на которой хотите запустить тесты (например, вашу PR ветку)
4. Настройте параметры (опционально):
   - **Run training**: снимите галочку, если НЕ хотите запускать обучение модели (только тесты)
   - **db_path**: оставьте по умолчанию или укажите путь к другой базе данных
5. Нажмите зелёную кнопку **"Run workflow"**

### Что будет выполнено

При ручном запуске:

1. **Job "test"** (всегда):
   - Сборка Docker образов
   - Линтинг кода с помощью ruff
   - Запуск pytest тестов

2. **Job "train-and-publish"** (только если включен параметр "Run training"):
   - Обучение модели
   - Загрузка модели на Hugging Face Hub

3. **Уведомления в Telegram** (если настроены)

## Просмотр результатов

После запуска:

1. Вы увидите новый workflow run в списке
2. Кликните на него для просмотра деталей
3. В каждом job вы можете развернуть шаги и посмотреть логи
4. Статус отобразится:
   - ✅ Зелёная галочка - всё прошло успешно
   - ❌ Красный крестик - есть ошибки
   - 🟡 Жёлтый кружок - в процессе выполнения

## Просмотр статуса тестов в PR

В самом Pull Request:

1. Внизу страницы PR вы увидите секцию **"Checks"**
2. Там отображается статус всех workflow runs
3. Кликните на **"Details"** рядом с любым check, чтобы посмотреть логи

## Локальный запуск тестов

Если вы хотите запустить тесты локально перед push:

```bash
# Сборка образов
docker compose -f docker-compose.yml build cvc-api
docker compose -f docker-compose.yml -f docker-compose.dev.yml build cvc-dev

# Линтинг
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm cvc-dev ruff check .

# Тесты
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm cvc-dev pytest tests/ -v --tb=short --cov=commands_classifier --cov-report=term-missing
```

## Полезные ссылки

- [GitHub Actions документация](https://docs.github.com/en/actions/using-workflows/manually-running-a-workflow)
- [Настройка CI/CD](cicd_setup.md)
- [README - раздел CI/CD](../README.md#cicd)
