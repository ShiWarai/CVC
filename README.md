# Commands Classifier

Мини-сервис для классификации текста в команды с использованием SetFit (few-shot learning). Позволяет обучать модель на малом датасете (8-16 примеров на класс) и классифицировать текстовые команды.

## Особенности

- **Few-shot learning**: Обучение на 5-50 примерах на класс
- **Поддержка русского языка**: Использует multilingual модели
- **Простой CLI интерфейс**: Легко использовать из командной строки
- **Гибкий формат датасета**: Поддержка CSV и JSON

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Подготовка датасета

Создайте CSV файл с колонками `text` и `command`:

```csv
text,command
равняйся,align
стань прямо,align
отставить,dismiss
лежать,lie_down
встать,stand_up
шагом марш,march
```

Или JSON файл:

```json
[
  {"text": "равняйся", "command": "align"},
  {"text": "стань прямо", "command": "align"},
  {"text": "отставить", "command": "dismiss"}
]
```

Пример датасета находится в `data/commands_example.csv`.

### 2. Обучение модели

```bash
python -m commands_classifier.cli train \
  --dataset data/commands_example.csv \
  --output models/my_model
```

Дополнительные параметры обучения:

```bash
python -m commands_classifier.cli train \
  --dataset data/commands_example.csv \
  --output models/my_model \
  --iterations 30 \
  --epochs 2 \
  --batch-size 16 \
  --learning-rate 2e-5
```

### 3. Классификация текста

Классификация одного текста:

```bash
python -m commands_classifier.cli predict \
  --model models/my_model \
  --text "равняйся"
```

Batch классификация (файл с текстами, по одному на строку):

```bash
python -m commands_classifier.cli predict \
  --model models/my_model \
  --file commands.txt
```

## Использование как библиотеки

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
- `--batch-size` (по умолчанию: 16) - размер батча
- `--learning-rate` (по умолчанию: 2e-5) - скорость обучения
- `--model-name` (по умолчанию: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) - предобученная модель

## Рекомендации

- **Размер датасета**: Для лучших результатов используйте 8-16 примеров на класс
- **Балансировка**: Старайтесь иметь примерно равное количество примеров для каждого класса
- **Качество данных**: Используйте разнообразные формулировки для каждой команды
- **Тестирование**: После обучения протестируйте модель на новых примерах

## Структура проекта

```
commands_transformer/
├── requirements.txt              # Зависимости
├── README.md                     # Документация
├── commands_classifier/
│   ├── __init__.py
│   ├── model.py                 # Класс CommandsClassifier
│   ├── dataset.py               # Утилиты для загрузки датасетов
│   └── cli.py                   # CLI интерфейс
├── data/
│   └── commands_example.csv     # Пример датасета
└── models/                      # Сохраненные модели (создается автоматически)
```

## Зависимости

- `setfit>=0.7.0` - few-shot learning фреймворк
- `sentence-transformers>=2.2.0` - эмбеддинги для русского языка
- `pandas>=1.5.0` - работа с CSV
- `scikit-learn>=1.0.0` - метрики и утилиты

## Лицензия

MIT

