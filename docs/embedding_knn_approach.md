# Embedding + kNN подход для классификации команд

## Обзор

Альтернатива текущему SetFit подходу, позволяющая добавлять новые классы команд без переобучения модели.

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Текущий SetFit                            │
├─────────────────────────────────────────────────────────────┤
│  Текст → [Encoder] → Embedding → [Classification Head] → Класс
│                                   (фиксированное N классов)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Embedding + kNN                           │
├─────────────────────────────────────────────────────────────┤
│  Текст → [Encoder] → Embedding → [kNN поиск в БД] → Класс    │
│                                   (динамическое N классов)   │
└─────────────────────────────────────────────────────────────┘
```

## Сравнение подходов

| Критерий | SetFit | Embedding + kNN |
|----------|--------|-----------------|
| Добавление нового класса | Переобучение модели | Добавить эмбеддинги в БД |
| Время добавления класса | 2-5 минут | Секунды |
| Скорость inference | ~10-50ms | ~50-200ms |
| Точность при большом датасете | Выше | Ниже |
| Few-shot (мало примеров) | Хуже | Лучше |
| Catastrophic forgetting | Есть риск | Нет |

## Компоненты реализации

### 1. Encoder (без изменений)
Используем тот же `google/embeddinggemma-300M` или `deepvk/USER-bge-m3` для генерации эмбеддингов.

### 2. Vector Store (новое)
FAISS индекс для быстрого поиска ближайших соседей:
- `IndexFlatIP` — точный поиск через inner product (cosine similarity после нормализации)
- `IndexIVFFlat` — приближённый поиск для больших датасетов (>100k)

### 3. Label Store (новое)
Хранение меток классов для каждого эмбеддинга:
- SQLite таблица: `(id, embedding_id, label, text, created_at)`
- Или in-memory list синхронизированный с FAISS индексом

### 4. Classifier Logic
```python
def predict(text, k=5):
    embedding = encoder.encode(text)
    distances, indices = faiss_index.search(embedding, k)
    neighbor_labels = [labels[i] for i in indices]
    return majority_vote(neighbor_labels)
```

## Варианты реализации

### Вариант A: Чистый kNN
- Храним все примеры в FAISS
- Классифицируем по k ближайшим соседям
- Простота, но медленнее при большом датасете

### Вариант B: Prototype-based (рекомендуется)
- Для каждого класса храним центроид (mean embedding)
- Классифицируем по ближайшему центроиду
- Быстрее, но менее гибко

### Вариант C: Гибрид SetFit + kNN (оптимально)
- SetFit для основных классов (высокая точность)
- kNN fallback для новых/редких классов
- Периодическое переобучение SetFit

## Схема данных

### Таблица embeddings
```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    label TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- numpy array serialized
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_prototype BOOLEAN DEFAULT FALSE  -- для prototype-based подхода
);
```

### FAISS индекс
```python
# Создание
d = 1024  # размерность эмбеддинга (зависит от модели)
index = faiss.IndexFlatIP(d)  # Inner Product для cosine similarity

# Добавление
embeddings = encoder.encode(texts).astype('float32')
faiss.normalize_L2(embeddings)  # нормализация для cosine
index.add(embeddings)

# Поиск
query_emb = encoder.encode([query]).astype('float32')
faiss.normalize_L2(query_emb)
distances, indices = index.search(query_emb, k=5)
```

## Алгоритм добавления нового класса

```
1. Получить примеры нового класса (тексты)
2. Сгенерировать эмбеддинги через encoder
3. Нормализовать эмбеддинги
4. Добавить в FAISS индекс
5. Сохранить метки в label store
6. (Опционально) Вычислить и сохранить prototype

Время: ~1-2 секунды на 100 примеров
```

## Алгоритм классификации

```
1. Получить текст запроса
2. Сгенерировать эмбеддинг
3. Найти k ближайших соседей в FAISS
4. Получить их метки из label store
5. Голосование большинством (или weighted voting по расстоянию)
6. Если confidence < threshold → "unknown"

Время: ~50-100ms
```

## Оптимизации

### Для скорости inference
- Использовать `IndexIVFFlat` вместо `IndexFlatIP` при >10k примеров
- Кэшировать частые запросы
- Batching для нескольких запросов

### Для точности
- Weighted voting (ближние соседи важнее)
- Увеличить k для классов с малым количеством примеров
- Использовать prototype + exemplars гибрид

### Для памяти
- Квантизация эмбеддингов (float32 → int8)
- Product Quantization (PQ) в FAISS
- Хранить только prototypes для больших классов

## Зависимости

```
faiss-cpu>=1.7.0  # или faiss-gpu для GPU
numpy>=1.20.0
sentence-transformers>=2.2.0  # уже есть
```

## Риски и митигация

| Риск | Митигация |
|------|-----------|
| Медленный inference при большом датасете | Использовать IVF индексы, prototypes |
| Низкая точность при малом k | Подобрать оптимальный k через валидацию |
| Дисбаланс классов | Weighted voting, oversampling |
| Drift эмбеддингов при смене encoder | Пересчитать все эмбеддинги |

## План внедрения

### Фаза 1: MVP
- [ ] Добавить FAISS в зависимости
- [ ] Создать `EmbeddingKNNClassifier` класс
- [ ] Реализовать fit/predict методы
- [ ] Тесты на текущем датасете

### Фаза 2: Интеграция
- [ ] API endpoint для добавления класса без переобучения
- [ ] Персистентность FAISS индекса (save/load)
- [ ] CLI команды для управления

### Фаза 3: Гибрид
- [ ] Fallback на kNN когда SetFit даёт low confidence
- [ ] Автоматическое переобучение SetFit по расписанию
- [ ] A/B тестирование подходов

## Ссылки

- [FAISS Tutorial (Pinecone)](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)
- [Embedding Is (Almost) All You Need (2025)](https://arxiv.org/abs/2508.04757)
- [Semantic Drift Compensation for Class-Incremental Learning](https://arxiv.org/abs/2004.00440)
- [Incremental Few-shot Text Classification](https://arxiv.org/abs/2104.11882)
