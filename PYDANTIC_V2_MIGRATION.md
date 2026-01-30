# Руководство по миграции на Pydantic v2

## Проблема: "Что-то пошло не так?"

### Причина

Кодовая база CVC использовала устаревший синтаксис Pydantic v1 при установленном Pydantic v2.12.5, что вызывало предупреждения об устаревании:

```
PydanticDeprecatedSince20: Валидаторы в стиле Pydantic V1 `@validator` устарели. 
Необходимо мигрировать на валидаторы в стиле Pydantic V2 `@field_validator`, 
см. руководство по миграции для получения дополнительной информации. 
Устарело в Pydantic V2.0, будет удалено в V3.0.
```

## Решение

Выполнена миграция всех валидаторов Pydantic с v1 на v2 API.

## Внесённые изменения

### 1. Миграция декораторов

**Было (v1):**
```python
from pydantic import validator

@validator('field_name')
def validate_field(cls, v):
    # логика валидации
    return v
```

**Стало (v2):**
```python
from pydantic import field_validator

@field_validator('field_name')
@classmethod
def validate_field(cls, v: str) -> str:
    # логика валидации
    return v
```

### 2. Ограничения полей для списков

**Было (v1):**
```python
from pydantic import Field

inputs: List[str] = Field(..., min_items=1, max_items=100)
```

**Стало (v2):**
```python
from pydantic import Field

inputs: List[str] = Field(..., min_length=1, max_length=100)
```

## Изменённые файлы

### 1. `commands_classifier/api/routes/examples.py`

**Изменения:**
- Заменено `@validator` на `@field_validator`
- Добавлен декоратор `@classmethod`
- Добавлены типы: `(cls, v: str) -> str`

**Валидатор:**
- `validate_no_control_chars` - проверяет поля text и command

### 2. `commands_classifier/api/routes/predict.py`

**Изменения:**
- Заменено `@validator` на `@field_validator`
- Добавлен декоратор `@classmethod`
- Изменено `min_items` → `min_length` и `max_items` → `max_length`
- Добавлены типы: `(cls, v: List[str]) -> List[str]`

**Валидаторы:**
- `EmbedRequest.validate_inputs` - проверяет входные данные для эмбеддингов
- `PredictBatchRequest.validate_texts` - проверяет тексты для пакетного предсказания

### 3. `commands_classifier/api/routes/load_from_hf.py`

**Изменения:**
- Заменено `@validator` на `@field_validator`
- Добавлен декоратор `@classmethod`
- Добавлены типы: `(cls, v: Optional[str]) -> Optional[str]`

**Валидаторы:**
- `LoadFromHfRequest.validate_repo_id` - проверяет формат ID репозитория Hugging Face
- `LoadFromHfRequest.validate_local_dir` - проверяет путь к локальной директории

## Ключевые различия между v1 и v2

| Функция | Pydantic v1 | Pydantic v2 |
|---------|-------------|-------------|
| Декоратор валидатора | `@validator` | `@field_validator` |
| Метод класса | Опционально | **Обязательно** `@classmethod` |
| Типизация | Опционально | Рекомендуется |
| Мин/макс для списков | `min_items`, `max_items` | `min_length`, `max_length` |
| Импорт | `from pydantic import validator` | `from pydantic import field_validator` |

## Преимущества миграции

1. ✅ **Нет предупреждений об устаревании** - Код готов к Pydantic v3
2. ✅ **Улучшенная типобезопасность** - Типы улучшают поддержку IDE и выявляют ошибки
3. ✅ **Повышенная производительность** - Pydantic v2 использует ядро на Rust для лучшей производительности
4. ✅ **Улучшенные сообщения об ошибках** - v2 предоставляет более детальные сообщения об ошибках валидации
5. ✅ **Современные лучшие практики** - Соответствует текущим стандартам Pydantic

## Тестирование

Все валидаторы были протестированы для обеспечения:
- ✅ Отсутствие синтаксических ошибок
- ✅ Отсутствие предупреждений об устаревании
- ✅ Логика валидации работает корректно
- ✅ Некорректный ввод корректно отклоняется
- ✅ Корректный ввод принимается

### Результаты тестирования

```
✓ Валидация ExampleRequest работает
✓ Валидация EmbedRequest работает
✓ Валидация PredictRequest работает
✓ Валидация PredictBatchRequest работает
✓ Валидация LoadFromHfRequest работает
✓ Валидация управляющих символов работает
✓ Валидация пустого списка работает
✅ Предупреждения об устаревании не обнаружены
```

## Чек-лист миграции

Для будущих миграций на Pydantic v2:

- [ ] Заменить `@validator` на `@field_validator`
- [ ] Добавить декоратор `@classmethod` ко всем валидаторам
- [ ] Добавить типизацию к методам валидаторов
- [ ] Изменить `min_items`/`max_items` на `min_length`/`max_length` для последовательностей
- [ ] Обновить импорты: `validator` → `field_validator`
- [ ] Протестировать работу валидаторов
- [ ] Проверить отсутствие предупреждений об устаревании

## Справочные материалы

- [Руководство по миграции Pydantic v2](https://docs.pydantic.dev/latest/migration/)
- [Документация по валидаторам полей](https://docs.pydantic.dev/latest/concepts/validators/#field-validators)
- [Заметки о выпуске Pydantic v2](https://docs.pydantic.dev/latest/changelog/)

## Обратная совместимость

Все изменения сохраняют ту же логику и поведение валидации. Миграция является чисто синтаксической и не изменяет:
- Что валидируется
- Как генерируются ошибки валидации
- Сообщения об ошибках валидации
- Поведение API

Это **обратно совместимое изменение**, которое только обновляет внутреннюю реализацию для использования лучших практик Pydantic v2.
