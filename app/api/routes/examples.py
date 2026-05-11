"""Эндпоинты для работы с примерами обучения."""

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.adapters import persistence as db
from app.api.state import get_config, get_examples_use_case
from app.api.utils import remove_punctuation

router = APIRouter(tags=["examples"])


# Модели запросов/ответов
class ExampleRequest(BaseModel):
    """Запрос для добавления примера."""

    text: str = Field(..., min_length=1, max_length=1000)
    command: str = Field(..., min_length=1, max_length=100)

    @field_validator("text", "command")
    @classmethod
    def validate_no_control_chars(cls, v: str) -> str:
        """Проверяет, что строка не содержит управляющих символов."""
        if any(ord(c) < 32 and c not in "\n\r\t" for c in v):
            raise ValueError("Строка содержит недопустимые управляющие символы")
        return v


class ExampleResponse(BaseModel):
    """Ответ с информацией о примере."""

    id: int
    text: str
    command: str


@router.get("/v1/examples", response_model=List[ExampleResponse])
async def get_examples():
    """
    Получает все примеры из базы данных.

    Returns:
        Список всех примеров
    """
    examples_uc = get_examples_use_case()
    if examples_uc is not None:
        examples = examples_uc.get_all()
    else:
        config = get_config()
        examples = db.get_all_examples(config["database"]["path"])
    return [ExampleResponse(id=ex[0], text=ex[1], command=ex[2]) for ex in examples]


@router.post("/v1/examples", response_model=ExampleResponse, status_code=201)
async def add_example(request: ExampleRequest):
    """
    Добавляет новый пример в базу данных.

    Args:
        request: Данные примера (text, command)

    Returns:
        Созданный пример с ID
    """
    examples_uc = get_examples_use_case()
    try:
        if examples_uc is not None:
            cleaned_text = remove_punctuation(request.text)
            if len(cleaned_text) == 0:
                raise HTTPException(
                    status_code=400, detail="После очистки строка оказалась пустой"
                )
            example_id = examples_uc.add(request.text, request.command)
            return ExampleResponse(id=example_id, text=cleaned_text, command=request.command)
        config = get_config()
        db_path = config["database"]["path"]
        cleaned_text = remove_punctuation(request.text)
        if len(cleaned_text) == 0:
            raise HTTPException(status_code=400, detail="Текст после очистки не может быть пустым")

        example_id = db.add_example(db_path, cleaned_text, request.command)
        return ExampleResponse(id=example_id, text=cleaned_text, command=request.command)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при добавлении примера: {str(e)}")


@router.delete("/v1/examples/{example_id}")
async def delete_example(example_id: int):
    """
    Удаляет пример по ID.

    Args:
        example_id: ID примера для удаления

    Returns:
        Сообщение об успешном удалении
    """
    # Валидация ID
    if example_id <= 0:
        raise HTTPException(status_code=400, detail="ID примера должен быть положительным числом")

    examples_uc = get_examples_use_case()
    if examples_uc is not None:
        deleted = examples_uc.delete(example_id)
    else:
        config = get_config()
        deleted = db.delete_example(config["database"]["path"], example_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Пример с ID {example_id} не найден")
    return {"message": f"Пример {example_id} успешно удален"}


@router.get("/v1/examples/{example_id}", response_model=ExampleResponse)
async def get_example(example_id: int):
    """
    Получает пример по ID.

    Args:
        example_id: ID примера

    Returns:
        Пример
    """
    # Валидация ID
    if example_id <= 0:
        raise HTTPException(status_code=400, detail="ID примера должен быть положительным числом")

    examples_uc = get_examples_use_case()
    if examples_uc is not None:
        example = examples_uc.get_by_id(example_id)
    else:
        config = get_config()
        example = db.get_example_by_id(config["database"]["path"], example_id)
    if example is None:
        raise HTTPException(status_code=404, detail=f"Пример с ID {example_id} не найден")
    return ExampleResponse(id=example[0], text=example[1], command=example[2])
