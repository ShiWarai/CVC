"""Эндпоинты для работы с примерами обучения."""

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from commands_classifier.api.state import get_config
from commands_classifier.api.utils import remove_punctuation
from commands_classifier import db

router = APIRouter(tags=["examples"])


# Модели запросов/ответов
class ExampleRequest(BaseModel):
    """Запрос для добавления примера."""
    text: str = Field(..., min_length=1, max_length=1000)
    command: str = Field(..., min_length=1, max_length=100)
    
    @field_validator('text', 'command')
    @classmethod
    def validate_no_control_chars(cls, v: str) -> str:
        """Проверяет, что строка не содержит управляющих символов."""
        if any(ord(c) < 32 and c not in '\n\r\t' for c in v):
            raise ValueError('Строка содержит недопустимые управляющие символы')
        return v


class ExampleResponse(BaseModel):
    """Ответ с информацией о примере."""
    id: int
    text: str
    command: str


@router.get("/examples", response_model=List[ExampleResponse])
async def get_examples():
    """
    Получает все примеры из базы данных.
    
    Returns:
        Список всех примеров
    """
    config = get_config()
    db_path = config["database"]["path"]
    examples = db.get_all_examples(db_path)
    return [ExampleResponse(id=ex[0], text=ex[1], command=ex[2]) for ex in examples]


@router.post("/examples", response_model=ExampleResponse, status_code=201)
async def add_example(request: ExampleRequest):
    """
    Добавляет новый пример в базу данных.
    
    Args:
        request: Данные примера (text, command)
        
    Returns:
        Созданный пример с ID
    """
    config = get_config()
    db_path = config["database"]["path"]
    try:
        # Очищаем знаки препинания из текста перед сохранением
        cleaned_text = remove_punctuation(request.text)
        
        # Проверяем, что после очистки текст не пустой
        if len(cleaned_text) == 0:
            raise HTTPException(status_code=400, detail="Текст после очистки не может быть пустым")
        
        example_id = db.add_example(db_path, cleaned_text, request.command)
        return ExampleResponse(id=example_id, text=cleaned_text, command=request.command)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при добавлении примера: {str(e)}")


@router.delete("/examples/{example_id}")
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
    
    config = get_config()
    db_path = config["database"]["path"]
    deleted = db.delete_example(db_path, example_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Пример с ID {example_id} не найден")
    return {"message": f"Пример {example_id} успешно удален"}


@router.get("/examples/{example_id}", response_model=ExampleResponse)
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
    
    config = get_config()
    db_path = config["database"]["path"]
    example = db.get_example_by_id(db_path, example_id)
    if example is None:
        raise HTTPException(status_code=404, detail=f"Пример с ID {example_id} не найден")
    return ExampleResponse(id=example[0], text=example[1], command=example[2])
