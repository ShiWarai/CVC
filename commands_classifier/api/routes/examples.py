"""Эндпоинты для работы с примерами обучения."""

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from commands_classifier.api.state import get_config
from commands_classifier.api.utils import remove_punctuation
from commands_classifier import db

router = APIRouter(tags=["examples"])


# Модели запросов/ответов
class ExampleRequest(BaseModel):
    """Запрос для добавления примера."""
    text: str
    command: str


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
        example_id = db.add_example(db_path, cleaned_text, request.command)
        return ExampleResponse(id=example_id, text=cleaned_text, command=request.command)
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
    config = get_config()
    db_path = config["database"]["path"]
    example = db.get_example_by_id(db_path, example_id)
    if example is None:
        raise HTTPException(status_code=404, detail=f"Пример с ID {example_id} не найден")
    return ExampleResponse(id=example[0], text=example[1], command=example[2])
