"""Эндпоинты для обучения и сброса модели."""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.adapters import persistence as db
from app.api.state import get_config, get_training_manager, unload_classifier

router = APIRouter(tags=["training"])


# Модели запросов/ответов
class TrainRequest(BaseModel):
    """Запрос для запуска обучения."""

    num_iterations: Optional[int] = Field(None, ge=1, le=1000)
    num_epochs: Optional[int] = Field(None, ge=1, le=100)
    batch_size: Optional[int] = Field(None, ge=1, le=512)
    learning_rate: Optional[float] = Field(None, gt=0, le=1.0)


class TrainResponse(BaseModel):
    """Ответ на запрос обучения."""

    training_id: str
    message: str


class ResetResponse(BaseModel):
    """Ответ на сброс обучения."""

    message: str
    reset_examples: int
    model_deleted: bool


@router.post("/v1/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    """
    Запускает обучение модели в фоновом режиме.

    Args:
        request: Параметры обучения (опционально, используются значения по умолчанию из config)
        - num_iterations: Количество итераций (по умолчанию из config.yaml)
        - num_epochs: Количество эпох (по умолчанию из config.yaml)
        - batch_size: Размер батча (по умолчанию из config.yaml)
        - learning_rate: Скорость обучения (по умолчанию из config.yaml)

    Returns:
        ID задачи обучения
    """
    training_manager = get_training_manager()
    config = get_config()

    if training_manager is None:
        raise HTTPException(status_code=500, detail="Training manager не инициализирован")

    if training_manager.is_training():
        raise HTTPException(status_code=409, detail="Обучение уже запущено")

    # Используем параметры из запроса или из конфига
    training_config = config["training"]
    num_iterations = request.num_iterations or training_config["iterations"]
    num_epochs = request.num_epochs or training_config["epochs"]
    batch_size = request.batch_size or training_config["batch_size"]
    learning_rate = request.learning_rate or training_config["learning_rate"]

    # Убеждаемся, что все числовые параметры имеют правильный тип
    num_iterations = int(num_iterations)
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)

    try:
        training_id = training_manager.start_training(
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        return TrainResponse(training_id=training_id, message="Обучение запущено в фоновом режиме")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при запуске обучения: {str(e)}")


@router.get("/v1/train/status")
async def get_training_status():
    """
    Возвращает статус текущего обучения.

    Returns:
        Статус обучения (id, status, progress, error, timestamps)
    """
    training_manager = get_training_manager()

    if training_manager is None:
        raise HTTPException(status_code=500, detail="Training manager не инициализирован")

    return training_manager.get_status()


@router.post("/v1/reset", response_model=ResetResponse)
async def reset_training():
    """
    Полностью сбрасывает обучение:
    - Помечает все примеры в БД как необученные (is_trained = 0)
    - Удаляет директорию с обученной моделью
    - Выгружает модель из памяти

    Returns:
        Информация о сбросе: количество сброшенных примеров и статус удаления модели
    """
    training_manager = get_training_manager()
    config = get_config()

    if training_manager is not None and training_manager.is_training():
        raise HTTPException(
            status_code=409, detail="Невозможно сбросить обучение во время активного обучения"
        )

    db_path = config["database"]["path"]
    model_path = config["model"]["path"]

    # 1. Сбрасываем статус обучения в БД
    reset_count = db.reset_training_status(db_path)

    # 2. Выгружаем модель из памяти
    unload_classifier()

    # 3. Удаляем директорию с моделью
    model_deleted = False
    model_path_obj = Path(model_path)
    if model_path_obj.exists():
        try:
            shutil.rmtree(model_path_obj)
            model_deleted = True
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Не удалось удалить директорию модели: {str(e)}"
            )

    return ResetResponse(
        message="Обучение успешно сброшено", reset_examples=reset_count, model_deleted=model_deleted
    )
