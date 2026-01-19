"""Эндпоинты для проверки здоровья и метрик."""

from fastapi import APIRouter

from commands_classifier.api.state import get_classifier, get_config, get_training_manager
from commands_classifier import db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """
    Проверка работоспособности сервера (TEI совместимый).
    
    Returns:
        Статус сервера
    """
    classifier = get_classifier()
    training_manager = get_training_manager()
    
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "training_active": training_manager.is_training() if training_manager else False
    }


@router.get("/metrics")
async def metrics():
    """
    Метрики сервера (TEI совместимый).
    
    Returns:
        Метрики сервера
    """
    classifier = get_classifier()
    training_manager = get_training_manager()
    config = get_config()
    
    db_path = config["database"]["path"]
    example_count = db.count_examples(db_path)
    training_stats = db.get_training_stats(db_path)
    
    return {
        "total_examples": example_count,
        "trained_examples": training_stats["trained"],
        "untrained_examples": training_stats["untrained"],
        "model_loaded": classifier is not None,
        "training_status": training_manager.get_status() if training_manager else None
    }
