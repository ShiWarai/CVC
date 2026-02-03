"""Эндпоинты для проверки здоровья и метрик."""

from fastapi import APIRouter, Response

from commands_classifier import db
from commands_classifier.api.state import get_classifier, get_config, get_training_manager

router = APIRouter(tags=["health"])


@router.get("/v1/health")
async def health(response: Response):
    """
    Проверка работоспособности сервера (TEI совместимый).
    Возвращает 503 при недоступности БД.
    """
    classifier = get_classifier()
    training_manager = get_training_manager()
    config = get_config()
    db_path = config["database"]["path"]
    db_ok = db.check_connection(db_path)

    if not db_ok:
        response.status_code = 503
        return {
            "status": "unhealthy",
            "model_loaded": classifier is not None,
            "training_active": training_manager.is_training() if training_manager else False,
            "database_available": False,
        }

    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "training_active": training_manager.is_training() if training_manager else False,
        "database_available": True,
    }


@router.get("/v1/metrics")
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
        "training_status": training_manager.get_status() if training_manager else None,
    }
