"""
E2E-тест: тестовая БД (3 класса), загрузка приложения, обучение, predict, reset.
Выполняется в контейнере с реальной моделью и реальным обучением.
"""

import time
from pathlib import Path  # noqa: F401 - используется в FIXTURES_DIR

import pytest
from fastapi import FastAPI

from app import db
from app.api.routes import (
    command_feedback_router,
    examples_router,
    health_router,
    load_from_hf_router,
    predict_router,
    training_router,
)
from app.api.state import (
    load_model,
    set_classifier,
    set_config,
    set_default_device,
    set_training_manager,
)
from app.api.training import TrainingManager

# Путь к фикстурам с тестовым датасетом (3 класса: lie_down, dismiss, unknown)
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CSV_MIGRATION_PATH = str(FIXTURES_DIR)
EXPECTED_CLASSES = {"lie_down", "dismiss", "unknown"}

# Примеры из датасета по одному на класс (для проверки predict после обучения)
SAMPLE_TEXTS_BY_CLASS = [
    "лягись",  # lie_down
    "отмена",  # dismiss
    "не знаю",  # unknown
]


def _setup_e2e_state(db_path: str, model_path: str):
    """Выставляет state для E2E (реальный TrainingManager)."""
    set_default_device("cpu")
    config = {
        "server": {"host": "0.0.0.0", "port": 20001},
        "model": {
            "path": model_path,
            "name": "cointegrated/rubert-tiny2",
            "confidence_threshold": 0.5,
        },
        "database": {
            "path": db_path,
            "csv_migration_path": CSV_MIGRATION_PATH,
        },
        "training": {
            "iterations": 2,
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 2e-5,
        },
    }
    set_config(config)
    db.init_db(db_path, CSV_MIGRATION_PATH)
    training_manager = TrainingManager(
        db_path,
        model_path,
        model_name=config["model"]["name"],
        confidence_threshold=config["model"]["confidence_threshold"],
        on_training_complete=load_model,
        default_device="cpu",
        cache_dir=None,
    )
    set_training_manager(training_manager)
    set_classifier(None)


@pytest.fixture
def e2e_app(temp_db_path, temp_model_dir):
    """Приложение для E2E с реальной БД и реальным TrainingManager."""
    _setup_e2e_state(temp_db_path, temp_model_dir)
    app = FastAPI(title="CVC E2E Test")
    app.include_router(predict_router)
    app.include_router(training_router)
    app.include_router(examples_router)
    app.include_router(load_from_hf_router)
    app.include_router(health_router)
    app.include_router(command_feedback_router)
    return app


@pytest.fixture
def e2e_client(e2e_app, temp_db_path, temp_model_dir):
    """HTTP-клиент для E2E-приложения."""
    from fastapi.testclient import TestClient

    _setup_e2e_state(temp_db_path, temp_model_dir)
    return TestClient(e2e_app)


def test_e2e_health(e2e_client):
    """GET /v1/health возвращает 200 и структуру."""
    response = e2e_client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_e2e_predict_without_model_fails(e2e_client):
    """POST /v1/predict без обученной модели возвращает ошибку (503)."""
    response = e2e_client.post("/v1/predict", json={"text": "лягись"})
    assert response.status_code == 503


def test_e2e_train_then_predict_then_reset_then_predict_fails(e2e_client):
    """
    Полный сценарий: обучение -> predict по 3 примерам -> reset -> predict снова ошибка.
    """
    # Запуск обучения (минимальные итерации для скорости)
    train_resp = e2e_client.post(
        "/v1/train",
        json={"num_iterations": 2, "num_epochs": 1, "batch_size": 32},
    )
    assert train_resp.status_code == 200
    training_id = train_resp.json()["training_id"]
    assert training_id

    # Ожидание завершения обучения (опрос статуса)
    max_wait = 600
    step = 5
    elapsed = 0
    while elapsed < max_wait:
        status_resp = e2e_client.get("/v1/train/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        if status_data.get("status") == "completed":
            break
        if status_data.get("status") == "failed":
            pytest.fail(f"Обучение завершилось с ошибкой: {status_data.get('error')}")
        time.sleep(step)
        elapsed += step
    else:
        pytest.fail(f"Обучение не завершилось за {max_wait} с")

    # Predict по трём строкам из датасета (по одной на класс)
    for text in SAMPLE_TEXTS_BY_CLASS:
        pred_resp = e2e_client.post("/v1/predict", json={"text": text})
        assert pred_resp.status_code == 200, f"predict для '{text}' вернул {pred_resp.status_code}"
        data = pred_resp.json()
        assert "command" in data
        assert data["command"] in EXPECTED_CLASSES, (
            f"ожидался один из {EXPECTED_CLASSES}, получено: {data['command']}"
        )

    # Сброс модели
    reset_resp = e2e_client.post("/v1/reset")
    assert reset_resp.status_code == 200
    assert "reset_examples" in reset_resp.json()

    # После reset predict снова должен вернуть ошибку (модель выгружена)
    predict_after_reset = e2e_client.post("/v1/predict", json={"text": "лягись"})
    assert predict_after_reset.status_code == 503
