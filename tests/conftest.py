"""Общие фикстуры для тестов CVC."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

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
    set_classifier,
    set_config,
    set_default_device,
    set_training_manager,
)


def _test_config(db_path: str, model_path: str):
    """Минимальный config для тестов (database, model, training)."""
    return {
        "server": {"host": "0.0.0.0", "port": 20001},
        "model": {
            "path": model_path,
            "name": "cointegrated/rubert-tiny2",
            "confidence_threshold": 0.5,
            "cache_dir": None,
        },
        "database": {"path": db_path, "csv_migration_path": None},
        "training": {
            "iterations": 20,
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 2e-5,
        },
    }


def _setup_test_state(db_path: str, model_path: str, classifier=None, training_manager=None):
    """Выставляет глобальный state синхронно (config, db, training_manager, classifier)."""
    set_default_device("cpu")
    config = _test_config(db_path, model_path)
    set_config(config)
    db.init_db(db_path, None)
    set_training_manager(
        training_manager if training_manager is not None else _make_mock_training_manager()
    )
    set_classifier(classifier)


@asynccontextmanager
async def _noop_lifespan(app: FastAPI):
    """Пустой lifespan (state уже задан в фикстуре)."""
    yield


def _make_mock_training_manager():
    """Мок TrainingManager для API-тестов."""
    mock = MagicMock()
    mock.is_training.return_value = False
    mock.get_status.return_value = {
        "status": "idle",
        "progress": 0.0,
        "training_id": None,
        "error": None,
        "started_at": None,
        "completed_at": None,
    }
    mock.start_training.return_value = "test-training-id"
    return mock


def _make_mock_classifier():
    """Мок CommandsClassifier для API-тестов predict/embed."""
    mock = MagicMock()
    mock.predict.return_value = "unknown"
    mock.predict.side_effect = (
        lambda text, return_confidence=False: ("unknown", 0.5) if return_confidence else "unknown"
    )
    mock.predict_batch.return_value = ["unknown", "unknown"]
    mock.predict_batch.side_effect = (
        lambda texts, return_confidence=False: (["unknown"] * len(texts), [0.5] * len(texts))
        if return_confidence
        else ["unknown"] * len(texts)
    )
    mock.get_embeddings.return_value = [[0.1] * 384]
    mock.get_embeddings.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
    return mock


@pytest.fixture
def temp_db_path(tmp_path):
    """Путь к временной БД для тестов."""
    path = tmp_path / "test.db"
    return str(path)


@pytest.fixture
def temp_model_dir(tmp_path):
    """Временная директория для модели (родительская должна существовать)."""
    d = tmp_path / "models" / "my_model"
    d.parent.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture
def app(temp_db_path, temp_model_dir):
    """Тестовое приложение с подменённым state (без реальной модели и HF)."""
    _setup_test_state(temp_db_path, temp_model_dir, classifier=None)
    test_app = FastAPI(title="CVC API Test", lifespan=_noop_lifespan)
    test_app.include_router(predict_router)
    test_app.include_router(training_router)
    test_app.include_router(examples_router)
    test_app.include_router(load_from_hf_router)
    test_app.include_router(health_router)
    test_app.include_router(command_feedback_router)
    return test_app


@pytest.fixture
def app_with_mock_classifier(temp_db_path, temp_model_dir):
    """Тестовое приложение с мок-классификатором (для predict/embed)."""
    mock_clf = _make_mock_classifier()
    _setup_test_state(temp_db_path, temp_model_dir, classifier=mock_clf)
    test_app = FastAPI(title="CVC API Test", lifespan=_noop_lifespan)
    test_app.include_router(predict_router)
    test_app.include_router(training_router)
    test_app.include_router(examples_router)
    test_app.include_router(load_from_hf_router)
    test_app.include_router(health_router)
    test_app.include_router(command_feedback_router)
    return test_app


@pytest.fixture
def client(app, temp_db_path, temp_model_dir):
    """HTTP-клиент для тестового приложения (без реальной модели)."""
    from fastapi.testclient import TestClient

    _setup_test_state(temp_db_path, temp_model_dir, classifier=None)
    return TestClient(app)


@pytest.fixture
def client_with_mock_classifier(app_with_mock_classifier, temp_db_path, temp_model_dir):
    """HTTP-клиент для приложения с мок-классификатором."""
    from fastapi.testclient import TestClient

    mock_clf = _make_mock_classifier()
    _setup_test_state(temp_db_path, temp_model_dir, classifier=mock_clf)
    return TestClient(app_with_mock_classifier)


@pytest.fixture
def mock_training_manager():
    """Мок TrainingManager для использования в тестах."""
    return _make_mock_training_manager()


@pytest.fixture
def mock_classifier():
    """Мок CommandsClassifier для использования в тестах."""
    return _make_mock_classifier()
