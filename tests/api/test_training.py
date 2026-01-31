"""API-тесты для эндпоинтов /train, /train/status, /reset."""

import pytest


def test_train_returns_200(client):
    """POST /train с моком TrainingManager возвращает 200 и training_id."""
    response = client.post("/train", json={})
    assert response.status_code == 200
    data = response.json()
    assert "training_id" in data
    assert data["training_id"] == "test-training-id"
    assert "message" in data


def test_train_with_params(client):
    """POST /train с параметрами принимает их."""
    response = client.post(
        "/train",
        json={
            "num_iterations": 5,
            "num_epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.0001,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "training_id" in data


def test_train_status_returns_200(client):
    """GET /train/status возвращает 200 и статус."""
    response = client.get("/train/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "idle"


def test_reset_returns_200(client):
    """POST /reset возвращает 200 и reset_examples, model_deleted."""
    response = client.post("/reset")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "reset_examples" in data
    assert "model_deleted" in data
