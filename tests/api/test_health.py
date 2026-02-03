"""API-тесты для эндпоинтов /v1/health и /v1/metrics."""


def test_health_returns_200(client):
    """GET /v1/health возвращает 200 и структуру status, model_loaded, training_active."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "training_active" in data


def test_health_model_not_loaded(client):
    """Без загруженной модели model_loaded = False."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_metrics_returns_200(client):
    """GET /v1/metrics возвращает 200 и счётчики примеров."""
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_examples" in data
    assert "trained_examples" in data
    assert "untrained_examples" in data
    assert "model_loaded" in data
    assert "training_status" in data


def test_metrics_empty_db(client):
    """При пустой БД total_examples = 0."""
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    assert response.json()["total_examples"] == 0
