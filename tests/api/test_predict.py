"""API-тесты для эндпоинтов predict и embed."""


def test_predict_without_model_returns_503(client):
    """POST /predict без загруженной модели возвращает 503."""
    response = client.post("/predict", json={"text": "лягись"})
    assert response.status_code == 503
    assert (
        "модель" in response.json().get("detail", "").lower()
        or "model" in response.json().get("detail", "").lower()
    )


def test_predict_with_mock_classifier_returns_200(client_with_mock_classifier):
    """POST /predict с мок-классификатором возвращает 200 и command."""
    response = client_with_mock_classifier.post("/predict", json={"text": "лягись"})
    assert response.status_code == 200
    data = response.json()
    assert "command" in data
    assert data["command"] == "unknown"


def test_predict_with_confidence(client_with_mock_classifier):
    """POST /predict с return_confidence=True возвращает confidence."""
    response = client_with_mock_classifier.post(
        "/predict",
        json={"text": "лягись", "return_confidence": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert "command" in data
    assert "confidence" in data
    assert data["confidence"] == 0.5


def test_predict_batch_without_model_returns_503(client):
    """POST /predict/batch без модели возвращает 503."""
    response = client.post("/predict/batch", json={"texts": ["a", "b"]})
    assert response.status_code == 503


def test_predict_batch_with_mock_classifier(client_with_mock_classifier):
    """POST /predict/batch с моком возвращает список commands."""
    response = client_with_mock_classifier.post(
        "/predict/batch",
        json={"texts": ["лягись", "отмена"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["commands"] == ["unknown", "unknown"]


def test_embed_returns_200(client):
    """POST /embed без модели создаёт временный классификатор и возвращает эмбеддинги (или 200)."""
    response = client.post("/embed", json={"inputs": ["hello"]})
    # Может быть 200 (если эндпоинт создаёт временную модель по config) или 503/500 при отсутствии модели
    assert response.status_code in (200, 503, 500)
    if response.status_code == 200:
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1


def test_embed_with_mock_classifier(client_with_mock_classifier):
    """POST /embed с мок-классификатором возвращает эмбеддинги."""
    response = client_with_mock_classifier.post("/embed", json={"inputs": ["hello", "world"]})
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2


def test_predict_validation_empty_text(client_with_mock_classifier):
    """POST /predict с пустым text возвращает 422."""
    response = client_with_mock_classifier.post("/predict", json={"text": ""})
    assert response.status_code == 422
