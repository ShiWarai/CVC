"""API-тесты для эндпоинтов /examples."""


def test_get_examples_empty(client):
    """GET /v1/examples при пустой БД возвращает []."""
    response = client.get("/v1/examples")
    assert response.status_code == 200
    assert response.json() == []


def test_add_example_returns_201(client):
    """POST /v1/examples создаёт пример и возвращает 201 с id, text, command."""
    response = client.post(
        "/v1/examples",
        json={"text": "лягись", "command": "lie_down"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["text"] == "лягись"
    assert data["command"] == "lie_down"


def test_get_examples_after_add(client):
    """GET /v1/examples после добавления возвращает список с одним примером."""
    client.post("/v1/examples", json={"text": "отмена", "command": "dismiss"})
    response = client.get("/v1/examples")
    assert response.status_code == 200
    examples = response.json()
    assert len(examples) == 1
    assert examples[0]["text"] == "отмена"
    assert examples[0]["command"] == "dismiss"


def test_get_example_by_id(client):
    """GET /v1/examples/{id} возвращает пример по ID."""
    add_resp = client.post("/v1/examples", json={"text": "неизвестно", "command": "unknown"})
    example_id = add_resp.json()["id"]
    response = client.get(f"/v1/examples/{example_id}")
    assert response.status_code == 200
    assert response.json()["id"] == example_id
    assert response.json()["text"] == "неизвестно"


def test_get_example_by_id_not_found(client):
    """GET /v1/examples/999 возвращает 404."""
    response = client.get("/v1/examples/999")
    assert response.status_code == 404


def test_delete_example(client):
    """DELETE /v1/examples/{id} удаляет пример и возвращает 200."""
    add_resp = client.post("/v1/examples", json={"text": "x", "command": "y"})
    example_id = add_resp.json()["id"]
    response = client.delete(f"/v1/examples/{example_id}")
    assert response.status_code == 200
    get_resp = client.get(f"/v1/examples/{example_id}")
    assert get_resp.status_code == 404


def test_add_example_validation_empty_text(client):
    """POST /v1/examples с пустым text возвращает 422."""
    response = client.post("/v1/examples", json={"text": "", "command": "cmd"})
    assert response.status_code == 422
