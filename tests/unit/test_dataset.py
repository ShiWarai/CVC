"""Unit-тесты для загрузки датасетов."""

import json

import pytest  # noqa: F401 - используется для фикстур (tmp_path)

from commands_classifier.dataset import load_dataset


def test_load_dataset_csv(tmp_path):
    """CSV с колонками text и command загружается."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("text,command\nпривет мир,greet\nлягись,lie_down", encoding="utf-8")
    texts, labels = load_dataset(str(csv_path))
    assert texts == ["привет мир", "лягись"]
    assert labels == ["greet", "lie_down"]


def test_load_dataset_csv_wrong_columns(tmp_path):
    """CSV без колонок text/command поднимает ValueError."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("foo,bar\n1,2", encoding="utf-8")
    with pytest.raises(ValueError, match="text.*command"):
        load_dataset(str(csv_path))


def test_load_dataset_json_list(tmp_path):
    """JSON в виде списка объектов [{\"text\", \"command\"}] загружается."""
    data = [
        {"text": "лягись", "command": "lie_down"},
        {"text": "отмена", "command": "dismiss"},
    ]
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    texts, labels = load_dataset(str(json_path))
    assert texts == ["лягись", "отмена"]
    assert labels == ["lie_down", "dismiss"]


def test_load_dataset_json_dict(tmp_path):
    """JSON в виде объекта {\"text\": [...], \"command\": [...]} загружается."""
    data = {"text": ["a", "b"], "command": ["cmd_a", "cmd_b"]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    texts, labels = load_dataset(str(json_path))
    assert texts == ["a", "b"]
    assert labels == ["cmd_a", "cmd_b"]


def test_load_dataset_json_dict_missing_keys(tmp_path):
    """JSON-объект без ключей text/command поднимает ValueError."""
    json_path = tmp_path / "bad.json"
    json_path.write_text('{"foo": [1], "bar": [2]}', encoding="utf-8")
    with pytest.raises(ValueError, match="text.*command"):
        load_dataset(str(json_path))


def test_load_dataset_file_not_found():
    """Отсутствующий файл поднимает FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_dataset("/nonexistent/path/data.csv")


def test_load_dataset_unsupported_format(tmp_path):
    """Неподдерживаемый формат поднимает ValueError."""
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        load_dataset(str(txt_path))


def test_load_dataset_length_mismatch_json(tmp_path):
    """Несовпадение длин text и command в JSON поднимает ValueError."""
    data = {"text": ["a", "b", "c"], "command": ["x", "y"]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="не совпадает"):
        load_dataset(str(json_path))
