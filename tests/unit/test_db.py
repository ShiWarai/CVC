"""Unit-тесты для модуля БД (init_db, add_example, count_examples, get_all_examples)."""

from commands_classifier import db


def test_init_db_and_count_examples(temp_db_path):
    """init_db создаёт БД; count_examples возвращает 0 для пустой БД."""
    db.init_db(temp_db_path, None)
    assert db.count_examples(temp_db_path) == 0


def test_add_example_and_get_all_examples(temp_db_path):
    """add_example добавляет запись; get_all_examples возвращает её."""
    db.init_db(temp_db_path, None)
    row_id = db.add_example(temp_db_path, "лягись", "lie_down")
    assert row_id >= 1
    examples = db.get_all_examples(temp_db_path)
    assert len(examples) == 1
    assert examples[0][0] == row_id
    assert examples[0][1] == "лягись"
    assert examples[0][2] == "lie_down"


def test_count_examples_after_add(temp_db_path):
    """count_examples увеличивается после add_example."""
    db.init_db(temp_db_path, None)
    assert db.count_examples(temp_db_path) == 0
    db.add_example(temp_db_path, "a", "cmd_a")
    assert db.count_examples(temp_db_path) == 1
    db.add_example(temp_db_path, "b", "cmd_b")
    assert db.count_examples(temp_db_path) == 2


def test_get_training_stats(temp_db_path):
    """get_training_stats возвращает trained/untrained счётчики."""
    db.init_db(temp_db_path, None)
    db.add_example(temp_db_path, "x", "unknown")
    stats = db.get_training_stats(temp_db_path)
    assert "trained" in stats
    assert "untrained" in stats
    assert stats["untrained"] == 1
    assert stats["trained"] == 0


def test_normalize_db_path_file(tmp_path):
    """_normalize_db_path для пути к файлу возвращает путь без изменений."""
    db_file = tmp_path / "existing.db"
    db_file.touch()
    result = db._normalize_db_path(str(db_file))
    assert result == str(db_file)


def test_normalize_db_path_nonexistent(tmp_path):
    """_normalize_db_path для несуществующего пути возвращает путь как есть."""
    path = tmp_path / "new.db"
    assert not path.exists()
    result = db._normalize_db_path(str(path))
    assert result == str(path)
