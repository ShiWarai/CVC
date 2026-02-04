"""Persistence-адаптер: SQLite-репозиторий примеров."""

from app.adapters.persistence.sqlite_repository import (
    SqliteExampleRepository,
    _default_repo,
    _normalize_db_path,
    add_example,
    check_connection,
    count_examples,
    delete_example,
    get_all_examples,
    get_example_by_id,
    get_examples_for_training,
    get_trained_examples_by_labels,
    get_training_stats,
    init_db,
    mark_examples_as_trained,
    reset_training_status,
)

__all__ = [
    "SqliteExampleRepository",
    "_default_repo",
    "_normalize_db_path",
    "init_db",
    "add_example",
    "get_all_examples",
    "get_example_by_id",
    "delete_example",
    "count_examples",
    "get_examples_for_training",
    "get_trained_examples_by_labels",
    "mark_examples_as_trained",
    "get_training_stats",
    "reset_training_status",
    "check_connection",
]
