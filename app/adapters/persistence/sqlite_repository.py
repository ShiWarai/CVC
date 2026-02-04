"""Реализация IExampleRepository для SQLite."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.domain.text_utils import remove_punctuation


def _normalize_db_path(db_path: str) -> str:
    path = Path(db_path)
    if path.exists() and path.is_dir():
        try:
            if not any(path.iterdir()):
                path.rmdir()
                return db_path
            return str(path / "training_data.db")
        except OSError:
            return str(path / "training_data.db")
    return db_path


def check_connection(db_path: str) -> bool:
    path = _normalize_db_path(db_path)
    try:
        conn = sqlite3.connect(path, timeout=2.0)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False


def _example_exists(cursor: sqlite3.Cursor, text: str, command: str) -> bool:
    cursor.execute("SELECT COUNT(*) FROM examples WHERE text = ? AND command = ?", (text, command))
    return cursor.fetchone()[0] > 0


class SqliteExampleRepository:
    """Реализация IExampleRepository для SQLite."""

    def init(self, db_path: str, csv_path: Optional[str] = None) -> None:
        db_path = _normalize_db_path(db_path)
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    command TEXT NOT NULL,
                    is_trained INTEGER DEFAULT 0
                )
            """)
            cursor.execute("PRAGMA table_info(examples)")
            columns = [column[1] for column in cursor.fetchall()]
            if "is_trained" not in columns:
                cursor.execute("ALTER TABLE examples ADD COLUMN is_trained INTEGER DEFAULT 0")
                cursor.execute("UPDATE examples SET is_trained = 0 WHERE is_trained IS NULL")
            conn.commit()
            if csv_path:
                csv_path_obj = Path(csv_path)
                if csv_path_obj.exists():
                    csv_files = list(csv_path_obj.glob("*.csv")) if csv_path_obj.is_dir() else (
                        [csv_path_obj] if csv_path_obj.suffix.lower() == ".csv" else []
                    )
                    if not csv_files and csv_path_obj.is_dir():
                        print(f"В директории {csv_path} не найдено CSV файлов")
                    for csv_file in csv_files:
                        try:
                            df = pd.read_csv(csv_file)
                            if "text" in df.columns and "command" in df.columns:
                                for _, row in df.iterrows():
                                    cleaned_text = remove_punctuation(str(row["text"]))
                                    command = str(row["command"])
                                    if not _example_exists(cursor, cleaned_text, command):
                                        cursor.execute(
                                            "INSERT INTO examples (text, command, is_trained) VALUES (?, ?, 0)",
                                            (cleaned_text, command),
                                        )
                                conn.commit()
                        except Exception as e:
                            print(f"Ошибка при синхронизации {csv_file.name}: {e}")
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                f"Не удалось создать/открыть базу данных по пути: {db_path}\nОшибка: {e}"
            ) from e
        finally:
            if conn:
                conn.close()

    def get_all(self, db_path: str) -> List[Tuple[int, str, str]]:
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, command FROM examples ORDER BY id")
        results = cursor.fetchall()
        conn.close()
        return results

    def get_by_id(self, db_path: str, example_id: int) -> Optional[Tuple[int, str, str]]:
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, command FROM examples WHERE id = ?", (example_id,))
        result = cursor.fetchone()
        conn.close()
        return result

    def add(self, db_path: str, text: str, command: str) -> int:
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO examples (text, command, is_trained) VALUES (?, ?, 0)", (text, command)
        )
        example_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return example_id

    def delete(self, db_path: str, example_id: int) -> bool:
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM examples WHERE id = ?", (example_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def get_examples_for_training(self, db_path: str) -> Tuple[List[str], List[str], List[int]]:
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, command FROM examples WHERE is_trained = 0 ORDER BY id")
        examples = cursor.fetchall()
        conn.close()
        return [ex[1] for ex in examples], [ex[2] for ex in examples], [ex[0] for ex in examples]

    def get_trained_examples_by_labels(
        self, db_path: str, labels: List[str], limit_per_label: int
    ) -> Tuple[List[str], List[str], List[int]]:
        if not labels:
            return [], [], []
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        texts, result_labels, ids = [], [], []
        for label in labels:
            cursor.execute(
                "SELECT id, text, command FROM examples WHERE command = ? AND is_trained = 1 ORDER BY id LIMIT ?",
                (label, limit_per_label),
            )
            for ex in cursor.fetchall():
                ids.append(ex[0])
                texts.append(ex[1])
                result_labels.append(ex[2])
        conn.close()
        return texts, result_labels, ids

    def mark_as_trained(self, db_path: str, example_ids: List[int]) -> None:
        if not example_ids:
            return
        validated_ids = [int(ex_id) for ex_id in example_ids]
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        if len(validated_ids) > 10000:
            conn.close()
            raise ValueError("Слишком много ID для одной операции (максимум 10000)")
        placeholders = ",".join("?" * len(validated_ids))
        cursor.execute(f"UPDATE examples SET is_trained = 1 WHERE id IN ({placeholders})", validated_ids)
        conn.commit()
        conn.close()

    def count(self, db_path: str) -> int:
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM examples")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_training_stats(self, db_path: str) -> Dict[str, Any]:
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM examples")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM examples WHERE is_trained = 1")
        trained = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM examples WHERE is_trained = 0")
        untrained = cursor.fetchone()[0]
        conn.close()
        return {"total": total, "trained": trained, "untrained": untrained}

    def reset_training_status(self, db_path: str) -> int:
        db_path = _normalize_db_path(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE examples SET is_trained = 0 WHERE is_trained = 1")
        reset_count = cursor.rowcount
        conn.commit()
        conn.close()
        return reset_count

    def check_connection(self, db_path: str) -> bool:
        path = _normalize_db_path(db_path)
        try:
            conn = sqlite3.connect(path, timeout=2.0)
            conn.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return False


_default_repo = SqliteExampleRepository()


def init_db(db_path: str, csv_path: Optional[str] = None) -> None:
    _default_repo.init(db_path, csv_path)


def get_all_examples(db_path: str) -> List[Tuple[int, str, str]]:
    return _default_repo.get_all(db_path)


def get_examples_for_training(db_path: str) -> Tuple[List[str], List[str], List[int]]:
    return _default_repo.get_examples_for_training(db_path)


def get_trained_examples_by_labels(
    db_path: str, labels: List[str], limit_per_label: int
) -> Tuple[List[str], List[str], List[int]]:
    return _default_repo.get_trained_examples_by_labels(db_path, labels, limit_per_label)


def add_example(db_path: str, text: str, command: str) -> int:
    return _default_repo.add(db_path, text, command)


def delete_example(db_path: str, example_id: int) -> bool:
    return _default_repo.delete(db_path, example_id)


def count_examples(db_path: str) -> int:
    return _default_repo.count(db_path)


def get_example_by_id(db_path: str, example_id: int) -> Optional[Tuple[int, str, str]]:
    return _default_repo.get_by_id(db_path, example_id)


def mark_examples_as_trained(db_path: str, example_ids: List[int]) -> None:
    _default_repo.mark_as_trained(db_path, example_ids)


def get_training_stats(db_path: str) -> Dict[str, Any]:
    return _default_repo.get_training_stats(db_path)


def reset_training_status(db_path: str) -> int:
    return _default_repo.reset_training_status(db_path)
