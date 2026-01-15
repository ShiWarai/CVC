"""Утилиты для работы с базой данных SQLite для хранения обучающих данных."""

import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd


def _normalize_db_path(db_path: str) -> str:
    """
    Нормализует путь к базе данных.
    Если путь указывает на директорию, пытается исправить это.
    
    Args:
        db_path: Путь к файлу базы данных
        
    Returns:
        Нормализованный путь к файлу базы данных
    """
    path = Path(db_path)
    
    # Если путь указывает на директорию
    if path.exists() and path.is_dir():
        # Пытаемся удалить директорию, если она пустая
        try:
            if not any(path.iterdir()):
                path.rmdir()
                # После удаления директории, путь свободен для создания файла
                return db_path
            else:
                # Если директория не пустая, создаем файл внутри неё
                return str(path / "training_data.db")
        except OSError:
            # Если не удалось удалить, создаем файл внутри
            return str(path / "training_data.db")
    
    return db_path


def init_db(db_path: str, csv_path: Optional[str] = None) -> None:
    """
    Инициализирует базу данных и создает таблицу examples.
    Если БД пустая и указан csv_path, выполняет миграцию данных из CSV.
    
    Args:
        db_path: Путь к файлу базы данных SQLite
        csv_path: Опциональный путь к CSV файлу для миграции
    """
    # Нормализуем путь к базе данных
    db_path = _normalize_db_path(db_path)
    path = Path(db_path)
    
    # Создаем родительскую директорию, если её нет
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Убеждаемся, что файл может быть создан (проверяем права доступа)
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Создаем таблицу examples
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                command TEXT NOT NULL
            )
        """)
        
        conn.commit()
        
        # Проверяем, пустая ли БД
        cursor.execute("SELECT COUNT(*) FROM examples")
        count = cursor.fetchone()[0]
        
        # Если БД пустая и указан CSV, выполняем миграцию
        if count == 0 and csv_path:
            csv_file = Path(csv_path)
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if 'text' in df.columns and 'command' in df.columns:
                        for _, row in df.iterrows():
                            cursor.execute(
                                "INSERT INTO examples (text, command) VALUES (?, ?)",
                                (str(row['text']), str(row['command']))
                            )
                        conn.commit()
                        print(f"Мигрировано {len(df)} примеров из {csv_path}")
                except Exception as e:
                    print(f"Ошибка при миграции CSV: {e}")
    except sqlite3.OperationalError as e:
        error_msg = (
            f"Не удалось создать/открыть базу данных по пути: {db_path}\n"
            f"Ошибка: {e}\n"
            f"Возможные причины:\n"
            f"  1. Нет прав на запись в директорию {path.parent}\n"
            f"  2. Путь указывает на директорию вместо файла (проблема Docker volume)\n"
            f"  3. Директория не существует и не может быть создана"
        )
        raise RuntimeError(error_msg) from e
    finally:
        if conn:
            conn.close()


def get_all_examples(db_path: str) -> List[Tuple[int, str, str]]:
    """
    Получает все примеры из базы данных.
    
    Args:
        db_path: Путь к файлу базы данных
        
    Returns:
        Список кортежей (id, text, command)
    """
    db_path = _normalize_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, command FROM examples ORDER BY id")
    results = cursor.fetchall()
    conn.close()
    return results


def get_examples_for_training(db_path: str) -> Tuple[List[str], List[str]]:
    """
    Получает примеры в формате для обучения (только text и command).
    
    Args:
        db_path: Путь к файлу базы данных
        
    Returns:
        Кортеж (texts, labels) - списки текстов и команд
    """
    examples = get_all_examples(db_path)
    texts = [ex[1] for ex in examples]
    labels = [ex[2] for ex in examples]
    return texts, labels


def add_example(db_path: str, text: str, command: str) -> int:
    """
    Добавляет новый пример в базу данных.
    
    Args:
        db_path: Путь к файлу базы данных
        text: Текст команды
        command: Метка команды
        
    Returns:
        ID добавленного примера
    """
    db_path = _normalize_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO examples (text, command) VALUES (?, ?)",
        (text, command)
    )
    example_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return example_id


def delete_example(db_path: str, example_id: int) -> bool:
    """
    Удаляет пример по ID.
    
    Args:
        db_path: Путь к файлу базы данных
        example_id: ID примера для удаления
        
    Returns:
        True если пример был удален, False если не найден
    """
    db_path = _normalize_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM examples WHERE id = ?", (example_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def count_examples(db_path: str) -> int:
    """
    Возвращает количество примеров в базе данных.
    
    Args:
        db_path: Путь к файлу базы данных
        
    Returns:
        Количество примеров
    """
    db_path = _normalize_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM examples")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_example_by_id(db_path: str, example_id: int) -> Optional[Tuple[int, str, str]]:
    """
    Получает пример по ID.
    
    Args:
        db_path: Путь к файлу базы данных
        example_id: ID примера
        
    Returns:
        Кортеж (id, text, command) или None если не найден
    """
    db_path = _normalize_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, command FROM examples WHERE id = ?", (example_id,))
    result = cursor.fetchone()
    conn.close()
    return result






