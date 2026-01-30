"""Утилиты для работы с базой данных SQLite для хранения обучающих данных."""

import sqlite3
import re
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd


def remove_punctuation(text: str) -> str:
    """
    Удаляет все знаки препинания из текста.
    
    Args:
        text: Исходный текст
        
    Returns:
        Текст без знаков препинания
    """
    # Удаляем все знаки препинания, оставляя только буквы, цифры и пробелы
    # Используем регулярное выражение для удаления всех знаков препинания
    text = re.sub(r'[^\w\s]', '', text)
    # Удаляем множественные пробелы и обрезаем
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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


def _example_exists(cursor: sqlite3.Cursor, text: str, command: str) -> bool:
    """
    Проверяет, существует ли пример с указанным text и command в БД.
    
    Args:
        cursor: Курсор базы данных
        text: Текст команды
        command: Метка команды
        
    Returns:
        True если пример существует, False иначе
    """
    cursor.execute(
        "SELECT COUNT(*) FROM examples WHERE text = ? AND command = ?",
        (text, command)
    )
    return cursor.fetchone()[0] > 0


def init_db(db_path: str, csv_path: Optional[str] = None) -> None:
    """
    Инициализирует базу данных и создает таблицу examples.
    При каждом запуске проверяет CSV файлы и добавляет отсутствующие строки в БД.
    
    Args:
        db_path: Путь к файлу базы данных SQLite
        csv_path: Опциональный путь к CSV файлу или директории с CSV файлами для миграции.
                  Если указана директория, загружаются все CSV файлы из неё.
                  При каждом запуске проверяются все CSV файлы и добавляются только новые строки.
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
                command TEXT NOT NULL,
                is_trained INTEGER DEFAULT 0
            )
        """)
        
        # Миграция: добавляем поле is_trained если его нет в существующей таблице
        cursor.execute("PRAGMA table_info(examples)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'is_trained' not in columns:
            cursor.execute("ALTER TABLE examples ADD COLUMN is_trained INTEGER DEFAULT 0")
            # Устанавливаем is_trained = 0 для всех существующих записей
            cursor.execute("UPDATE examples SET is_trained = 0 WHERE is_trained IS NULL")
        
        conn.commit()
        
        # Если указан CSV путь, выполняем синхронизацию (проверяем при каждом запуске)
        if csv_path:
            csv_path_obj = Path(csv_path)
            if csv_path_obj.exists():
                csv_files = []
                
                # Если это директория, находим все CSV файлы в ней
                if csv_path_obj.is_dir():
                    csv_files = list(csv_path_obj.glob("*.csv"))
                    if not csv_files:
                        print(f"В директории {csv_path} не найдено CSV файлов")
                # Если это файл, используем его
                elif csv_path_obj.is_file() and csv_path_obj.suffix.lower() == '.csv':
                    csv_files = [csv_path_obj]
                else:
                    print(f"Путь {csv_path} не является директорией или CSV файлом")
                
                # Синхронизируем данные из всех найденных CSV файлов
                total_added = 0
                total_skipped = 0
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        if 'text' in df.columns and 'command' in df.columns:
                            added_count = 0
                            skipped_count = 0
                            for _, row in df.iterrows():
                                # Очищаем знаки препинания из текста перед сохранением
                                cleaned_text = remove_punctuation(str(row['text']))
                                command = str(row['command'])
                                
                                # Проверяем, существует ли уже такая строка
                                if not _example_exists(cursor, cleaned_text, command):
                                    cursor.execute(
                                        "INSERT INTO examples (text, command, is_trained) VALUES (?, ?, 0)",
                                        (cleaned_text, command)
                                    )
                                    added_count += 1
                                else:
                                    skipped_count += 1
                            
                            conn.commit()
                            total_added += added_count
                            total_skipped += skipped_count
                    except Exception as e:
                        print(f"Ошибка при синхронизации {csv_file.name}: {e}")
                
                if total_added > 0:
                    print(f"Синхронизация CSV: добавлено {total_added} новых примеров из {len(csv_files)} файл(ов)")
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


def get_examples_for_training(db_path: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Получает необученные примеры в формате для обучения (только text и command).
    
    Args:
        db_path: Путь к файлу базы данных
        
    Returns:
        Кортеж (texts, labels, ids) - списки текстов, команд и ID строк
    """
    db_path = _normalize_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Получаем только необученные примеры (is_trained = 0)
    cursor.execute("SELECT id, text, command FROM examples WHERE is_trained = 0 ORDER BY id")
    examples = cursor.fetchall()
    conn.close()
    
    texts = [ex[1] for ex in examples]
    labels = [ex[2] for ex in examples]
    ids = [ex[0] for ex in examples]
    return texts, labels, ids


def get_trained_examples_by_labels(db_path: str, labels: List[str], limit_per_label: int) -> Tuple[List[str], List[str], List[int]]:
    """
    Получает обученные примеры из указанных классов для дополнения недостающих примеров.
    
    Args:
        db_path: Путь к файлу базы данных
        labels: Список меток классов, для которых нужно получить примеры
        limit_per_label: Максимальное количество примеров на класс
        
    Returns:
        Кортеж (texts, labels, ids) - списки текстов, команд и ID строк
    """
    if not labels:
        return [], [], []
    
    db_path = _normalize_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    texts = []
    result_labels = []
    ids = []
    
    for label in labels:
        # Получаем обученные примеры из этого класса (is_trained = 1)
        cursor.execute(
            "SELECT id, text, command FROM examples WHERE command = ? AND is_trained = 1 ORDER BY id LIMIT ?",
            (label, limit_per_label)
        )
        examples = cursor.fetchall()
        
        for ex in examples:
            ids.append(ex[0])
            texts.append(ex[1])
            result_labels.append(ex[2])
    
    conn.close()
    return texts, result_labels, ids


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
        "INSERT INTO examples (text, command, is_trained) VALUES (?, ?, 0)",
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


def mark_examples_as_trained(db_path: str, example_ids: List[int]) -> None:
    """
    Отмечает примеры как обученные (устанавливает is_trained = 1).
    
    Args:
        db_path: Путь к файлу базы данных
        example_ids: Список ID примеров для отметки
    """
    if not example_ids:
        return
    
    # Валидация: убеждаемся, что все ID являются целыми числами
    try:
        validated_ids = [int(ex_id) for ex_id in example_ids]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Некорректные ID примеров: {e}")
    
    db_path = _normalize_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Используем параметризованный запрос для безопасности
    # Ограничиваем количество ID для предотвращения DoS
    if len(validated_ids) > 10000:
        raise ValueError("Слишком много ID для одной операции (максимум 10000)")
    
    placeholders = ','.join('?' * len(validated_ids))
    query = f"UPDATE examples SET is_trained = 1 WHERE id IN ({placeholders})"
    cursor.execute(query, validated_ids)
    
    conn.commit()
    conn.close()


def get_training_stats(db_path: str) -> dict:
    """
    Получает статистику по обученным и необученным примерам.
    
    Args:
        db_path: Путь к файлу базы данных
        
    Returns:
        Словарь со статистикой: total, trained, untrained
    """
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
    
    return {
        "total": total,
        "trained": trained,
        "untrained": untrained
    }


def reset_training_status(db_path: str) -> int:
    """
    Сбрасывает статус обучения для всех примеров (устанавливает is_trained = 0).
    
    Args:
        db_path: Путь к файлу базы данных
        
    Returns:
        Количество сброшенных записей
    """
    db_path = _normalize_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("UPDATE examples SET is_trained = 0 WHERE is_trained = 1")
    reset_count = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    return reset_count






