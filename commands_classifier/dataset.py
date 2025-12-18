"""Утилиты для загрузки и подготовки датасетов."""

import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple


def load_dataset(dataset_path: str) -> Tuple[List[str], List[str]]:
    """
    Загружает датасет из CSV или JSON файла.
    
    Args:
        dataset_path: Путь к файлу датасета
        
    Returns:
        Кортеж (texts, labels) - списки текстов и меток
        
    Raises:
        ValueError: Если формат файла не поддерживается
    """
    path = Path(dataset_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {dataset_path}")
    
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(dataset_path)
        
        # Проверяем наличие нужных колонок
        if 'text' not in df.columns or 'command' not in df.columns:
            raise ValueError(
                "CSV файл должен содержать колонки 'text' и 'command'. "
                f"Найдены колонки: {list(df.columns)}"
            )
        
        texts = df['text'].astype(str).tolist()
        labels = df['command'].astype(str).tolist()
        
    elif path.suffix.lower() == '.json':
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Поддерживаем два формата JSON:
        # 1. Список объектов: [{"text": "...", "command": "..."}, ...]
        # 2. Объект с ключами: {"texts": [...], "commands": [...]}
        if isinstance(data, list):
            texts = [item['text'] for item in data]
            labels = [item['command'] for item in data]
        elif isinstance(data, dict):
            if 'text' in data and 'command' in data:
                texts = data['text']
                labels = data['command']
            else:
                raise ValueError(
                    "JSON должен содержать ключи 'text' и 'command' или быть списком объектов"
                )
        else:
            raise ValueError("Неверный формат JSON файла")
    else:
        raise ValueError(
            f"Неподдерживаемый формат файла: {path.suffix}. "
            "Поддерживаются только .csv и .json"
        )
    
    if len(texts) != len(labels):
        raise ValueError(
            f"Количество текстов ({len(texts)}) не совпадает с количеством меток ({len(labels)})"
        )
    
    return texts, labels

