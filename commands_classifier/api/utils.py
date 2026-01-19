"""Утилиты для API."""

import re


def remove_punctuation(text: str) -> str:
    """
    Удаляет все знаки препинания из текста.
    
    Args:
        text: Исходный текст
        
    Returns:
        Текст без знаков препинания
    """
    # Удаляем все знаки препинания, оставляя только буквы, цифры и пробелы
    text = re.sub(r'[^\w\s]', '', text)
    # Удаляем множественные пробелы и обрезаем
    text = re.sub(r'\s+', ' ', text).strip()
    return text
