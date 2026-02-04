"""Утилиты для работы с текстом. Без внешних зависимостей."""

import re


def remove_punctuation(text: str) -> str:
    """
    Удаляет все знаки препинания из текста.

    Args:
        text: Исходный текст

    Returns:
        Текст без знаков препинания
    """
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
