"""Утилиты для API. Реэкспорт из domain для обратной совместимости."""

from app.domain.text_utils import remove_punctuation

__all__ = ["remove_punctuation"]
