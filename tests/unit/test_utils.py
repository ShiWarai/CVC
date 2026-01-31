"""Unit-тесты для утилит (remove_punctuation)."""

import pytest

from commands_classifier.api.utils import remove_punctuation


def test_remove_punctuation_empty_string():
    """Пустая строка остаётся пустой (после strip)."""
    assert remove_punctuation("") == ""


def test_remove_punctuation_strips_whitespace():
    """Пробелы по краям убираются."""
    assert remove_punctuation("  hello  ") == "hello"


def test_remove_punctuation_removes_punctuation():
    """Знаки препинания удаляются (без вставки пробелов)."""
    assert remove_punctuation("привет, мир!") == "привет мир"
    assert remove_punctuation("hello, world!") == "hello world"
    assert remove_punctuation("a.b,c;d:e?f/g") == "abcdefg"


def test_remove_punctuation_multiple_spaces():
    """Несколько пробелов схлопываются в один."""
    assert remove_punctuation("a   b    c") == "a b c"


def test_remove_punctuation_unicode():
    """Юникод (кириллица) сохраняется."""
    assert remove_punctuation("Лягись!") == "Лягись"
    assert remove_punctuation("Отмена.") == "Отмена"


def test_remove_punctuation_digits_preserved():
    """Цифры и буквы сохраняются."""
    assert remove_punctuation("команда 123") == "команда 123"


def test_remove_punctuation_only_punctuation():
    """Только знаки препинания дают пустую строку."""
    assert remove_punctuation("...!!!???") == ""
