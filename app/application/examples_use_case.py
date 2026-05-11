"""Сценарий CRUD примеров обучения."""

from typing import Callable, List, Optional, Tuple

from app.domain.ports import IExampleRepository
from app.domain.text_utils import remove_punctuation


class ExamplesUseCase:
    """Сценарий работы с примерами: get_all, get_by_id, add, delete. Нормализация текста при add."""

    def __init__(
        self,
        repository: IExampleRepository,
        get_db_path: Callable[[], str],
        normalizer: Callable[[str], str] = remove_punctuation,
    ):
        self._repo = repository
        self._get_db_path = get_db_path
        self._normalize = normalizer

    def get_all(self) -> List[Tuple[int, str, str]]:
        """Возвращает все примеры (id, text, command)."""
        return self._repo.get_all(self._get_db_path())

    def get_by_id(self, example_id: int) -> Optional[Tuple[int, str, str]]:
        """Возвращает пример по ID или None."""
        return self._repo.get_by_id(self._get_db_path(), example_id)

    def add(self, text: str, command: str) -> int:
        """Добавляет пример после нормализации текста. Возвращает ID."""
        cleaned = self._normalize(text)
        return self._repo.add(self._get_db_path(), cleaned, command)

    def delete(self, example_id: int) -> bool:
        """Удаляет пример по ID. Возвращает True если удалён."""
        return self._repo.delete(self._get_db_path(), example_id)
