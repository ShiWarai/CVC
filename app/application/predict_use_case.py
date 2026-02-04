"""Сценарий предсказания и эмбеддингов."""

from typing import Callable, List, Optional, Tuple, Union

from app.domain.ports import IClassifier
from app.domain.text_utils import remove_punctuation


class PredictUseCase:
    """Сценарий предсказания: нормализация текста и вызов классификатора."""

    def __init__(
        self,
        get_classifier: Callable[[], Optional[IClassifier]],
        normalizer: Callable[[str], str] = remove_punctuation,
    ):
        self._get_classifier = get_classifier
        self._normalize = normalizer

    def execute_single(
        self, text: str, return_confidence: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """Классифицирует один текст."""
        classifier = self._get_classifier()
        if classifier is None:
            return ("unknown", 0.0) if return_confidence else "unknown"
        cleaned = self._normalize(text)
        return classifier.predict(cleaned, return_confidence=return_confidence)

    def execute_batch(
        self, texts: List[str], return_confidence: bool = False
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """Классифицирует список текстов."""
        classifier = self._get_classifier()
        if classifier is None:
            if return_confidence:
                return (["unknown"] * len(texts), [0.0] * len(texts))
            return ["unknown"] * len(texts)
        cleaned = [self._normalize(t) for t in texts]
        return classifier.predict_batch(cleaned, return_confidence=return_confidence)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Возвращает эмбеддинги для текстов. При отсутствии классификатора создаётся временный по config (вызывающая сторона)."""
        classifier = self._get_classifier()
        if classifier is None:
            raise ValueError("Классификатор не доступен для эмбеддингов")
        cleaned = [self._normalize(t) for t in texts]
        return classifier.get_embeddings(cleaned)
