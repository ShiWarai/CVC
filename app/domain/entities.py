"""Доменные сущности. Без зависимостей от фреймворков."""

from dataclasses import dataclass
from enum import Enum


class TrainingStatus(str, Enum):
    """Статусы обучения."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class Example:
    """Пример для обучения: текст команды и метка."""

    id: int
    text: str
    command: str
    is_trained: bool = False


@dataclass(frozen=True)
class PredictionResult:
    """Результат предсказания: команда и уверенность."""

    command: str
    confidence: float
