"""Domain layer: entities, ports, text utilities. No external dependencies."""

from app.domain.entities import Example, PredictionResult, TrainingStatus
from app.domain.ports import IClassifier, IExampleRepository
from app.domain.text_utils import remove_punctuation

__all__ = [
    "Example",
    "PredictionResult",
    "TrainingStatus",
    "IClassifier",
    "IExampleRepository",
    "remove_punctuation",
]
