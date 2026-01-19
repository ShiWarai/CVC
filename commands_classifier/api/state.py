"""Глобальное состояние приложения."""

from typing import Optional, Dict, Any
from commands_classifier.model import CommandsClassifier
from commands_classifier.api.training import TrainingManager

# Глобальные переменные состояния
_classifier: Optional[CommandsClassifier] = None
_training_manager: Optional[TrainingManager] = None
_config: Dict[str, Any] = {}
_default_device: str = "cpu"


def get_classifier() -> Optional[CommandsClassifier]:
    """Возвращает текущий классификатор."""
    return _classifier


def set_classifier(classifier: Optional[CommandsClassifier]) -> None:
    """Устанавливает классификатор."""
    global _classifier
    _classifier = classifier


def unload_classifier() -> None:
    """Выгружает классификатор из памяти с очисткой GPU."""
    global _classifier
    
    if _classifier is not None:
        try:
            import torch
            import gc
            
            # Перемещаем модель на CPU перед удалением
            if _classifier.model is not None:
                if hasattr(_classifier.model, 'to'):
                    try:
                        _classifier.model = _classifier.model.to('cpu')
                    except:
                        pass
                if hasattr(_classifier.model, 'model_body') and hasattr(_classifier.model.model_body, 'to'):
                    try:
                        _classifier.model.model_body = _classifier.model.model_body.to('cpu')
                    except:
                        pass
            
            del _classifier
            _classifier = None
            gc.collect()
            
            # Очищаем кэш CUDA
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
        except Exception:
            _classifier = None


def get_training_manager() -> Optional[TrainingManager]:
    """Возвращает менеджер обучения."""
    return _training_manager


def set_training_manager(manager: Optional[TrainingManager]) -> None:
    """Устанавливает менеджер обучения."""
    global _training_manager
    _training_manager = manager


def get_config() -> Dict[str, Any]:
    """Возвращает конфигурацию."""
    return _config


def set_config(config: Dict[str, Any]) -> None:
    """Устанавливает конфигурацию."""
    global _config
    _config = config


def get_default_device() -> str:
    """Возвращает устройство по умолчанию."""
    return _default_device


def set_default_device(device: str) -> None:
    """Устанавливает устройство по умолчанию."""
    global _default_device
    _default_device = device
