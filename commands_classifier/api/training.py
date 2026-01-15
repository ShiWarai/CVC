"""Модуль для управления фоновым обучением модели."""

import threading
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from enum import Enum
from commands_classifier import db
from commands_classifier.model import CommandsClassifier

# Настраиваем логгер для обучения
logger = logging.getLogger("commands_classifier.training")

def _log_training(message: str, level: str = "info"):
    """Выводит сообщение и в лог, и в консоль для гарантированного отображения."""
    if level == "info":
        logger.info(message)
        print(message, flush=True)
    elif level == "error":
        logger.error(message)
        print(message, file=__import__('sys').stderr, flush=True)
    elif level == "warning":
        logger.warning(message)
        print(message, flush=True)


class TrainingStatus(str, Enum):
    """Статусы обучения."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingManager:
    """Менеджер для управления фоновым обучением модели."""
    
    def __init__(
        self, 
        db_path: str, 
        model_path: str, 
        model_name: str = "google/embeddinggemma-300M",
        confidence_threshold: float = 0.5,
        on_training_complete: Optional[Callable[[], None]] = None,
        default_device: str = "cpu"
    ):
        """
        Инициализирует менеджер обучения.
        
        Args:
            db_path: Путь к базе данных SQLite
            model_path: Путь для сохранения обученной модели
            model_name: Имя базовой модели для обучения
            confidence_threshold: Порог уверенности для классификации
            on_training_complete: Callback функция, вызываемая после успешного обучения
            default_device: Устройство для обучения (определяется автоматически при старте)
        """
        self.db_path = db_path
        self.model_path = model_path
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)  # Убеждаемся, что это float
        self.on_training_complete = on_training_complete
        self.default_device = default_device  # Устройство определяется автоматически при старте
        self.lock = threading.Lock()
        self.training_thread: Optional[threading.Thread] = None
        
        # Статус обучения
        self.training_id: Optional[str] = None
        self.status = TrainingStatus.IDLE
        self.progress = 0.0
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
    
    def start_training(
        self,
        num_iterations: int = 20,
        num_epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> str:
        """
        Запускает обучение в фоновом режиме.
        
        Args:
            num_iterations: Количество итераций контрастного обучения
            num_epochs: Количество эпох fine-tuning
            batch_size: Размер батча
            learning_rate: Скорость обучения
            Примечание: Устройство (CPU/CUDA) определяется автоматически при старте приложения
            
        Returns:
            ID задачи обучения
            
        Raises:
            RuntimeError: Если обучение уже запущено
        """
        with self.lock:
            if self.status == TrainingStatus.RUNNING:
                raise RuntimeError("Обучение уже запущено")
            
            # Создаем новый ID задачи
            self.training_id = str(uuid.uuid4())
            self.status = TrainingStatus.RUNNING
            self.progress = 0.0
            self.error = None
            self.started_at = datetime.now()
            self.completed_at = None
            
            # Запускаем обучение в отдельном потоке
            self.training_thread = threading.Thread(
                target=self._train_in_background,
                args=(num_iterations, num_epochs, batch_size, learning_rate),
                daemon=True
            )
            self.training_thread.start()
            
            return self.training_id
    
    def _train_in_background(
        self,
        num_iterations: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float
    ):
        """Выполняет обучение в фоновом потоке."""
        try:
            _log_training(f"[Обучение {self.training_id}] Начало обучения...")
            
            # Обновляем прогресс: загрузка данных
            self.progress = 0.1
            _log_training(f"[Обучение {self.training_id}] Загрузка данных из БД...")
            
            # Загружаем данные из БД
            texts, labels = db.get_examples_for_training(self.db_path)
            
            if len(texts) == 0:
                raise ValueError("Нет данных для обучения в базе данных")
            
            _log_training(f"[Обучение {self.training_id}] Загружено {len(texts)} примеров")
            
            # Проверяем минимальные требования для SetFit
            unique_labels = set(labels)
            label_counts = {label: labels.count(label) for label in unique_labels}
            
            _log_training(f"[Обучение {self.training_id}] Классы: {label_counts}")
            
            # SetFit требует минимум 2 класса для создания отрицательных пар
            if len(unique_labels) < 2:
                raise ValueError(
                    f"Недостаточно классов для обучения. Найдено классов: {len(unique_labels)}. "
                    f"Требуется минимум 2 класса. Добавьте примеры с разными метками команд."
                )
            
            # Проверяем, что в каждом классе достаточно примеров
            # SetFit создает пары примеров, поэтому нужно минимум 2 примера в каждом классе
            min_examples_per_class = min(label_counts.values())
            if min_examples_per_class < 2:
                classes_with_one_example = [label for label, count in label_counts.items() if count < 2]
                raise ValueError(
                    f"Недостаточно примеров в некоторых классах. "
                    f"Классы с менее чем 2 примерами: {classes_with_one_example}. "
                    f"Добавьте больше примеров для этих классов."
                )
            
            # Рекомендуем минимум 4 примера на класс для стабильного обучения
            if min_examples_per_class < 4:
                _log_training(
                    f"[Обучение {self.training_id}] Предупреждение: минимальное количество примеров на класс: {min_examples_per_class}. "
                    f"Рекомендуется минимум 4 примера на класс для лучших результатов.",
                    "warning"
                )
            
            # Обновляем прогресс: инициализация модели
            self.progress = 0.2
            _log_training(f"[Обучение {self.training_id}] Инициализация модели {self.model_name}...")
            
            # Создаем и обучаем модель
            # Убеждаемся, что confidence_threshold - это float
            threshold_float = float(self.confidence_threshold)
            _log_training(f"[Обучение {self.training_id}] Создание классификатора с confidence_threshold={threshold_float} (тип: {type(threshold_float)})")
            
            classifier = CommandsClassifier(
                model_name=self.model_name,
                confidence_threshold=threshold_float
            )
            
            # Проверяем, что threshold правильно установлен
            _log_training(f"[Обучение {self.training_id}] Проверка: classifier.confidence_threshold={classifier.confidence_threshold} (тип: {type(classifier.confidence_threshold)})")
            
            # Обновляем прогресс: начало обучения
            self.progress = 0.3
            
            # Убеждаемся, что все параметры имеют правильный тип перед обучением
            num_iterations_int = int(num_iterations)
            num_epochs_int = int(num_epochs)
            batch_size_int = int(batch_size)
            learning_rate_float = float(learning_rate)  # Критично: learning_rate должен быть float
            
            # Используем автоматически определённое устройство
            device = self.default_device
            
            # Логируем информацию о выбранном устройстве
            if device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        _log_training(f"[Обучение {self.training_id}] ✓ Обучение на GPU: {torch.cuda.get_device_name(0)}")
                        _log_training(f"[Обучение {self.training_id}] ✓ Количество GPU: {torch.cuda.device_count()}")
                        _log_training(f"[Обучение {self.training_id}] ✓ CUDA версия: {torch.version.cuda}")
                        _log_training(f"[Обучение {self.training_id}] ✓ Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                    else:
                        _log_training(f"[Обучение {self.training_id}] ⚠ CUDA была выбрана, но недоступна. Переключаемся на CPU.", "warning")
                        device = "cpu"
                except Exception as e:
                    _log_training(f"[Обучение {self.training_id}] ⚠ Ошибка при проверке CUDA: {e}. Используется CPU.", "warning")
                    device = "cpu"
            
            if device == "cpu":
                _log_training(f"[Обучение {self.training_id}] ℹ Обучение на CPU")
            
            _log_training(f"[Обучение {self.training_id}] Начало обучения (iterations={num_iterations_int}, epochs={num_epochs_int}, batch_size={batch_size_int}, lr={learning_rate_float}, device={device})...")
            _log_training(f"[Обучение {self.training_id}] Типы параметров: iterations={type(num_iterations_int)}, epochs={type(num_epochs_int)}, batch_size={type(batch_size_int)}, lr={type(learning_rate_float)}")
            
            # Обучаем модель
            try:
                classifier.train(
                    texts=texts,
                    labels=labels,
                    num_iterations=num_iterations_int,
                    num_epochs=num_epochs_int,
                    batch_size=batch_size_int,
                    learning_rate=learning_rate_float,
                    device=device
                )
            except Exception as train_error:
                import traceback
                _log_training(f"[Обучение {self.training_id}] Ошибка в classifier.train(): {train_error}", "error")
                _log_training(f"[Обучение {self.training_id}] Тип confidence_threshold: {type(classifier.confidence_threshold)}, значение: {classifier.confidence_threshold}", "error")
                _log_training(f"[Обучение {self.training_id}] Полная трассировка:\n{traceback.format_exc()}", "error")
                raise
            
            # Обновляем прогресс: сохранение модели
            self.progress = 0.9
            _log_training(f"[Обучение {self.training_id}] Сохранение модели в {self.model_path}...")
            
            # Сохраняем модель
            classifier.save(self.model_path)
            _log_training(f"[Обучение {self.training_id}] Модель успешно сохранена")
            
            # Обучение завершено успешно
            with self.lock:
                self.status = TrainingStatus.COMPLETED
                self.progress = 1.0
                self.completed_at = datetime.now()
            
            _log_training(f"[Обучение {self.training_id}] Обучение завершено успешно! Модель сохранена в {self.model_path}")
            
            # Автоматически перезагружаем модель, если указан callback
            if self.on_training_complete:
                try:
                    _log_training(f"[Обучение {self.training_id}] Перезагрузка модели...")
                    self.on_training_complete()
                    _log_training(f"[Обучение {self.training_id}] Модель успешно перезагружена и готова к использованию")
                except Exception as reload_error:
                    _log_training(
                        f"[Обучение {self.training_id}] Предупреждение: не удалось перезагрузить модель автоматически: {reload_error}. "
                        f"Перезапустите сервер вручную.",
                        "warning"
                    )
            else:
                _log_training(f"[Обучение {self.training_id}] ВНИМАНИЕ: Для использования новой модели перезапустите сервер")
                
        except Exception as e:
            # Ошибка при обучении
            import traceback
            error_traceback = traceback.format_exc()
            _log_training(f"[Обучение {self.training_id}] ОШИБКА: {e}", "error")
            _log_training(f"[Обучение {self.training_id}] Трассировка:\n{error_traceback}", "error")
            with self.lock:
                self.status = TrainingStatus.FAILED
                self.error = str(e)
                self.completed_at = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус обучения.
        
        Returns:
            Словарь со статусом обучения
        """
        with self.lock:
            return {
                "training_id": self.training_id,
                "status": self.status.value,
                "progress": self.progress,
                "error": self.error,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            }
    
    def is_training(self) -> bool:
        """
        Проверяет, идет ли обучение.
        
        Returns:
            True если обучение запущено
        """
        with self.lock:
            return self.status == TrainingStatus.RUNNING

