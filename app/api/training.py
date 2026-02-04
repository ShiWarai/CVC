"""Модуль для управления фоновым обучением модели."""

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.adapters import persistence as db
from app.adapters.ml import CommandsClassifier
from app.domain.entities import TrainingStatus

# Настраиваем логгер для обучения
logger = logging.getLogger("app.training")


def _default_classifier_factory(
    model_name: str, confidence_threshold: float, cache_dir: Optional[str]
):
    """Фабрика по умолчанию: создаёт CommandsClassifier с заданными параметрами."""

    def factory():
        return CommandsClassifier(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            cache_dir=cache_dir,
        )

    return factory


class TrainingManager:
    """Менеджер для управления фоновым обучением модели."""

    def __init__(
        self,
        db_path: str,
        model_path: str,
        model_name: str,
        confidence_threshold: float = 0.5,
        on_training_complete: Optional[Callable[[], None]] = None,
        default_device: str = "cpu",
        cache_dir: Optional[str] = None,
        example_repository: Optional[Any] = None,
        classifier_factory: Optional[Callable[[], Any]] = None,
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
            cache_dir: Путь для кэширования базовой модели (опционально)
            example_repository: Реализация IExampleRepository (если None — app.adapters.persistence._default_repo)
            classifier_factory: Callable[[], IClassifier] (если None — создаётся CommandsClassifier внутри)
        """
        self.db_path = db_path
        self.model_path = model_path
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.on_training_complete = on_training_complete
        self.default_device = default_device
        self.cache_dir = cache_dir
        self._example_repository = example_repository
        self._classifier_factory = classifier_factory
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
        learning_rate: float = 2e-5,
    ) -> str:
        """
        Запускает обучение в фоновом режиме.

        Args:
            num_iterations: Количество итераций контрастного обучения
            num_epochs: Количество эпох fine-tuning
            batch_size: Размер батча
            learning_rate: Скорость обучения

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
                daemon=True,
            )
            self.training_thread.start()

            return self.training_id

    def _train_in_background(
        self, num_iterations: int, num_epochs: int, batch_size: int, learning_rate: float
    ):
        """Выполняет обучение в фоновом потоке."""
        model_path_obj = Path(self.model_path)

        try:
            logger.info(f"[Обучение {self.training_id}] Начало обучения...")

            # Обновляем прогресс: загрузка данных
            self.progress = 0.1

            # Загружаем данные из БД (только необученные)
            if self._example_repository is not None:
                texts, labels, example_ids = self._example_repository.get_examples_for_training(
                    self.db_path
                )
            else:
                texts, labels, example_ids = db.get_examples_for_training(self.db_path)

            if len(texts) == 0:
                raise ValueError("Нет необученных данных для обучения в базе данных")

            # Проверяем минимальные требования для SetFit
            unique_labels = set(labels)
            label_counts = {label: labels.count(label) for label in unique_labels}

            logger.info(
                f"[Обучение {self.training_id}] Загружено {len(texts)} примеров, классов: {len(unique_labels)}"
            )

            # SetFit требует минимум 2 класса для создания отрицательных пар
            if len(unique_labels) < 2:
                raise ValueError(
                    f"Недостаточно классов для обучения. Найдено классов: {len(unique_labels)}. "
                    f"Требуется минимум 2 класса. Добавьте примеры с разными метками команд."
                )

            # Проверяем, что в каждом классе достаточно примеров
            # SetFit создает пары примеров, поэтому нужно минимум 2 примера в каждом классе
            min_examples_per_class = min(label_counts.values())
            classes_with_insufficient_examples = [
                label for label, count in label_counts.items() if count < 2
            ]

            # Если в некоторых классах недостаточно примеров, дополняем их обученными примерами
            if classes_with_insufficient_examples:
                logger.warning(
                    f"[Обучение {self.training_id}] Обнаружены классы с недостаточным количеством примеров: {classes_with_insufficient_examples}"
                )

                # Для каждого класса с недостаточным количеством примеров дополняем обученными
                for label in classes_with_insufficient_examples:
                    needed = 2 - label_counts[label]  # Сколько нужно добавить до минимума

                    # Получаем обученные примеры из этого класса
                    trained_texts, trained_labels, trained_ids = db.get_trained_examples_by_labels(
                        self.db_path, [label], needed
                    )

                    if len(trained_texts) > 0:
                        # Добавляем обученные примеры к основному набору
                        texts.extend(trained_texts)
                        labels.extend(trained_labels)
                        # Важно: не добавляем trained_ids в example_ids, так как эти примеры уже обучены
                        # и не должны быть отмечены как is_trained = 1 после обучения

                # Пересчитываем статистику после дополнения
                unique_labels = set(labels)
                label_counts = {label: labels.count(label) for label in unique_labels}

                # Проверяем еще раз после дополнения
                min_examples_per_class = min(label_counts.values())
                if min_examples_per_class < 2:
                    classes_still_insufficient = [
                        label for label, count in label_counts.items() if count < 2
                    ]
                    raise ValueError(
                        f"Недостаточно примеров в некоторых классах даже после дополнения обученными примерами. "
                        f"Классы с менее чем 2 примерами: {classes_still_insufficient}. "
                        f"Добавьте больше примеров для этих классов."
                    )

            # Рекомендуем минимум 4 примера на класс для стабильного обучения
            if min_examples_per_class < 4:
                logger.warning(
                    f"[Обучение {self.training_id}] Предупреждение: минимальное количество примеров на класс: {min_examples_per_class}. "
                    f"Рекомендуется минимум 4 примера на класс для лучших результатов."
                )

            # Проверяем, существует ли уже модель (переобучение отключено)
            model_exists = model_path_obj.exists()

            if model_exists:
                raise ValueError(
                    f"Модель уже существует в {self.model_path}. "
                    f"Переобучение отключено для защиты от потери старых знаний. "
                    f"Удалите существующую модель вручную, если хотите обучить заново."
                )

            # Обновляем прогресс: инициализация модели
            self.progress = 0.2

            # Создаем и обучаем модель
            threshold_float = float(self.confidence_threshold)
            if self._classifier_factory is not None:
                classifier = self._classifier_factory()
            else:
                classifier = CommandsClassifier(
                    model_name=self.model_name,
                    confidence_threshold=threshold_float,
                    cache_dir=self.cache_dir,
                )

            # Обновляем прогресс: начало обучения
            self.progress = 0.3

            # Убеждаемся, что все параметры имеют правильный тип перед обучением
            num_iterations_int = int(num_iterations)
            num_epochs_int = int(num_epochs)
            batch_size_int = int(batch_size)
            learning_rate_float = float(learning_rate)

            # Используем устройство
            device = self.default_device

            # Обучаем модель
            try:
                classifier.train(
                    texts=texts,
                    labels=labels,
                    num_iterations=num_iterations_int,
                    num_epochs=num_epochs_int,
                    batch_size=batch_size_int,
                    learning_rate=learning_rate_float,
                    device=device,
                )
            except Exception as train_error:
                import traceback

                logger.error(f"[Обучение {self.training_id}] Ошибка при обучении: {train_error}")
                logger.error(
                    f"[Обучение {self.training_id}] Трассировка:\n{traceback.format_exc()}"
                )
                raise

            # Обновляем прогресс: сохранение модели
            self.progress = 0.9

            # Сохраняем модель
            classifier.save(self.model_path)

            # Освобождаем память GPU после обучения
            if device == "cuda":
                try:
                    import gc

                    import torch

                    # Перемещаем модель на CPU перед удалением
                    if classifier.model is not None:
                        if hasattr(classifier.model, "to"):
                            classifier.model = classifier.model.to("cpu")
                        if hasattr(classifier.model, "model_body") and hasattr(
                            classifier.model.model_body, "to"
                        ):
                            classifier.model.model_body = classifier.model.model_body.to("cpu")
                    # Удаляем ссылку на модель
                    del classifier
                    # Принудительная сборка мусора
                    gc.collect()
                    # Очищаем кэш CUDA
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass

            # ЛОГИКА УДАЛЕНИЯ BACKUP ОТКЛЮЧЕНА (закомментировано)
            # # Удаляем backup после успешного сохранения
            # if backup_path:
            #     backup_path_obj = Path(backup_path)
            #     if backup_path_obj.exists():
            #         shutil.rmtree(backup_path_obj)
            #         logger.info(f"[Обучение {self.training_id}] Backup удален")

            # Отмечаем использованные строки как обученные
            if example_ids:
                if self._example_repository is not None:
                    self._example_repository.mark_as_trained(self.db_path, example_ids)
                else:
                    db.mark_examples_as_trained(self.db_path, example_ids)

            # Обучение завершено успешно
            with self.lock:
                self.status = TrainingStatus.COMPLETED
                self.progress = 1.0
                self.completed_at = datetime.now()

            logger.info(f"[Обучение {self.training_id}] Обучение завершено успешно")

            # Автоматически перезагружаем модель, если указан callback
            if self.on_training_complete:
                try:
                    self.on_training_complete()
                except Exception as reload_error:
                    logger.warning(
                        f"[Обучение {self.training_id}] Предупреждение: не удалось перезагрузить модель: {reload_error}"
                    )

        except Exception as e:
            # Ошибка при обучении
            import traceback

            logger.error(f"[Обучение {self.training_id}] ОШИБКА: {e}")
            logger.error(f"[Обучение {self.training_id}] Трассировка:\n{traceback.format_exc()}")

            # ЛОГИКА ВОССТАНОВЛЕНИЯ ИЗ BACKUP ОТКЛЮЧЕНА (закомментировано)
            # # В случае ошибки восстанавливаем старую модель из backup, если она была
            # if backup_path is not None:
            #     try:
            #         backup_path_obj = Path(backup_path)
            #         if backup_path_obj.exists():
            #             # Удаляем поврежденную модель, если она есть
            #             if model_path_obj.exists():
            #                 shutil.rmtree(model_path_obj)
            #             # Восстанавливаем из backup
            #             shutil.move(str(backup_path_obj), str(model_path_obj))
            #             logger.warning(f"[Обучение {self.training_id}] Старая модель восстановлена из backup")
            #     except Exception as restore_error:
            #         logger.error(f"[Обучение {self.training_id}] Не удалось восстановить модель из backup: {restore_error}")

            with self.lock:
                self.status = TrainingStatus.FAILED
                self.error = str(e)
                self.completed_at = datetime.now()

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус обучения.

        Returns:
            Словарь со статусом обучения и метриками качества
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
