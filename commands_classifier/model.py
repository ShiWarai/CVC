"""Класс для работы с SetFit моделью классификации команд."""

from pathlib import Path
from typing import List, Optional, Tuple
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
import numpy as np


class CommandsClassifier:
    """Классификатор команд на основе SetFit для few-shot learning."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        confidence_threshold: float = 0.5
    ):
        """
        Инициализирует классификатор.
        
        Args:
            model_name: Имя предобученной модели из sentence-transformers
            confidence_threshold: Порог уверенности для отбраковки (0.0-1.0). 
                                  Если уверенность ниже порога, возвращается "unknown"
        """
        self.model_name = model_name
        self.model: Optional[SetFitModel] = None
        self.is_trained = False
        self.confidence_threshold = confidence_threshold
    
    def train(
        self,
        texts: List[str],
        labels: List[str],
        num_iterations: int = 20,
        num_epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """
        Обучает модель на предоставленных данных.
        
        Args:
            texts: Список текстов для обучения
            labels: Список меток (команд) для каждого текста
            num_iterations: Количество итераций контрастного обучения (используется как num_epochs для body)
            num_epochs: Количество эпох для fine-tuning head
            batch_size: Размер батча
            learning_rate: Скорость обучения
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Количество текстов ({len(texts)}) не совпадает с количеством меток ({len(labels)})"
            )
        
        # Создаем модель
        self.model = SetFitModel.from_pretrained(self.model_name)
        
        # Создаем датасет из текстов и меток
        train_dataset = Dataset.from_dict({"text": texts, "label": labels})
        
        # Создаем тренер с параметрами напрямую
        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_dataset,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            column_mapping={"text": "text", "label": "label"},
        )
        
        # Обучаем модель
        trainer.train()
        
        self.is_trained = True
    
    def predict(self, text: str, return_confidence: bool = False) -> str | Tuple[str, float]:
        """
        Классифицирует один текст.
        
        Args:
            text: Текст для классификации
            return_confidence: Если True, возвращает (команда, уверенность)
            
        Returns:
            Предсказанная команда или (команда, уверенность) если return_confidence=True
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        # Получаем предсказания с вероятностями
        predictions, confidences = self._predict_with_confidence([text])
        command = predictions[0]
        confidence = confidences[0]
        
        # Применяем порог уверенности
        if confidence < self.confidence_threshold:
            command = "unknown"
        
        if return_confidence:
            return command, confidence
        return command
    
    def _predict_with_confidence(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        Внутренний метод для получения предсказаний с уверенностью.
        
        Args:
            texts: Список текстов для классификации
            
        Returns:
            Кортеж (предсказания, уверенности)
        """
        # Получаем вероятности для всех классов
        probs = self.model.predict_proba(texts)
        
        # Получаем предсказания (классы)
        preds = self.model.predict(texts)
        
        # Находим максимальную вероятность для каждого предсказания
        predictions = []
        confidences = []
        
        # Обрабатываем probs (может быть numpy array или список)
        if hasattr(probs, 'tolist'):
            probs = probs.tolist()
        
        # Обрабатываем preds (может быть numpy array или список)
        if hasattr(preds, 'tolist'):
            preds = preds.tolist()
        else:
            preds = list(preds)
        
        for i, prob in enumerate(probs):
            # Находим индекс максимальной вероятности
            if isinstance(prob, (list, np.ndarray)):
                max_idx = np.argmax(prob)
                max_prob = float(prob[max_idx])
            else:
                max_prob = float(prob)
            
            # Используем предсказание модели
            predictions.append(str(preds[i]))
            confidences.append(max_prob)
        
        return predictions, confidences
    
    def predict_batch(self, texts: List[str], return_confidence: bool = False) -> List[str] | Tuple[List[str], List[float]]:
        """
        Классифицирует список текстов.
        
        Args:
            texts: Список текстов для классификации
            return_confidence: Если True, возвращает (команды, уверенности)
            
        Returns:
            Список предсказанных команд или (команды, уверенности) если return_confidence=True
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        predictions, confidences = self._predict_with_confidence(texts)
        
        # Применяем порог уверенности
        commands = []
        for pred, conf in zip(predictions, confidences):
            if conf < self.confidence_threshold:
                commands.append("unknown")
            else:
                commands.append(pred)
        
        if return_confidence:
            return commands, confidences
        return commands
    
    def save(self, model_path: str):
        """
        Сохраняет обученную модель.
        
        Args:
            model_path: Путь для сохранения модели
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена. Нечего сохранять.")
        
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(path))
    
    def load(self, model_path: str, confidence_threshold: Optional[float] = None):
        """
        Загружает сохраненную модель.
        
        Args:
            model_path: Путь к сохраненной модели
            confidence_threshold: Порог уверенности (если None, используется текущий)
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        self.model = SetFitModel.from_pretrained(str(path))
        self.is_trained = True
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

