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
        model_name: str = "google/embeddinggemma-300M",
        confidence_threshold: float = 0.5,
        cache_dir: Optional[str] = None
    ):
        """
        Инициализирует классификатор.
        
        Args:
            model_name: Имя предобученной модели (по умолчанию: google/embeddinggemma-300M)
            confidence_threshold: Порог уверенности для отбраковки (0.0-1.0). 
                                  Если уверенность ниже порога, возвращается "unknown"
            cache_dir: Путь для кэширования базовой модели (опционально)
        """
        self.model_name = model_name
        self.model: Optional[SetFitModel] = None
        self.is_trained = False
        # Убеждаемся, что confidence_threshold всегда float
        self.confidence_threshold = float(confidence_threshold)
        self.cache_dir = cache_dir
    
    def train(
        self,
        texts: List[str],
        labels: List[str],
        num_iterations: int = 20,
        num_epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        device: Optional[str] = None
    ):
        """
        Обучает модель на предоставленных данных.
        
        Args:
            texts: Список текстов для обучения
            labels: Список меток (команд) для каждого текста
            num_iterations: Количество итераций контрастного обучения (используется как num_epochs для body)
            num_epochs: Количество эпох для fine-tuning head
            batch_size: Размер батча (больше = быстрее, но требует больше памяти)
            learning_rate: Скорость обучения
            device: Устройство для обучения ('cpu', 'cuda' или None - определяется автоматически)
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Количество текстов ({len(texts)}) не совпадает с количеством меток ({len(labels)})"
            )
        
        
        # Определяем устройство
        if device is None:
            # Проверяем доступность CUDA, если не указано явно
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    device = "cuda"
                    print(f"✓ CUDA доступна. Используется GPU: {torch.cuda.get_device_name(0)}")
                else:
                    device = "cpu"
                    print("ℹ CUDA недоступна. Используется CPU.")
            except ImportError:
                device = "cpu"
                print("ℹ PyTorch не установлен. Используется CPU.")
        
        # Создаем модель с кэшированием в указанную директорию
        cache_dir_path = None
        if self.cache_dir:
            cache_dir_path = Path(self.cache_dir)
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            cache_dir_path = str(cache_dir_path)
        
        # Создаем модель и сразу перемещаем на нужное устройство
        # Используем use_safetensors=True для обхода требования torch>=2.6
        try:
            self.model = SetFitModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir_path,
                use_safetensors=True
            )
        except Exception as e:
            # Если safetensors не доступны, пробуем без них
            # Это может вызвать ошибку если transformers>=4.48 и torch<2.6
            try:
                self.model = SetFitModel.from_pretrained(
                    self.model_name,
                    cache_dir=cache_dir_path
                )
            except ValueError as ve:
                if "torch>=2.6" in str(ve).lower() or "torch >= 2.6" in str(ve).lower():
                    raise ValueError(
                        f"Модель {self.model_name} требует torch>=2.6, но установлена {__import__('torch').__version__}. "
                        f"Используйте альтернативную модель, например: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    ) from ve
                raise
        
        # Перемещаем модель на устройство
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    # Перемещаем модель на GPU
                    self.model = self.model.to(device)
                    # Проверяем, что модель действительно на GPU
                    # SetFitModel содержит SentenceTransformer, который нужно проверить отдельно
                    if hasattr(self.model, 'model_body') and hasattr(self.model.model_body, 'to'):
                        self.model.model_body = self.model.model_body.to(device)
                    print(f"✓ Модель перемещена на GPU: {torch.cuda.get_device_name(0)}")
                    print(f"✓ CUDA версия: {torch.version.cuda}")
                    print(f"✓ Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                else:
                    device = "cpu"
                    print("⚠ CUDA была запрошена, но недоступна. Используется CPU.")
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(device)
            except Exception as e:
                device = "cpu"
                print(f"⚠ Ошибка при использовании CUDA: {e}. Используется CPU.")
                if hasattr(self.model, 'to'):
                    self.model = self.model.to(device)
        else:
            if hasattr(self.model, 'to'):
                self.model = self.model.to(device)
            print(f"ℹ Обучение на {device.upper()}")
        
        # Создаем датасет из текстов и меток
        train_dataset = Dataset.from_dict({"text": texts, "label": labels})
        
        # Создаем тренер с параметрами напрямую
        # Модель уже перемещена на нужное устройство выше
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
        # Временно сохраняем confidence_threshold, чтобы избежать проблем во время обучения
        # (SetFit может вызывать внутренние методы, которые могут использовать этот атрибут)
        original_threshold = self.confidence_threshold
        try:
            # Убеждаемся, что threshold - это float перед обучением
            self.confidence_threshold = float(original_threshold)
            trainer.train()
        finally:
            # Восстанавливаем значение
            self.confidence_threshold = float(original_threshold)
        
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
        confidence = float(confidences[0])  # Убеждаемся, что это float
        
        # Применяем порог уверенности
        # Убеждаемся, что оба значения - float
        if float(confidence) < float(self.confidence_threshold):
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
            # Убеждаемся, что confidence всегда float
            confidences.append(float(max_prob))
        
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
            # Убеждаемся, что оба значения - float
            conf_float = float(conf)
            threshold_float = float(self.confidence_threshold)
            if conf_float < threshold_float:
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
        
        import shutil
        import tempfile
        
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем во временную директорию, чтобы избежать проблем с открытыми файлами
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / path.name
            self.model.save_pretrained(str(temp_path))
            
            # Если целевая директория существует, удаляем её
            if path.exists():
                shutil.rmtree(path)
            
            # Перемещаем из временной директории в целевую
            shutil.move(str(temp_path), str(path))
    
    def load(self, model_path: str, confidence_threshold: Optional[float] = None):
        """
        Загружает сохраненную модель.
        
        Args:
            model_path: Путь к сохраненной модели
            confidence_threshold: Порог уверенности (если None, используется текущий)
        """
        import warnings
        
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Подавляем предупреждение о токенизаторе Mistral (если модель была обучена на Mistral)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*mistral.*regex.*", category=UserWarning)
            # Пытаемся загрузить с safetensors, если доступно
            try:
                self.model = SetFitModel.from_pretrained(str(path), use_safetensors=True)
            except:
                self.model = SetFitModel.from_pretrained(str(path))
        
        self.is_trained = True
        
        if confidence_threshold is not None:
            # Убеждаемся, что confidence_threshold всегда float
            self.confidence_threshold = float(confidence_threshold)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Получает эмбеддинги для списка текстов.
        Использует базовую модель эмбеддингов (без классификатора).
        
        Args:
            texts: Список текстов для получения эмбеддингов
            
        Returns:
            Список эмбеддингов (каждый эмбеддинг - список float)
            
        Raises:
            ValueError: Если модель не инициализирована
        """
        # Если модель не загружена, загружаем базовую модель
        if self.model is None:
            # Пытаемся загрузить с safetensors, если доступно
            try:
                self.model = SetFitModel.from_pretrained(self.model_name, use_safetensors=True)
            except:
                self.model = SetFitModel.from_pretrained(self.model_name)
        
        # Получаем базовую модель эмбеддингов (sentence-transformers)
        # SetFitModel имеет атрибут model_body для доступа к базовой модели
        if hasattr(self.model, 'model_body'):
            embedding_model = self.model.model_body
        elif hasattr(self.model, 'model'):
            # Альтернативный способ доступа
            embedding_model = self.model.model
        else:
            # Если нет доступа к базовой модели, используем весь model
            embedding_model = self.model
        
        # Получаем эмбеддинги через encode (стандартный метод sentence-transformers)
        if hasattr(embedding_model, 'encode'):
            embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        else:
            # Fallback: если encode недоступен, создаем новую базовую модель
            from sentence_transformers import SentenceTransformer
            base_model = SentenceTransformer(self.model_name)
            embeddings = base_model.encode(texts, convert_to_numpy=True)
        
        # Преобразуем в список списков float
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
        
        # Убеждаемся, что все элементы - списки float
        result = []
        for emb in embeddings:
            if isinstance(emb, (list, np.ndarray)):
                result.append([float(x) for x in emb])
            else:
                result.append([float(emb)])
        
        return result

