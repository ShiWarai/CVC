"""FastAPI сервер для классификатора команд."""

import yaml
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from commands_classifier.model import CommandsClassifier
from commands_classifier import db
from commands_classifier.api.training import TrainingManager

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def remove_punctuation(text: str) -> str:
    """
    Удаляет все знаки препинания из текста.
    
    Args:
        text: Исходный текст
        
    Returns:
        Текст без знаков препинания
    """
    # Удаляем все знаки препинания, оставляя только буквы, цифры и пробелы
    # Используем регулярное выражение для удаления всех знаков препинания
    text = re.sub(r'[^\w\s]', '', text)
    # Удаляем множественные пробелы и обрезаем
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Модели запросов/ответов
class EmbedRequest(BaseModel):
    """Запрос для получения эмбеддингов (TEI совместимый)."""
    inputs: List[str]


class EmbedResponse(BaseModel):
    """Ответ с эмбеддингами (TEI совместимый)."""
    embeddings: List[List[float]]


class PredictRequest(BaseModel):
    """Запрос для классификации команд."""
    text: str
    return_confidence: bool = False


class PredictResponse(BaseModel):
    """Ответ с предсказанием команды."""
    command: str
    confidence: Optional[float] = None


class PredictBatchRequest(BaseModel):
    """Запрос для batch классификации."""
    texts: List[str]
    return_confidence: bool = False


class PredictBatchResponse(BaseModel):
    """Ответ с batch предсказаниями."""
    commands: List[str]
    confidences: Optional[List[float]] = None


class TrainRequest(BaseModel):
    """Запрос для запуска обучения."""
    num_iterations: Optional[int] = None
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None


class TrainResponse(BaseModel):
    """Ответ на запрос обучения."""
    training_id: str
    message: str


class ExampleRequest(BaseModel):
    """Запрос для добавления примера."""
    text: str
    command: str


class ExampleResponse(BaseModel):
    """Ответ с информацией о примере."""
    id: int
    text: str
    command: str


# Глобальные переменные
classifier: Optional[CommandsClassifier] = None
training_manager: Optional[TrainingManager] = None
config: Dict[str, Any] = {}
# Автоматически определённое устройство для обучения (определяется при старте)
default_device: str = "cpu"


def load_config(config_path: str = "config.yaml"):
    """Загружает конфигурацию из YAML файла."""
    global config
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # Конфигурация по умолчанию
        config = {
            "server": {"host": "0.0.0.0", "port": 20001},
            "model": {
                "path": "models/my_model",
                "name": "deepvk/USER-bge-m3",
                "confidence_threshold": 0.5
            },
            "database": {
                "path": "db/training_data.db",
                "csv_migration_path": "data"
            },
            "training": {
                "iterations": 20,
                "epochs": 1,
                "batch_size": 16,
                "learning_rate": 2e-5
            }
        }


def load_model():
    """Загружает модель из файла."""
    global classifier
    
    model_path = config["model"]["path"]
    model_path_obj = Path(model_path)
    
    if model_path_obj.exists():
        try:
            # Убеждаемся, что confidence_threshold - это float
            confidence_threshold = float(config["model"].get("confidence_threshold", 0.5))
            
            # Выгружаем старую модель из памяти
            if classifier is not None:
                del classifier
                import gc
                gc.collect()
            
            # Загружаем новую модель
            cache_dir = config["model"].get("cache_dir")
            classifier = CommandsClassifier(
                confidence_threshold=confidence_threshold,
                cache_dir=cache_dir
            )
            classifier.load(model_path, confidence_threshold=confidence_threshold)
            print(f"Модель успешно загружена из {model_path}")
            return True
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить модель: {e}")
            classifier = None
            return False
    else:
        classifier = None
        return False


def init_app():
    """Инициализирует приложение при запуске."""
    global classifier, training_manager
    
    # Инициализируем токен Hugging Face (только при запуске сервера, не при импорте клиента)
    try:
        import os
        import huggingface_hub
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    except ImportError:
        # huggingface_hub не установлен, используем только переменные окружения
        pass
    except Exception:
        pass
    
    load_config()
    
    # Автоматически определяем устройство для обучения
    global default_device
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        torch_version = str(torch.__version__)
        if cuda_available:
            default_device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            # Проверяем, является ли это AMD GPU через ROCm
            if "rocm" in torch_version.lower() or "+rocmsdk" in torch_version.lower():
                print(f"✓ AMD ROCm доступен. Обучение будет выполняться на AMD GPU: {device_name}")
            else:
                print(f"✓ CUDA доступна. Обучение будет выполняться на GPU: {device_name}")
        else:
            default_device = "cpu"
            # Проверяем, не CPU-only ли версия PyTorch
            if "+cpu" in torch_version.lower():
                print("⚠ Обнаружена CPU-only версия PyTorch.")
                print("   Для использования NVIDIA CUDA установите зависимости из requirements-cuda.txt:")
                print("   pip install -r requirements-cuda.txt")
                print("   Для использования AMD ROCm установите зависимости из requirements-rocm.txt:")
                print("   pip install -r requirements-rocm.txt")
            print("ℹ GPU недоступен. Обучение будет выполняться на CPU")
    except ImportError:
        default_device = "cpu"
        print("ℹ PyTorch не установлен. Обучение будет выполняться на CPU")
    except Exception as e:
        default_device = "cpu"
        print(f"ℹ Ошибка при проверке GPU: {e}. Обучение будет выполняться на CPU")
    
    # Инициализируем базу данных
    db_path = config["database"]["path"]
    csv_path = config["database"].get("csv_migration_path")
    db.init_db(db_path, csv_path)
    
    # Инициализируем менеджер обучения с callback для перезагрузки модели
    model_path = config["model"]["path"]
    model_name = config["model"]["name"]
    confidence_threshold = float(config["model"].get("confidence_threshold", 0.5))
    cache_dir = config["model"].get("cache_dir")  # Путь для кэширования базовой модели
    training_manager = TrainingManager(
        db_path, 
        model_path, 
        model_name, 
        confidence_threshold,
        on_training_complete=load_model,  # Передаем callback для перезагрузки
        default_device=default_device,  # Передаем автоматически определённое устройство
        cache_dir=cache_dir  # Передаем путь для кэширования
    )
    
    # Пытаемся загрузить модель
    load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Startup
    init_app()
    yield
    # Shutdown (если нужно что-то очистить)


app = FastAPI(
    title="CVC API",
    description="API для классификации голосовых команд",
    lifespan=lifespan
)


# Эндпоинты

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Получает эмбеддинги для текстов (TEI совместимый эндпоинт).
    
    Args:
        request: Запрос с текстами для эмбеддингов
        
    Returns:
        Эмбеддинги для каждого текста
    """
    if classifier is None:
        # Если модель не загружена, создаем базовую модель для эмбеддингов
        cache_dir = config["model"].get("cache_dir")
        temp_classifier = CommandsClassifier(
            model_name=config["model"]["name"],
            cache_dir=cache_dir
        )
        # Очищаем знаки препинания из всех текстов
        cleaned_inputs = [remove_punctuation(text) for text in request.inputs]
        embeddings = temp_classifier.get_embeddings(cleaned_inputs)
    else:
        # Очищаем знаки препинания из всех текстов
        cleaned_inputs = [remove_punctuation(text) for text in request.inputs]
        embeddings = classifier.get_embeddings(cleaned_inputs)
    
    return EmbedResponse(embeddings=embeddings)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Классифицирует один текст в команду.
    
    Args:
        request: Запрос с текстом для классификации
        
    Returns:
        Предсказанная команда и опционально уверенность
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Сначала обучите модель.")
    
    try:
        # Очищаем знаки препинания из текста
        cleaned_text = remove_punctuation(request.text)
        
        if request.return_confidence:
            command, confidence = classifier.predict(cleaned_text, return_confidence=True)
            return PredictResponse(command=command, confidence=confidence)
        else:
            command = classifier.predict(cleaned_text)
            return PredictResponse(command=command)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")


@app.post("/predict/batch", response_model=PredictBatchResponse)
async def predict_batch(request: PredictBatchRequest):
    """
    Классифицирует список текстов в команды.
    
    Args:
        request: Запрос с текстами для классификации
        
    Returns:
        Предсказанные команды и опционально уверенности
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Сначала обучите модель.")
    
    try:
        # Очищаем знаки препинания из всех текстов
        cleaned_texts = [remove_punctuation(text) for text in request.texts]
        
        if request.return_confidence:
            commands, confidences = classifier.predict_batch(cleaned_texts, return_confidence=True)
            return PredictBatchResponse(commands=commands, confidences=confidences)
        else:
            commands = classifier.predict_batch(cleaned_texts)
            return PredictBatchResponse(commands=commands)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")


@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    """
    Запускает обучение модели в фоновом режиме.
    
    Args:
        request: Параметры обучения (опционально, используются значения по умолчанию из config)
        - num_iterations: Количество итераций (по умолчанию из config.yaml)
        - num_epochs: Количество эпох (по умолчанию из config.yaml)
        - batch_size: Размер батча (по умолчанию из config.yaml)
        - learning_rate: Скорость обучения (по умолчанию из config.yaml)
        
    Returns:
        ID задачи обучения
    """
    if training_manager is None:
        raise HTTPException(status_code=500, detail="Training manager не инициализирован")
    
    if training_manager.is_training():
        raise HTTPException(status_code=409, detail="Обучение уже запущено")
    
    # Используем параметры из запроса или из конфига
    training_config = config["training"]
    num_iterations = request.num_iterations or training_config["iterations"]
    num_epochs = request.num_epochs or training_config["epochs"]
    batch_size = request.batch_size or training_config["batch_size"]
    learning_rate = request.learning_rate or training_config["learning_rate"]
    
    # Убеждаемся, что все числовые параметры имеют правильный тип
    num_iterations = int(num_iterations)
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)  # Важно: преобразуем в float
    
    try:
        training_id = training_manager.start_training(
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        return TrainResponse(
            training_id=training_id,
            message="Обучение запущено в фоновом режиме"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при запуске обучения: {str(e)}")


@app.get("/train/status")
async def get_training_status():
    """
    Возвращает статус текущего обучения.
    
    Returns:
        Статус обучения (id, status, progress, error, timestamps)
    """
    if training_manager is None:
        raise HTTPException(status_code=500, detail="Training manager не инициализирован")
    
    return training_manager.get_status()


@app.get("/health")
async def health():
    """
    Проверка работоспособности сервера (TEI совместимый).
    
    Returns:
        Статус сервера
    """
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "training_active": training_manager.is_training() if training_manager else False
    }


@app.get("/metrics")
async def metrics():
    """
    Метрики сервера (TEI совместимый).
    
    Returns:
        Метрики сервера
    """
    db_path = config["database"]["path"]
    example_count = db.count_examples(db_path)
    training_stats = db.get_training_stats(db_path)
    
    return {
        "total_examples": example_count,
        "trained_examples": training_stats["trained"],
        "untrained_examples": training_stats["untrained"],
        "model_loaded": classifier is not None,
        "training_status": training_manager.get_status() if training_manager else None
    }


@app.get("/examples", response_model=List[ExampleResponse])
async def get_examples():
    """
    Получает все примеры из базы данных.
    
    Returns:
        Список всех примеров
    """
    db_path = config["database"]["path"]
    examples = db.get_all_examples(db_path)
    return [ExampleResponse(id=ex[0], text=ex[1], command=ex[2]) for ex in examples]


@app.post("/examples", response_model=ExampleResponse, status_code=201)
async def add_example(request: ExampleRequest):
    """
    Добавляет новый пример в базу данных.
    
    Args:
        request: Данные примера (text, command)
        
    Returns:
        Созданный пример с ID
    """
    db_path = config["database"]["path"]
    try:
        # Очищаем знаки препинания из текста перед сохранением
        cleaned_text = remove_punctuation(request.text)
        example_id = db.add_example(db_path, cleaned_text, request.command)
        return ExampleResponse(id=example_id, text=cleaned_text, command=request.command)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при добавлении примера: {str(e)}")


@app.delete("/examples/{example_id}")
async def delete_example(example_id: int):
    """
    Удаляет пример по ID.
    
    Args:
        example_id: ID примера для удаления
        
    Returns:
        Сообщение об успешном удалении
    """
    db_path = config["database"]["path"]
    deleted = db.delete_example(db_path, example_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Пример с ID {example_id} не найден")
    return {"message": f"Пример {example_id} успешно удален"}


@app.get("/examples/{example_id}", response_model=ExampleResponse)
async def get_example(example_id: int):
    """
    Получает пример по ID.
    
    Args:
        example_id: ID примера
        
    Returns:
        Пример
    """
    db_path = config["database"]["path"]
    example = db.get_example_by_id(db_path, example_id)
    if example is None:
        raise HTTPException(status_code=404, detail=f"Пример с ID {example_id} не найден")
    return ExampleResponse(id=example[0], text=example[1], command=example[2])

