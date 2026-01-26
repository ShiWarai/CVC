"""FastAPI сервер для классификатора команд."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI

from commands_classifier.model import CommandsClassifier
from commands_classifier import db
from commands_classifier.api.training import TrainingManager
from commands_classifier.api.state import (
    get_classifier, set_classifier, unload_classifier,
    get_config, set_config,
    get_training_manager, set_training_manager,
    get_default_device, set_default_device,
    load_model
)
from commands_classifier.api.routes import (
    predict_router,
    training_router,
    examples_router,
    package_router,
    download_router,
    health_router
)

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Загружает конфигурацию из YAML файла."""
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
    set_config(config)
    return config


def init_app():
    """Инициализирует приложение при запуске."""
    # Инициализируем токен Hugging Face
    try:
        import os
        import huggingface_hub
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    except ImportError:
        pass
    except Exception:
        pass
    
    config = load_config()
    
    # Автоматически определяем устройство для обучения
    try:
        import torch
        if torch.cuda.is_available():
            set_default_device("cuda")
        else:
            set_default_device("cpu")
    except ImportError:
        set_default_device("cpu")
    except Exception:
        set_default_device("cpu")
    
    # Инициализируем базу данных
    db_path = config["database"]["path"]
    csv_path = config["database"].get("csv_migration_path")
    db.init_db(db_path, csv_path)
    
    # Инициализируем менеджер обучения с callback для перезагрузки модели
    model_path = config["model"]["path"]
    model_name = config["model"]["name"]
    confidence_threshold = float(config["model"].get("confidence_threshold", 0.5))
    cache_dir = config["model"].get("cache_dir")
    
    training_manager = TrainingManager(
        db_path, 
        model_path, 
        model_name, 
        confidence_threshold,
        on_training_complete=load_model,
        default_device=get_default_device(),
        cache_dir=cache_dir
    )
    set_training_manager(training_manager)
    
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

# Подключаем роутеры
app.include_router(predict_router)
app.include_router(training_router)
app.include_router(examples_router)
app.include_router(package_router)
app.include_router(download_router)
app.include_router(health_router)
