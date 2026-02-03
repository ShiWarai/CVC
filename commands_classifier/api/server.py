"""FastAPI сервер для классификатора команд."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI

from commands_classifier import db
from commands_classifier.api.routes import (
    command_feedback_router,
    examples_router,
    health_router,
    load_from_hf_router,
    predict_router,
    training_router,
)
from commands_classifier.api.state import (
    get_default_device,
    load_model,
    set_config,
    set_default_device,
    set_training_manager,
)
from commands_classifier.api.training import TrainingManager

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Загружает конфигурацию из YAML файла. Требуется наличие config.yaml."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Конфигурация не найдена: {config_file}. Создайте config.yaml в корне проекта (см. config.yaml.example или README)."
        )
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError(f"Конфигурация в {config_file} пуста или невалидна.")
    # Проверка обязательных параметров
    model = config.get("model") or {}
    database = config.get("database") or {}
    if not (model.get("name") or "").strip():
        raise ValueError(
            f"В {config_file} отсутствует или пуст обязательный параметр model.name (базовая модель для обучения/эмбеддингов)."
        )
    if not (model.get("path") or "").strip():
        raise ValueError(
            f"В {config_file} отсутствует или пуст обязательный параметр model.path."
        )
    if not (database.get("path") or "").strip():
        raise ValueError(
            f"В {config_file} отсутствует или пуст обязательный параметр database.path."
        )
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
        cache_dir=cache_dir,
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
    title="CVC API", description="API для классификации голосовых команд", lifespan=lifespan
)

# Подключаем роутеры (версионирование указано в самих роутах)
app.include_router(predict_router)
app.include_router(training_router)
app.include_router(examples_router)
app.include_router(load_from_hf_router)
app.include_router(health_router)
app.include_router(command_feedback_router)
