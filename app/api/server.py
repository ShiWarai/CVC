"""FastAPI сервер для классификатора команд."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI

from app.adapters import persistence as db
from app.api.routes import (
    command_feedback_router,
    examples_router,
    health_router,
    load_from_hf_router,
    predict_router,
    training_router,
)
from app.api.state import (
    get_classifier,
    get_config,
    get_default_device,
    load_model,
    set_config,
    set_default_device,
    set_examples_use_case,
    set_predict_use_case,
    set_training_manager,
)
from app.api.training import TrainingManager
from app.application.examples_use_case import ExamplesUseCase
from app.application.predict_use_case import PredictUseCase

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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

        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))

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

    # Сценарии (use cases) для роутов
    set_predict_use_case(PredictUseCase(get_classifier))
    set_examples_use_case(
        ExamplesUseCase(db._default_repo, lambda: get_config()["database"]["path"])
    )

    # Менеджер обучения с callback для перезагрузки модели
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
        example_repository=db._default_repo,
    )
    set_training_manager(training_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    init_app()

    t0 = time.monotonic()
    loaded = await asyncio.to_thread(load_model)
    load_sec = time.monotonic() - t0
    if loaded:
        logger.info("Модель загружена за %.1f с", load_sec)
        classifier = get_classifier()
        if classifier:
            t1 = time.monotonic()
            await asyncio.to_thread(classifier.predict, "warmup", False)
            logger.info("Warmup predict завершён за %.1f с", time.monotonic() - t1)
    else:
        logger.warning("Модель не загружена (%.1f с)", load_sec)

    yield


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
