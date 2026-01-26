"""Эндпоинты для загрузки модели с Hugging Face Hub."""

import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from commands_classifier.api.state import get_config, get_training_manager, load_model

router = APIRouter(tags=["load_from_hf"])


# Модели запросов/ответов
class LoadFromHfRequest(BaseModel):
    """Запрос на загрузку модели с Hugging Face."""
    repo_id: Optional[str] = None  # Например: "username/model-name". Если не указан, используется HF_REPO_ID из конфигурации сервера
    local_dir: Optional[str] = None  # Путь для сохранения (опционально, используется из config если не указан)


class LoadFromHfResponse(BaseModel):
    """Ответ на запрос загрузки модели."""
    message: str
    load_id: str


class LoadFromHfStatusResponse(BaseModel):
    """Ответ со статусом загрузки модели."""
    load_id: str
    status: str  # idle, pending, running, completed, failed
    progress: float
    local_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# Статус загрузки модели (модульный уровень)
_load_from_hf_status: Dict[str, Any] = {
    "load_id": None,
    "status": "idle",
    "progress": 0.0,
    "local_path": None,
    "error": None,
    "started_at": None,
    "completed_at": None
}


def _run_load_from_hf_task(repo_id: str, local_dir: str, load_id: str):
    """
    Фоновая задача для загрузки модели с Hugging Face Hub.
    """
    global _load_from_hf_status
    
    try:
        _load_from_hf_status["status"] = "running"
        _load_from_hf_status["progress"] = 0.1
        
        # Импортируем huggingface_hub
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            _load_from_hf_status["status"] = "failed"
            _load_from_hf_status["error"] = "huggingface-hub не установлен. Установите: pip install huggingface-hub"
            _load_from_hf_status["completed_at"] = datetime.now().isoformat()
            return
        
        # Получаем токен из переменных окружения
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        _load_from_hf_status["progress"] = 0.2
        
        # Создаем директорию для модели
        local_path_obj = Path(local_dir)
        local_path_obj.mkdir(parents=True, exist_ok=True)
        
        _load_from_hf_status["progress"] = 0.3
        
        # Загружаем модель
        try:
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path_obj),
                token=hf_token,
                local_dir_use_symlinks=False  # Не используем симлинки для Docker
            )
            _load_from_hf_status["progress"] = 0.9
            
            # Проверяем, что модель загружена
            if not local_path_obj.exists() or not any(local_path_obj.iterdir()):
                _load_from_hf_status["status"] = "failed"
                _load_from_hf_status["error"] = "Модель не была загружена или директория пуста"
                _load_from_hf_status["completed_at"] = datetime.now().isoformat()
                return
            
            _load_from_hf_status["status"] = "completed"
            _load_from_hf_status["progress"] = 1.0
            _load_from_hf_status["local_path"] = str(local_path_obj)
            _load_from_hf_status["completed_at"] = datetime.now().isoformat()
            
            # Перезагружаем модель в память
            try:
                load_model()
            except Exception as e:
                # Не критично, модель загружена на диск
                print(f"Предупреждение: не удалось перезагрузить модель в память: {e}")
            
        except Exception as e:
            _load_from_hf_status["status"] = "failed"
            _load_from_hf_status["error"] = f"Ошибка при загрузке модели: {str(e)}"
            _load_from_hf_status["completed_at"] = datetime.now().isoformat()
            
    except Exception as e:
        _load_from_hf_status["status"] = "failed"
        _load_from_hf_status["error"] = str(e)
        _load_from_hf_status["completed_at"] = datetime.now().isoformat()


@router.post("/load_from_hf", response_model=LoadFromHfResponse)
async def load_from_hf(request: LoadFromHfRequest):
    """
    Загружает модель с Hugging Face Hub.
    Запускается в фоновом режиме.
    
    Args:
        request: Параметры загрузки
        - repo_id: ID репозитория на Hugging Face (например: "username/model-name")
        - local_dir: Путь для сохранения (опционально, используется из config если не указан)
        
    Returns:
        ID задачи загрузки
    """
    global _load_from_hf_status
    
    training_manager = get_training_manager()
    config = get_config()
    
    # Проверяем, что не идёт загрузка
    if _load_from_hf_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Загрузка уже выполняется")
    
    # Проверяем, что не идёт обучение
    if training_manager is not None and training_manager.is_training():
        raise HTTPException(status_code=409, detail="Невозможно загрузить модель во время обучения")
    
    # Определяем repo_id - из запроса или из конфигурации сервера
    repo_id = request.repo_id
    if not repo_id:
        repo_id = os.getenv("HF_REPO_ID")
        if not repo_id:
            raise HTTPException(status_code=400, detail="repo_id не указан в запросе и HF_REPO_ID не найден в конфигурации сервера")
    
    # Определяем путь для сохранения
    if request.local_dir:
        local_dir = request.local_dir
    else:
        # Используем путь из конфига
        model_path = config["model"]["path"]
        local_dir = str(Path(model_path).parent / Path(model_path).name)
    
    # Генерируем ID задачи
    load_id = str(uuid.uuid4())[:8]
    
    # Сбрасываем статус
    _load_from_hf_status = {
        "load_id": load_id,
        "status": "pending",
        "progress": 0.0,
        "local_path": None,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    # Запускаем в фоновом потоке
    thread = threading.Thread(
        target=_run_load_from_hf_task,
        args=(repo_id, local_dir, load_id),
        daemon=True
    )
    thread.start()
    
    return LoadFromHfResponse(
        message="Загрузка модели запущена в фоновом режиме",
        load_id=load_id
    )


@router.get("/load_from_hf/status", response_model=LoadFromHfStatusResponse)
async def get_load_from_hf_status():
    """
    Возвращает статус загрузки модели.
    
    Returns:
        Статус загрузки (id, status, progress, local_path, error, timestamps)
    """
    return LoadFromHfStatusResponse(
        load_id=_load_from_hf_status["load_id"] or "",
        status=_load_from_hf_status["status"],
        progress=_load_from_hf_status["progress"],
        local_path=_load_from_hf_status["local_path"],
        error=_load_from_hf_status["error"],
        started_at=_load_from_hf_status["started_at"],
        completed_at=_load_from_hf_status["completed_at"]
    )
