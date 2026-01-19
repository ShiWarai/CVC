"""Эндпоинты для упаковки модели."""

import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from commands_classifier.api.state import get_config, get_training_manager

router = APIRouter(tags=["package"])


# Модели запросов/ответов
class PackageResponse(BaseModel):
    """Ответ на запрос упаковки модели."""
    message: str
    package_id: str


class PackageStatusResponse(BaseModel):
    """Ответ со статусом упаковки модели."""
    package_id: str
    status: str  # idle, pending, running, completed, failed
    progress: float
    output_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# Статус упаковки модели (модульный уровень)
_package_status: Dict[str, Any] = {
    "package_id": None,
    "status": "idle",
    "progress": 0.0,
    "output_path": None,
    "error": None,
    "started_at": None,
    "completed_at": None
}


def _run_package_task(model_path: str, output_path: str, package_id: str):
    """
    Фоновая задача для упаковки модели в tar.gz (только Linux).
    Использует системную команду tar для упаковки.
    """
    global _package_status
    
    try:
        _package_status["status"] = "running"
        _package_status["progress"] = 0.1
        
        model_path_obj = Path(model_path)
        output_path_obj = Path(output_path)
        
        # Убеждаемся, что выходная директория существует
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Удаляем старый архив, если существует
        if output_path_obj.exists():
            output_path_obj.unlink()
        
        _package_status["progress"] = 0.2
        
        # Используем tar для упаковки (Linux)
        # tar -czvf output.tar.gz -C parent_dir model_dir_name
        parent_dir = model_path_obj.parent
        model_dir_name = model_path_obj.name
        
        cmd = [
            "tar", "-czvf", str(output_path_obj),
            "-C", str(parent_dir),
            model_dir_name
        ]
        
        _package_status["progress"] = 0.3
        
        # Запускаем процесс
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        _package_status["progress"] = 0.5
        
        # Ждём завершения
        stdout, stderr = process.communicate()
        
        _package_status["progress"] = 0.9
        
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='replace')
            _package_status["status"] = "failed"
            _package_status["error"] = f"tar вернул код {process.returncode}: {error_msg}"
            _package_status["completed_at"] = datetime.now().isoformat()
            return
        
        # Проверяем, что файл создан
        if not output_path_obj.exists():
            _package_status["status"] = "failed"
            _package_status["error"] = "Архив не был создан"
            _package_status["completed_at"] = datetime.now().isoformat()
            return
        
        _package_status["status"] = "completed"
        _package_status["progress"] = 1.0
        _package_status["output_path"] = str(output_path_obj)
        _package_status["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        _package_status["status"] = "failed"
        _package_status["error"] = str(e)
        _package_status["completed_at"] = datetime.now().isoformat()


@router.post("/package", response_model=PackageResponse)
async def package_model():
    """
    Упаковывает обученную модель в tar.gz архив (только Linux).
    Запускается в фоновом режиме.
    
    Returns:
        ID задачи упаковки
    """
    global _package_status
    
    training_manager = get_training_manager()
    config = get_config()
    
    # Проверяем, что не идёт упаковка
    if _package_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Упаковка уже выполняется")
    
    # Проверяем, что не идёт обучение
    if training_manager is not None and training_manager.is_training():
        raise HTTPException(status_code=409, detail="Невозможно упаковать модель во время обучения")
    
    model_path = config["model"]["path"]
    model_path_obj = Path(model_path)
    
    # Проверяем, что модель существует
    if not model_path_obj.exists():
        raise HTTPException(status_code=404, detail="Модель не найдена. Сначала обучите модель.")
    
    # Генерируем имя архива
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{model_path_obj.name}_{timestamp}.tar.gz"
    output_path = model_path_obj.parent / output_filename
    
    # Генерируем ID задачи
    package_id = str(uuid.uuid4())[:8]
    
    # Сбрасываем статус
    _package_status = {
        "package_id": package_id,
        "status": "pending",
        "progress": 0.0,
        "output_path": None,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    # Запускаем в фоновом потоке
    thread = threading.Thread(
        target=_run_package_task,
        args=(str(model_path_obj), str(output_path), package_id),
        daemon=True
    )
    thread.start()
    
    return PackageResponse(
        message="Упаковка модели запущена в фоновом режиме",
        package_id=package_id
    )


@router.get("/package/status", response_model=PackageStatusResponse)
async def get_package_status():
    """
    Возвращает статус упаковки модели.
    
    Returns:
        Статус упаковки (id, status, progress, output_path, error, timestamps)
    """
    return PackageStatusResponse(
        package_id=_package_status["package_id"] or "",
        status=_package_status["status"],
        progress=_package_status["progress"],
        output_path=_package_status["output_path"],
        error=_package_status["error"],
        started_at=_package_status["started_at"],
        completed_at=_package_status["completed_at"]
    )
