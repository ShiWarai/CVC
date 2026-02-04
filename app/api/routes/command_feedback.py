"""Эндпоинт для загрузки репорта «исправить команду» из RDS-2P-Salute."""

import logging
from typing import Any, List

import requests
from fastapi import APIRouter, HTTPException

from app.api.state import get_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["command-feedback"])

# URL по умолчанию (внутри Docker-сети robot-services-network)
DEFAULT_COMMAND_FEEDBACK_URL = "http://rds-2p-salute-app:8000/v1/admin/command-feedback"
REQUEST_TIMEOUT = 30


@router.get("/v1/command-feedback")
def get_command_feedback() -> List[dict]:
    """
    Загружает записи обратной связи по командам из приложения RDS-2P-Salute.

    Эндпоинт доступен только из локальной/внутренней сети на стороне RDS;
    при запросе с публичного IP RDS вернёт 403. CVC проксирует ответ как есть.
    """
    config = get_config()
    url = (
        config.get("command_feedback", {}).get("url")
        or config.get("rds", {}).get("command_feedback_url")
        or DEFAULT_COMMAND_FEEDBACK_URL
    ).strip()

    if not url:
        raise HTTPException(
            status_code=503,
            detail="command_feedback.url не задан в конфигурации",
        )

    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data: Any = resp.json()
        if not isinstance(data, list):
            raise HTTPException(
                status_code=502,
                detail="Ответ RDS не является массивом",
            )
        return data
    except requests.exceptions.Timeout:
        logger.warning("Timeout при запросе command-feedback: %s", url)
        raise HTTPException(
            status_code=504,
            detail="Таймаут при обращении к сервису RDS",
        )
    except requests.exceptions.ConnectionError as e:
        logger.warning("Ошибка соединения с RDS: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Сервис RDS недоступен (проверьте сеть и имя хоста)",
        )
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise HTTPException(
                status_code=502,
                detail="RDS вернул 403 Forbidden (доступ только из внутренней сети)",
            )
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка RDS: {e.response.status_code if e.response else str(e)}",
        )
    except ValueError as e:
        logger.warning("Невалидный JSON от RDS: %s", e)
        raise HTTPException(status_code=502, detail="Невалидный JSON в ответе RDS")
