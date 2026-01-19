"""Эндпоинты для предсказаний и эмбеддингов."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from commands_classifier.api.state import get_classifier, get_config
from commands_classifier.api.utils import remove_punctuation
from commands_classifier.model import CommandsClassifier

router = APIRouter(tags=["predict"])


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


@router.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Получает эмбеддинги для текстов (TEI совместимый эндпоинт).
    
    Args:
        request: Запрос с текстами для эмбеддингов
        
    Returns:
        Эмбеддинги для каждого текста
    """
    classifier = get_classifier()
    config = get_config()
    
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


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Классифицирует один текст в команду.
    
    Args:
        request: Запрос с текстом для классификации
        
    Returns:
        Предсказанная команда и опционально уверенность
    """
    classifier = get_classifier()
    
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


@router.post("/predict/batch", response_model=PredictBatchResponse)
async def predict_batch(request: PredictBatchRequest):
    """
    Классифицирует список текстов в команды.
    
    Args:
        request: Запрос с текстами для классификации
        
    Returns:
        Предсказанные команды и опционально уверенности
    """
    classifier = get_classifier()
    
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
