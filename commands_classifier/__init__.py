"""CVC - Classification of Voice Commands. Few-shot learning классификатор голосовых команд с использованием SetFit."""

from pathlib import Path
import os

# Загружаем переменные окружения из .env файла (для локального использования)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        # Устанавливаем токен Hugging Face из .env, если он есть
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
except ImportError:
    # python-dotenv не установлен, используем системные переменные окружения
    pass

__version__ = "0.1.0"

