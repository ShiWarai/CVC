"""CVC - Classification of Voice Commands. Few-shot learning классификатор голосовых команд с использованием SetFit."""

from pathlib import Path

# Загружаем переменные окружения из .env файла (для локального использования)
# Только загружаем переменные, без вызова huggingface_hub.login() - это будет сделано
# только когда действительно нужно (при инициализации сервера)
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        # Удаляем BOM (UTF-8 Byte Order Mark) если он есть, так как он мешает load_dotenv()
        try:
            with open(env_path, "r", encoding="utf-8-sig") as f:
                env_content = f.read()
            # Временно перезаписываем файл без BOM
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)
        except Exception:
            pass

        load_dotenv(env_path, override=True)
except ImportError:
    # python-dotenv не установлен, используем системные переменные окружения
    pass
except Exception:
    pass

__version__ = "0.1.0"
