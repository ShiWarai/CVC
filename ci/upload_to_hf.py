#!/usr/bin/env python3
"""
Скрипт для загрузки обученной модели на Hugging Face Hub.
"""

import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # noqa: E402

import yaml  # noqa: E402
from huggingface_hub import HfApi, create_repo, login  # noqa: E402


def load_config(config_path: str = "config.yaml") -> dict:
    """Загружает конфигурацию из YAML файла."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Основная функция для загрузки модели на HF."""
    print("=" * 60)
    print("Загрузка модели на Hugging Face Hub")
    print("=" * 60)

    # Получаем токен из переменных окружения
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("✗ Ошибка: HF_TOKEN не установлен в переменных окружения", file=sys.stderr)
        return 1

    # Получаем репозиторий из переменных окружения или конфига
    hf_repo_id = os.getenv("HF_REPO_ID")
    if not hf_repo_id:
        print("✗ Ошибка: HF_REPO_ID не установлен в переменных окружения", file=sys.stderr)
        print(
            "Установите переменную окружения HF_REPO_ID (например: username/model-name)",
            file=sys.stderr,
        )
        return 1

    # Получаем SHA коммита из GitHub для уникального коммита в HF
    github_sha = os.getenv("GITHUB_SHA", "")
    if github_sha:
        # Используем короткий SHA (первые 7 символов)
        commit_sha = github_sha[:7]
    else:
        commit_sha = None

    # Загружаем конфигурацию для определения пути к модели
    config = load_config()
    model_path = config.get("model", {}).get("path", "models/panda_commands")
    model_path_obj = Path(model_path)

    if not model_path_obj.exists():
        print(f"✗ Ошибка: Модель не найдена в {model_path}", file=sys.stderr)
        print("Сначала обучите модель", file=sys.stderr)
        return 1

    print("\nПараметры загрузки:")
    print(f"  Репозиторий HF: {hf_repo_id}")
    print(f"  Путь к модели: {model_path}")
    if commit_sha:
        print(f"  Коммит GitHub: {commit_sha}")

    try:
        # Авторизуемся в HF
        print("\nАвторизация в Hugging Face...")
        login(token=hf_token, add_to_git_credential=False)
        print("✓ Авторизация успешна")

        # Создаем API клиент
        api = HfApi()

        # Проверяем существование репозитория, создаем если нужно
        print(f"\nПроверка репозитория {hf_repo_id}...")
        try:
            api.repo_info(repo_id=hf_repo_id, repo_type="model")
            print("✓ Репозиторий существует")
        except Exception:
            print(f"Создание репозитория {hf_repo_id}...")
            create_repo(
                repo_id=hf_repo_id,
                repo_type="model",
                private=True,  # По умолчанию приватный репозиторий
                token=hf_token,
            )
            print("✓ Репозиторий создан")

        # Загружаем модель
        print("\nЗагрузка модели на Hugging Face Hub...")
        # Формируем уникальное сообщение коммита с SHA из GitHub
        if commit_sha:
            commit_message = f"CI: модель обучена и загружена ({commit_sha})"
        else:
            commit_message = "CI: модель обучена и загружена"

        api.upload_folder(
            folder_path=str(model_path_obj),
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.pyc", "__pycache__", "*.log"],
        )

        print("\n✓ Модель успешно загружена на Hugging Face Hub")
        print(f"  Репозиторий: https://huggingface.co/{hf_repo_id}")
        return 0

    except Exception as e:
        print(f"\n✗ Ошибка при загрузке модели: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
