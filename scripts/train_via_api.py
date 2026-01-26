#!/usr/bin/env python3
"""
Скрипт для обучения модели в CI/CD пайплайне.
Может работать через API или напрямую (без API).
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Добавляем корневую директорию проекта в путь
# В Docker контейнере это /app, на хосте - родительская директория скрипта
if Path("/app").exists():
    # Работаем внутри Docker контейнера
    project_root = Path("/app")
else:
    # Работаем на хосте
    project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from commands_classifier.client import CVCApiClient
from commands_classifier.model import CommandsClassifier
from commands_classifier import db as db_module
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Загружает конфигурацию из YAML файла."""
    # В Docker контейнере используем /app/config.yaml, на хосте - относительно project_root
    if Path("/app").exists():
        config_file = Path("/app") / "config.yaml" if config_path == "config.yaml" else Path(config_path)
    else:
        config_file = project_root / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def wait_for_server(url: str = "http://localhost:20001", timeout: int = 120) -> bool:
    """Ожидает готовности сервера."""
    import requests
    
    print(f"Ожидание готовности сервера на {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Сервер готов")
                return True
        except Exception:
            pass
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(f"\n✗ Сервер не ответил за {timeout} секунд")
    return False


def main():
    """Основная функция для обучения модели в CI/CD пайплайне."""
    print("=" * 60)
    print("Запуск обучения модели (CI/CD)")
    print("=" * 60)
    
    # Загружаем конфигурацию
    config = load_config()
    
    # Определяем путь к БД для обучения
    # Если TRAINING_DB_PATH не указан, используем отдельную БД для CI/CD по умолчанию
    training_db_path = os.getenv("TRAINING_DB_PATH", "").strip()
    if not training_db_path:
        # По умолчанию для CI/CD используем отдельную БД
        training_db_path = "db/training_data_ci.db"
    
    original_db_path = config.get("database", {}).get("path", "db/training_data.db")
    
    print(f"\nБаза данных:")
    print(f"  Основная БД (из config): {original_db_path}")
    print(f"  БД для обучения: {training_db_path}")
    
    # Инициализируем БД для обучения (загружает данные из CSV)
    # В Docker пути работают относительно /app, но скрипт запускается на хосте
    # Поэтому используем пути относительно project_root
    csv_path = config.get("database", {}).get("csv_migration_path", "data")
    db_module.init_db(training_db_path, csv_path)
    print(f"✓ База данных для обучения инициализирована")
    
    # Параметры обучения из конфига или переменных окружения
    training_config = config.get("training", {})
    # Обрабатываем пустые строки: если переменная окружения пустая, используем значение из конфига
    num_iterations_env = os.getenv("NUM_ITERATIONS", "").strip()
    num_iterations = int(num_iterations_env) if num_iterations_env else training_config.get("iterations", 20)
    
    num_epochs_env = os.getenv("NUM_EPOCHS", "").strip()
    num_epochs = int(num_epochs_env) if num_epochs_env else training_config.get("epochs", 1)
    
    batch_size_env = os.getenv("BATCH_SIZE", "").strip()
    batch_size = int(batch_size_env) if batch_size_env else training_config.get("batch_size", 128)
    
    learning_rate_env = os.getenv("LEARNING_RATE", "").strip()
    learning_rate = float(learning_rate_env) if learning_rate_env else training_config.get("learning_rate", 2e-5)
    
    print(f"\nПараметры обучения:")
    print(f"  Итераций: {num_iterations}")
    print(f"  Эпох: {num_epochs}")
    print(f"  Размер батча: {batch_size}")
    print(f"  Скорость обучения: {learning_rate}")
    
    # Сбрасываем статус обучения напрямую в БД (не через API)
    print(f"\nСброс статуса обучения в БД {training_db_path}...")
    try:
        reset_count = db_module.reset_training_status(training_db_path)
        print(f"✓ Сброс выполнен: {reset_count} примеров помечено как необученные")
    except Exception as e:
        print(f"⚠️  Предупреждение: не удалось выполнить reset: {e}")
        print("Продолжаем обучение...")
    
    # Загружаем данные из БД для обучения
    print(f"\nЗагрузка данных из БД для обучения...")
    texts, labels, example_ids = db_module.get_examples_for_training(training_db_path)
    
    if len(texts) == 0:
        print("✗ Ошибка: нет необученных данных в базе данных", file=sys.stderr)
        print(f"  Проверьте БД: {training_db_path}", file=sys.stderr)
        return 1
    
    print(f"✓ Загружено {len(texts)} примеров для обучения")
    
    # Обучаем модель напрямую (не через API)
    print(f"\nЗапуск обучения модели...")
    try:
        model_config = config.get("model", {})
        model_path = model_config.get("path", "models/panda_commands")
        model_name = model_config.get("name", "google/embeddinggemma-300M")
        confidence_threshold = float(model_config.get("confidence_threshold", 0.5))
        cache_dir = model_config.get("cache_dir")
        
        # Определяем устройство для обучения
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Устройство: {device}")
        
        # Создаем и обучаем модель
        classifier = CommandsClassifier(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            cache_dir=cache_dir
        )
        
        print("  Обучение модели...")
        metrics = classifier.train(
            texts=texts,
            labels=labels,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )
        
        # Сохраняем модель
        print(f"  Сохранение модели в {model_path}...")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        classifier.save(model_path)
        
        print(f"\n✓ Обучение завершено успешно!")
        if metrics:
            print("\nМетрики качества модели:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Помечаем примеры как обученные
        print(f"\nОбновление статуса примеров в БД...")
        from commands_classifier.db import mark_examples_as_trained
        mark_examples_as_trained(training_db_path, example_ids)
        print(f"✓ {len(example_ids)} примеров помечено как обученные")
        
        # Упаковка модели выполняется через API (нужен сервер для /package endpoint)
        # Запускаем сервер, если он не запущен
        server_config = config.get("server", {})
        api_url = f"http://{server_config.get('host', 'localhost')}:{server_config.get('port', 20001)}"
        
        print(f"\nПодготовка к упаковке модели...")
        client = None
        try:
            client = CVCApiClient(api_url)
            client.health()
            print("✓ Сервер уже запущен")
        except Exception:
            # Внутри Docker контейнера сервер должен быть уже запущен
            # Просто ждем его готовности
            print("Ожидание готовности сервера...")
            if wait_for_server(api_url, timeout=60):
                client = CVCApiClient(api_url)
            else:
                print("⚠️  Сервер не готов, пропускаем упаковку...")
        
        # Упаковываем модель через API
        if client:
            print(f"\nУпаковка модели через API...")
            try:
                package_result = client.package()
                package_id = package_result.get("package_id")
                print(f"✓ Упаковка запущена. ID задачи: {package_id}")
                
                # Ждем завершения упаковки
                while True:
                    time.sleep(2)
                    status = client.get_package_status()
                    
                    if status.get("status") == "completed":
                        output_path = status.get("output_path")
                        print(f"\n✓ Модель упакована: {output_path}")
                        # Сохраняем путь для использования в CI/CD
                        archive_file = os.getenv("GITHUB_OUTPUT", "/tmp/archive_path.txt")
                        with open(archive_file, "a") as f:
                            f.write(f"archive_path={output_path}\n")
                        print(f"ARCHIVE_PATH={output_path}")
                        break
                    elif status.get("status") == "failed":
                        error = status.get("error", "Неизвестная ошибка")
                        print(f"\n✗ Упаковка завершилась с ошибкой: {error}", file=sys.stderr)
                        return 1
                    elif status.get("status") == "running":
                        progress = status.get("progress", 0)
                        print(f"Прогресс упаковки: {progress:.1%}", end='\r')
            except Exception as e:
                print(f"\n⚠️  Ошибка при упаковке: {e}", file=sys.stderr)
                print("Продолжаем без упаковки...")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Ошибка при обучении: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Если мы запускали сервер, не останавливаем его (может использоваться дальше)
        # Можно добавить опцию для остановки, если нужно
        pass


if __name__ == "__main__":
    sys.exit(main())
