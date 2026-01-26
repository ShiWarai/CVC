#!/usr/bin/env python3
"""
Скрипт для обучения модели через API в CI/CD пайплайне.
Запускает сервер через docker-compose, затем использует API для обучения.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from commands_classifier.client import CVCApiClient
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Загружает конфигурацию из YAML файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
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
    """Основная функция для обучения модели через API."""
    print("=" * 60)
    print("Запуск обучения модели через API (CI/CD)")
    print("=" * 60)
    
    # Загружаем конфигурацию
    config = load_config()
    
    # Параметры обучения из конфига или переменных окружения
    training_config = config.get("training", {})
    num_iterations = int(os.getenv("NUM_ITERATIONS", training_config.get("iterations", 20)))
    num_epochs = int(os.getenv("NUM_EPOCHS", training_config.get("epochs", 1)))
    batch_size = int(os.getenv("BATCH_SIZE", training_config.get("batch_size", 128)))
    learning_rate = float(os.getenv("LEARNING_RATE", training_config.get("learning_rate", 2e-5)))
    
    server_config = config.get("server", {})
    api_url = f"http://{server_config.get('host', 'localhost')}:{server_config.get('port', 20001)}"
    
    print(f"\nПараметры обучения:")
    print(f"  Итераций: {num_iterations}")
    print(f"  Эпох: {num_epochs}")
    print(f"  Размер батча: {batch_size}")
    print(f"  Скорость обучения: {learning_rate}")
    print(f"  API URL: {api_url}")
    
    # Проверяем, запущен ли сервер
    print(f"\nПроверка сервера...")
    try:
        client = CVCApiClient(api_url)
        client.health()
        print("✓ Сервер уже запущен")
        server_started = False
    except Exception:
        print("Сервер не запущен, запускаем через docker-compose...")
        
        # Запускаем docker-compose
        print("Запуск docker-compose up -d...")
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"✗ Ошибка при запуске docker-compose: {result.stderr}", file=sys.stderr)
            return 1
        
        print("✓ Docker-compose запущен")
        server_started = True
        
        # Ждем готовности сервера
        if not wait_for_server(api_url):
            return 1
    
    # Создаем клиент
    client = CVCApiClient(api_url)
    
    # Проверяем статус обучения
    try:
        status = client.get_training_status()
        if status.get("status") == "running":
            print("⚠️  Обучение уже запущено. Ожидание завершения...")
            # Ждем завершения текущего обучения
            while status.get("status") == "running":
                time.sleep(5)
                status = client.get_training_status()
                progress = status.get("progress", 0)
                print(f"Прогресс: {progress:.1%}", end='\r')
            
            if status.get("status") == "completed":
                print("\n✓ Предыдущее обучение завершено")
            elif status.get("status") == "failed":
                print(f"\n⚠️  Предыдущее обучение завершилось с ошибкой: {status.get('error')}")
                print("Продолжаем с новым обучением...")
    except Exception as e:
        print(f"⚠️  Не удалось проверить статус: {e}")
    
    # Запускаем обучение
    print(f"\nЗапуск обучения через API...")
    try:
        result = client.train(
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        training_id = result.get("training_id")
        print(f"✓ Обучение запущено. ID задачи: {training_id}")
        print("Ожидание завершения обучения...")
        
        # Ждем завершения обучения
        while True:
            time.sleep(5)
            status = client.get_training_status()
            
            if status.get("status") == "completed":
                print(f"\n✓ Обучение завершено успешно!")
                if "metrics" in status and status["metrics"]:
                    print("\nМетрики качества модели:")
                    for metric, value in status["metrics"].items():
                        print(f"  {metric}: {value:.4f}")
                break
            elif status.get("status") == "failed":
                error = status.get("error", "Неизвестная ошибка")
                print(f"\n✗ Обучение завершилось с ошибкой: {error}", file=sys.stderr)
                return 1
            elif status.get("status") == "running":
                progress = status.get("progress", 0)
                print(f"Прогресс: {progress:.1%}", end='\r')
        
        # После успешного обучения упаковываем модель
        print(f"\nУпаковка модели...")
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
                    # Также выводим в stdout для совместимости
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
