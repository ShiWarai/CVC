"""Консольный клиент для работы с API сервером CVC."""

import argparse
import sys
import requests
import json
from typing import Optional


class CVCApiClient:
    """Клиент для работы с CVC API сервером."""
    
    def __init__(self, base_url: str = "http://localhost:20001", use_proxy: bool = False):
        """
        Инициализирует клиент.
        
        Args:
            base_url: Базовый URL API сервера (по умолчанию: http://localhost:20001)
            use_proxy: Использовать ли системный прокси (по умолчанию: False)
        """
        self.base_url = base_url.rstrip('/')
        # Создаем сессию с отключенным прокси для локальных запросов
        self.session = requests.Session()
        if not use_proxy:
            # Отключаем прокси полностью
            self.session.proxies = {
                'http': None,
                'https': None
            }
            # Отключаем использование переменных окружения для прокси
            self.session.trust_env = False
    
    def predict(self, text: str, return_confidence: bool = False) -> dict:
        """
        Классифицирует текст.
        
        Args:
            text: Текст для классификации
            return_confidence: Возвращать ли уверенность
            
        Returns:
            Результат классификации
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json={"text": text, "return_confidence": return_confidence}
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, texts: list, return_confidence: bool = False) -> dict:
        """
        Классифицирует список текстов.
        
        Args:
            texts: Список текстов
            return_confidence: Возвращать ли уверенность
            
        Returns:
            Результаты классификации
        """
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json={"texts": texts, "return_confidence": return_confidence}
        )
        response.raise_for_status()
        return response.json()
    
    def embed(self, texts: list) -> dict:
        """
        Получает эмбеддинги для текстов.
        
        Args:
            texts: Список текстов
            
        Returns:
            Эмбеддинги
        """
        response = self.session.post(
            f"{self.base_url}/embed",
            json={"inputs": texts}
        )
        response.raise_for_status()
        return response.json()
    
    def train(
        self,
        num_iterations: Optional[int] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> dict:
        """
        Запускает обучение модели.
        
        Args:
            num_iterations: Количество итераций
            num_epochs: Количество эпох
            batch_size: Размер батча
            learning_rate: Скорость обучения
            Примечание: Устройство (CPU/CUDA) определяется автоматически при старте приложения
            
        Returns:
            Результат запуска обучения
        """
        payload = {}
        if num_iterations is not None:
            payload["num_iterations"] = num_iterations
        if num_epochs is not None:
            payload["num_epochs"] = num_epochs
        if batch_size is not None:
            payload["batch_size"] = batch_size
        if learning_rate is not None:
            payload["learning_rate"] = learning_rate
        
        response = self.session.post(f"{self.base_url}/train", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_training_status(self) -> dict:
        """Получает статус обучения."""
        response = self.session.get(f"{self.base_url}/train/status")
        response.raise_for_status()
        return response.json()
    
    def get_examples(self) -> list:
        """Получает все примеры."""
        response = self.session.get(f"{self.base_url}/examples")
        response.raise_for_status()
        return response.json()
    
    def add_example(self, text: str, command: str) -> dict:
        """
        Добавляет пример.
        
        Args:
            text: Текст команды
            command: Метка команды
            
        Returns:
            Созданный пример
        """
        response = self.session.post(
            f"{self.base_url}/examples",
            json={"text": text, "command": command}
        )
        response.raise_for_status()
        return response.json()
    
    def delete_example(self, example_id: int) -> dict:
        """
        Удаляет пример.
        
        Args:
            example_id: ID примера
            
        Returns:
            Результат удаления
        """
        response = self.session.delete(f"{self.base_url}/examples/{example_id}")
        response.raise_for_status()
        return response.json()
    
    def get_example(self, example_id: int) -> dict:
        """
        Получает пример по ID.
        
        Args:
            example_id: ID примера
            
        Returns:
            Пример
        """
        response = self.session.get(f"{self.base_url}/examples/{example_id}")
        response.raise_for_status()
        return response.json()
    
    def health(self) -> dict:
        """Проверяет работоспособность сервера."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def metrics(self) -> dict:
        """Получает метрики сервера."""
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()
    
    def reset(self) -> dict:
        """
        Сбрасывает обучение модели:
        - Помечает все примеры в БД как необученные
        - Удаляет обученную модель
        
        Returns:
            Результат сброса (reset_examples, model_deleted)
        """
        response = self.session.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()
    
    def package(self) -> dict:
        """
        Запускает упаковку модели в tar.gz архив (только Linux).
        
        Returns:
            Результат запуска (package_id, message)
        """
        response = self.session.post(f"{self.base_url}/package")
        response.raise_for_status()
        return response.json()
    
    def get_package_status(self) -> dict:
        """
        Получает статус упаковки модели.
        
        Returns:
            Статус упаковки (package_id, status, progress, output_path, error)
        """
        response = self.session.get(f"{self.base_url}/package/status")
        response.raise_for_status()
        return response.json()
    
    def download_model(self, repo_id: str, local_dir: Optional[str] = None) -> dict:
        """
        Запускает загрузку модели с Hugging Face Hub.
        
        Args:
            repo_id: ID репозитория на Hugging Face (например: "username/model-name")
            local_dir: Путь для сохранения (опционально, используется из config если не указан)
            
        Returns:
            Результат запуска (download_id, message)
        """
        payload = {"repo_id": repo_id}
        if local_dir:
            payload["local_dir"] = local_dir
        
        response = self.session.post(f"{self.base_url}/download", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_download_status(self) -> dict:
        """
        Получает статус загрузки модели с Hugging Face Hub.
        
        Returns:
            Статус загрузки (download_id, status, progress, local_path, error)
        """
        response = self.session.get(f"{self.base_url}/download/status")
        response.raise_for_status()
        return response.json()


def predict_command(args):
    """Команда для классификации текста."""
    try:
        client = CVCApiClient(args.url)
        
        if args.text:
            result = client.predict(args.text, return_confidence=args.show_confidence)
            if args.show_confidence:
                print(f"Команда: {result['command']} (уверенность: {result['confidence']:.2%})")
            else:
                print(f"Команда: {result['command']}")
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            result = client.predict_batch(texts, return_confidence=args.show_confidence)
            if args.show_confidence:
                for text, command, conf in zip(texts, result['commands'], result['confidences']):
                    print(f"{text} -> {command} (уверенность: {conf:.2%})")
            else:
                for text, command in zip(texts, result['commands']):
                    print(f"{text} -> {command}")
        else:
            print("Ошибка: необходимо указать --text или --file", file=sys.stderr)
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при обращении к API: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Детали: {error_detail}", file=sys.stderr)
            except:
                print(f"Ответ сервера: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


def train_command(args):
    """Команда для запуска обучения."""
    try:
        client = CVCApiClient(args.url)
        
        print("Запуск обучения через API...")
        result = client.train(
            num_iterations=args.iterations,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        print(f"Обучение запущено. ID задачи: {result['training_id']}")
        print(f"Сообщение: {result['message']}")
        print("\nОжидание завершения обучения...")
        
        # Ждем завершения обучения и показываем метрики
        import time
        while True:
            time.sleep(2)  # Проверяем каждые 2 секунды
            status = client.get_training_status()
            
            if status['status'] == 'completed':
                print(f"\n✓ Обучение завершено успешно!")
                if 'metrics' in status and status['metrics']:
                    print("\nМетрики качества модели:")
                    for metric, value in status['metrics'].items():
                        print(f"  {metric}: {value:.4f}")
                break
            elif status['status'] == 'failed':
                print(f"\n✗ Обучение завершилось с ошибкой: {status.get('error', 'Неизвестная ошибка')}")
                sys.exit(1)
            elif status['status'] == 'running':
                print(f"Прогресс: {status['progress']:.1%}", end='\r')
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            print(f"Ошибка: {e.response.json().get('detail', 'Обучение уже запущено')}", file=sys.stderr)
        else:
            print(f"Ошибка при запуске обучения: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


def train_status_command(args):
    """Команда для проверки статуса обучения."""
    try:
        client = CVCApiClient(args.url)
        status = client.get_training_status()
        
        print(f"ID задачи: {status['training_id']}")
        print(f"Статус: {status['status']}")
        print(f"Прогресс: {status['progress']:.1%}")
        
        if status['started_at']:
            print(f"Начато: {status['started_at']}")
        if status['completed_at']:
            print(f"Завершено: {status['completed_at']}")
        if status['error']:
            print(f"Ошибка: {status['error']}")
        
        # Показываем метрики качества, если они есть
        if 'metrics' in status and status['metrics']:
            print("\nМетрики качества модели:")
            for metric, value in status['metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


def examples_command(args):
    """Команда для работы с примерами."""
    try:
        client = CVCApiClient(args.url)
        
        if args.action == 'list':
            examples = client.get_examples()
            print(f"Всего примеров: {len(examples)}")
            for ex in examples:
                print(f"  [{ex['id']}] {ex['text']} -> {ex['command']}")
        elif args.action == 'add':
            result = client.add_example(args.text, args.command)
            print(f"Добавлен пример [{result['id']}]: {result['text']} -> {result['command']}")
        elif args.action == 'delete':
            result = client.delete_example(args.id)
            print(result['message'])
        elif args.action == 'get':
            example = client.get_example(args.id)
            print(f"[{example['id']}] {example['text']} -> {example['command']}")
            
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Главная функция CLI клиента."""
    parser = argparse.ArgumentParser(
        description="CVC API клиент - консольная обёртка для работы с API сервером",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:20001',
        help='URL API сервера (по умолчанию: http://localhost:20001)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Команда predict
    predict_parser = subparsers.add_parser('predict', help='Классифицировать текст')
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument('--text', type=str, help='Текст для классификации')
    predict_group.add_argument('--file', type=str, help='Файл с текстами (по одному на строку)')
    predict_parser.add_argument('--show-confidence', action='store_true', help='Показывать уверенность')
    
    # Команда train
    train_parser = subparsers.add_parser('train', help='Запустить обучение модели через API')
    train_parser.add_argument('--iterations', type=int, help='Количество итераций')
    train_parser.add_argument('--epochs', type=int, help='Количество эпох')
    train_parser.add_argument('--batch-size', type=int, help='Размер батча (больше = быстрее, но требует больше памяти)')
    train_parser.add_argument('--learning-rate', type=float, help='Скорость обучения')
    
    # Команда train-status
    subparsers.add_parser('train-status', help='Проверить статус обучения')
    
    # Команда examples
    examples_parser = subparsers.add_parser('examples', help='Работа с примерами')
    examples_subparsers = examples_parser.add_subparsers(dest='action', help='Действие')
    
    examples_subparsers.add_parser('list', help='Список всех примеров')
    
    add_parser = examples_subparsers.add_parser('add', help='Добавить пример')
    add_parser.add_argument('--text', type=str, required=True, help='Текст команды')
    add_parser.add_argument('--command', type=str, required=True, help='Метка команды')
    
    delete_parser = examples_subparsers.add_parser('delete', help='Удалить пример')
    delete_parser.add_argument('--id', type=int, required=True, help='ID примера')
    
    get_parser = examples_subparsers.add_parser('get', help='Получить пример по ID')
    get_parser.add_argument('--id', type=int, required=True, help='ID примера')
    
    # Команда health
    subparsers.add_parser('health', help='Проверить работоспособность сервера')
    
    # Команда metrics
    subparsers.add_parser('metrics', help='Получить метрики сервера')
    
    # Команда reset
    subparsers.add_parser('reset', help='Сбросить обучение (удалить модель, пометить все примеры как необученные)')
    
    # Команда package
    subparsers.add_parser('package', help='Упаковать модель в tar.gz архив (только Linux)')
    
    # Команда package-status
    subparsers.add_parser('package-status', help='Проверить статус упаковки модели')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'predict':
        predict_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'train-status':
        train_status_command(args)
    elif args.command == 'examples':
        examples_command(args)
    elif args.command == 'health':
        try:
            client = CVCApiClient(args.url)
            result = client.health()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == 'metrics':
        try:
            client = CVCApiClient(args.url)
            result = client.metrics()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == 'reset':
        try:
            client = CVCApiClient(args.url)
            result = client.reset()
            print(f"✓ {result['message']}")
            print(f"  Сброшено примеров: {result['reset_examples']}")
            print(f"  Модель удалена: {'да' if result['model_deleted'] else 'нет (не существовала)'}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 409:
                print(f"Ошибка: {e.response.json().get('detail', 'Невозможно сбросить во время обучения')}", file=sys.stderr)
            else:
                print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == 'package':
        try:
            client = CVCApiClient(args.url)
            
            print("Запуск упаковки модели...")
            result = client.package()
            print(f"Упаковка запущена. ID задачи: {result['package_id']}")
            print(f"Сообщение: {result['message']}")
            print("\nОжидание завершения упаковки...")
            
            # Ждем завершения упаковки
            import time
            while True:
                time.sleep(1)
                status = client.get_package_status()
                
                if status['status'] == 'completed':
                    print(f"\n✓ Упаковка завершена успешно!")
                    print(f"  Архив: {status['output_path']}")
                    break
                elif status['status'] == 'failed':
                    print(f"\n✗ Упаковка завершилась с ошибкой: {status.get('error', 'Неизвестная ошибка')}")
                    sys.exit(1)
                elif status['status'] == 'running':
                    print(f"Прогресс: {status['progress']:.0%}", end='\r')
                    
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 409:
                print(f"Ошибка: {e.response.json().get('detail', 'Упаковка уже выполняется')}", file=sys.stderr)
            elif e.response.status_code == 404:
                print(f"Ошибка: {e.response.json().get('detail', 'Модель не найдена')}", file=sys.stderr)
            else:
                print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == 'package-status':
        try:
            client = CVCApiClient(args.url)
            status = client.get_package_status()
            
            print(f"ID задачи: {status['package_id'] or 'нет активной задачи'}")
            print(f"Статус: {status['status']}")
            print(f"Прогресс: {status['progress']:.0%}")
            
            if status['started_at']:
                print(f"Начато: {status['started_at']}")
            if status['completed_at']:
                print(f"Завершено: {status['completed_at']}")
            if status['output_path']:
                print(f"Архив: {status['output_path']}")
            if status['error']:
                print(f"Ошибка: {status['error']}")
                
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()

