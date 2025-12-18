"""CLI интерфейс для классификатора команд."""

import argparse
import sys
from pathlib import Path
from .model import CommandsClassifier
from .dataset import load_dataset


def train_command(args):
    """Команда для обучения модели."""
    print(f"Загрузка датасета из {args.dataset}...")
    try:
        texts, labels = load_dataset(args.dataset)
        print(f"Загружено {len(texts)} примеров")
        
        # Показываем статистику по классам
        from collections import Counter
        label_counts = Counter(labels)
        print(f"Классы: {dict(label_counts)}")
        
        print("Инициализация модели...")
        classifier = CommandsClassifier(model_name=args.model_name)
        
        print("Обучение модели...")
        classifier.train(
            texts,
            labels,
            num_iterations=args.iterations,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        print(f"Сохранение модели в {args.output}...")
        classifier.save(args.output)
        print("Модель успешно обучена и сохранена!")
        
    except Exception as e:
        print(f"Ошибка при обучении: {e}", file=sys.stderr)
        sys.exit(1)


def predict_command(args):
    """Команда для предсказания."""
    print(f"Загрузка модели из {args.model}...")
    try:
        classifier = CommandsClassifier(confidence_threshold=args.confidence_threshold)
        classifier.load(args.model, confidence_threshold=args.confidence_threshold)
        
        if args.text:
            # Классификация одного текста
            if args.show_confidence:
                result, confidence = classifier.predict(args.text, return_confidence=True)
                print(f"Команда: {result} (уверенность: {confidence:.2%})")
            else:
                result = classifier.predict(args.text)
                print(f"Команда: {result}")
        elif args.file:
            # Batch классификация
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Классификация {len(texts)} текстов...")
            if args.show_confidence:
                results, confidences = classifier.predict_batch(texts, return_confidence=True)
                # Выводим результаты с уверенностью
                for text, command, conf in zip(texts, results, confidences):
                    print(f"{text} -> {command} (уверенность: {conf:.2%})")
            else:
                results = classifier.predict_batch(texts)
                # Выводим результаты
                for text, command in zip(texts, results):
                    print(f"{text} -> {command}")
        else:
            print("Ошибка: необходимо указать --text или --file", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Ошибка при предсказании: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="SetFit классификатор команд для few-shot learning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Команда train
    train_parser = subparsers.add_parser('train', help='Обучить модель на датасете')
    train_parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Путь к файлу датасета (CSV или JSON)'
    )
    train_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Путь для сохранения обученной модели'
    )
    train_parser.add_argument(
        '--model-name',
        type=str,
        default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        help='Имя предобученной модели (по умолчанию: multilingual mpnet)'
    )
    train_parser.add_argument(
        '--iterations',
        type=int,
        default=20,
        help='Количество итераций контрастного обучения (по умолчанию: 20)'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Количество эпох fine-tuning (по умолчанию: 1)'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Размер батча (по умолчанию: 16)'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Скорость обучения (по умолчанию: 2e-5)'
    )
    
    # Команда predict
    predict_parser = subparsers.add_parser('predict', help='Классифицировать текст')
    predict_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Путь к обученной модели'
    )
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument(
        '--text',
        type=str,
        help='Текст для классификации'
    )
    predict_group.add_argument(
        '--file',
        type=str,
        help='Файл с текстами (по одному на строку) для batch классификации'
    )
    predict_parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Порог уверенности для отбраковки (0.0-1.0). Если уверенность ниже, возвращается "unknown" (по умолчанию: 0.5)'
    )
    predict_parser.add_argument(
        '--show-confidence',
        action='store_true',
        help='Показывать уверенность модели для каждого предсказания'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)


if __name__ == '__main__':
    main()

