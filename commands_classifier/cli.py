"""CLI для запуска API сервера."""

import argparse
import sys


def serve_command(args):
    """Команда для запуска API сервера."""
    # Проверяем импорт uvicorn отдельно
    try:
        import uvicorn
    except ImportError:
        print("Ошибка: uvicorn не установлен. Установите его: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    print(f"Запуск API сервера на {args.host}:{args.port}")
    print(f"Конфигурация: {args.config}")
    print(f"Документация API: http://{args.host}:{args.port}/docs")

    try:
        # Для reload нужно передавать строку импорта, а не объект
        if args.reload:
            uvicorn.run(
                "commands_classifier.api.server:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
            )
        else:
            # Без reload можно использовать объект напрямую
            from commands_classifier.api.server import app

            uvicorn.run(app, host=args.host, port=args.port, reload=False)
    except ImportError as e:
        print(f"Ошибка импорта модуля: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при запуске сервера: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    """Главная функция CLI для запуска сервера."""
    parser = argparse.ArgumentParser(
        description="CVC API сервер - запуск сервера для классификации голосовых команд",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Игнорируем команду "serve" если она передана (для обратной совместимости)
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        sys.argv.pop(1)

    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Хост для сервера (по умолчанию: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=20001, help="Порт для сервера (по умолчанию: 20001)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к конфигурационному файлу (по умолчанию: config.yaml)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Включить автоматическую перезагрузку при изменении кода (для разработки)",
    )

    args = parser.parse_args()
    serve_command(args)


if __name__ == "__main__":
    main()
