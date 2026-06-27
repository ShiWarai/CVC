"""Точка входа: запуск API сервера CVC."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="CVC API — сервер классификации голосовых команд",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Хост (по умолчанию: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=20001, help="Порт (по умолчанию: 20001)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к config.yaml (по умолчанию: config.yaml)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Автоперезагрузка при изменении кода (для разработки)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Число worker-процессов Uvicorn (по умолчанию: UVICORN_WORKERS или 1)",
    )
    args = parser.parse_args()

    workers = args.workers
    if workers is None:
        workers = int(os.getenv("UVICORN_WORKERS", "1"))

    try:
        import uvicorn
    except ImportError:
        print("Ошибка: uvicorn не установлен. Установите: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    print(f"Запуск API на {args.host}:{args.port}")
    print(f"Конфигурация: {args.config}")
    if not args.reload:
        print(f"Workers: {workers}")
    print(f"Документация: http://{args.host}:{args.port}/docs")

    try:
        if args.reload:
            uvicorn.run(
                "app.api.server:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
            )
        else:
            uvicorn.run(
                "app.api.server:app",
                host=args.host,
                port=args.port,
                reload=False,
                workers=workers,
            )
    except Exception as e:
        print(f"Ошибка при запуске: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
