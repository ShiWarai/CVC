"""Retry с экспоненциальным backoff для вызовов Hugging Face Hub."""

import time
from typing import Callable, TypeVar

T = TypeVar("T")

# Задержки в секундах: 1, 2, 4
DEFAULT_BACKOFF = (1.0, 2.0, 4.0)


def retry_hf(
    fn: Callable[[], T],
    max_retries: int = 3,
    backoff: tuple[float, ...] = DEFAULT_BACKOFF,
) -> T:
    """
    Выполняет callable с повторами при ошибке (сетевые сбои при обращении к HF).

    Args:
        fn: Безаргументный callable (например, lambda: from_pretrained(...)).
        max_retries: Максимальное число попыток (включая первую).
        backoff: Кортеж задержек в секундах между попытками (длина должна быть >= max_retries - 1).

    Returns:
        Результат вызова fn().

    Raises:
        Последнее исключение при исчерпании попыток.
    """
    last_exc = None
    delays = backoff[: max_retries - 1] if max_retries > 1 else ()
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < len(delays):
                time.sleep(delays[attempt])
            else:
                raise
    raise last_exc  # type: ignore[misc]
