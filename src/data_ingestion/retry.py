"""Small bounded retry helper for transient I/O operations."""

from collections.abc import Callable
import random
import time
from typing import TypeVar


T = TypeVar("T")


def retry_call(
    operation: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay_seconds: float = 1.0,
) -> T:
    """Run an operation with bounded exponential backoff and jitter.

    Only connection, timeout, and OS I/O errors are retried.  Configuration,
    parsing, and integrity errors should surface immediately instead of being
    hidden behind repeated attempts.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return operation()
        except (ConnectionError, TimeoutError, OSError):
            if attempt == attempts - 1:
                raise
            # Jitter prevents multiple workers restarted at the same time from
            # retrying in lockstep and creating another load spike.
            delay = base_delay_seconds * (2**attempt) * random.uniform(0.8, 1.2)
            time.sleep(delay)
    raise AssertionError("unreachable")
