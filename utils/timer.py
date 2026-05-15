"""Timing decorator for pipeline stages."""

import time
import functools
from utils.logger import get_logger

logger = get_logger("timer")


def timed(label: str = ""):
    """Decorator: logs how long a function takes."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tag = label or fn.__name__
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"[{tag}] completed in {elapsed:.2f}s")
            return result
        return wrapper
    return decorator