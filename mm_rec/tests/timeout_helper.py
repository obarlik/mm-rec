"""
Manual timeout helper for unittest.TestCase tests
Since pytest timeout doesn't always work with unittest.TestCase,
we provide a manual timeout mechanism using threading.
"""

import threading
import time
import signal
import os
from typing import Callable, Any, Optional


class TimeoutError(Exception):
    """Raised when a timeout occurs."""
    pass


def _timeout_signal_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Function exceeded timeout")


def timeout_handler(timeout_seconds: float, func: Callable, *args, **kwargs) -> Any:
    """
    Execute a function with a timeout using signal (more reliable).
    
    Args:
        timeout_seconds: Maximum time to wait
        func: Function to execute
        *args, **kwargs: Arguments to pass to function
    
    Returns:
        Function result
    
    Raises:
        TimeoutError: If function doesn't complete in time
    """
    # Use signal-based timeout (only works in main thread)
    old_handler = signal.signal(signal.SIGALRM, _timeout_signal_handler)
    signal.alarm(int(timeout_seconds))
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)
        return result
    except TimeoutError:
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)
        raise
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)
        raise


def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to a test method.
    
    Usage:
        @with_timeout(60)
        def test_something(self):
            # test code
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # First argument is 'self' for instance methods
            self = args[0] if args else None
            try:
                return timeout_handler(timeout_seconds, func, *args, **kwargs)
            except TimeoutError as e:
                if self and hasattr(self, 'fail'):
                    self.fail(str(e))
                else:
                    raise
        return wrapper
    return decorator


class TimeoutContext:
    """
    Context manager for timeout.
    
    Usage:
        with TimeoutContext(60):
            # code that should complete in 60 seconds
    """
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            raise TimeoutError(
                f"Code block exceeded timeout of {self.timeout_seconds} seconds "
                f"(took {elapsed:.2f} seconds)"
            )
        return False

