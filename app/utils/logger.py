"""
Logger utility module for easy injection into other modules.

This module provides decorators and helper functions to easily add
logging capabilities to any module or function in the application.
"""

import functools
import inspect
import time
from typing import Any, Callable, Optional, Dict
from app.logging_config import get_logger


def log_function_call(
    logger_name: Optional[str] = None,
    log_args: bool = True,
    log_result: bool = True,
    log_performance: bool = True,
    log_level: str = "INFO"
):
    """
    Decorator to automatically log function calls, arguments, results, and performance.

    This decorator wraps any function to provide automatic logging of:
    - Function entry and exit
    - Input arguments (optional)
    - Return values (optional)
    - Execution time (optional)
    - Any exceptions that occur

    Args:
        logger_name: Name of the logger to use (defaults to module name)
        log_args: Whether to log function arguments
        log_result: Whether to log function return values
        log_performance: Whether to log execution time
        log_level: Log level to use for successful calls

    Returns:
        Decorated function with automatic logging

    Example:
        >>> @log_function_call(log_args=True, log_performance=True)
        ... def process_document(doc_id: str, content: str) -> dict:
        ...     # Function implementation
        ...     return {"status": "processed", "doc_id": doc_id}

        >>> # This will automatically log:
        >>> # - Function entry with arguments
        >>> # - Execution time
        >>> # - Return value
        >>> # - Any exceptions
    """
    def decorator(func: Callable) -> Callable:
        # Get logger for this function's module
        logger = get_logger(logger_name or func.__module__)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _execute_with_logging(
                func, args, kwargs, logger, log_args, log_result, log_performance, log_level
            )

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _execute_with_logging_async(
                func, args, kwargs, logger, log_args, log_result, log_performance, log_level
            )

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _execute_with_logging(
    func: Callable,
    args: tuple,
    kwargs: dict,
    logger,
    log_args: bool,
    log_result: bool,
    log_performance: bool,
    log_level: str
) -> Any:
    """Execute a synchronous function with logging."""
    func_name = f"{func.__module__}.{func.__qualname__}"
    start_time = time.time()

    # Prepare log data
    log_data = {"function": func_name}
    if log_args:
        log_data["args"] = args
        log_data["kwargs"] = kwargs

    # Log function entry
    import logging
    logger.log(
        getattr(logging, log_level.upper(), 20),  # Default to INFO level
        f"Entering function: {func_name}",
        extra={'extra_fields': {
            "event_type": "function_entry",
            **log_data
        }}
    )

    try:
        # Execute the function
        result = func(*args, **kwargs)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Prepare success log data
        success_data = {"function": func_name}
        if log_result:
            success_data["result"] = result
        if log_performance:
            success_data["execution_time_ms"] = round(execution_time * 1000, 2)

        # Log successful completion
        logger.log(
            getattr(logger, log_level.upper(), 20),
            f"Function completed successfully: {func_name}",
            extra={'extra_fields': {
                "event_type": "function_success",
                **success_data
            }}
        )

        return result

    except Exception as e:
        # Calculate execution time even for failures
        execution_time = time.time() - start_time

        # Log the exception
        logger.error(
            f"Function failed: {func_name}",
            extra={'extra_fields': {
                "event_type": "function_error",
                "function": func_name,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": round(execution_time * 1000, 2)
            }},
            exc_info=True
        )

        # Re-raise the exception
        raise


async def _execute_with_logging_async(
    func: Callable,
    args: tuple,
    kwargs: dict,
    logger,
    log_args: bool,
    log_result: bool,
    log_performance: bool,
    log_level: str
) -> Any:
    """Execute an asynchronous function with logging."""
    func_name = f"{func.__module__}.{func.__qualname__}"
    start_time = time.time()

    # Prepare log data
    log_data = {"function": func_name}
    if log_args:
        log_data["args"] = args
        log_data["kwargs"] = kwargs

    # Log function entry
    import logging
    logger.log(
        getattr(logging, log_level.upper(), 20),
        f"Entering async function: {func_name}",
        extra={'extra_fields': {
            "event_type": "function_entry",
            **log_data
        }}
    )

    try:
        # Execute the async function
        result = await func(*args, **kwargs)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Prepare success log data
        success_data = {"function": func_name}
        if log_result:
            success_data["result"] = result
        if log_performance:
            success_data["execution_time_ms"] = round(execution_time * 1000, 2)

        # Log successful completion
        logger.log(
            getattr(logger, log_level.upper(), 20),
            f"Async function completed successfully: {func_name}",
            extra={'extra_fields': {
                "event_type": "function_success",
                **success_data
            }}
        )

        return result

    except Exception as e:
        # Calculate execution time even for failures
        execution_time = time.time() - start_time

        # Log the exception
        logger.error(
            f"Async function failed: {func_name}",
            extra={'extra_fields': {
                "event_type": "function_error",
                "function": func_name,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": round(execution_time * 1000, 2)
            }},
            exc_info=True
        )

        # Re-raise the exception
        raise


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.

    This mixin provides a logger instance and helper methods
    for consistent logging across all methods in a class.

    Usage:
        >>> class MyService(LoggerMixin):
        ...     def process_data(self, data):
        ...         self.log_info("Processing data", extra_data={"size": len(data)})
        ...         # Process the data
        ...         self.log_success("Data processed successfully")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(self.__class__.__module__)

    @property
    def logger(self):
        """Get the logger instance for this class."""
        return self._logger

    def log_info(self, message: str, extra_data: Optional[Dict] = None):
        """Log an info message with optional extra data."""
        self._log_with_extra(self.logger.info, message, extra_data)

    def log_debug(self, message: str, extra_data: Optional[Dict] = None):
        """Log a debug message with optional extra data."""
        self._log_with_extra(self.logger.debug, message, extra_data)

    def log_warning(self, message: str, extra_data: Optional[Dict] = None):
        """Log a warning message with optional extra data."""
        self._log_with_extra(self.logger.warning, message, extra_data)

    def log_error(self, message: str, extra_data: Optional[Dict] = None, exc_info: bool = False):
        """Log an error message with optional extra data and exception info."""
        extra = {'extra_fields': extra_data} if extra_data else {}
        self.logger.error(message, extra=extra, exc_info=exc_info)

    def log_success(self, message: str, extra_data: Optional[Dict] = None):
        """Log a success message with optional extra data."""
        if extra_data:
            extra_data["event_type"] = "success"
        else:
            extra_data = {"event_type": "success"}
        self._log_with_extra(self.logger.info, message, extra_data)

    def _log_with_extra(self, log_func: Callable, message: str, extra_data: Optional[Dict]):
        """Helper method to log with extra data."""
        extra = {'extra_fields': extra_data} if extra_data else {}
        log_func(message, extra=extra)


def inject_logger(cls):
    """
    Class decorator to inject a logger into any class.

    This decorator adds a `logger` property to the class that
    returns a properly configured logger instance.

    Args:
        cls: The class to inject the logger into

    Returns:
        The class with an injected logger property

    Example:
        >>> @inject_logger
        ... class DocumentProcessor:
        ...     def process(self, doc):
        ...         self.logger.info("Processing document")
        ...         # Process the document
    """
    cls._logger = get_logger(cls.__module__)

    @property
    def logger(self):
        return cls._logger

    cls.logger = logger
    return cls
