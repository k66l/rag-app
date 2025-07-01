"""
Logging configuration module for the RAG Application.

This module provides centralized logging configuration with structured formatting,
multiple handlers, and different log levels for development and production environments.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Converts log records into JSON format for better parsing and analysis.
    Useful for log aggregation systems like ELK stack, Datadog, etc.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON formatted log string
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields that were passed to the logger
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


class ColorFormatter(logging.Formatter):
    """
    Custom colored formatter for console output.

    Adds color coding to different log levels for better readability
    during development and debugging.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with colors.

        Args:
            record: The log record to format

        Returns:
            Colored log string
        """
        # Get the original formatted message
        formatted = super().format(record)

        # Add color to the log level
        color = self.COLORS.get(record.levelname, '')
        if color:
            # Replace the level name with colored version
            formatted = formatted.replace(
                record.levelname,
                f"{color}{record.levelname}{self.RESET}"
            )

        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    enable_console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    This function configures the root logger with appropriate handlers
    and formatters based on the environment and requirements.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        json_format: Whether to use JSON formatting (useful for production)
        enable_console: Whether to enable console logging

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(log_level="DEBUG", log_file="app.log")
        >>> logger.info("Application started", extra={'extra_fields': {'user_id': 123}})
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set up formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            console_handler.setFormatter(formatter)
        else:
            # Use colored formatter for console in non-JSON mode
            console_formatter = ColorFormatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress some noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    This function returns a logger with the specified name,
    which will inherit the configuration from the root logger.

    Args:
        name: Name of the logger (usually __name__ of the module)

    Returns:
        Logger instance for the specified name

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module operation completed")
    """
    return logging.getLogger(name)


# Create a default logger instance for the application
app_logger = get_logger("rag_app")
