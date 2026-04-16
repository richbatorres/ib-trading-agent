"""Logging configuration for the IB Trading Agent.

Configures dual output (console + rotating file) using Python's standard
logging module. Log files are stored in the ``logs/`` directory with daily
rotation and 30-day retention.

Log format example::

    2025-01-15 09:30:01,234 | INFO     | src.services.llm_service | LLM call: purpose=sentiment, model=claude-sonnet-4-6, input_tokens=500, output_tokens=200, total_tokens=700
"""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = os.path.join("logs")
LOG_FILE = os.path.join(LOG_DIR, "agent.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with console and rotating file handlers.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"INFO"``, ``"DEBUG"``, ``"WARNING"``).
        Applied to the root logger; individual loggers may override.

    The function is idempotent — calling it multiple times will not add
    duplicate handlers.
    """
    # Ensure the logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers on repeated calls.
    # Check for our specific handler types rather than any handler,
    # because test frameworks (pytest) may add their own handlers.
    has_our_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, TimedRotatingFileHandler)
        and getattr(h, "stream", None) is sys.stdout
        for h in root_logger.handlers
    )
    has_our_file = any(
        isinstance(h, TimedRotatingFileHandler) for h in root_logger.handlers
    )
    if has_our_console and has_our_file:
        return

    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler — writes to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Rotating file handler — rotates daily, keeps 30 backup files
    file_handler = TimedRotatingFileHandler(
        filename=LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
