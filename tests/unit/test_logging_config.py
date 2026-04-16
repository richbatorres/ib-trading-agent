"""Unit tests for src/logging_config.py."""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from unittest.mock import patch

import pytest

from src.logging_config import LOG_DIR, LOG_FILE, LOG_FORMAT, setup_logging


def _remove_our_handlers():
    """Remove only the handlers added by setup_logging (console to stdout + file)."""
    root = logging.getLogger()
    to_remove = []
    for h in root.handlers:
        if isinstance(h, TimedRotatingFileHandler):
            to_remove.append(h)
        elif (
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, TimedRotatingFileHandler)
            and getattr(h, "stream", None) is sys.stdout
        ):
            to_remove.append(h)
    for h in to_remove:
        root.removeHandler(h)
        h.close()


@pytest.fixture(autouse=True)
def _clean_root_logger():
    """Remove our handlers before and after each test."""
    _remove_our_handlers()
    original_level = logging.getLogger().level
    yield
    _remove_our_handlers()
    logging.getLogger().setLevel(original_level)


def _our_stream_handlers():
    """Return StreamHandlers writing to stdout (ours, not pytest's)."""
    return [
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, TimedRotatingFileHandler)
        and getattr(h, "stream", None) is sys.stdout
    ]


def _our_file_handlers():
    """Return TimedRotatingFileHandlers (ours)."""
    return [
        h for h in logging.getLogger().handlers
        if isinstance(h, TimedRotatingFileHandler)
    ]


class TestSetupLogging:
    """Tests for the setup_logging() function."""

    def test_creates_logs_directory(self, tmp_path, monkeypatch):
        log_dir = str(tmp_path / "logs")
        log_file = os.path.join(log_dir, "agent.log")
        monkeypatch.setattr("src.logging_config.LOG_DIR", log_dir)
        monkeypatch.setattr("src.logging_config.LOG_FILE", log_file)
        assert not os.path.exists(log_dir)
        setup_logging()
        assert os.path.isdir(log_dir)

    def test_adds_console_handler(self):
        setup_logging()
        assert len(_our_stream_handlers()) == 1

    def test_adds_file_handler(self):
        setup_logging()
        assert len(_our_file_handlers()) == 1

    def test_file_handler_rotates_daily_with_30_backups(self):
        setup_logging()
        fh = _our_file_handlers()[0]
        assert fh.when == "MIDNIGHT"
        assert fh.interval == 86400
        assert fh.backupCount == 30

    def test_default_level_is_info(self):
        setup_logging()
        assert logging.getLogger().level == logging.INFO

    def test_custom_level(self):
        setup_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_custom_level_case_insensitive(self):
        setup_logging(level="warning")
        assert logging.getLogger().level == logging.WARNING

    def test_log_format_applied(self):
        setup_logging()
        for h in _our_stream_handlers() + _our_file_handlers():
            assert h.formatter._fmt == LOG_FORMAT

    def test_idempotent(self):
        setup_logging()
        setup_logging()
        assert len(_our_stream_handlers()) == 1
        assert len(_our_file_handlers()) == 1

    def test_log_format_contains_required_fields(self):
        assert "%(asctime)s" in LOG_FORMAT
        assert "%(levelname)" in LOG_FORMAT
        assert "%(name)s" in LOG_FORMAT
        assert "%(message)s" in LOG_FORMAT

    def test_llm_token_log_message_format(self, capfd):
        setup_logging()
        lgr = logging.getLogger("src.services.llm_service")
        lgr.info(
            "LLM call: purpose=sentiment, model=claude-sonnet-4-6, "
            "input_tokens=500, output_tokens=200, total_tokens=700"
        )
        captured = capfd.readouterr()
        assert "purpose=sentiment" in captured.out
        assert "src.services.llm_service" in captured.out

    def test_log_file_path_is_relative(self):
        assert not os.path.isabs(LOG_FILE)
        assert LOG_FILE == os.path.join("logs", "agent.log")

    def test_log_dir_path_is_relative(self):
        assert not os.path.isabs(LOG_DIR)
        assert LOG_DIR == "logs"
