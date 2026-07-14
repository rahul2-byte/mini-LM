"""Structured, dependency-free logging setup."""

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Serialize log records as stable JSON objects for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert one log record into a compact, newline-safe JSON string."""
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Install the application-wide stderr handler used by CLI pipelines.

    ``force=True`` intentionally replaces handlers installed by a caller or
    test runner.  Without that reset, repeated smoke runs can print each event
    multiple times and make operational logs misleading.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
