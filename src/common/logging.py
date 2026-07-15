"""Human-readable logging setup for local pipeline runs."""

import logging
import sys
from datetime import datetime


_STANDARD_RECORD_FIELDS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class HumanFormatter(logging.Formatter):
    """Format application events as concise lines for a local terminal."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the event and useful source/run/shard context."""
        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS and key != "event" and not key.startswith("_")
        }
        context = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
        message = (
            f"{datetime.now().strftime('%H:%M:%S')} {record.levelname:<7} {record.getMessage()}"
        )
        if context:
            message += f" {context}"
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        return message


def configure_logging(level: int = logging.INFO) -> None:
    """Install the application-wide human-readable stderr handler.

    ``force=True`` intentionally replaces handlers installed by a caller or
    test runner so repeated smoke runs do not duplicate every event.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(HumanFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
