"""
Structured logging configuration for the application.
Provides JSON-formatted logs with consistent fields.
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from contextvars import ContextVar

from .settings import LOG_LEVEL, LOG_FORMAT

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class StructuredLogger:
    """Structured JSON logger with consistent formatting."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add our custom handler
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.propagate = False  # Don't propagate to root logger

    def _get_base_fields(self) -> Dict[str, Any]:
        """Get common fields for all log entries."""
        fields = {
            "timestamp": time.time(),
            "level": self.logger.level,
            "logger": self.logger.name,
        }

        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            fields["request_id"] = request_id

        session_id = session_id_var.get()
        if session_id:
            fields["session_id"] = session_id

        user_id = user_id_var.get()
        if user_id:
            fields["user_id"] = user_id

        return fields

    def debug(self, message: str, **kwargs):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(message, extra={"structured_data": kwargs})

    def info(self, message: str, **kwargs):
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(message, extra={"structured_data": kwargs})

    def warning(self, message: str, **kwargs):
        if self.logger.isEnabledFor(logging.WARNING):
            self.logger.warning(message, extra={"structured_data": kwargs})

    def error(self, message: str, **kwargs):
        if self.logger.isEnabledFor(logging.ERROR):
            self.logger.error(message, extra={"structured_data": kwargs})

    def critical(self, message: str, **kwargs):
        if self.logger.isEnabledFor(logging.CRITICAL):
            self.logger.critical(message, extra={"structured_data": kwargs})


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record) -> str:
        # Start with base log record fields
        log_entry = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add structured data if present
        if hasattr(record, "structured_data"):
            log_entry.update(record.structured_data)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        if LOG_FORMAT == "json":
            return json.dumps(log_entry, default=str)
        else:
            # Text format for development
            extra_fields = {
                k: v
                for k, v in log_entry.items()
                if k not in ["timestamp", "level", "logger", "message"]
            }
            extra_str = (
                " ".join(f"{k}={v}" for k, v in extra_fields.items())
                if extra_fields
                else ""
            )
            return f"{log_entry['timestamp']} {log_entry['level']} {log_entry['logger']} {log_entry['message']} {extra_str}".strip()


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


def set_request_context(
    request_id: str, session_id: Optional[str] = None, user_id: Optional[str] = None
):
    """Set context variables for the current request."""
    request_id_var.set(request_id)
    if session_id:
        session_id_var.set(session_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context():
    """Clear request context variables."""
    request_id_var.set(None)
    session_id_var.set(None)
    user_id_var.set(None)
