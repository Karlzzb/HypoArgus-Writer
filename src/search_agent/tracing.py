"""Langfuse tracing wiring.

``get_langfuse_callback`` builds a LangChain ``CallbackHandler`` from the
``LANGFUSE_*`` environment (loaded from ``.env``). The subgraph bakes this
handler into the compiled graph, so every node and LLM generation is traced
with no per-call arguments. It is non-fatal by design: when credentials or the
``langfuse`` package are absent, it returns ``None`` and the subgraph runs
untraced.

Env:
- ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` — required to activate
- ``LANGFUSE_BASE_URL``                            — Langfuse host
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .env import env_str

logger = logging.getLogger(__name__)

_client_configured = False
_export_status_lock = threading.Lock()
_export_status: dict[str, Any] = {
    "status": "not_configured", "error_count": 0,
    "last_error_logger": None, "last_error_level": None,
}


class _NonFatalExportStatusHandler(logging.Handler):
    """Capture OTLP exporter failures without writing them to stderr."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING:
            return
        with _export_status_lock:
            _export_status["status"] = "degraded"
            _export_status["error_count"] += 1
            _export_status["last_error_logger"] = record.name
            _export_status["last_error_level"] = record.levelname


_export_status_handler = _NonFatalExportStatusHandler()


def _install_nonfatal_export_logging() -> None:
    """Prevent a background telemetry failure from polluting CLI stderr."""
    exporter_logger = logging.getLogger(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    )
    exporter_logger.handlers = [_export_status_handler]
    exporter_logger.propagate = False


def get_langfuse_export_status() -> dict[str, Any]:
    """Return secret-free, process-local export health information."""
    with _export_status_lock:
        return dict(_export_status)


def get_langfuse_callback():
    """Return a Langfuse ``CallbackHandler``, or ``None`` if not configured."""
    public_key = env_str("LANGFUSE_PUBLIC_KEY")
    secret_key = env_str("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except Exception as exc:  # pragma: no cover - optional integration guard
        logger.warning("Langfuse configured but import failed; tracing disabled: %s", exc)
        return None

    global _client_configured
    if not _client_configured:
        _install_nonfatal_export_logging()
        with _export_status_lock:
            _export_status.update(
                status="configured", error_count=0,
                last_error_logger=None, last_error_level=None,
            )
        # Configures the process-wide Langfuse client that CallbackHandler uses.
        timeout_raw = env_str("LANGFUSE_TIMEOUT_SECONDS")
        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=env_str("LANGFUSE_BASE_URL"),
            timeout=int(timeout_raw) if timeout_raw and timeout_raw.isdigit() else 5,
        )
        _client_configured = True

    return CallbackHandler()


__all__ = ["get_langfuse_callback", "get_langfuse_export_status"]
