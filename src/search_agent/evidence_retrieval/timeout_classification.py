"""Consistent timeout classification across providers, metrics and traces."""

from __future__ import annotations

import asyncio

import httpx

from .errors import ErrorCode, RetrievalError


TIMEOUT_LAYERS = {
    "searchagent_wait_for",
    "http_read_timeout",
    "provider_timeout",
    "gateway_timeout",
    "request_deadline",
}


def timeout_layer(exc: BaseException | None) -> str | None:
    """Return a stable timeout layer, following wrapped exception causes."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, RetrievalError):
            if current.timeout_layer in TIMEOUT_LAYERS:
                return current.timeout_layer
            if current.code in {ErrorCode.KB_TIMEOUT, ErrorCode.WEB_TIMEOUT, ErrorCode.STRUCTURED_TIMEOUT, ErrorCode.JUDGE_TIMEOUT, ErrorCode.TASK_TIMEOUT}:
                return "provider_timeout"
        if isinstance(current, httpx.ReadTimeout):
            return "http_read_timeout"
        if isinstance(current, httpx.TimeoutException):
            return "provider_timeout"
        if isinstance(current, httpx.HTTPStatusError) and current.response.status_code == 504:
            return "gateway_timeout"
        if isinstance(current, (asyncio.TimeoutError, TimeoutError)):
            return "searchagent_wait_for"
        current = current.__cause__ or current.__context__
    return None


def is_timeout(exc: BaseException | None) -> bool:
    return timeout_layer(exc) is not None


__all__ = ["TIMEOUT_LAYERS", "is_timeout", "timeout_layer"]
