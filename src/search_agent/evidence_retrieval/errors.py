"""Structured, redaction-safe errors for the evidence retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    WEB_TIMEOUT = "WEB_TIMEOUT"
    WEB_PROVIDER_ERROR = "WEB_PROVIDER_ERROR"
    # Search completed without a provider error but yielded no usable rows
    # after the bounded query-variant fallback chain.
    WEB_NO_RESULT = "WEB_NO_RESULT"
    WEB_RESPONSE_PARSE_ERROR = "WEB_RESPONSE_PARSE_ERROR"
    WEB_FETCH_ERROR = "WEB_FETCH_ERROR"
    WEB_CONTENT_UNSUPPORTED = "WEB_CONTENT_UNSUPPORTED"
    WEB_CONTENT_TYPE_MISMATCH = "WEB_CONTENT_TYPE_MISMATCH"
    WEB_WHITELIST_BLOCKED = "WEB_WHITELIST_BLOCKED"
    KB_UNAUTHORIZED = "KB_UNAUTHORIZED"
    KB_NOT_FOUND = "KB_NOT_FOUND"
    KB_TIMEOUT = "KB_TIMEOUT"
    KB_PROVIDER_ERROR = "KB_PROVIDER_ERROR"
    STRUCTURED_UNAVAILABLE = "STRUCTURED_UNAVAILABLE"
    STRUCTURED_SCENARIO_NOT_FOUND = "STRUCTURED_SCENARIO_NOT_FOUND"
    STRUCTURED_PARAM_INVALID = "STRUCTURED_PARAM_INVALID"
    STRUCTURED_PERMISSION_DENIED = "STRUCTURED_PERMISSION_DENIED"
    STRUCTURED_TIMEOUT = "STRUCTURED_TIMEOUT"
    RERANK_ANOMALY = "RERANK_ANOMALY"
    MODEL_ERROR = "MODEL_ERROR"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TASK_TIMEOUT = "TASK_TIMEOUT"
    JUDGE_ERROR = "JUDGE_ERROR"
    JUDGE_TIMEOUT = "JUDGE_TIMEOUT"
    JUDGE_EMPTY_RESPONSE = "JUDGE_EMPTY_RESPONSE"
    JUDGE_PARTIAL_VALIDATION_ERROR = "JUDGE_PARTIAL_VALIDATION_ERROR"
    JUDGE_VALIDATION_ERROR = "JUDGE_VALIDATION_ERROR"
    JUDGE_REPAIR_RETRY_ERROR = "JUDGE_REPAIR_RETRY_ERROR"


@dataclass(slots=True)
class RetrievalError(Exception):
    code: ErrorCode
    message: str
    node: str
    tool: str | None = None
    retryable: bool = False
    timeout_layer: str | None = None
    details: dict[str, object] | None = None

    def __str__(self) -> str:
        return f"{self.code.value} at {self.node}: {self.message}"
