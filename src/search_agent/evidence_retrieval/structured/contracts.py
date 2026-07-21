"""Structured Tool Calling schemas shared by registry, executor and tracing."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field

from ..schemas import StrictModel


class StructuredToolResult(StrictModel):
    tool_call_id: str
    tool_name: str
    scenario_key: str
    status: Literal["SUCCESS", "NO_DATA", "INVALID_ARGUMENT", "PERMISSION_DENIED", "TOOL_ERROR"]
    query_summary: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    dataset_id: str | None = None
    query_execution_id: str | None = None
    server_elapsed_ms: int | None = None
    target_task_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


class StructuredToolCallRecord(StrictModel):
    tool_call_id: str
    tool_name: str
    scenario_key: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str = "PENDING"
    target_task_ids: list[str] = Field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None
    elapsed_ms: int = 0
    error: str | None = None
    row_count: int = 0
    dataset_id: str | None = None
    query_execution_id: str | None = None
    server_elapsed_ms: int | None = None


class StructuredQueryResponse(StrictModel):
    """Authoritative response envelope returned by the Structured HTTP API."""

    rows: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    dataset_id: str | None = None
    query_execution_id: str | None = None
    server_elapsed_ms: int | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class NoStructuredQueryArgs(StrictModel):
    reason: str = Field(description="为什么该段落不适用于任何已注册结构化场景")
    evaluated_task_ids: list[str] = Field(description="已经评估的全部 SearchAgent task_id")

    tool_call_id: str | None = Field(default=None, description="Internal ToolNode call identifier; callers must omit it.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "NoStructuredQueryArgs", "StructuredQueryResponse", "StructuredToolCallRecord", "StructuredToolResult", "utc_now",
]
