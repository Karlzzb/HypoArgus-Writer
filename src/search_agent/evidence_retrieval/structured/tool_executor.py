"""Compatibility adapter that delegates to the registered Function Tool.

Production V12 uses LangGraph ToolNode.  This module remains only for older
callers/tests and intentionally contains no scenario execution logic.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .contracts import StructuredToolCallRecord, StructuredToolResult, utc_now
from .intent_agent import RequestedToolCall
from .registry import StructuredToolDefinition


async def execute_tool_call(
    call: RequestedToolCall,
    registry: dict[str, StructuredToolDefinition],
    client: Any = None,
) -> tuple[StructuredToolResult, StructuredToolCallRecord]:
    del client
    definition = next((row for row in registry.values() if row.tool.name == call.tool_name), None)
    started_at = utc_now()
    started = time.monotonic()
    if definition is None:
        result = StructuredToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            scenario_key="unknown",
            status="INVALID_ARGUMENT",
            query_summary="Tool is not allowlisted",
            error_code="STRUCTURED_TOOL_NOT_ALLOWED",
            error_message="Tool is not allowlisted",
        )
    else:
        try:
            message = await definition.tool.ainvoke({
                "name": call.tool_name,
                "args": call.arguments,
                "id": call.tool_call_id,
                "type": "tool_call",
            })
            raw = getattr(message, "content", message)
            result = StructuredToolResult.model_validate(json.loads(raw) if isinstance(raw, str) else raw)
        except Exception as exc:
            result = StructuredToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                scenario_key=definition.scenario_key,
                status="TOOL_ERROR",
                query_summary="Registered tool invocation failed",
                arguments=call.arguments,
                target_task_ids=list(call.arguments.get("target_task_ids") or call.arguments.get("evaluated_task_ids") or []),
                error_code=type(exc).__name__,
                error_message=f"{type(exc).__name__}: {exc}",
            )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    record = StructuredToolCallRecord(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        scenario_key=result.scenario_key,
        arguments=result.arguments,
        status=result.status,
        target_task_ids=result.target_task_ids,
        started_at=started_at,
        ended_at=utc_now(),
        elapsed_ms=elapsed_ms,
        error=result.error_message,
        row_count=result.row_count,
        dataset_id=result.dataset_id,
        query_execution_id=result.query_execution_id,
        server_elapsed_ms=result.server_elapsed_ms,
    )
    return result, record


async def execute_tool_calls(calls, registry, client=None):
    return await asyncio.gather(*(execute_tool_call(call, registry, client) for call in calls))


__all__ = ["execute_tool_call", "execute_tool_calls"]
