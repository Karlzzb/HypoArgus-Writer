"""V12 paragraph-level Structured Tool Calling LangGraph subgraph.

The execution contract is deliberately the standard LangGraph sequence:
AIMessage.tool_calls -> ToolNode -> ToolMessage.  There is no second/manual
scenario executor.
"""
from __future__ import annotations

import json
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ..schemas import EvidenceCandidate, RetrievalTask
from .contracts import StructuredToolCallRecord, StructuredToolResult, utc_now
from .intent_agent import RequestedToolCall, StructuredIntentAgent
from .registry import StructuredToolDefinition, build_structured_tool_registry, tools_from_registry
from .result_mapper import map_structured_tool_result


class StructuredSubgraphState(TypedDict, total=False):
    request_id: str
    paragraph_id: str
    paragraph_text: str
    tasks: list[RetrievalTask]
    organization_context: dict[str, Any]
    context: dict[str, Any]
    structured_messages: Annotated[list[BaseMessage], add_messages]
    requested_calls: list[RequestedToolCall]
    tool_results: list[StructuredToolResult]
    tool_call_records: list[StructuredToolCallRecord]
    candidates: list[EvidenceCandidate]
    warnings: list[str]
    metrics: dict[str, Any]
    tool_round: int
    repair_feedback: str


class StructuredToolCallingSubgraph:
    def __init__(self, model: Any, client: Any, trace: Any, config: Any = None):
        self.model = model
        self.client = client
        self.trace = trace
        self.config = config

    def _build_graph(self, registry: dict[str, StructuredToolDefinition]):
        intent = StructuredIntentAgent(self.model, registry, config=self.config)
        graph = StateGraph(StructuredSubgraphState)
        tool_node = ToolNode(tools_from_registry(registry), messages_key="structured_messages")

        async def prepare_structured_context(state: StructuredSubgraphState):
            tasks = state["tasks"]
            return {"context": {
                "paragraph_text": state["paragraph_text"],
                "organization_context": state.get("organization_context", {}),
                "task_ids": [task.task_id for task in tasks],
                "required_slots": {task.task_id: task.required_slots for task in tasks},
                "atomic_claims": {
                    task.task_id: [claim.model_dump(mode="json") for claim in task.atomic_claims]
                    for task in tasks
                },
            }}

        async def structured_intent_agent(state: StructuredSubgraphState):
            started = time.monotonic()
            calls, warnings = await intent.select(
                state["paragraph_text"],
                state["tasks"],
                feedback=state.get("repair_feedback"),
                organization_context=state.get("organization_context"),
            )
            message = AIMessage(
                content="",
                tool_calls=[{
                    "id": call.tool_call_id,
                    "name": call.tool_name,
                    "args": {**call.arguments, "tool_call_id": call.tool_call_id},
                    "type": "tool_call",
                } for call in calls],
            )
            elapsed = int((time.monotonic() - started) * 1000)
            async with self.trace.span("structured.intent", {
                "request_id": state["request_id"],
                "paragraph_id": state["paragraph_id"],
                "task_count": len(state["tasks"]),
                "available_tool_count": len(registry),
                "tool_call_count": len(calls),
                "tool_names": [call.tool_name for call in calls],
                "tool_choice": "required",
                "parallel_tool_calls": True,
            }):
                pass
            metrics = dict(state.get("metrics", {}))
            metrics["intent_elapsed_ms"] = metrics.get("intent_elapsed_ms", 0) + elapsed
            deterministic = any(
                warning in {"STRUCTURED_DETERMINISTIC_ROUTE", "STRUCTURED_NO_MATCH"}
                for warning in warnings
            )
            metrics["intent_llm_call_count"] = (
                metrics.get("intent_llm_call_count", 0) + int(not deterministic)
            )
            metrics["intent_mode"] = (
                "no_match"
                if "STRUCTURED_NO_MATCH" in warnings
                else "deterministic"
                if deterministic
                else "llm"
            )
            return {
                "requested_calls": calls,
                "structured_messages": [message],
                "warnings": list(dict.fromkeys([*state.get("warnings", []), *warnings])),
                "metrics": metrics,
            }

        async def validate_tool_results(state: StructuredSubgraphState):
            started = time.monotonic()
            current_ids = {call.tool_call_id for call in state.get("requested_calls", [])}
            by_call = {call.tool_call_id: call for call in state.get("requested_calls", [])}
            parsed: list[StructuredToolResult] = []
            records: list[StructuredToolCallRecord] = []
            warnings = list(state.get("warnings", []))
            for message in state.get("structured_messages", []):
                if not isinstance(message, ToolMessage) or message.tool_call_id not in current_ids:
                    continue
                call = by_call[message.tool_call_id]
                try:
                    raw = message.content
                    if isinstance(raw, list):
                        raw = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in raw)
                    value = json.loads(raw) if isinstance(raw, str) else raw
                    result = StructuredToolResult.model_validate(value)
                    if result.tool_call_id != message.tool_call_id:
                        raise ValueError("tool_call_id invariant violated")
                except Exception as exc:
                    definition = next((row for row in registry.values() if row.tool.name == call.tool_name), None)
                    result = StructuredToolResult(
                        tool_call_id=message.tool_call_id,
                        tool_name=call.tool_name,
                        scenario_key=definition.scenario_key if definition else "unknown",
                        status="TOOL_ERROR",
                        query_summary="ToolMessage parsing failed",
                        arguments=call.arguments,
                        target_task_ids=list(call.arguments.get("target_task_ids") or call.arguments.get("evaluated_task_ids") or []),
                        error_code="STRUCTURED_TOOL_MESSAGE_INVALID",
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                parsed.append(result)
                trace_meta = {
                    "request_id": state["request_id"],
                    "paragraph_id": state["paragraph_id"],
                    "tool_call_id": result.tool_call_id,
                    "tool_name": result.tool_name,
                    "scenario_key": result.scenario_key,
                    "status": result.status,
                    "row_count": result.row_count,
                    "dataset_id": result.dataset_id,
                    "query_execution_id": result.query_execution_id,
                }
                async with self.trace.span("structured.tool_call", trace_meta):
                    async with self.trace.span("structured.tool_execute", trace_meta):
                        pass
                    async with self.trace.span("structured.tool_result", trace_meta):
                        pass
                records.append(StructuredToolCallRecord(
                    tool_call_id=result.tool_call_id,
                    tool_name=result.tool_name,
                    scenario_key=result.scenario_key,
                    arguments=result.arguments,
                    status=result.status,
                    target_task_ids=result.target_task_ids,
                    started_at=utc_now(),
                    ended_at=utc_now(),
                    elapsed_ms=result.server_elapsed_ms or 0,
                    error=result.error_message,
                    row_count=result.row_count,
                    dataset_id=result.dataset_id,
                    query_execution_id=result.query_execution_id,
                    server_elapsed_ms=result.server_elapsed_ms,
                ))
                if result.status not in {"SUCCESS", "NO_DATA"}:
                    warnings.append(f"STRUCTURED_{result.status}:{result.tool_call_id}")

            current_invalid = [row for row in parsed if row.status == "INVALID_ARGUMENT"]
            round_number = int(state.get("tool_round", 0)) + 1
            feedback = "; ".join(
                f"{row.tool_name}: {row.error_message or row.error_code or 'INVALID_ARGUMENT'}"
                for row in current_invalid
            )
            metrics = dict(state.get("metrics", {}))
            metrics["tool_node_elapsed_ms"] = metrics.get("tool_node_elapsed_ms", 0) + int((time.monotonic() - started) * 1000)
            return {
                "tool_results": [*state.get("tool_results", []), *parsed],
                "tool_call_records": [*state.get("tool_call_records", []), *records],
                "warnings": list(dict.fromkeys(warnings)),
                "tool_round": round_number,
                "repair_feedback": feedback,
                "metrics": metrics,
            }

        def need_more_tools(state: StructuredSubgraphState):
            return "repair" if state.get("repair_feedback") and state.get("tool_round", 0) < 2 else "done"

        async def map_structured_candidates(state: StructuredSubgraphState):
            task_by_id = {task.task_id: task for task in state["tasks"]}
            candidates = [
                candidate
                for result in state.get("tool_results", [])
                for candidate in map_structured_tool_result(result, task_by_id)
            ]
            metrics = dict(state.get("metrics", {}))
            metrics.update({
                "tool_call_count": len(state.get("tool_call_records", [])),
                "structured_query_count": sum(
                    record.scenario_key != "no_structured_query"
                    for record in state.get("tool_call_records", [])
                ),
                "candidate_count": len(candidates),
            })
            async with self.trace.span("structured.candidate_map", {
                "request_id": state["request_id"],
                "paragraph_id": state["paragraph_id"],
                "candidate_count": len(candidates),
            }):
                pass
            return {"candidates": candidates, "metrics": metrics}

        graph.add_node("prepare_structured_context", prepare_structured_context)
        graph.add_node("structured_intent_agent", structured_intent_agent)
        graph.add_node("structured_tool_node", tool_node)
        graph.add_node("validate_tool_results", validate_tool_results)
        graph.add_node("map_structured_candidates", map_structured_candidates)
        graph.add_edge(START, "prepare_structured_context")
        graph.add_edge("prepare_structured_context", "structured_intent_agent")
        graph.add_edge("structured_intent_agent", "structured_tool_node")
        graph.add_edge("structured_tool_node", "validate_tool_results")
        graph.add_conditional_edges(
            "validate_tool_results",
            need_more_tools,
            {"repair": "structured_intent_agent", "done": "map_structured_candidates"},
        )
        graph.add_edge("map_structured_candidates", END)
        return graph.compile()

    async def run(
        self,
        request_id: str,
        paragraph_id: str,
        paragraph_text: str,
        tasks: list[RetrievalTask],
        organization_context: dict[str, Any] | None = None,
        *,
        available_scenario_keys: set[str] | None = None,
    ):
        registry = build_structured_tool_registry(
            self.client,
            organization_context,
            available_scenario_keys=available_scenario_keys,
        )
        graph = self._build_graph(registry)
        return await graph.ainvoke({
            "request_id": request_id,
            "paragraph_id": paragraph_id,
            "paragraph_text": paragraph_text,
            "tasks": tasks,
            "organization_context": organization_context or {},
            "structured_messages": [],
            "tool_results": [],
            "tool_call_records": [],
            "warnings": [],
            "metrics": {},
            "tool_round": 0,
            "repair_feedback": "",
        })


__all__ = ["StructuredSubgraphState", "StructuredToolCallingSubgraph"]
