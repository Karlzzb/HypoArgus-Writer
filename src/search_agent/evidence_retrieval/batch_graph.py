"""Batch graph: validation, task flattening, shared prefetch and parallel retrieval."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from .config import EvidenceRetrievalConfig
from .dependencies import EvidenceRetrievalDependencies
from .errors import ErrorCode
from .flows import ParallelSourcesFlow
from .schemas import (
    ErrorDetail,
    ExecutionStatus,
    ExecutionSummary,
    ParagraphSearchOutput,
    RetrievalTask,
    RetrievalTaskResult,
    SearchAgentBatchInput,
    SearchAgentBatchOutput,
    build_retrieval_tasks,
)
from .tracing import SafeTraceEmitter, redact


def build_evidence_retrieval_graph(
    config: EvidenceRetrievalConfig | None = None,
    dependencies: Any = None,
    *,
    callbacks: list[Any] | None = None,
    trace_sanitizer=redact,
):
    config = config or EvidenceRetrievalConfig.from_env()
    if dependencies is None:
        # SearchAgent Judge uses the explicitly configured OpenAI-compatible
        # gateway. This avoids an unrelated Anthropic credential silently
        # taking precedence for this latency-sensitive production path.
        # Keep the heavyweight general LangChain/OpenAI SDK out of the normal
        # evidence-retrieval import path.  This fallback is only reached when
        # callers did not inject dependencies explicitly.
        from .providers.openai_compatible_chat import OpenAICompatibleChatClient

        llm = OpenAICompatibleChatClient.from_env(
            model=config.judge_model,
            timeout_seconds=config.parallel_batch_judge_timeout_ms / 1000,
            max_connections=config.judge_batch_concurrency,
        )
        deps = (
            EvidenceRetrievalDependencies.with_capabilities(
                config,
                llm,
                structured_intent_llm_enabled=True,
                evidence_judge_llm_enabled=True,
            )
            if llm is not None
            else EvidenceRetrievalDependencies.defaults(config)
        )
    else:
        deps = dependencies.complete(config) if hasattr(dependencies, "complete") else dependencies
    trace = SafeTraceEmitter(config, callbacks, trace_sanitizer)

    # TypedDict state for LangGraph compatibility
    from typing import TypedDict

    class _BatchState(TypedDict, total=False):
        request: dict[str, Any]
        tasks: list[dict[str, Any]]
        task_results: list[dict[str, Any]]
        shared_resources: dict[str, Any]
        errors: list[dict[str, Any]]
        started_at: float
        output: dict[str, Any]
        validation_failed: bool
        flow_metrics: dict[str, Any]

    graph = StateGraph(_BatchState)

    async def batch_validate(state):
        try:
            request = SearchAgentBatchInput.model_validate(state["request"])
            trace.bind_parent(request.request_id, request.trace_context)
            task_count = sum(
                len(p.forward_items) + len(p.reverse_items) for p in request.paragraphs
            )
            async with trace.span(
                "request.validate",
                {
                    "request_id": request.request_id,
                    "document_id": request.document_id,
                    "paragraph_count": len(request.paragraphs),
                    "task_count": task_count,
                    "flow_mode": "parallel_sources",
                },
            ) as span:
                if (
                    config.max_tasks_per_request is not None
                    and task_count > config.max_tasks_per_request
                ):
                    raise ValueError("request exceeds max_tasks_per_request")
                span["output"] = {"valid": True, "task_count": task_count}
            return {
                "request": request.model_dump(mode="json"),
                "started_at": time.monotonic(),
                "errors": [],
                "validation_failed": False,
                "flow_metrics": {},
            }
        except Exception as exc:
            raw = state.get("request", {})
            error = ErrorDetail(
                code=ErrorCode.INVALID_INPUT.value,
                node="batch_validate",
                retryable=False,
                reason=f"{type(exc).__name__}: request validation failed",
            )
            return {
                "request": raw if isinstance(raw, dict) else {},
                "started_at": time.monotonic(),
                "errors": [error.model_dump(mode="json")],
                "validation_failed": True,
            }

    async def invalid_finalize(state):
        raw = state.get("request", {})
        output = SearchAgentBatchOutput(
            request_id=str(raw.get("request_id") or "invalid-request"),
            document_id=str(raw.get("document_id") or "invalid-document"),
            execution_summary=ExecutionSummary(
                paragraph_count=0,
                task_count=0,
                success_count=0,
                partial_count=0,
                error_count=1,
                elapsed_ms=int((time.monotonic() - state["started_at"]) * 1000),
            ),
            paragraph_results=[],
            errors=[ErrorDetail.model_validate(x) for x in state.get("errors", [])],
            trace_id=None,
            integration_guard={
                "shadow_mode": config.shadow_mode,
                "influence_propagation_allowed": False,
                "automatic_writeback_allowed": False,
                "reason": "INVALID_INPUT",
            },
        )
        await trace.emit(
            "request.invalid",
            {
                "request_id": output.request_id,
                "document_id": output.document_id,
                "error_codes": [error.code for error in output.errors],
            },
        )
        if callbacks:
            output.trace_id = trace.external_trace_id(output.request_id)
        await trace.flush(3000)
        await trace.finish(
            output.request_id, {"request_id": output.request_id, "status": "invalid"}
        )
        await trace.flush(3000)
        return {"output": output.model_dump(mode="json")}

    async def batch_prepare(state):
        request = SearchAgentBatchInput.model_validate(state["request"])
        async with trace.span(
            "request.prepare",
            {"request_id": request.request_id, "document_id": request.document_id},
        ) as span:
            tasks = build_retrieval_tasks(request)
            span["output"] = {"task_count": len(tasks), "paragraph_count": len(request.paragraphs)}
        return {"tasks": [task.model_dump(mode="json") for task in tasks]}

    async def prefetch_shared_resources(state):
        scenarios: dict[str, Any] = {}
        structured_healthy = False
        health_meta = {
            "cache_hit": False,
            "check_executed": False,
            "ttl_seconds": 0.0,
            "status": "unknown",
        }
        errors = list(state.get("errors", []))
        request = SearchAgentBatchInput.model_validate(state["request"])
        async with trace.span(
            "shared.prefetch",
            {"request_id": request.request_id, "document_id": request.document_id},
        ) as span:
            try:

                async def resolved(value):
                    return value

                async def resolve_health():
                    if hasattr(deps.structured_client, "healthy_with_meta"):
                        return await deps.structured_client.healthy_with_meta()
                    if hasattr(deps.structured_client, "healthy"):
                        result = bool(await deps.structured_client.healthy())
                        return result, {
                            "cache_hit": False,
                            "check_executed": True,
                            "ttl_seconds": 0.0,
                            "status": "healthy" if result else "unhealthy",
                        }
                    return True, {
                        "cache_hit": False,
                        "check_executed": False,
                        "ttl_seconds": 0.0,
                        "status": "unsupported",
                    }

                if hasattr(deps.structured_client, "detailed_scenarios"):
                    scenarios_call = deps.structured_client.detailed_scenarios()
                elif hasattr(deps.structured_client, "scenarios"):
                    scenarios_call = deps.structured_client.scenarios()
                else:
                    scenarios_call = resolved({})
                health_result, scenarios = await asyncio.gather(resolve_health(), scenarios_call)
                structured_healthy, health_meta = health_result
            except Exception:
                health_meta["status"] = "error"
                errors.append(
                    ErrorDetail(
                        code=ErrorCode.STRUCTURED_UNAVAILABLE.value,
                        node="prefetch_shared_resources",
                        tool="structured_query",
                        retryable=True,
                        reason="Structured registry unavailable; other sources remain usable.",
                    ).model_dump(mode="json")
                )
            span["output"] = {
                "structured_healthy": structured_healthy,
                "structured_scenario_count": len(scenarios),
                "structured_health_cache_hit": bool(health_meta["cache_hit"]),
                "structured_health_check_executed": bool(health_meta["check_executed"]),
                "structured_health_cache_ttl_seconds": float(health_meta["ttl_seconds"]),
                "structured_health_status": str(health_meta["status"]),
                # Retain the old field for trace consumers while adding the
                # explicit structured-prefixed observability contract.
                "scenario_count": len(scenarios),
                "error_codes": [x["code"] for x in errors],
            }
        return {
            "shared_resources": {"scenarios": scenarios, "structured_healthy": structured_healthy},
            "errors": errors,
        }

    async def run_retrieval_tasks(state):
        scenarios = state.get("shared_resources", {}).get("scenarios", {})
        tasks = [RetrievalTask.model_validate(raw) for raw in state.get("tasks", [])]
        # One cache per report-level Batch: all paragraphs/tasks share it, but
        # unrelated requests never reuse stale search or page content.
        flow = ParallelSourcesFlow(config, deps, trace, {})
        structured_healthy = bool(
            state.get("shared_resources", {}).get("structured_healthy", False)
        )
        results, metrics = await flow.run(
            tasks,
            scenarios,
            structured_healthy=structured_healthy,
        )
        return {
            "task_results": [result.model_dump(mode="json") for result in results],
            "flow_metrics": metrics,
        }

    async def batch_finalize(state):
        request = SearchAgentBatchInput.model_validate(state["request"])
        results = [RetrievalTaskResult.model_validate(x) for x in state.get("task_results", [])]
        by_paragraph: dict[str, list[RetrievalTaskResult]] = {
            p.paragraph_id: [] for p in request.paragraphs
        }
        task_to_paragraph = {
            task.task_id: task.paragraph_id for task in build_retrieval_tasks(request)
        }
        for result in results:
            by_paragraph[task_to_paragraph[result.task_id]].append(result)
        elapsed_values = sorted(r.elapsed_ms for r in results)

        def percentile(p: float) -> int:
            if not elapsed_values:
                return 0
            index = min(
                len(elapsed_values) - 1,
                max(0, math.ceil(len(elapsed_values) * p) - 1),
            )
            return elapsed_values[index]

        summary = ExecutionSummary(
            paragraph_count=len(request.paragraphs),
            task_count=len(results),
            success_count=sum(r.execution_status == ExecutionStatus.SUCCESS for r in results),
            partial_count=sum(r.execution_status == ExecutionStatus.PARTIAL for r in results),
            error_count=sum(r.execution_status == ExecutionStatus.ERROR for r in results),
            elapsed_ms=int((time.monotonic() - state["started_at"]) * 1000),
            task_elapsed_p50_ms=percentile(0.50),
            task_elapsed_p95_ms=percentile(0.95),
        )
        guarded_tasks = []
        for result in results:
            conclusive = result.verification.verdict.value in {"SUPPORTED", "REFUTED"}
            high_quality = (
                result.execution_status == ExecutionStatus.SUCCESS
                and conclusive
                and result.verification.confidence >= config.min_final_evidence_score
            )
            allowed = high_quality and not config.shadow_mode
            if config.shadow_mode:
                reason = "SHADOW_MODE"
            elif result.execution_status != ExecutionStatus.SUCCESS:
                reason = result.execution_status.value
            elif result.verification.verdict.value == "CONFLICT":
                reason = "MANUAL_REVIEW_REQUIRED"
            elif not conclusive:
                reason = result.verification.verdict.value
            elif not high_quality:
                reason = "LOW_QUALITY"
            else:
                reason = "ALLOWED"
            guarded_tasks.append(
                {
                    "task_id": result.task_id,
                    "verdict": result.verification.verdict.value,
                    "execution_status": result.execution_status.value,
                    "downstream_allowed": allowed,
                    "reason": reason,
                }
            )
        output = SearchAgentBatchOutput(
            request_id=request.request_id,
            document_id=request.document_id,
            execution_summary=summary,
            paragraph_results=[
                ParagraphSearchOutput(
                    paragraph_id=p.paragraph_id, results=by_paragraph[p.paragraph_id]
                )
                for p in request.paragraphs
            ],
            errors=[ErrorDetail.model_validate(x) for x in state.get("errors", [])],
            trace_id=trace.external_trace_id(request.request_id) if callbacks else None,
            flow_metrics=dict(state.get("flow_metrics", {})),
            integration_guard={
                "shadow_mode": config.shadow_mode,
                "influence_propagation_allowed": bool(guarded_tasks)
                and all(row["downstream_allowed"] for row in guarded_tasks),
                "automatic_writeback_allowed": bool(guarded_tasks)
                and all(row["downstream_allowed"] for row in guarded_tasks),
                "task_actions": guarded_tasks,
            },
        )
        async with trace.span(
            "request.finalize",
            {
                "request_id": request.request_id,
                "document_id": request.document_id,
                "task_count": len(results),
                "paragraph_count": len(request.paragraphs),
                "success_count": summary.success_count,
                "partial_count": summary.partial_count,
                "error_count": summary.error_count,
            },
        ) as span:
            span["output"] = {
                "execution_summary": summary.model_dump(mode="json"),
                "verdict_counts": {
                    verdict: sum(r.verification.verdict.value == verdict for r in results)
                    for verdict in ("SUPPORTED", "REFUTED", "CONFLICT", "INCONCLUSIVE")
                },
            }
        # Use perf_counter (monotonic, high-resolution) and float ms so
        # sub-second flushes are not truncated to 0.
        flush_started = time.perf_counter()
        children_flush_ok = await trace.flush(3000)
        children_flush_ms = round((time.perf_counter() - flush_started) * 1000, 3)
        if callbacks:
            output.trace_id = trace.external_trace_id(request.request_id)
        before_root_end = time.perf_counter()
        await trace.finish(
            request.request_id, {"request_id": request.request_id, "status": "complete"}
        )
        root_end_ms = round((time.perf_counter() - before_root_end) * 1000, 3)
        root_flush_started = time.perf_counter()
        root_flush_ok = await trace.flush(3000)
        root_flush_ms = round((time.perf_counter() - root_flush_started) * 1000, 3)
        output.flow_metrics["langfuse_flush_ms"] = round(children_flush_ms + root_flush_ms, 3)
        output.flow_metrics["langfuse_children_flush_ms"] = children_flush_ms
        output.flow_metrics["langfuse_root_end_ms"] = root_end_ms
        output.flow_metrics["langfuse_root_flush_ms"] = root_flush_ms
        output.flow_metrics["langfuse_flush_before_root_end_ms"] = children_flush_ms
        output.flow_metrics["langfuse_flush_after_root_end_ms"] = root_flush_ms
        output.flow_metrics["observability_degraded"] = not (children_flush_ok and root_flush_ok)
        return {"output": output.model_dump(mode="json")}

    graph.add_node("batch_validate", batch_validate)
    graph.add_node("invalid_finalize", invalid_finalize)
    graph.add_node("batch_prepare", batch_prepare)
    graph.add_node("prefetch_shared_resources", prefetch_shared_resources)
    graph.add_node("run_retrieval_tasks", run_retrieval_tasks)
    graph.add_node("batch_finalize", batch_finalize)
    graph.add_edge(START, "batch_validate")
    graph.add_conditional_edges(
        "batch_validate",
        lambda state: "invalid" if state.get("validation_failed") else "valid",
        {"invalid": "invalid_finalize", "valid": "batch_prepare"},
    )
    graph.add_edge("invalid_finalize", END)
    graph.add_edge("batch_prepare", "prefetch_shared_resources")
    graph.add_edge("prefetch_shared_resources", "run_retrieval_tasks")
    graph.add_edge("run_retrieval_tasks", "batch_finalize")
    graph.add_edge("batch_finalize", END)
    return graph.compile()
