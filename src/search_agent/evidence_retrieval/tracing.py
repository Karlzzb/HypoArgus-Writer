"""Bounded, secret-safe trace payload helpers."""

from __future__ import annotations

import asyncio
import re
import inspect
import time
import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from .config import EvidenceRetrievalConfig


_SECRET_KEYS = re.compile(r"(authorization|api[_-]?key|cookie|password|secret|token)", re.I)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
_CURRENT_SPAN: ContextVar[tuple[str, uuid.UUID] | None] = ContextVar("evidence_retrieval_current_span", default=None)


def redact(value: Any, config: EvidenceRetrievalConfig, key: str = "") -> Any:
    if _SECRET_KEYS.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, config, str(k)) for k, v in value.items() if str(k) != "raw_payload"}
    if isinstance(value, (list, tuple)):
        return [redact(v, config, key) for v in value[:50]]
    if isinstance(value, str):
        value = _BEARER.sub("Bearer [REDACTED]", value)
        for secret in (config.volcano_api_key, config.bisheng_token, config.doris_password):
            if secret is not None:
                raw = secret.get_secret_value()
                if raw:
                    value = value.replace(raw, "[REDACTED]")
        if key.lower() == "raw_response_preview":
            # V6 judge diagnostics need a bounded response-format preview.
            # It is still secret-scrubbed above and never enters business output.
            return value[:config.batch_judge_raw_preview_chars]
        if not config.trace_content and key.lower() in {"content", "text", "paragraph_text", "target_text"}:
            return "[CONTENT OMITTED]"
        return value[:config.trace_max_chars]
    return value


def metric_payload(*, request_id: str, task_id: str | None, node: str, elapsed_ms: int, **counts: Any) -> dict[str, Any]:
    return {"request_id": request_id, "task_id": task_id, "node": node, "elapsed_ms": elapsed_ms, **counts}


def _build_metadata(payload: dict[str, Any], config: EvidenceRetrievalConfig, sanitizer) -> dict[str, Any]:
    """Build Langfuse observation metadata from a business payload.

    Business metadata is sanitized and merged with the trace-context marker.
    """
    metadata: dict[str, Any] = {"sanitized": True}
    if not payload:
        return metadata
    sanitized = sanitizer(payload, config)
    if isinstance(sanitized, dict):
        for key, value in sanitized.items():
            # Inputs are emitted separately; metadata is the diagnostic surface.
            if key in {"raw_response_preview", "raw_payload"}:
                continue
            metadata[key] = value
    return metadata


class SafeTraceEmitter:
    """Emit only sanitized, bounded observations to Langfuse-compatible callbacks."""

    def __init__(self, config: EvidenceRetrievalConfig, callbacks: list[Any] | None = None, sanitizer=redact):
        self.config = config
        self.callbacks = list(callbacks or [])
        self.sanitizer = sanitizer
        self.run_id = uuid.uuid4()
        self._run_ids: dict[str, uuid.UUID] = {}
        self._parent_run_ids: dict[str, uuid.UUID] = {}
        self._trace_contexts: dict[str, dict[str, Any]] = {}
        self._roots_started: set[tuple[int, str]] = set()
        self._pending: set[asyncio.Task] = set()
        self._closed_roots: set[tuple[int, str]] = set()
        # task_id → task span run_id (so candidate.merge / verification can be
        # parented under the already-opened task span without re-opening it).
        self._task_span_ids: dict[str, uuid.UUID] = {}
        # Explicit trace identity is part of the retrieval contract.  Keep a
        # public, request-independent root id and task observation map so
        # asynchronous callbacks never have to infer parents from a
        # ContextVar or from a callback run id.
        self.root_observation_id: uuid.UUID | None = None
        self.task_observation_ids: dict[str, uuid.UUID] = {}

    def bind_parent(self, request_id: str, trace_context: dict[str, Any] | None) -> uuid.UUID:
        run_id = self._run_ids.setdefault(request_id, uuid.uuid4())
        if self.root_observation_id is None:
            self.root_observation_id = run_id
        context = dict(trace_context or {})
        self._trace_contexts[request_id] = context
        raw_parent = context.get("parent_run_id") or context.get("parent_observation_id")
        if raw_parent:
            try:
                self._parent_run_ids[request_id] = uuid.UUID(str(raw_parent))
            except (ValueError, TypeError, AttributeError):
                pass
        return run_id

    def run_id_for(self, request_id: str | None) -> uuid.UUID:
        if not request_id:
            return self.run_id
        value = self._run_ids.setdefault(request_id, uuid.uuid4())
        if self.root_observation_id is None:
            self.root_observation_id = value
        return value

    def register_task_span(self, task_id: str, run_id: uuid.UUID) -> None:
        """Remember the run_id of a task span so later sub-spans parent under it."""
        if task_id:
            self._task_span_ids[task_id] = run_id
            self.task_observation_ids[task_id] = run_id

    def task_span_id(self, task_id: str | None) -> uuid.UUID | None:
        if not task_id:
            return None
        return self._task_span_ids.get(task_id)

    def _ensure_root(self, callback, request_id: str, run_id: uuid.UUID) -> None:
        callback_key = (id(callback), request_id)
        if callback_key in self._roots_started:
            return
        trace_context = self.sanitizer(self._trace_contexts.get(request_id, {}), self.config)
        parent_run_id = self._parent_run_ids.get(request_id)
        root = callback.on_chain_start(
            {"name": "SearchAgentEvidenceRetrieval"},
            {"request_id": request_id or None},
            run_id=run_id, parent_run_id=parent_run_id,
            name="SearchAgentEvidenceRetrieval",
            tags=["evidence_retrieval"],
            metadata={"sanitized": True, "trace_context": trace_context},
        )
        if inspect.isawaitable(root):
            asyncio.ensure_future(root)
        self._roots_started.add(callback_key)

    def _update_observation_metadata(self, callback, run_id: uuid.UUID, metadata: dict[str, Any]) -> None:
        """Explicitly update an observation's metadata via the StatefulSpanClient.

        Some Langfuse SDK paths filter or defer metadata set via on_chain_start.
        Calling .update(metadata=...) on the stored StatefulSpanClient is the
        direct, reliable way to ensure the business metadata reaches the server.
        """
        if not metadata:
            return
        runs = getattr(callback, "_runs", None)
        if not isinstance(runs, dict):
            return
        observation = runs.get(run_id)
        if observation is None:
            return
        update = getattr(observation, "update", None)
        if update is None:
            return
        try:
            update(metadata=metadata)
        except Exception:
            # Observability must never break business flow.
            pass

    async def emit(self, event: str, payload: dict[str, Any]) -> None:
        """Emit a diagnostic Event observation (no fake duration)."""
        if not self.callbacks:
            return
        safe = self.sanitizer(payload, self.config)
        request_id = str(safe.get("request_id") or "")
        run_id = self.run_id_for(request_id)
        # Standalone diagnostic events belong to this request's root. The
        # external parent is only for the root itself.
        parent_run_id = run_id
        current = _CURRENT_SPAN.get()
        if current and current[0] == request_id:
            parent_run_id = current[1]
        for callback in self.callbacks:
            try:
                if callback.__class__.__module__.startswith("langfuse") and hasattr(callback, "on_chain_start"):
                    self._ensure_root(callback, request_id, run_id)
                    child_id = uuid.uuid4()
                    metadata = _build_metadata(safe, self.config, self.sanitizer)
                    started = callback.on_chain_start(
                        {"name": event}, safe, run_id=child_id,
                        parent_run_id=parent_run_id, name=event,
                        tags=["evidence_retrieval", "sanitized", "diagnostic"], metadata=metadata,
                    )
                    if inspect.isawaitable(started):
                        await started
                    # Diagnostic events are instantaneous by design; do not
                    # pad to 1ms to fake a duration on Langfuse's timeline.
                    ended = callback.on_chain_end(
                        {"event": event, **safe}, run_id=child_id, parent_run_id=parent_run_id,
                    )
                    if inspect.isawaitable(ended):
                        await ended
                elif hasattr(callback, "on_custom_event"):
                    try:
                        result = callback.on_custom_event(event, safe, run_id=run_id, tags=["evidence_retrieval"])
                    except TypeError:
                        result = callback.on_custom_event(event, safe)
                    if inspect.isawaitable(result):
                        await result
                elif hasattr(callback, "emit"):
                    result = callback.emit(event, safe)
                    if inspect.isawaitable(result):
                        await result
                elif callable(callback):
                    result = callback(event, safe)
                    if inspect.isawaitable(result):
                        await result
            except Exception:
                # Observability failure must not alter a factual verdict.
                continue

    def emit_nowait(self, event: str, payload: dict[str, Any]) -> bool:
        """Queue a sanitized diagnostic event without extending the retrieval deadline."""
        if not self.callbacks or len(self._pending) >= self.config.trace_queue_max:
            return False
        task = asyncio.create_task(self.emit(event, payload))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return True

    async def start_span(
        self, name: str, metadata: dict[str, Any] | None = None,
        *, parent_run_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Start a span whose lifecycle may cross multiple async phases."""
        safe = self.sanitizer(metadata or {}, self.config)
        request_id = str(safe.get("request_id") or "")
        root_run_id = self.run_id_for(request_id)
        current = _CURRENT_SPAN.get()
        if parent_run_id is None:
            parent_run_id = current[1] if current and current[0] == request_id else root_run_id
        child_id = uuid.uuid4()
        observation_metadata = _build_metadata(safe, self.config, self.sanitizer)
        data: dict[str, Any] = {
            "_child_id": child_id,
            "_started": time.monotonic(),
            "_parent_run_id": parent_run_id,
            "_request_id": request_id,
            "_name": name,
            "_metadata": dict(metadata or {}),
            "_observation_metadata": observation_metadata,
            "_ended": False,
        }
        for callback in self.callbacks:
            try:
                if callback.__class__.__module__.startswith("langfuse") and hasattr(callback, "on_chain_start"):
                    self._ensure_root(callback, request_id, root_run_id)
                    value = callback.on_chain_start(
                        {"name": name}, safe, run_id=child_id,
                        parent_run_id=parent_run_id, name=name,
                        tags=["evidence_retrieval", "sanitized"],
                        metadata=observation_metadata,
                    )
                    if inspect.isawaitable(value):
                        await value
                    self._update_observation_metadata(callback, child_id, observation_metadata)
            except Exception:
                continue
        return data

    def activate_span(self, span_data: dict[str, Any]):
        """Activate an explicitly started span for child creation."""
        return _CURRENT_SPAN.set((str(span_data.get("_request_id") or ""), span_data["_child_id"]))

    @staticmethod
    def deactivate_span(token) -> None:
        _CURRENT_SPAN.reset(token)

    async def end_span(
        self, span_data: dict[str, Any], *, output: Any = None,
        error: BaseException | None = None, final_metadata: dict[str, Any] | None = None,
    ) -> None:
        """End a span exactly once, preserving its original parent."""
        if span_data.get("_ended"):
            return
        span_data["_ended"] = True
        child_id = span_data["_child_id"]
        parent_run_id = span_data["_parent_run_id"]
        observation_metadata = span_data.get("_observation_metadata") or {}
        if final_metadata:
            merged = {**observation_metadata, **self.sanitizer(final_metadata, self.config)}
            for callback in self.callbacks:
                self._update_observation_metadata(callback, child_id, merged)
        elapsed_ms = max(0.0, round((time.monotonic() - span_data["_started"]) * 1000, 3))
        payload = self.sanitizer({
            **span_data.get("_metadata", {}),
            "elapsed_ms": elapsed_ms,
            "output": output,
            "error": error is not None,
        }, self.config)
        for callback in self.callbacks:
            try:
                if error is not None and hasattr(callback, "on_chain_error"):
                    value = callback.on_chain_error(
                        error, run_id=child_id, parent_run_id=parent_run_id,
                    )
                elif hasattr(callback, "on_chain_end"):
                    value = callback.on_chain_end(
                        payload, run_id=child_id, parent_run_id=parent_run_id,
                    )
                else:
                    continue
                if inspect.isawaitable(value):
                    await value
            except Exception:
                continue

    @asynccontextmanager
    async def span(
        self, name: str, metadata: dict[str, Any] | None = None,
        *, parent_run_id: uuid.UUID | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async context manager wrapping a real operation with start/end timestamps.

        If ``parent_run_id`` is provided, the span is parented under that
        observation explicitly. Otherwise the current span ContextVar is used,
        falling back to the request root.
        """
        if not self.callbacks:
            yield {}
            return
        safe = self.sanitizer(metadata or {}, self.config)
        request_id = str(safe.get("request_id") or "")
        run_id = self.run_id_for(request_id)
        current = _CURRENT_SPAN.get()
        if parent_run_id is None:
            parent_run_id = current[1] if current and current[0] == request_id else self.run_id_for(request_id)
        started_at = time.monotonic()
        child_id = uuid.uuid4()
        span_data: dict[str, Any] = {"_child_id": child_id, "_started": started_at}
        observation_metadata = _build_metadata(safe, self.config, self.sanitizer)

        for callback in self.callbacks:
            try:
                if callback.__class__.__module__.startswith("langfuse") and hasattr(callback, "on_chain_start"):
                    self._ensure_root(callback, request_id, run_id)
                    cb_started = callback.on_chain_start(
                        {"name": name}, safe, run_id=child_id,
                        parent_run_id=parent_run_id, name=name,
                        tags=["evidence_retrieval", "sanitized"], metadata=observation_metadata,
                    )
                    if inspect.isawaitable(cb_started):
                        await cb_started
                    # Defensively persist metadata on the StatefulSpanClient too,
                    # since some SDK paths filter what on_chain_start records.
                    self._update_observation_metadata(callback, child_id, observation_metadata)
            except Exception:
                continue

        context_token = _CURRENT_SPAN.set((request_id, child_id))
        caught: Exception | None = None
        try:
            yield span_data
        except Exception as exc:
            caught = exc
            span_data["_error"] = True
            span_data["_error_message"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _CURRENT_SPAN.reset(context_token)
            elapsed_ms = max(0.0, round((time.monotonic() - started_at) * 1000, 3))
            output = self.sanitizer({
                **(metadata or {}), "elapsed_ms": elapsed_ms,
                "output": span_data.get("output"),
                "error": span_data.get("_error", False),
            }, self.config)
            # If caller captured extra metadata to attach at end (counts that
            # are only known after the operation), apply it now via update().
            final_metadata = span_data.get("_final_metadata")
            if final_metadata:
                merged = {**observation_metadata, **final_metadata}
                for callback in self.callbacks:
                    self._update_observation_metadata(callback, child_id, merged)
            for callback in self.callbacks:
                try:
                    if callback.__class__.__module__.startswith("langfuse") and hasattr(callback, "on_chain_end"):
                        if span_data.get("_error") and hasattr(callback, "on_chain_error"):
                            cb_error = callback.on_chain_error(
                                caught or RuntimeError(span_data.get("_error_message", "operation failed")),
                                run_id=child_id, parent_run_id=parent_run_id,
                            )
                        else:
                            cb_error = callback.on_chain_end(output, run_id=child_id, parent_run_id=parent_run_id)
                        if inspect.isawaitable(cb_error):
                            await cb_error
                except Exception:
                    continue

    @asynccontextmanager
    async def generation(
        self, name: str, metadata: dict[str, Any] | None = None,
        *, model: str | None = None, provider: str | None = None,
        parent_run_id: uuid.UUID | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Open a Langfuse GENERATION observation for an LLM call."""
        if not self.callbacks:
            yield {}
            return
        safe = self.sanitizer(metadata or {}, self.config)
        request_id = str(safe.get("request_id") or "")
        run_id = self.run_id_for(request_id)
        current = _CURRENT_SPAN.get()
        if parent_run_id is None:
            parent_run_id = current[1] if current and current[0] == request_id else self.run_id_for(request_id)
        started_at = time.monotonic()
        child_id = uuid.uuid4()
        span_data: dict[str, Any] = {"_child_id": child_id, "_started": started_at}
        observation_metadata = _build_metadata(safe, self.config, self.sanitizer)
        for callback in self.callbacks:
            try:
                if callback.__class__.__module__.startswith("langfuse") and hasattr(callback, "on_chat_model_start"):
                    self._ensure_root(callback, request_id, run_id)
                    invocation_params = {
                        "_type": "ChatOpenAI",
                        "model": model or "unavailable",
                        "model_name": model or "unavailable",
                        "provider": provider or "openai_compatible",
                        "temperature": span_data.get("temperature"),
                        "max_tokens": span_data.get("max_tokens"),
                        "top_p": span_data.get("top_p"),
                    }
                    serialized = {
                        "name": name, "id": [name],
                        "model": model or "unavailable",
                        "provider": provider or "openai_compatible",
                    }
                    cb_started = callback.on_chat_model_start(
                        serialized, [[]], run_id=child_id,
                        parent_run_id=parent_run_id, name=name,
                        tags=["evidence_retrieval", "sanitized", "generation"],
                        metadata=observation_metadata,
                        invocation_params=invocation_params,
                    )
                    if inspect.isawaitable(cb_started):
                        await cb_started
                    # Defensive: explicitly attach metadata to the Generation
                    # observation via the StatefulSpanClient. on_chat_model_start
                    # sometimes only sets model + input, not the full metadata.
                    self._update_observation_metadata(callback, child_id, observation_metadata)
                elif callback.__class__.__module__.startswith("langfuse") and hasattr(callback, "on_llm_start"):
                    self._ensure_root(callback, request_id, run_id)
                    serialized = {"name": name, "id": [name], "model": model or {}}
                    cb_started = callback.on_llm_start(
                        serialized, [], run_id=child_id,
                        parent_run_id=parent_run_id, name=name,
                        tags=["evidence_retrieval", "sanitized", "generation"],
                        metadata=observation_metadata,
                        invocation_params={
                            "_type": "OpenAI", "model": model or "unavailable",
                            "model_name": model or "unavailable",
                        },
                    )
                    if inspect.isawaitable(cb_started):
                        await cb_started
            except Exception:
                continue

        context_token = _CURRENT_SPAN.set((request_id, child_id))
        caught: Exception | None = None
        try:
            yield span_data
        except Exception as exc:
            caught = exc
            span_data["_error"] = True
            span_data["_error_message"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _CURRENT_SPAN.reset(context_token)
            elapsed_ms = max(0.0, round((time.monotonic() - started_at) * 1000, 3))
            usage = span_data.get("usage") or {}
            finish_reason = span_data.get("finish_reason")
            model_name = span_data.get("model") or model or "unavailable"
            provider_name = span_data.get("provider") or provider or "unavailable"
            # Build final metadata: include usage_status so the trace makes
            # clear whether token usage was actually reported by the provider.
            usage_status = "available" if (isinstance(usage, dict) and usage) else "unavailable"
            final_metadata = {
                **observation_metadata,
                "model": model_name,
                "provider": provider_name,
                "finish_reason": finish_reason or "unavailable",
                "usage_status": usage_status,
                "latency_ms": elapsed_ms,
                "elapsed_ms": elapsed_ms,
                "status": "ERROR" if span_data.get("_error", False) else "SUCCESS",
            }
            extra_final_metadata = span_data.get("_final_metadata") or {}
            if isinstance(extra_final_metadata, dict):
                sanitized_extra = self.sanitizer(extra_final_metadata, self.config)
                if isinstance(sanitized_extra, dict):
                    final_metadata.update(sanitized_extra)
            if isinstance(usage, dict) and usage:
                final_metadata["usage"] = usage
                # Flatten the provider usage fields as well as preserving the
                # original usage object.  Do not synthesize zeroes when the
                # provider did not return usage; `usage_status=unavailable`
                # is the explicit signal for that case.
                usage_aliases = {
                    "actual_input_tokens": ("prompt_tokens", "input_tokens"),
                    "actual_output_tokens": ("completion_tokens", "output_tokens"),
                    "total_tokens": ("total_tokens",),
                }
                for output_key, input_keys in usage_aliases.items():
                    for input_key in input_keys:
                        if usage.get(input_key) is not None:
                            final_metadata[output_key] = usage[input_key]
                            break
            output = self.sanitizer({
                **(metadata or {}), "elapsed_ms": elapsed_ms,
                "model": model_name, "provider": provider_name,
                "finish_reason": finish_reason or "unavailable",
                "usage": usage if (isinstance(usage, dict) and usage) else "unavailable",
                "usage_status": usage_status,
                "error": span_data.get("_error", False),
            }, self.config)
            # Update the Generation observation with usage/model/metadata before
            # on_llm_end fires. This is the path Langfuse SDK persists.
            for callback in self.callbacks:
                try:
                    if callback.__class__.__module__.startswith("langfuse") and hasattr(callback, "on_llm_end"):
                        # StatefulSpanClient may be removed from callback._runs
                        # by on_llm_end, so persist final metadata first.
                        self._update_observation_metadata(callback, child_id, final_metadata)
                        if span_data.get("_error") and hasattr(callback, "on_llm_error"):
                            cb_error = callback.on_llm_error(
                                caught or RuntimeError(span_data.get("_error_message", "operation failed")),
                                run_id=child_id, parent_run_id=parent_run_id,
                            )
                            if inspect.isawaitable(cb_error):
                                await cb_error
                        else:
                            try:
                                from langchain_core.outputs import LLMResult, ChatGeneration
                                from langchain_core.messages import AIMessage
                                response_metadata = {
                                    "model": model_name,
                                    "provider": provider_name,
                                    "stop_reason": finish_reason or "unavailable",
                                }
                                if isinstance(usage, dict) and usage:
                                    response_metadata["usage"] = usage
                                message = AIMessage(
                                    content=str(span_data.get("output_summary", "")),
                                    response_metadata=response_metadata,
                                )
                                generation_obj = ChatGeneration(message=message)
                                llm_result = LLMResult(
                                    generations=[[generation_obj]],
                                    llm_output={
                                        "model_name": model_name,
                                        "token_usage": usage if isinstance(usage, dict) and usage else {},
                                    },
                                )
                                cb_end = callback.on_llm_end(
                                    llm_result, run_id=child_id, parent_run_id=parent_run_id,
                                )
                            except Exception:
                                cb_end = callback.on_chain_end(
                                    output, run_id=child_id, parent_run_id=parent_run_id,
                                )
                            if inspect.isawaitable(cb_end):
                                await cb_end
                    elif callback.__class__.__module__.startswith("langfuse") and hasattr(callback, "on_chain_end"):
                        self._update_observation_metadata(callback, child_id, final_metadata)
                        cb_end = callback.on_chain_end(output, run_id=child_id, parent_run_id=parent_run_id)
                        if inspect.isawaitable(cb_end):
                            await cb_end
                except Exception:
                    continue

    async def flush(self, timeout_ms: int = 3000) -> bool:
        """Best-effort lifecycle drain for service shutdown and E2E tests."""
        pending: set[asyncio.Task] = set()
        if self._pending:
            _, pending = await asyncio.wait(tuple(self._pending), timeout=max(0, timeout_ms) / 1000)
        for task in pending:
            task.cancel()
        ok = not pending
        remaining = max(0.001, timeout_ms / 1000)
        for callback in self.callbacks:
            target = callback if hasattr(callback, "flush") else getattr(callback, "client", None)
            if target is None or not hasattr(target, "flush"):
                continue
            try:
                value = target.flush()
                if inspect.isawaitable(value):
                    await asyncio.wait_for(value, remaining)
            except Exception:
                ok = False
        return ok

    async def finish(self, request_id: str, payload: dict[str, Any] | None = None) -> None:
        """Close the request root only after every child and exporter flush."""
        run_id = self.run_id_for(request_id)
        parent_run_id = self._parent_run_ids.get(request_id)
        safe = self.sanitizer(payload or {"request_id": request_id}, self.config)
        for callback in self.callbacks:
            key = (id(callback), request_id)
            if key not in self._roots_started or key in self._closed_roots:
                continue
            try:
                value = callback.on_chain_end(safe, run_id=run_id, parent_run_id=parent_run_id)
                if inspect.isawaitable(value):
                    await value
                self._closed_roots.add(key)
            except Exception:
                continue

    def external_trace_id(self, request_id: str | None = None) -> str:
        for callback in self.callbacks:
            trace_id = getattr(callback, "last_trace_id", None)
            if trace_id:
                return str(trace_id)
        return str(self.run_id_for(request_id))
