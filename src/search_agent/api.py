"""Stable public API for SearchAgent V12.

Usage (long-lived runtime):
    from search_agent import SearchAgentRuntime
    runtime = SearchAgentRuntime.from_env(
        structured_intent_llm_enabled=True,
        evidence_judge_llm_enabled=True,
    )
    output = await runtime.ainvoke(input_dict)
    await runtime.aclose()

Usage (one-shot):
    from search_agent import ainvoke_search_agent
    output = await ainvoke_search_agent(
        input_dict,
        structured_intent_llm_enabled=True,
        evidence_judge_llm_enabled=True,
    )
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


class SearchAgentContractError(Exception):
    """Raised when input/output contract validation fails."""


class SearchAgentClosedError(RuntimeError):
    """Raised when ainvoke is called after aclose."""


class SearchAgentConfigurationError(RuntimeError):
    """Raised when environment configuration is insufficient."""


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SearchAgentConfigurationError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off"
    )


def _resolve_capability(
    explicit: bool | None,
    env_name: str,
) -> bool:
    if explicit is not None:
        return explicit
    # Preserve the historical from_env() default when the new variables are absent.
    return _env_bool(env_name, default=True)


class SearchAgentRuntime:
    """Reusable runtime that compiles the LangGraph once and serves multiple invokes."""

    def __init__(
        self,
        *,
        config: Any,
        dependencies: Any,
        graph: Any,
        batch_graph: Any,
        callbacks: list[Any] | None = None,
    ) -> None:
        self._config = config
        self._dependencies = dependencies
        self._graph = graph
        self._batch_graph = batch_graph
        self._callbacks = callbacks or []
        self._closed = False

    @classmethod
    def from_env(
        cls,
        *,
        structured_intent_llm_enabled: bool | None = None,
        evidence_judge_llm_enabled: bool | None = None,
        evidence_output_mode: str | None = None,
        callbacks: list[Any] | None = None,
    ) -> SearchAgentRuntime:
        """Create runtime with independently configurable LLM capabilities.

        Explicit capability flags take precedence, then the two
        ``SEARCH_AGENT_*_LLM_ENABLED`` environment values (default True if unset).
        """
        from .evidence_retrieval.batch_graph import build_evidence_retrieval_graph
        from .evidence_retrieval.config import EvidenceRetrievalConfig
        from .evidence_retrieval.dependencies import EvidenceRetrievalDependencies
        from .evidence_retrieval.providers.openai_compatible_chat import OpenAICompatibleChatClient
        from .evidence_retrieval.search_agent_graph import build_search_agent_graph
        from .tracing import get_langfuse_callback

        config = EvidenceRetrievalConfig.from_env(**(
            {"evidence_output_mode": evidence_output_mode}
            if evidence_output_mode is not None else {}
        ))
        # callbacks=None (default) → auto-add V12's own langfuse handler (standalone mode).
        # callbacks=[] or [external_handler] → caller-managed, skip auto-add (integration mode
        # where host system passes its own langfuse handler via contextvar per-request).
        if callbacks is None:
            cb = []
            callback = get_langfuse_callback()
            if callback is not None:
                cb.append(callback)
        else:
            cb = list(callbacks)

        structured_intent_enabled = _resolve_capability(
            structured_intent_llm_enabled,
            "SEARCH_AGENT_STRUCTURED_INTENT_LLM_ENABLED",
        )
        evidence_judge_enabled = _resolve_capability(
            evidence_judge_llm_enabled,
            "SEARCH_AGENT_EVIDENCE_JUDGE_LLM_ENABLED",
        )
        if structured_intent_enabled or evidence_judge_enabled:
            llm = OpenAICompatibleChatClient.from_env(
                model=config.judge_model,
                timeout_seconds=config.parallel_batch_judge_timeout_ms / 1000,
                max_connections=config.judge_batch_concurrency,
            )
            if llm is None:
                capabilities = ", ".join(
                    name
                    for name, enabled in (
                        ("Structured Intent", structured_intent_enabled),
                        ("Evidence Judge", evidence_judge_enabled),
                    )
                    if enabled
                )
                raise SearchAgentConfigurationError(
                    f"{capabilities} requires the shared SearchAgent LLM client, but "
                    "LLM_KEY / LLM_BASE_URL / LLM_MODEL is not configured."
                )
            dependencies = EvidenceRetrievalDependencies.with_capabilities(
                config,
                llm,
                structured_intent_llm_enabled=structured_intent_enabled,
                evidence_judge_llm_enabled=evidence_judge_enabled,
            )
        else:
            dependencies = EvidenceRetrievalDependencies.defaults(config)

        batch_graph = build_evidence_retrieval_graph(
            config, dependencies, callbacks=cb or None
        )
        graph = build_search_agent_graph(
            config,
            dependencies,
            callbacks=cb or None,
            batch_graph=batch_graph,
        )
        return cls(
            config=config,
            dependencies=dependencies,
            graph=graph,
            batch_graph=batch_graph,
            callbacks=cb,
        )

    @classmethod
    def create(
        cls,
        *,
        config: Any,
        dependencies: Any,
        callbacks: list[Any] | None = None,
    ) -> SearchAgentRuntime:
        """Dependency-injection entry point for testing and host systems."""
        from .evidence_retrieval.batch_graph import build_evidence_retrieval_graph
        from .evidence_retrieval.search_agent_graph import build_search_agent_graph

        batch_graph = build_evidence_retrieval_graph(
            config, dependencies, callbacks=callbacks or None
        )
        graph = build_search_agent_graph(
            config,
            dependencies,
            callbacks=callbacks or None,
            batch_graph=batch_graph,
        )
        return cls(
            config=config,
            dependencies=dependencies,
            graph=graph,
            batch_graph=batch_graph,
            callbacks=callbacks,
        )

    async def ainvoke(self, payload: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Invoke the SearchAgent graph.

        Input: search-agent-input/v1 (dict or SearchAgentInputState).
        Output: search-agent-output/v1 (JSON-serializable dict).
        """
        if self._closed:
            raise SearchAgentClosedError("SearchAgentRuntime has been closed. Create a new instance.")

        from .evidence_retrieval.public_contracts import (
            SearchAgentInputState,
            SearchAgentOutputState,
        )

        if isinstance(payload, Mapping):
            input_state = SearchAgentInputState.model_validate(dict(payload))
        else:
            input_state = SearchAgentInputState.model_validate(payload)

        state = await self._graph.ainvoke({"input": input_state.model_dump(mode="json")})

        raw_output = state.get("public_output") or state
        if isinstance(raw_output, Mapping) and "public_output" in raw_output:
            raw_output = raw_output["public_output"]

        output = SearchAgentOutputState.model_validate(raw_output)

        # Hard ID consistency check
        if output.request_id != input_state.request_id:
            raise SearchAgentContractError(
                f"request_id mismatch: input={input_state.request_id} output={output.request_id}"
            )
        if output.document_id != input_state.document_id:
            raise SearchAgentContractError(
                f"document_id mismatch: input={input_state.document_id} output={output.document_id}"
            )
        if output.paragraph_id != input_state.paragraph.paragraph_id:
            raise SearchAgentContractError(
                f"paragraph_id mismatch: input={input_state.paragraph.paragraph_id} output={output.paragraph_id}"
            )

        return output.model_dump(mode="json")

    async def ainvoke_batch(self, payloads: list[Mapping[str, Any] | Any]) -> dict[str, Any]:
        """Invoke all active paragraphs in one request-level retrieval graph.

        The per-paragraph public output contract is preserved in ``outputs``;
        ``diagnostic_output`` is returned separately for the host trace and is
        never mixed into citations or decisions.
        """

        if self._closed:
            raise SearchAgentClosedError(
                "SearchAgentRuntime has been closed. Create a new instance."
            )
        from langchain_core.runnables.config import var_child_runnable_config

        from .evidence_retrieval.batch_graph import build_evidence_retrieval_graph
        from .evidence_retrieval.output_adapter import (
            build_public_batch_outputs,
            to_internal_batch_request_many,
        )
        from .evidence_retrieval.public_contracts import SearchAgentInputState

        values = [
            SearchAgentInputState.model_validate(dict(payload))
            if isinstance(payload, Mapping)
            else SearchAgentInputState.model_validate(payload)
            for payload in payloads
        ]
        request = to_internal_batch_request_many(values)
        # SafeTraceEmitter holds request-local parent/span bookkeeping. Build
        # only the light graph wrapper per request while reusing every HTTP,
        # DB and LLM dependency. This also lets custom Web/KB/Judge spans use
        # the host's current Langfuse handlers instead of disappearing from
        # the parent trace.
        raw_callbacks = (var_child_runnable_config.get() or {}).get("callbacks")
        active_callbacks = list(self._callbacks)
        if isinstance(raw_callbacks, (list, tuple)):
            active_callbacks = list(raw_callbacks)
        else:
            handlers = getattr(raw_callbacks, "handlers", None)
            if handlers:
                active_callbacks = list(handlers)
        parent_run_id = getattr(raw_callbacks, "parent_run_id", None)
        if parent_run_id:
            request = request.model_copy(
                update={
                    "trace_context": {
                        **request.trace_context,
                        "parent_run_id": str(parent_run_id),
                    }
                }
            )
        batch_graph = build_evidence_retrieval_graph(
            self._config,
            self._dependencies,
            callbacks=active_callbacks or None,
        )
        state = await batch_graph.ainvoke(
            {"request": request.model_dump(mode="json")}
        )
        diagnostic = state.get("output") or state
        outputs = build_public_batch_outputs(values, diagnostic, self._config)
        return {
            "request_id": request.request_id,
            "document_id": request.document_id,
            "outputs": [output.model_dump(mode="json") for output in outputs],
            "diagnostic_output": diagnostic,
        }

    async def aclose(self) -> None:
        """Idempotently close all dependency resources (HTTP/LLM/KB clients)."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self._dependencies, "aclose", None)
        if close is not None:
            await close()

    async def __aenter__(self) -> SearchAgentRuntime:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


async def ainvoke_search_agent(
    payload: Mapping[str, Any] | Any,
    *,
    structured_intent_llm_enabled: bool | None = None,
    evidence_judge_llm_enabled: bool | None = None,
) -> dict[str, Any]:
    """One-shot call. Creates a Runtime, invokes, and closes."""
    async with SearchAgentRuntime.from_env(
        structured_intent_llm_enabled=structured_intent_llm_enabled,
        evidence_judge_llm_enabled=evidence_judge_llm_enabled,
    ) as runtime:
        return await runtime.ainvoke(payload)


__all__ = [
    "SearchAgentRuntime",
    "ainvoke_search_agent",
    "SearchAgentContractError",
    "SearchAgentClosedError",
    "SearchAgentConfigurationError",
]
