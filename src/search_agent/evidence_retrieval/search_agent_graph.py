"""Single-paragraph SearchAgent public graph with diagnostic isolation."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .batch_graph import build_evidence_retrieval_graph
from .config import EvidenceRetrievalConfig
from .output_adapter import adapt_v11_input_to_legacy, build_public_output
from .public_contracts import SearchAgentGraphState, SearchAgentInputState


def build_search_agent_graph(
    config: EvidenceRetrievalConfig | None = None,
    dependencies: Any = None,
    *,
    callbacks: list[Any] | None = None,
    batch_graph: Any | None = None,
):
    config = config or EvidenceRetrievalConfig.from_env()
    legacy_graph = batch_graph or build_evidence_retrieval_graph(
        config, dependencies, callbacks=callbacks
    )
    graph = StateGraph(SearchAgentGraphState)

    async def validate_input(state: SearchAgentGraphState):
        value = SearchAgentInputState.model_validate(state["input"])
        return {"input": value.model_dump(mode="json")}

    async def execute_core(state: SearchAgentGraphState):
        value = SearchAgentInputState.model_validate(state["input"])
        legacy = adapt_v11_input_to_legacy(value)
        diagnostic = (await legacy_graph.ainvoke({"request": legacy}))["output"]
        public = build_public_output(value, diagnostic, config)
        return {
            "public_output": public.model_dump(mode="json"),
            "diagnostic_output": diagnostic,
        }

    graph.add_node("validate_input", validate_input)
    graph.add_node("execute_core", execute_core)
    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "execute_core")
    graph.add_edge("execute_core", END)
    return graph.compile()


__all__ = ["build_search_agent_graph"]
