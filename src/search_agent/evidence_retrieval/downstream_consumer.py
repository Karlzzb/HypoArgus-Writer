"""Reference downstream consumer that depends only on OutputState v1."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .public_contracts import SearchAgentOutputState


class DownstreamResult(BaseModel):
    request_id: str
    paragraph_id: str
    verdicts: dict[str, str]
    citations_by_task: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    inconclusive_task_ids: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)


def consume_search_agent_output(output: SearchAgentOutputState | dict[str, Any]) -> DownstreamResult:
    value = output if isinstance(output, SearchAgentOutputState) else SearchAgentOutputState.model_validate(output)
    citation_by_id = {citation.citation_id: citation for citation in value.citations}
    citations_by_task: dict[str, list[dict[str, Any]]] = {}
    for decision in value.results:
        citations_by_task[decision.task_id] = [
            {
                "citation_id": citation_by_id[citation_id].citation_id,
                "relation": citation_by_id[citation_id].relation,
                "content": citation_by_id[citation_id].content,
                "summary": citation_by_id[citation_id].summary,
            }
            for citation_id in decision.citation_ids
            if citation_id in citation_by_id
        ]
    return DownstreamResult(
        request_id=value.request_id,
        paragraph_id=value.paragraph_id,
        verdicts={decision.task_id: decision.verdict for decision in value.results},
        citations_by_task=citations_by_task,
        inconclusive_task_ids=[decision.task_id for decision in value.results if decision.verdict == "INCONCLUSIVE"],
        warning_codes=list(dict.fromkeys(warning.code for warning in value.warnings)),
    )


__all__ = ["DownstreamResult", "consume_search_agent_output"]
