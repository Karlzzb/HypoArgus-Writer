"""LangGraph reducers for V11 append/upsert channels."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .public_contracts import CitationRecord, PublicWarning, SearchAgentInputState, TaskDecision
from .schemas import CandidateJudgeRecord, ErrorDetail, EvidenceCandidate, EvidenceItem, source_evidence_fingerprint


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def merge_candidate_channel(left, right) -> list[EvidenceCandidate]:
    output: dict[tuple[str, str, str], EvidenceCandidate] = {}
    for raw in [*_as_list(left), *_as_list(right)]:
        candidate = raw if isinstance(raw, EvidenceCandidate) else EvidenceCandidate.model_validate(raw)
        key = (candidate.task_id, candidate.source_type.value, source_evidence_fingerprint(candidate))
        previous = output.get(key)
        if previous is None:
            output[key] = candidate
            continue
        metadata = {**previous.metadata, **candidate.metadata}
        for field in ("query_ids", "query_variant_ids", "candidate_source_query_ids", "matched_task_ids", "provenance"):
            values = [*_as_list(previous.metadata.get(field)), *_as_list(candidate.metadata.get(field))]
            if values:
                metadata[field] = list(dict.fromkeys(str(value) for value in values))
        output[key] = previous.model_copy(update={"metadata": metadata})
    return list(output.values())


def merge_citation_channel(left, right) -> list[CitationRecord]:
    output: dict[str, CitationRecord] = {}
    for raw in [*_as_list(left), *_as_list(right)]:
        value = raw if isinstance(raw, CitationRecord) else CitationRecord.model_validate(raw)
        previous = output.get(value.citation_id)
        if previous is not None:
            value = value.model_copy(update={
                "task_ids": list(dict.fromkeys([*previous.task_ids, *value.task_ids])),
            })
        output[value.citation_id] = value
    return list(output.values())


def merge_tool_call_channel(left, right):
    output: dict[str, Any] = {}
    for value in [*_as_list(left), *_as_list(right)]:
        key = str(getattr(value, "tool_call_id", None) or value.get("tool_call_id"))
        output[key] = value
    return list(output.values())


def merge_judge_result_channel(left, right):
    output: dict[tuple[str, str], Any] = {}
    for value in [*_as_list(left), *_as_list(right)]:
        if isinstance(value, dict):
            key = (str(value.get("task_id", "")), str(value.get("candidate_id", "")))
        else:
            key = (str(getattr(value, "task_id", "")), str(getattr(value, "candidate_id", "")))
        output[key] = value
    return list(output.values())


def merge_evidence_item_channel(left, right) -> list[EvidenceItem]:
    output: dict[str, EvidenceItem] = {}
    for raw in [*_as_list(left), *_as_list(right)]:
        value = raw if isinstance(raw, EvidenceItem) else EvidenceItem.model_validate(raw)
        output[value.evidence_id] = value
    return list(output.values())


def merge_dict_channel(left, right) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


def merge_task_decision_channel(left, right) -> dict[str, TaskDecision]:
    output: dict[str, TaskDecision] = {}
    for mapping in (left or {}, right or {}):
        for task_id, raw in mapping.items():
            output[str(task_id)] = raw if isinstance(raw, TaskDecision) else TaskDecision.model_validate(raw)
    return output


def _fingerprint(value: Any) -> tuple[str, str, tuple[str, ...]]:
    if isinstance(value, PublicWarning):
        return value.code, value.message, tuple(sorted(value.task_ids))
    if isinstance(value, ErrorDetail):
        return value.code, value.reason, tuple(sorted(value.affected_candidate_ids))
    raw = value if isinstance(value, dict) else value.model_dump(mode="json")
    return (
        str(raw.get("code", "")),
        str(raw.get("message") or raw.get("reason") or ""),
        tuple(sorted(str(item) for item in raw.get("task_ids", raw.get("affected_candidate_ids", [])))),
    )


def _append_unique(left, right):
    output = []
    seen = set()
    for value in [*_as_list(left), *_as_list(right)]:
        key = _fingerprint(value)
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


append_warning_channel = _append_unique
append_error_channel = _append_unique


class SearchAgentInternalState(TypedDict, total=False):
    input: SearchAgentInputState
    tasks: list[Any]
    atomic_claims: dict[str, list[Any]]
    web_candidates: Annotated[list[EvidenceCandidate], merge_candidate_channel]
    kb_candidates: Annotated[list[EvidenceCandidate], merge_candidate_channel]
    structured_candidates: Annotated[list[EvidenceCandidate], merge_candidate_channel]
    structured_messages: Annotated[list[BaseMessage], add_messages]
    structured_tool_calls: Annotated[list[Any], merge_tool_call_channel]
    all_candidates: Annotated[list[EvidenceCandidate], merge_candidate_channel]
    gap_candidates: Annotated[list[EvidenceCandidate], merge_candidate_channel]
    judge_results: Annotated[list[CandidateJudgeRecord], merge_judge_result_channel]
    evidence_items: Annotated[list[EvidenceItem], merge_evidence_item_channel]
    citations: Annotated[list[CitationRecord], merge_citation_channel]
    task_decisions: Annotated[dict[str, TaskDecision], merge_task_decision_channel]
    warnings: Annotated[list[PublicWarning], append_warning_channel]
    errors: Annotated[list[ErrorDetail], append_error_channel]
    diagnostics: Annotated[dict[str, Any], merge_dict_channel]
    public_output: dict[str, Any]


__all__ = [
    "SearchAgentInternalState", "append_error_channel", "append_warning_channel",
    "merge_candidate_channel", "merge_citation_channel", "merge_dict_channel", "merge_evidence_item_channel", "merge_judge_result_channel",
    "merge_task_decision_channel", "merge_tool_call_channel",
]
