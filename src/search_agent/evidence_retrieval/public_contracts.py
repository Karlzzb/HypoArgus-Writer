"""Frozen production input/output contracts.

Only models in this module are part of the downstream protocol. Retrieval
candidates, raw Judge responses, funnels and timings remain diagnostic data.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import Field, model_validator

from .schemas import KnowledgeContext, ParagraphSearchInput, StrictModel

INPUT_SCHEMA_VERSION = "search-agent-input/v1"
OUTPUT_SCHEMA_VERSION = "search-agent-output/v1"


class ParagraphInput(ParagraphSearchInput):
    """Single-paragraph boundary used by the production SearchAgent graph."""


class OrganizationContext(StrictModel):
    """Tenant boundary supplied by the upstream orchestration graph."""

    school_id: str | None = None
    tenant_id: str | None = None
    organization_name: str | None = None
    region: str | None = None

    @model_validator(mode="after")
    def reject_blank_identifiers(self):
        for name in ("school_id", "tenant_id", "organization_name", "region"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be blank")
        return self


class SearchAgentInputState(StrictModel):
    schema_version: Literal["search-agent-input/v1"] = INPUT_SCHEMA_VERSION
    request_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    user_id: str | None = None
    paragraph: ParagraphInput
    organization_context: OrganizationContext | None = None
    knowledge_context: KnowledgeContext | None = None
    retrieval_policy: dict[str, Any] | None = None
    trace_context: dict[str, Any] | None = None


class PublicWarning(StrictModel):
    code: str
    message: str
    task_ids: list[str] = Field(default_factory=list)
    retryable: bool = False


class TraceReference(StrictModel):
    trace_id: str | None = None
    trace_url: str | None = None


class AgentRunStatus(StrictModel):
    status: Literal["SUCCESS", "PARTIAL", "ERROR"]
    completed_task_count: int = 0
    partial_task_count: int = 0
    error_task_count: int = 0
    message: str | None = None


class CitationJudgment(StrictModel):
    confidence: float = Field(ge=0, le=1)
    directness: float = Field(ge=0, le=1)
    supported_claim_ids: list[str] = Field(default_factory=list)
    refuted_claim_ids: list[str] = Field(default_factory=list)
    reason: str
    scope_compatible: bool
    scope_mismatch_reasons: list[str] = Field(default_factory=list)
    quote_match_mode: Literal["EXACT", "NORMALIZED", "SNIPPET", "STRUCTURED_ROW"]


class CitationProvenance(StrictModel):
    query_ids: list[str] = Field(default_factory=list)
    tool_call_id: str | None = None
    scenario_key: str | None = None
    dataset_id: str | None = None
    query_execution_id: str | None = None
    retrieved_at: str
    published_at: str | None = None
    content_fingerprint: str
    source_evidence_fingerprint: str


class CitationRecord(StrictModel):
    citation_id: str
    task_ids: list[str]
    content: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    title: str | None = None
    source_type: Literal["WEB", "KNOWLEDGE_BASE", "STRUCTURED_DATA"]
    source_name: str
    url: str | None = None
    source_ref: dict[str, Any] | None = None
    document_id: str | None = None
    knowledge_id: str | None = None
    file_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    relation: Literal["SUPPORT", "REFUTE", "SUPPLEMENT"]
    status: Literal["ACCEPTED", "DEGRADED"]
    judgment: CitationJudgment
    provenance: CitationProvenance


class EvidenceGap(StrictModel):
    reason: str
    missing_slots: list[str] = Field(default_factory=list)
    gap_retrieval_triggered: bool = False
    resolved: bool = False


class AtomicClaimDecision(StrictModel):
    claim_id: str
    claim_text: str
    logic_role: str | None = None
    verdict: Literal["SUPPORTED", "REFUTED", "CONFLICT", "INCONCLUSIVE"]
    citation_ids: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    reason: str


class TaskDecision(StrictModel):
    task_id: str
    item_id: str
    node_id: str
    hypothesis_id: str | None = None
    line_type: Literal["forward", "reverse"]
    target_text: str
    run_status: Literal["SUCCESS", "PARTIAL", "ERROR"]
    verdict: Literal["SUPPORTED", "REFUTED", "CONFLICT", "INCONCLUSIVE"]
    confidence: float = Field(ge=0, le=1)
    conclusion_summary: str
    citation_ids: list[str] = Field(default_factory=list)
    supporting_citation_ids: list[str] = Field(default_factory=list)
    refuting_citation_ids: list[str] = Field(default_factory=list)
    supplementary_citation_ids: list[str] = Field(default_factory=list)
    atomic_claim_results: list[AtomicClaimDecision] = Field(default_factory=list)
    evidence_gap: EvidenceGap | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def conclusion_matches_verdict(self):
        summary = self.conclusion_summary.strip()
        if not summary:
            raise ValueError("conclusion_summary must not be empty")
        forbidden = {
            "SUPPORTED": ("不足以确认", "相互冲突", "反驳"),
            "REFUTED": ("不足以确认", "相互冲突", "支持该主张"),
            "CONFLICT": ("一致支持", "一致反驳"),
            "INCONCLUSIVE": ("已确认", "一致支持", "一致反驳"),
        }
        if any(token in summary for token in forbidden[self.verdict]):
            raise ValueError("conclusion_summary conflicts with verdict")
        return self


class SearchAgentOutputState(StrictModel):
    schema_version: Literal["search-agent-output/v1"] = OUTPUT_SCHEMA_VERSION
    request_id: str
    document_id: str
    paragraph_id: str
    run_status: AgentRunStatus
    results: list[TaskDecision]
    citations: list[CitationRecord]
    warnings: list[PublicWarning] = Field(default_factory=list)
    trace: TraceReference = Field(default_factory=TraceReference)

    @model_validator(mode="after")
    def validate_public_references(self):
        task_ids = [row.task_id for row in self.results]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("results.task_id must be unique")
        citations = {row.citation_id: row for row in self.citations}
        if len(citations) != len(self.citations):
            raise ValueError("citations.citation_id must be unique")
        for result in self.results:
            for citation_id in result.citation_ids:
                citation = citations.get(citation_id)
                if citation is None:
                    raise ValueError(f"unknown citation_id: {citation_id}")
                if result.task_id not in citation.task_ids:
                    raise ValueError(
                        f"citation {citation_id} is not bound to task {result.task_id}"
                    )
            for claim in result.atomic_claim_results:
                if any(citation_id not in citations for citation_id in claim.citation_ids):
                    raise ValueError(f"claim {claim.claim_id} references an unknown citation")
        return self


class SearchAgentGraphState(TypedDict, total=False):
    input: dict[str, Any]
    public_output: dict[str, Any]
    diagnostic_output: dict[str, Any]


__all__ = [
    "INPUT_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "AgentRunStatus",
    "AtomicClaimDecision",
    "CitationJudgment",
    "CitationProvenance",
    "CitationRecord",
    "EvidenceGap",
    "OrganizationContext",
    "ParagraphInput",
    "PublicWarning",
    "SearchAgentGraphState",
    "SearchAgentInputState",
    "SearchAgentOutputState",
    "TaskDecision",
    "TraceReference",
]
