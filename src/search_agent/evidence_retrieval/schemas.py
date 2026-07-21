"""Public and internal Pydantic contracts for evidence retrieval."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .claim_logic import AtomicClaim, ClaimLogicOperator, atomize_claim, normalize_reverse_hypothesis


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LineType(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


class RetrievalGoal(str, Enum):
    VERIFY_ORIGINAL = "verify_original"
    VERIFY_HYPOTHESIS = "verify_hypothesis"


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"


class TerminationReason(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    EXHAUSTED = "EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    NO_AVAILABLE_ROUTE = "NO_AVAILABLE_ROUTE"
    INVALID_INPUT = "INVALID_INPUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TOOL_ERROR = "TOOL_ERROR"


class VerificationVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    CONFLICT = "CONFLICT"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceRelation(str, Enum):
    SUPPORT = "SUPPORT"
    REFUTE = "REFUTE"
    SUPPLEMENT = "SUPPLEMENT"
    NEUTRAL = "NEUTRAL"


class NeutralReason(str, Enum):
    """Judge 判定 NEUTRAL 时的标准化原因枚举。"""
    WRONG_ENTITY = "WRONG_ENTITY"
    WRONG_YEAR = "WRONG_YEAR"
    WRONG_REGION = "WRONG_REGION"
    WRONG_METRIC = "WRONG_METRIC"
    WRONG_MARKET_SCOPE = "WRONG_MARKET_SCOPE"
    WRONG_STATISTICAL_SCOPE = "WRONG_STATISTICAL_SCOPE"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    MISSING_NUMERIC_VALUE = "MISSING_NUMERIC_VALUE"
    BACKGROUND_ONLY = "BACKGROUND_ONLY"
    CONTEXT_TRUNCATED = "CONTEXT_TRUNCATED"
    QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"
    IRRELEVANT = "IRRELEVANT"


class SourceType(str, Enum):
    WEB = "web"
    KNOWLEDGE_BASE = "knowledge_base"
    STRUCTURED = "structured"


class SourceRef(StrictModel):
    url: str | None = None
    knowledge_id: str | None = None
    knowledge_origin: Literal["upstream_selected", "configured_public"] | None = None
    file_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    scenario_name: str | None = None
    record_id: str | None = None
    query_id: str | None = None
    dataset_id: str | None = None
    query_execution_id: str | None = None
    query_params_hash: str | None = None


class SourceRange(StrictModel):
    raw_store_id: str | None = None
    byte_start: int = Field(default=0, ge=0)
    byte_end: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.byte_end < self.byte_start:
            raise ValueError("byte_end must be >= byte_start")
        return self


class ArgumentPathItem(StrictModel):
    level: int = Field(ge=1)
    node_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ArgumentContext(StrictModel):
    argument_path: list[ArgumentPathItem] = Field(default_factory=list)
    boundary: str | None = None
    max_depth: int = Field(default=3, ge=1)


class ForwardItem(StrictModel):
    item_id: str = Field(min_length=1)
    item_type: Literal["claim", "evidence"] = "claim"
    target_text: str = Field(min_length=1)
    existing_evidence_text: str | None = None
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    required_slots: list[str] = Field(default_factory=list)


class ReverseItem(StrictModel):
    item_id: str = Field(min_length=1)
    target_text: str = Field(min_length=1)
    relation_to_original: Literal["oppose", "advance", "expand"] = "oppose"
    required_slots: list[str] = Field(default_factory=list)


class ParagraphSearchInput(StrictModel):
    paragraph_id: str = Field(min_length=1)
    paragraph_text: str
    source_ref: SourceRange | None = None
    argument_context: ArgumentContext = Field(default_factory=ArgumentContext)
    forward_items: list[ForwardItem] = Field(default_factory=list)
    reverse_items: list[ReverseItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_item_ids(self):
        ids = [x.item_id for x in [*self.forward_items, *self.reverse_items]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate item_id in paragraph {self.paragraph_id}")
        return self


class KnowledgeContext(StrictModel):
    selected_knowledge_ids: list[str] = Field(default_factory=list)

    @field_validator("selected_knowledge_ids")
    @classmethod
    def unique_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("selected_knowledge_ids must be unique")
        return value


class SearchAgentBatchInput(StrictModel):
    request_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    paragraphs: list[ParagraphSearchInput] = Field(min_length=1)
    knowledge_context: KnowledgeContext = Field(default_factory=KnowledgeContext)
    retrieval_policy: dict[str, Any] = Field(default_factory=dict)
    trace_context: dict[str, Any] = Field(default_factory=dict)
    organization_context: dict[str, Any] | None = None

    @model_validator(mode="after")
    def globally_unique(self):
        paragraph_ids = [p.paragraph_id for p in self.paragraphs]
        if len(paragraph_ids) != len(set(paragraph_ids)):
            raise ValueError("paragraph_id must be unique")
        item_ids = [i.item_id for p in self.paragraphs for i in [*p.forward_items, *p.reverse_items]]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("item_id must be unique within a request")
        return self


class RetrievalTask(StrictModel):
    task_id: str
    request_id: str
    document_id: str
    user_id: str
    paragraph_id: str
    line_type: LineType
    node_id: str
    item_id: str
    hypothesis_id: str | None = None
    target_text: str
    paragraph_text: str
    argument_path: list[ArgumentPathItem] = Field(default_factory=list)
    boundary: str | None = None
    required_slots: list[str] = Field(default_factory=list)
    existing_evidence_text: str | None = None
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    relation_to_original: Literal["oppose", "advance", "expand"] | None = None
    retrieval_goal: RetrievalGoal
    selected_knowledge_ids: list[str] = Field(default_factory=list)
    atomic_claims: list[AtomicClaim] = Field(default_factory=list)
    claim_logic_operator: ClaimLogicOperator = ClaimLogicOperator.SINGLE
    normalized_hypothesis: str | None = None
    neutral_retrieval_query: str | None = None
    polarity: str = "POSITIVE"
    argument_type: str = "QUALITATIVE_CLAIM"
    organization_context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reverse_has_hypothesis(self):
        if self.line_type == LineType.REVERSE and not self.hypothesis_id:
            self.hypothesis_id = f"h_{self.item_id}"
        return self


class QueryItem(StrictModel):
    query_id: str
    query: str = Field(min_length=1)
    purpose: str = "fact verification"


class EvidenceCandidate(StrictModel):
    candidate_id: str
    task_id: str
    source_type: SourceType
    source_name: str
    source_ref: SourceRef = Field(default_factory=SourceRef)
    title: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    initial_score: float = 0.0
    rerank_score: float = 0.0
    snippet_only: bool = False
    context_window: dict[str, Any] = Field(default_factory=dict)


class EvidenceScores(StrictModel):
    relevance: float = Field(default=0, ge=0, le=1)
    authority: float = Field(default=0.5, ge=0, le=1)
    directness: float = Field(default=0, ge=0, le=1)
    slot_coverage: float = Field(default=0, ge=0, le=1)
    freshness: float = Field(default=0.5, ge=0, le=1)
    traceability: float = Field(default=0, ge=0, le=1)
    final: float = Field(default=0, ge=0, le=1)


class EvidenceItem(StrictModel):
    evidence_id: str
    task_id: str
    source_type: SourceType
    source_name: str
    source_ref: SourceRef = Field(default_factory=SourceRef)
    title: str = ""
    content: str
    quoted_spans: list[str] = Field(default_factory=list)
    snippet_only: bool = False
    relation: EvidenceRelation
    judge_confidence: float = Field(ge=0, le=1)
    scores: EvidenceScores = Field(default_factory=EvidenceScores)
    covered_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    reason: str = ""
    content_fingerprint: str = ""
    source_evidence_fingerprint: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    slot_evidence: dict[str, dict[str, Any]] = Field(default_factory=dict)
    numeric_relation: EvidenceRelation | None = None
    neutral_reason: str | None = None
    scope_compatible: bool = True
    scope_mismatch_reasons: list[str] = Field(default_factory=list)
    context_window: dict[str, Any] = Field(default_factory=dict)


class PreparedContext(StrictModel):
    target_text: str
    paragraph_text: str
    argument_path: list[ArgumentPathItem] = Field(default_factory=list)
    boundary: str | None = None
    required_slots: list[str] = Field(default_factory=list)
    subject_terms: list[str] = Field(default_factory=list)
    time_scope: list[str] = Field(default_factory=list)
    region_scope: list[str] = Field(default_factory=list)
    metric_terms: list[str] = Field(default_factory=list)
    parent_argument_summary: str = ""
    existing_evidence_summary: str = ""
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    query_context: str = ""


class JudgeResult(StrictModel):
    relation: EvidenceRelation
    confidence: float = Field(ge=0, le=1)
    directness: float = Field(ge=0, le=1)
    reason: str = ""
    quoted_spans: list[str] = Field(default_factory=list)
    supported_claim_ids: list[str] = Field(default_factory=list)
    refuted_claim_ids: list[str] = Field(default_factory=list)
    covered_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    slot_evidence: dict[str, dict[str, Any]] = Field(default_factory=dict)
    llm_relation: EvidenceRelation | None = None
    numeric_facts: list[dict[str, Any]] = Field(default_factory=list)
    numeric_relation: EvidenceRelation | None = None
    relation_conflict: bool = False
    final_relation: EvidenceRelation | None = None
    override_reason: str | None = None
    neutral_reason: NeutralReason | None = None
    quote_match_mode: str | None = None
    scope_compatible: bool = True
    scope_mismatch_reasons: list[str] = Field(default_factory=list)
    claim_results: list["ClaimJudgeResult"] = Field(default_factory=list)


class ClaimJudgeResult(StrictModel):
    """Judge output for one candidate × one atomic claim.

    Candidate-level relations are retained for backwards compatibility, but
    V12 decisions and citations consume this record exclusively.
    """

    claim_id: str
    relation: EvidenceRelation
    confidence: float = Field(ge=0, le=1)
    directness: float = Field(ge=0, le=1)
    reason: str = ""
    quoted_spans: list[str] = Field(default_factory=list)
    covered_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    slot_evidence: dict[str, dict[str, Any]] = Field(default_factory=dict)
    numeric_facts: list[dict[str, Any]] = Field(default_factory=list)
    numeric_relation: EvidenceRelation | None = None
    neutral_reason: NeutralReason | None = None
    scope_compatible: bool = True
    scope_mismatch_reasons: list[str] = Field(default_factory=list)
    quote_match_mode: str | None = None
    matched_claim_id: str | None = None
    override_reason: str | None = None
    numeric_override_allowed: bool = False


class CandidateJudgeRecord(StrictModel):
    task_id: str
    candidate_id: str
    judgment: JudgeResult


class EvidenceQuality(StrictModel):
    effective_evidence_count: int = 0
    direct_evidence_count: int = 0
    authoritative_evidence_count: int = 0
    independent_source_count: int = 0
    independent_document_count: int = 0
    claim_coverage_score: float = 0
    support_weight: float = 0
    refute_weight: float = 0
    supplement_weight: float = 0
    conflict_score: float = 0
    final_evidence_score: float = 0
    noise_ratio: float = 0
    missing_slots: list[str] = Field(default_factory=list)
    only_snippets: bool = False


class VerificationResult(StrictModel):
    verdict: VerificationVerdict
    upstream_status: str
    confidence: float = Field(ge=0, le=1)
    reason: str
    conflict_detected: bool = False
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    refuting_evidence_ids: list[str] = Field(default_factory=list)
    supplementary_evidence_ids: list[str] = Field(default_factory=list)
    # Sufficiency path: GENERIC (multi-source rule) or AUTHORITATIVE_STRUCTURED_OVERRIDE.
    sufficiency_path: str = "GENERIC"
    override_applied: bool = False
    aggregated_slot_coverage: dict[str, list[str]] = Field(default_factory=dict)
    coverage_conflicts: list[str] = Field(default_factory=list)
    claim_logic_operator: str = "SINGLE"
    atomic_claim_verdicts: dict[str, str] = Field(default_factory=dict)


class ErrorDetail(StrictModel):
    code: str
    node: str
    tool: str | None = None
    retryable: bool = False
    reason: str
    batch_id: str | None = None
    affected_candidate_ids: list[str] = Field(default_factory=list)
    timeout_layer: str | None = None
    configured_timeout_ms: int | None = None
    actual_elapsed_ms: int | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolUsage(StrictModel):
    web_rounds: int = 0
    selected_kb_calls: int = 0
    public_kb_calls: int = 0
    structured_calls: int = 0
    attempted_tools: list[str] = Field(default_factory=list)
    tool_elapsed_ms: dict[str, int] = Field(default_factory=dict)


class RetrievalTaskResult(StrictModel):
    task_id: str
    item_id: str
    line_type: LineType
    node_id: str
    hypothesis_id: str | None = None
    target_text: str
    execution_status: ExecutionStatus
    termination_reason: TerminationReason
    verification: VerificationResult
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    evidence_quality: EvidenceQuality = Field(default_factory=EvidenceQuality)
    tool_usage: ToolUsage = Field(default_factory=ToolUsage)
    evidence_gap: str | None = None
    errors: list[ErrorDetail] = Field(default_factory=list)
    elapsed_ms: int = 0
    node_timings_ms: dict[str, int] = Field(default_factory=dict)
    judge_batches: list[str] = Field(default_factory=list)
    judge_candidate_total: int = 0
    judge_candidate_completed: int = 0
    judge_candidate_failed: int = 0
    judge_completeness_ratio: float = Field(default=1.0, ge=0, le=1)
    retrieved_candidate_count: int = 0
    shared_candidate_count: int = 0
    adjacent_chunk_candidate_count: int = 0
    support_count: int = 0
    refute_count: int = 0
    supplement_count: int = 0
    neutral_count: int = 0
    neutral_reasons: list[str] = Field(default_factory=list)
    missing_slots_before_gap_retrieval: list[str] = Field(default_factory=list)
    missing_slots_after_gap_retrieval: list[str] = Field(default_factory=list)
    gap_retrieval_triggered: bool = False
    gap_queries: list[str] = Field(default_factory=list)
    gap_new_candidates: int = 0
    gap_new_evidence: int = 0
    gap_resolved_slot_count: int = 0
    gap_verdict_changed: bool = False
    gap_resolved: bool = False
    atomic_claim_count: int = 0
    kb_query_variant_count: int = 0
    kb_query_result_count_by_query: dict[str, int] = Field(default_factory=dict)
    kb_query_zero_hit_count: int = 0
    kb_raw_candidate_count: int = 0
    kb_exact_duplicate_count: int = 0
    kb_adjacent_chunk_count: int = 0
    kb_shared_candidate_count: int = 0
    kb_final_candidate_count: int = 0
    atomic_claim_candidate_coverage: dict[str, int] = Field(default_factory=dict)
    required_slot_candidate_coverage: dict[str, int] = Field(default_factory=dict)


class ParagraphSearchOutput(StrictModel):
    paragraph_id: str
    results: list[RetrievalTaskResult] = Field(default_factory=list)


class ExecutionSummary(StrictModel):
    paragraph_count: int
    task_count: int
    success_count: int
    partial_count: int
    error_count: int
    elapsed_ms: int
    task_elapsed_p50_ms: int = 0
    task_elapsed_p95_ms: int = 0


class SearchAgentBatchOutput(StrictModel):
    request_id: str
    document_id: str
    execution_summary: ExecutionSummary
    paragraph_results: list[ParagraphSearchOutput]
    errors: list[ErrorDetail] = Field(default_factory=list)
    trace_id: str | None = None
    flow_metrics: dict[str, Any] = Field(default_factory=dict)
    integration_guard: dict[str, Any] = Field(default_factory=dict)


def build_retrieval_tasks(request: SearchAgentBatchInput) -> list[RetrievalTask]:
    """Flatten a validated request while preserving paragraph/item order."""
    tasks: list[RetrievalTask] = []
    ordinal = 0
    for paragraph in request.paragraphs:
        common = dict(
            request_id=request.request_id,
            document_id=request.document_id,
            user_id=request.user_id,
            paragraph_id=paragraph.paragraph_id,
            paragraph_text=paragraph.paragraph_text,
            argument_path=paragraph.argument_context.argument_path,
            boundary=paragraph.argument_context.boundary,
            selected_knowledge_ids=request.knowledge_context.selected_knowledge_ids,
            organization_context=request.organization_context or {},
        )
        for item in paragraph.forward_items:
            ordinal += 1
            claim_group = atomize_claim(item.target_text, line_type="forward")
            tasks.append(RetrievalTask(
                task_id=f"{request.request_id}:task:{ordinal}", line_type=LineType.FORWARD,
                node_id=item.item_id, item_id=item.item_id, target_text=item.target_text,
                required_slots=item.required_slots, existing_evidence_text=item.existing_evidence_text,
                source_refs=item.source_refs,
                retrieval_goal=RetrievalGoal.VERIFY_ORIGINAL, **common,
                atomic_claims=claim_group.atomic_claims,
                claim_logic_operator=claim_group.logic_operator,
                argument_type="NUMERIC_FACT" if any(c.value is not None for c in claim_group.atomic_claims) else "QUALITATIVE_CLAIM",
            ))
        for item in paragraph.reverse_items:
            ordinal += 1
            normalized = normalize_reverse_hypothesis(item.target_text)
            claim_group = atomize_claim(normalized["normalized_hypothesis"], line_type="reverse")
            tasks.append(RetrievalTask(
                task_id=f"{request.request_id}:task:{ordinal}", line_type=LineType.REVERSE,
                node_id=item.item_id, item_id=item.item_id, hypothesis_id=getattr(item, 'hypothesis_id', None) or f"h_{item.item_id}",
                target_text=item.target_text, required_slots=item.required_slots,
                relation_to_original=item.relation_to_original,
                retrieval_goal=RetrievalGoal.VERIFY_HYPOTHESIS, **common,
                atomic_claims=claim_group.atomic_claims,
                claim_logic_operator=claim_group.logic_operator,
                normalized_hypothesis=normalized["normalized_hypothesis"],
                neutral_retrieval_query=normalized["neutral_retrieval_query"],
                polarity=normalized["polarity"],
                argument_type="NEGATIVE_HYPOTHESIS",
            ))
    return tasks


def map_verdict_to_upstream(verdict: VerificationVerdict, line_type: LineType) -> tuple[str, bool]:
    if verdict == VerificationVerdict.CONFLICT:
        return "doubtful", True
    if verdict == VerificationVerdict.INCONCLUSIVE:
        return "doubtful", False
    if line_type == LineType.FORWARD:
        return ("credible", False) if verdict == VerificationVerdict.SUPPORTED else ("error", False)
    return ("supported", False) if verdict == VerificationVerdict.SUPPORTED else ("refuted", False)


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    port = parts.port
    netloc = host if port is None or (parts.scheme == "http" and port == 80) or (parts.scheme == "https" and port == 443) else f"{host}:{port}"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")))
    return urlunsplit((parts.scheme.lower(), netloc, parts.path.rstrip("/") or "/", query, ""))


def stable_evidence_key(candidate: EvidenceCandidate | EvidenceItem) -> str:
    """Globally unique within a request: hashed from request_id+task_id+source.

    Same underlying source content used in different tasks (e.g. Forward
    SUPPORT vs Reverse REFUTE) yields different evidence_id, preventing
    relation overwrite or cache collision in downstream consumers.
    """
    ref = candidate.source_ref
    # task_id encodes the request_id+task_id scope (see RetrievalTask schema).
    scope = f"{candidate.task_id}"
    if candidate.source_type == SourceType.WEB and ref.url:
        identity = f"web|{scope}|{canonical_url(ref.url)}|{hashlib.sha256(candidate.content.encode()).hexdigest()}"
    elif candidate.source_type == SourceType.KNOWLEDGE_BASE:
        identity = f"kb|{scope}|{ref.knowledge_id}|{ref.file_id}|{ref.chunk_id}"
    else:
        identity = f"structured|{scope}|{ref.scenario_name}|{ref.record_id}|{hashlib.sha256(candidate.content.encode()).hexdigest()}"
    return hashlib.sha256(identity.encode()).hexdigest()


def source_evidence_fingerprint(candidate: EvidenceCandidate | EvidenceItem) -> str:
    """Stable fingerprint of the underlying source content, ignoring task scope.

    Use this to recognize "same underlying source content" across tasks
    (e.g. a Structured row reused by Forward and Reverse). It must NOT
    replace the task-scoped `evidence_id`.
    """
    ref = candidate.source_ref
    if candidate.source_type == SourceType.WEB and ref.url:
        identity = f"web|{canonical_url(ref.url)}|{hashlib.sha256(candidate.content.encode()).hexdigest()}"
    elif candidate.source_type == SourceType.KNOWLEDGE_BASE:
        identity = f"kb|{ref.knowledge_id}|{ref.file_id}|{ref.chunk_id}"
    else:
        identity = f"structured|{ref.scenario_name}|{ref.record_id}|{hashlib.sha256(candidate.content.encode()).hexdigest()}"
    return hashlib.sha256(identity.encode()).hexdigest()


def stable_evidence_item_key(
    candidate: EvidenceCandidate | EvidenceItem,
    relation: EvidenceRelation | str | None = None,
) -> str:
    """Task/source/fact/relation key shared by initial and gap rounds."""
    normalized_relation = relation or getattr(candidate, "relation", EvidenceRelation.NEUTRAL)
    if isinstance(normalized_relation, EvidenceRelation):
        normalized_relation = normalized_relation.value
    identity = "|".join([
        candidate.task_id,
        candidate.source_type.value,
        source_evidence_fingerprint(candidate),
        str(getattr(candidate, "metadata", {}).get("matched_claim_id") or ""),
        str(normalized_relation).upper(),
    ])
    return hashlib.sha256(identity.encode()).hexdigest()


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
