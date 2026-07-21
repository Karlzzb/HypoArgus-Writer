"""Argument-aware, multi-source evidence retrieval public API."""

from .config import EvidenceRetrievalConfig
from .batch_graph import build_evidence_retrieval_graph
from .dependencies import EvidenceRetrievalDependencies
from .claim_logic import AtomicClaim, AtomicClaimGroup, ClaimLogicOperator, apply_claim_logic, atomize_claim, normalize_reverse_hypothesis
from .numeric_relation import NumericRelationResult, NumericRelationVerifier
from .retrieval_queries import build_kb_query_variants, dedupe_query_variants
from .candidate_sharing import match_candidate_to_task, share_candidates
from .chunk_context import build_adjacent_context
from .slot_aggregation import (
    SlotType, TimeSlotEvidence, aggregate_slot_evidence, infer_slot_evidence, infer_slot_type,
    normalize_slot_evidence, validate_slot_value,
)
from .scope_guard import ScopeCompatibility, apply_scope_guard, check_scope_compatibility
from .pair_consistency import check_pair_consistency
from .gap_retrieval import GapRetrievalPlan, plan_gap_retrieval
from .schemas import (
    EvidenceCandidate, EvidenceItem, EvidenceQuality, EvidenceRelation,
    ExecutionStatus, ForwardItem, LineType, ParagraphSearchInput,
    RetrievalTask, RetrievalTaskResult, ReverseItem,
    SearchAgentBatchInput, SearchAgentBatchOutput, TerminationReason,
    VerificationResult, VerificationVerdict, build_retrieval_tasks,
    map_verdict_to_upstream, stable_evidence_item_key, stable_evidence_key,
)
from .public_contracts import (
    AgentRunStatus, AtomicClaimDecision, CitationJudgment, CitationProvenance,
    CitationRecord, EvidenceGap, ParagraphInput, PublicWarning,
    SearchAgentInputState, SearchAgentOutputState, TaskDecision, TraceReference,
)
from .search_agent_graph import build_search_agent_graph
from .semantic_validation import validate_public_output_semantics

__all__ = [
    "EvidenceRetrievalConfig", "EvidenceRetrievalDependencies", "build_evidence_retrieval_graph",
    "build_search_agent_graph", "validate_public_output_semantics",
    "EvidenceCandidate", "EvidenceItem", "EvidenceQuality",
    "EvidenceRelation", "ExecutionStatus", "ForwardItem", "LineType",
    "ParagraphSearchInput", "RetrievalTask", "RetrievalTaskResult", "ReverseItem",
    "SearchAgentBatchInput", "SearchAgentBatchOutput",
    "TerminationReason", "VerificationResult", "VerificationVerdict",
    "build_retrieval_tasks", "map_verdict_to_upstream", "stable_evidence_key", "stable_evidence_item_key",
    "AtomicClaim", "AtomicClaimGroup", "ClaimLogicOperator", "atomize_claim",
    "normalize_reverse_hypothesis", "apply_claim_logic", "NumericRelationResult", "NumericRelationVerifier",
    "build_kb_query_variants", "dedupe_query_variants",
    "match_candidate_to_task", "share_candidates",
    "build_adjacent_context",
    "SlotType", "TimeSlotEvidence", "aggregate_slot_evidence", "infer_slot_evidence", "infer_slot_type",
    "normalize_slot_evidence", "validate_slot_value",
    "ScopeCompatibility", "apply_scope_guard", "check_scope_compatibility",
    "check_pair_consistency",
    "GapRetrievalPlan", "plan_gap_retrieval",
    "AgentRunStatus", "AtomicClaimDecision", "CitationJudgment", "CitationProvenance",
    "CitationRecord", "EvidenceGap", "ParagraphInput", "PublicWarning",
    "SearchAgentInputState", "SearchAgentOutputState", "TaskDecision", "TraceReference",
]
