from search_agent.evidence_retrieval.admission import AdmissionReason, decide_admission
from search_agent.evidence_retrieval.config import EvidenceRetrievalConfig
from search_agent.evidence_retrieval.schemas import EvidenceItem, EvidenceRelation, EvidenceScores, SourceType
from search_agent.evidence_retrieval.adaptive_stop import decide_reverse_stop


def _item(**changes):
    value = EvidenceItem(
        evidence_id="ev-1", task_id="task-1", source_type=SourceType.WEB,
        source_name="web", content="2025 年薪资为 5000 元。", quoted_spans=["2025 年薪资为 5000 元。"],
        relation=EvidenceRelation.SUPPORT, judge_confidence=.8,
        scores=EvidenceScores(directness=.8), metadata={"supported_claim_ids": ["c1"]},
    )
    return value.model_copy(update=changes)


def test_admission_matrix_records_all_applicable_blockers_in_stable_order() -> None:
    decision = decide_admission(_item(judge_confidence=.2, scores=EvidenceScores(directness=.2), scope_compatible=False), {"c1": "2025 年薪资为 5000 元。"}, EvidenceRetrievalConfig())
    assert decision.admitted is False
    assert decision.reasons == (AdmissionReason.SCOPE_OR_QUOTE, AdmissionReason.LOW_CONFIDENCE, AdmissionReason.LOW_DIRECTNESS)


def test_admission_accepts_complete_direct_mapped_evidence() -> None:
    decision = decide_admission(_item(), {"c1": "2025 年薪资为 5000 元。"}, EvidenceRetrievalConfig())
    assert decision.reasons == (AdmissionReason.ADMITTED,)


def test_reverse_stop_requires_repeated_zero_yield_and_never_overrides_protection() -> None:
    decision = decide_reverse_stop(enabled=True, prior_attempts=2, prior_admitted=0, minimum_attempts=2, unresolved=False, high_priority=False, coverage_protected=False)
    assert decision.skip is True
    protected = decide_reverse_stop(enabled=True, prior_attempts=9, prior_admitted=0, minimum_attempts=2, unresolved=True, high_priority=False, coverage_protected=False)
    assert protected.skip is False
    assert protected.reason == "PROTECTED_REQUIRED_PATH"
