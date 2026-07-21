"""Task-level verdict aggregation. Verdict and tool execution are decoupled.

A verdict is computed independently from the available effective evidence. The
execution_status (SUCCESS / PARTIAL / ERROR) and termination_reason (SUFFICIENT
/ EXHAUSTED / TOOL_ERROR / TIMEOUT) describe whether the *retrieval pipeline*
ran cleanly, not whether the verdict is conclusive. It is therefore valid to
end with ``PARTIAL`` + ``TOOL_ERROR`` + ``REFUTED``: some upstream tools
failed, but the surviving Structured/KB evidence already suffices to refute
the hypothesis.
"""

from __future__ import annotations

from .config import EvidenceRetrievalConfig
from .claim_logic import apply_claim_logic, ClaimLogicOperator
from .schemas import (
    EvidenceItem, EvidenceQuality, EvidenceRelation, LineType, SourceType,
    VerificationResult, VerificationVerdict, map_verdict_to_upstream,
)


def _authoritative_structured_sufficient(
    items: list[EvidenceItem], quality: EvidenceQuality, config: EvidenceRetrievalConfig,
) -> tuple[bool, str | None]:
    """A single authoritative Structured source may satisfy a fact-retrieval task."""
    if not config.authoritative_structured_override_enabled:
        return False, None
    structured_items = [x for x in items if x.source_type == SourceType.STRUCTURED]
    if not structured_items:
        return False, None
    if quality.refute_weight > 0 and quality.support_weight > 0:
        return False, None
    if quality.missing_slots and config.authoritative_structured_require_full_slot_coverage:
        return False, None
    has_high_authority = any(
        item.scores.authority >= config.authoritative_structured_min_authority
        and item.scores.directness >= config.authoritative_structured_min_directness
        for item in structured_items
    )
    if not has_high_authority:
        return False, None
    if quality.effective_evidence_count == 0:
        return False, None
    return True, "AUTHORITATIVE_STRUCTURED_OVERRIDE"


def aggregate_verification(
    items: list[EvidenceItem],
    quality: EvidenceQuality,
    line_type: LineType,
    config: EvidenceRetrievalConfig,
    *,
    atomic_claim_verdicts: dict[str, str] | None = None,
    claim_logic_operator: str | ClaimLogicOperator = ClaimLogicOperator.SINGLE,
) -> VerificationResult:
    """Compute the final verdict from evidence quality AND atomic claim logic.

    P0-1: Atomic Claim verdicts control the final verdict via apply_claim_logic().
    P0-2: missing_slots blocks SUPPORTED/REFUTED in the GENERIC path.
    P0-8: Atomic Claim level conflict (both SUPPORT and REFUTE) → CONFLICT.
    """
    unique_items: list[EvidenceItem] = []
    seen_items: set[tuple[str, str, str, str]] = set()
    for item in items:
        key = (
            item.task_id,
            item.source_type.value,
            item.source_evidence_fingerprint or item.content_fingerprint or item.evidence_id,
            item.relation.value,
        )
        if key in seen_items:
            continue
        seen_items.add(key)
        unique_items.append(item)
    items = unique_items

    # P0-2: required_slots missing blocks definitive verdicts in GENERIC path
    slots_missing = bool(quality.missing_slots)

    direct_ok = quality.direct_evidence_count >= config.min_direct_evidence_count
    count_ok = (
        quality.effective_evidence_count >= config.min_effective_evidence_count
        and quality.independent_document_count >= config.min_independent_document_count
    )
    exceptional = any(x.scores.final >= config.high_authority_single_evidence_score and x.scores.directness >= 0.8 for x in items)
    sufficient = (
        direct_ok and (count_ok or exceptional)
        and quality.independent_source_count >= config.min_independent_source_count
        and quality.claim_coverage_score >= config.min_claim_coverage_score
        and quality.final_evidence_score >= config.min_final_evidence_score
        and quality.noise_ratio <= config.max_noise_ratio
        and not quality.only_snippets
        and not slots_missing  # P0-2: missing_slots blocks GENERIC sufficiency
    )
    structured_override, override_reason = _authoritative_structured_sufficient(items, quality, config)

    # P0-1: Compute claim-logic verdict from atomic claim verdicts
    claim_verdict: str | None = None
    if atomic_claim_verdicts:
        relations = list(atomic_claim_verdicts.values())
        claim_verdict = apply_claim_logic(relations, claim_logic_operator)

    # AND follows Boolean short-circuit semantics: one reliably REFUTED branch
    # refutes the conjunction; it is not an evidence conflict.

    # Compute quality-based verdict
    if quality.support_weight >= config.conflict_weight_threshold and quality.refute_weight >= config.conflict_weight_threshold:
        quality_verdict = VerificationVerdict.CONFLICT
    elif sufficient and quality.support_weight - quality.refute_weight >= config.verdict_margin:
        quality_verdict = VerificationVerdict.SUPPORTED
    elif sufficient and quality.refute_weight - quality.support_weight >= config.verdict_margin:
        quality_verdict = VerificationVerdict.REFUTED
    elif structured_override:
        if quality.support_weight > quality.refute_weight:
            quality_verdict = VerificationVerdict.SUPPORTED
        else:
            quality_verdict = VerificationVerdict.REFUTED
    else:
        quality_verdict = VerificationVerdict.INCONCLUSIVE

    # P0-1: Atomic Claim verdict takes precedence when it's definitive
    final_verdict = quality_verdict
    override_applied = bool(
        structured_override
        and quality_verdict in {VerificationVerdict.SUPPORTED, VerificationVerdict.REFUTED}
    )
    if claim_verdict and claim_verdict in {"SUPPORTED", "REFUTED", "CONFLICT"}:
        final_verdict = VerificationVerdict(claim_verdict)
        override_applied = True

    # P0-2: If slots are missing, downgrade to INCONCLUSIVE regardless
    decisive_and_refute = (
        str(getattr(claim_logic_operator, "value", claim_logic_operator)).upper() == "AND"
        and claim_verdict == "REFUTED"
    )
    if slots_missing and final_verdict in {VerificationVerdict.SUPPORTED, VerificationVerdict.REFUTED} and not decisive_and_refute:
        final_verdict = VerificationVerdict.INCONCLUSIVE

    # Compute confidence and reason
    if final_verdict == VerificationVerdict.CONFLICT:
        confidence = min(1.0, quality.conflict_score)
        reason = "当前同时存在支持证据和反驳证据，双方证据强度接近，暂无法形成单一确定结论。"
    elif final_verdict == VerificationVerdict.SUPPORTED:
        confidence = min(1.0, quality.final_evidence_score)
        reason = "已获得多条相互独立且直接相关的证据，证据质量达到配置要求，支持目标内容。"
    elif final_verdict == VerificationVerdict.REFUTED:
        confidence = min(1.0, quality.final_evidence_score)
        reason = "已获得多条相互独立且直接相关的证据，证据质量达到配置要求，反驳目标内容。"
    else:
        confidence = min(0.59, quality.final_evidence_score)
        if slots_missing:
            reason = "证据尚不充分，部分必填槽位未覆盖，因此暂不能形成确定结论。"
        else:
            reason = "证据尚不充分，未达到配置的充分性阈值，因此暂不能形成确定结论。"

    upstream, conflict = map_verdict_to_upstream(final_verdict, line_type)
    return VerificationResult(
        verdict=final_verdict, upstream_status=upstream, confidence=confidence, reason=reason,
        conflict_detected=conflict,
        supporting_evidence_ids=list(dict.fromkeys(x.evidence_id for x in items if x.relation == EvidenceRelation.SUPPORT)),
        refuting_evidence_ids=list(dict.fromkeys(x.evidence_id for x in items if x.relation == EvidenceRelation.REFUTE)),
        supplementary_evidence_ids=list(dict.fromkeys(x.evidence_id for x in items if x.relation == EvidenceRelation.SUPPLEMENT)),
        sufficiency_path=override_reason or "GENERIC",
        override_applied=override_applied,
    )
