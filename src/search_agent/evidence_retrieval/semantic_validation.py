"""Semantic gates applied immediately before returning a public response."""
from __future__ import annotations

import re
from typing import Any

from .claim_logic import apply_claim_logic
from .config import EvidenceRetrievalConfig
from .output_adapter import _complete_fact_sentence
from .public_contracts import SearchAgentInputState, SearchAgentOutputState
from .scope_guard import check_scope_compatibility


class PublicOutputSemanticError(ValueError):
    pass


def validate_search_agent_input_semantics(value: SearchAgentInputState) -> None:
    text = " ".join([
        value.paragraph.paragraph_text,
        *(row.target_text for row in value.paragraph.forward_items),
        *(row.target_text for row in value.paragraph.reverse_items),
    ])
    if re.search(r"本校|我校|本单位|本企业", text):
        organization = value.organization_context
        if organization is None or not (organization.school_id or organization.tenant_id):
            raise PublicOutputSemanticError("MISSING_ORGANIZATION_CONTEXT")


def validate_public_output_semantics(
    output: SearchAgentOutputState | dict[str, Any],
    public_input: SearchAgentInputState,
    config: EvidenceRetrievalConfig | None = None,
    *,
    structured_tool_records: list[Any] | None = None,
) -> SearchAgentOutputState:
    config = config or EvidenceRetrievalConfig.from_env()
    value = output if isinstance(output, SearchAgentOutputState) else SearchAgentOutputState.model_validate(output)
    errors: list[str] = []
    if value.request_id != public_input.request_id:
        errors.append("request_id mismatch")
    if value.document_id != public_input.document_id:
        errors.append("document_id mismatch")
    if value.paragraph_id != public_input.paragraph.paragraph_id:
        errors.append("paragraph_id mismatch")
    prefix = f"{public_input.request_id}:task:"
    if any(not row.task_id.startswith(prefix) for row in value.results):
        errors.append("task_id prefix mismatch")

    citation_by_id = {row.citation_id: row for row in value.citations}
    for citation in value.citations:
        judgment = citation.judgment
        if citation.relation == "SUPPORT" and not judgment.supported_claim_ids:
            errors.append(f"{citation.citation_id}: SUPPORT without claim mapping")
        if citation.relation == "REFUTE" and not judgment.refuted_claim_ids:
            errors.append(f"{citation.citation_id}: REFUTE without claim mapping")
        if citation.relation == "SUPPLEMENT" and (judgment.supported_claim_ids or judgment.refuted_claim_ids):
            errors.append(f"{citation.citation_id}: SUPPLEMENT has conclusive claim mapping")
        if citation.relation in {"SUPPORT", "REFUTE"}:
            if judgment.confidence < config.public_citation_min_confidence:
                errors.append(f"{citation.citation_id}: confidence below admission threshold")
            if judgment.directness < config.public_citation_min_directness:
                errors.append(f"{citation.citation_id}: directness below admission threshold")
            if not judgment.scope_compatible:
                errors.append(f"{citation.citation_id}: incompatible scope")
        if not citation.content.strip() or not citation.summary.strip() or citation.summary.strip() == citation.content.strip():
            errors.append(f"{citation.citation_id}: invalid citation summary")

    for result in value.results:
        all_ids = set(result.citation_ids)
        if any(citation_id not in citation_by_id for citation_id in all_ids):
            errors.append(f"{result.task_id}: unresolved citation_id")
        claim_by_id = {row.claim_id: row for row in result.atomic_claim_results}
        claim_ids = set(claim_by_id)
        for citation_id in all_ids:
            citation = citation_by_id[citation_id]
            if result.task_id not in citation.task_ids:
                errors.append(f"{result.task_id}: citation is bound to another task")
            mapped = set(citation.judgment.supported_claim_ids + citation.judgment.refuted_claim_ids)
            if not mapped.issubset(claim_ids):
                errors.append(f"{result.task_id}: citation mapped to unknown claim")
            for claim_id in mapped & claim_ids:
                if not _complete_fact_sentence(citation.content, claim_by_id[claim_id].claim_text):
                    errors.append(
                        f"{result.task_id}: citation {citation_id} is not a complete claim-bound fact"
                    )
                report = check_scope_compatibility(claim_by_id[claim_id].claim_text, citation.content)
                if not report.compatible:
                    errors.append(
                        f"{result.task_id}: citation {citation_id} failed semantic scope recheck "
                        f"({','.join(report.mismatch_reasons)})"
                    )
        atomic_logic = apply_claim_logic(
            [row.verdict for row in result.atomic_claim_results],
            result.atomic_claim_results[0].logic_role if result.atomic_claim_results else "SINGLE",
        )
        if result.verdict in {"SUPPORTED", "REFUTED", "CONFLICT"} and atomic_logic != result.verdict:
            errors.append(f"{result.task_id}: verdict conflicts with atomic claim logic")
        expected_word = {
            "SUPPORTED": "支持",
            "REFUTED": "反驳",
            "CONFLICT": "冲突",
            "INCONCLUSIVE": "不足",
        }[result.verdict]
        if expected_word not in result.conclusion_summary:
            errors.append(f"{result.task_id}: summary conflicts with verdict")
        if result.evidence_gap and not result.evidence_gap.resolved and result.verdict in {"SUPPORTED", "REFUTED"}:
            decisive = atomic_logic == "REFUTED" and any(row.logic_role == "AND" for row in result.atomic_claim_results)
            if not decisive:
                errors.append(f"{result.task_id}: unresolved gap conflicts with verdict")

    for record in structured_tool_records or []:
        raw = record if isinstance(record, dict) else record.model_dump(mode="json")
        if raw.get("scenario_key") == "no_structured_query" and raw.get("status") != "SUCCESS":
            errors.append("no_structured_query must succeed")
    if errors:
        raise PublicOutputSemanticError("; ".join(errors))
    return value


__all__ = [
    "PublicOutputSemanticError", "validate_public_output_semantics",
    "validate_search_agent_input_semantics",
]
