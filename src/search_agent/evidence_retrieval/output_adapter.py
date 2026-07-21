"""Fail-closed mapper from internal retrieval diagnostics to the public API."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from .claim_logic import apply_claim_logic
from .config import EvidenceRetrievalConfig
from .public_contracts import (
    AgentRunStatus,
    AtomicClaimDecision,
    CitationJudgment,
    CitationProvenance,
    CitationRecord,
    EvidenceGap,
    PublicWarning,
    SearchAgentInputState,
    SearchAgentOutputState,
    TaskDecision,
    TraceReference,
)
from .schemas import (
    EvidenceItem,
    EvidenceRelation,
    RetrievalTask,
    RetrievalTaskResult,
    SearchAgentBatchInput,
    SourceType,
    build_retrieval_tasks,
    stable_json_hash,
)


def to_internal_batch_request(value: SearchAgentInputState) -> SearchAgentBatchInput:
    """Build the internal retrieval request without changing public IDs."""
    return SearchAgentBatchInput(
        request_id=value.request_id,
        document_id=value.document_id,
        user_id=value.user_id or "anonymous",
        paragraphs=[value.paragraph],
        knowledge_context=value.knowledge_context or {},
        retrieval_policy=value.retrieval_policy or {},
        trace_context=value.trace_context or {},
        organization_context=value.organization_context.model_dump(mode="json")
        if value.organization_context
        else None,
    )


def to_internal_batch_request_many(
    values: list[SearchAgentInputState],
) -> SearchAgentBatchInput:
    """Combine compatible public paragraph inputs into one internal request."""

    if not values:
        raise ValueError("SearchAgent batch requires at least one paragraph input")
    first = values[0]
    for value in values[1:]:
        if value.request_id != first.request_id:
            raise ValueError("all batch inputs must share request_id")
        if value.document_id != first.document_id:
            raise ValueError("all batch inputs must share document_id")
        if value.user_id != first.user_id:
            raise ValueError("all batch inputs must share user_id")
        if value.organization_context != first.organization_context:
            raise ValueError("all batch inputs must share organization_context")
    knowledge_ids = list(
        dict.fromkeys(
            knowledge_id
            for value in values
            for knowledge_id in (
                value.knowledge_context.selected_knowledge_ids if value.knowledge_context else []
            )
        )
    )
    return SearchAgentBatchInput(
        request_id=first.request_id,
        document_id=first.document_id,
        user_id=first.user_id or "anonymous",
        paragraphs=[value.paragraph for value in values],
        knowledge_context={"selected_knowledge_ids": knowledge_ids},
        retrieval_policy=first.retrieval_policy or {},
        trace_context=first.trace_context or {},
        organization_context=(
            first.organization_context.model_dump(mode="json")
            if first.organization_context
            else None
        ),
    )


def adapt_v11_input_to_legacy(value: SearchAgentInputState) -> dict[str, Any]:
    """Deprecated compatibility alias; production code uses the formal input."""
    return to_internal_batch_request(value).model_dump(mode="json")


def _quote_mode(item: EvidenceItem) -> str:
    if item.source_type == SourceType.STRUCTURED:
        return "STRUCTURED_ROW"
    if item.snippet_only:
        return "SNIPPET"
    return (
        "NORMALIZED"
        if str(item.metadata.get("quote_match_mode") or "").casefold() == "normalized"
        else "EXACT"
    )


def _query_ids(item: EvidenceItem) -> list[str]:
    values: list[Any] = [item.source_ref.query_id]
    for key in ("query_ids", "query_variant_ids", "candidate_source_query_ids"):
        raw = item.metadata.get(key) or []
        values.extend(raw if isinstance(raw, list) else [raw])
    return list(dict.fromkeys(str(value) for value in values if value))


def _optional_str(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None or value == "" else int(value)
    except (TypeError, ValueError):
        return None


_PUBLIC_CITATION_BLOCKERS = {
    "WRONG_YEAR",
    "WRONG_TIME_SCOPE",
    "WRONG_REGION",
    "WRONG_REGION_SCOPE",
    "WRONG_SUBJECT",
    "WRONG_ENTITY",
    "WRONG_SUBJECT_SCOPE",
    "WRONG_METRIC",
    "WRONG_METRIC_SCOPE",
    "WRONG_MARKET_SCOPE",
    "WRONG_STATISTICAL_SCOPE",
    "INCOMPARABLE_UNIT",
    "QUOTE_NOT_FOUND",
}


_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_MEASUREMENT_PATTERN = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?\s*(?:万亿|千亿|百亿|亿|万|千)?\s*"
    r"(?:美元|人民币|元|台|人|家|个|%|％|个百分点|倍)(?![\w%％])",
    re.I,
)
_METRIC_PATTERNS = (
    (
        re.compile(r"市场规模|market[_\s-]?size", re.I),
        re.compile(r"市场规模|market[_\s-]?size", re.I),
    ),
    (re.compile(r"出货量|shipment", re.I), re.compile(r"出货量|shipment", re.I)),
    (
        re.compile(r"同比|环比|增长率|复合增长|CAGR|growth", re.I),
        re.compile(r"同比|环比|增长率|复合增长|CAGR|growth", re.I),
    ),
    (
        re.compile(r"份额|占比|market[_\s-]?share", re.I),
        re.compile(r"份额|占比|market[_\s-]?share", re.I),
    ),
    (re.compile(r"保有量|inventory", re.I), re.compile(r"保有量|inventory", re.I)),
    (
        re.compile(r"招聘|接收人数|实习生人数|actual[_\s-]?count", re.I),
        re.compile(r"招聘|接收人数|实习生人数|actual[_\s-]?count", re.I),
    ),
    (re.compile(r"薪资|工资|salary", re.I), re.compile(r"薪资|工资|salary", re.I)),
)


def _complete_fact_sentence(value: str, claim_text: str = "") -> bool:
    text = " ".join(str(value or "").split()).strip()
    if len(text) < 8 or not re.search(r"[\u3400-\u9fffA-Za-z]", text):
        return False
    if re.search(
        r"(?:达到|约为|预计|截至|因为|以及|并且|为|到|至|and|or|because)\s*$", text, re.I
    ):
        return False
    # Continuation fragments are not independently consumable citations.
    if re.match(r"^(?:同比|环比|其中|此外|同时|并且|以及|而且|该比例|该数值)", text, re.I):
        return False
    claim = " ".join(str(claim_text or "").split()).strip()
    if not claim:
        return True
    # A dated claim must retain every explicit year in the public quote. This
    # prevents surrounding candidate context from laundering an undated span.
    claim_years = set(_YEAR_PATTERN.findall(claim))
    if claim_years and not claim_years.issubset(set(_YEAR_PATTERN.findall(text))):
        return False
    # Numeric conclusions require a real measured value in the quote, not a
    # qualitative phrase such as "占据主导".
    if _MEASUREMENT_PATTERN.search(claim) and not _MEASUREMENT_PATTERN.search(text):
        return False
    # Keep the claimed metric explicit so a bare number cannot support market
    # size, shipment, share, growth, recruitment, or salary claims.
    for claim_pattern, quote_pattern in _METRIC_PATTERNS:
        if claim_pattern.search(claim) and not quote_pattern.search(text):
            return False
    return True


def _citation_summary(relation: str, claim_text: str) -> str:
    claim = " ".join(claim_text.split()).strip()[:160]
    if relation == "SUPPORT":
        return f"该来源给出了支持“{claim}”的直接事实。"
    if relation == "REFUTE":
        return f"该来源给出了反驳“{claim}”的直接事实。"
    return f"该来源提供了与“{claim}”相关的背景信息。"


def _citation(
    item: EvidenceItem,
    claim_text_by_id: dict[str, str],
    config: EvidenceRetrievalConfig,
) -> CitationRecord | None:
    content = " ".join(span.strip() for span in item.quoted_spans if span.strip())[:600]
    if item.metadata.get("retrieval_candidate_passthrough") is True:
        if not content:
            return None
        source_type = {
            SourceType.WEB: "WEB",
            SourceType.KNOWLEDGE_BASE: "KNOWLEDGE_BASE",
            SourceType.STRUCTURED: "STRUCTURED_DATA",
        }[item.source_type]
        citation_id = f"cit-{stable_json_hash([item.evidence_id, 'CANDIDATE_PASSTHROUGH'])[:20]}"
        retrieved_at = str(
            item.metadata.get("retrieved_at")
            or datetime.now(UTC).isoformat().replace("+00:00", "Z")
        )
        return CitationRecord(
            citation_id=citation_id,
            task_ids=[item.task_id],
            content=content,
            summary="未裁决的检索候选素材，交由上层 HypoArgus Judgment 判断。",
            title=item.title or None,
            source_type=source_type,
            source_name=item.source_name,
            url=item.source_ref.url,
            document_id=_optional_str(item.metadata.get("document_id")),
            knowledge_id=_optional_str(item.source_ref.knowledge_id),
            file_id=_optional_str(item.source_ref.file_id),
            chunk_id=_optional_str(item.source_ref.chunk_id),
            page=_optional_int(item.metadata.get("page")),
            # Compatibility-only public value. No relation was judged.
            relation="SUPPLEMENT",
            status="DEGRADED" if item.snippet_only else "ACCEPTED",
            judgment=CitationJudgment(
                confidence=0.0,
                directness=0.0,
                supported_claim_ids=[],
                refuted_claim_ids=[],
                reason="RETRIEVAL_CANDIDATE_PASSTHROUGH: unjudged material",
                scope_compatible=True,
                scope_mismatch_reasons=[],
                quote_match_mode=_quote_mode(item),
            ),
            provenance=CitationProvenance(
                query_ids=_query_ids(item),
                tool_call_id=item.metadata.get("tool_call_id"),
                scenario_key=item.metadata.get("scenario_key") or item.source_ref.scenario_name,
                dataset_id=item.source_ref.dataset_id,
                query_execution_id=item.source_ref.query_execution_id,
                retrieved_at=retrieved_at,
                published_at=item.metadata.get("published_at"),
                content_fingerprint=item.content_fingerprint,
                source_evidence_fingerprint=item.source_evidence_fingerprint,
            ),
        )

    if item.relation == EvidenceRelation.NEUTRAL:
        return None
    if not _complete_fact_sentence(content):
        return None
    if any(reason in _PUBLIC_CITATION_BLOCKERS for reason in item.scope_mismatch_reasons):
        return None
    if re.search(
        r"无法直接(?:支持|确认|反驳|否定)|cannot directly (?:support|refute)",
        item.reason or "",
        re.I,
    ):
        return None

    supported = (
        list(item.metadata.get("supported_claim_ids") or [])
        if item.relation == EvidenceRelation.SUPPORT
        else []
    )
    refuted = (
        list(item.metadata.get("refuted_claim_ids") or [])
        if item.relation == EvidenceRelation.REFUTE
        else []
    )
    mapped_ids = supported or refuted
    if item.relation in {EvidenceRelation.SUPPORT, EvidenceRelation.REFUTE}:
        if not mapped_ids or any(value not in claim_text_by_id for value in mapped_ids):
            return None
        if not item.scope_compatible:
            return None
        if item.judge_confidence < config.public_citation_min_confidence:
            return None
        if item.scores.directness < config.public_citation_min_directness:
            return None
        if any(
            not _complete_fact_sentence(content, claim_text_by_id[claim_id])
            for claim_id in mapped_ids
        ):
            return None
    elif item.relation == EvidenceRelation.SUPPLEMENT:
        matched_claim_id = str(item.metadata.get("matched_claim_id") or "")
        if matched_claim_id not in claim_text_by_id:
            return None
        if item.judge_confidence < config.public_supplement_min_confidence:
            return None
        supported, refuted, mapped_ids = [], [], [matched_claim_id]
    else:
        return None

    source_type = {
        SourceType.WEB: "WEB",
        SourceType.KNOWLEDGE_BASE: "KNOWLEDGE_BASE",
        SourceType.STRUCTURED: "STRUCTURED_DATA",
    }[item.source_type]
    citation_id = f"cit-{stable_json_hash([item.evidence_id, item.relation.value])[:20]}"
    retrieved_at = str(
        item.metadata.get("retrieved_at") or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    return CitationRecord(
        citation_id=citation_id,
        task_ids=[item.task_id],
        content=content,
        summary=_citation_summary(item.relation.value, claim_text_by_id[mapped_ids[0]]),
        title=item.title or None,
        source_type=source_type,
        source_name=item.source_name,
        url=item.source_ref.url,
        document_id=_optional_str(item.metadata.get("document_id")),
        knowledge_id=_optional_str(item.source_ref.knowledge_id),
        file_id=_optional_str(item.source_ref.file_id),
        chunk_id=_optional_str(item.source_ref.chunk_id),
        page=_optional_int(item.metadata.get("page")),
        relation=item.relation.value,
        status="DEGRADED" if item.snippet_only else "ACCEPTED",
        judgment=CitationJudgment(
            confidence=item.judge_confidence,
            directness=item.scores.directness,
            supported_claim_ids=supported,
            refuted_claim_ids=refuted,
            reason=item.reason or item.relation.value,
            scope_compatible=item.scope_compatible,
            scope_mismatch_reasons=item.scope_mismatch_reasons,
            quote_match_mode=_quote_mode(item),
        ),
        provenance=CitationProvenance(
            query_ids=_query_ids(item),
            tool_call_id=item.metadata.get("tool_call_id"),
            scenario_key=item.metadata.get("scenario_key") or item.source_ref.scenario_name,
            dataset_id=item.source_ref.dataset_id,
            query_execution_id=item.source_ref.query_execution_id,
            retrieved_at=retrieved_at,
            published_at=item.metadata.get("published_at"),
            content_fingerprint=item.content_fingerprint,
            source_evidence_fingerprint=item.source_evidence_fingerprint,
        ),
    )


def _conclusion_summary(
    verdict: str,
    target_text: str,
    atomic: list[AtomicClaimDecision],
    decisive_and_refuted: bool,
) -> str:
    target = " ".join(target_text.split()).strip()[:180]
    if verdict == "SUPPORTED":
        return f"现有有效引用支持“{target}”。"
    if verdict == "REFUTED":
        if decisive_and_refuted:
            claim = next((row.claim_text for row in atomic if row.verdict == "REFUTED"), target)
            return f"现有有效引用反驳原子主张“{claim}”，因此该 AND 主张整体被反驳。"
        return f"现有有效引用反驳“{target}”。"
    if verdict == "CONFLICT":
        return f"关于“{target}”的有效引用相互冲突。"
    return f"关于“{target}”的有效引用不足，无法形成确定结论。"


def build_public_output(
    public_input: SearchAgentInputState,
    diagnostic: dict[str, Any],
    config: EvidenceRetrievalConfig | None = None,
    *,
    task_by_id: dict[str, RetrievalTask] | None = None,
) -> SearchAgentOutputState:
    config = config or EvidenceRetrievalConfig.from_env()
    if task_by_id is None:
        tasks = build_retrieval_tasks(to_internal_batch_request(public_input))
        task_by_id = {task.task_id: task for task in tasks}
    raw_results = [
        raw
        for paragraph in diagnostic.get("paragraph_results", [])
        if paragraph.get("paragraph_id") == public_input.paragraph.paragraph_id
        for raw in paragraph.get("results", [])
    ]
    results = [RetrievalTaskResult.model_validate(raw) for raw in raw_results]
    citations: dict[str, CitationRecord] = {}
    decisions: list[TaskDecision] = []
    public_warnings: dict[tuple[str, str], PublicWarning] = {}
    web_task_metrics = diagnostic.get("flow_metrics", {}).get("web_task", {})

    for result in results:
        task = task_by_id[result.task_id]
        claim_text_by_id = {
            claim.claim_id: claim.source_text_span or claim.qualifier or claim.subject
            for claim in task.atomic_claims
        }
        task_citations: list[CitationRecord] = []
        for item in result.evidence_items:
            citation = _citation(item, claim_text_by_id, config)
            if citation is None:
                continue
            previous = citations.get(citation.citation_id)
            if previous:
                citation = citation.model_copy(
                    update={
                        "task_ids": list(dict.fromkeys([*previous.task_ids, *citation.task_ids])),
                    }
                )
            citations[citation.citation_id] = citation
            task_citations.append(citation)

        supporting = [row.citation_id for row in task_citations if row.relation == "SUPPORT"]
        refuting = [row.citation_id for row in task_citations if row.relation == "REFUTE"]
        supplementary = [row.citation_id for row in task_citations if row.relation == "SUPPLEMENT"]
        atomic: list[AtomicClaimDecision] = []
        for claim in task.atomic_claims:
            claim_citations = [
                row
                for row in task_citations
                if claim.claim_id
                in (row.judgment.supported_claim_ids + row.judgment.refuted_claim_ids)
            ]
            has_support = any(row.relation == "SUPPORT" for row in claim_citations)
            has_refute = any(row.relation == "REFUTE" for row in claim_citations)
            verdict = (
                "CONFLICT"
                if has_support and has_refute
                else "REFUTED"
                if has_refute
                else "SUPPORTED"
                if has_support
                else "INCONCLUSIVE"
            )
            _claim_reason = {
                "SUPPORTED": "已获得有效证据支持该原子主张。",
                "REFUTED": "已获得有效证据反驳该原子主张。",
                "CONFLICT": "同时存在支持和反驳证据，无法形成确定结论。",
                "INCONCLUSIVE": "当前检索未获得可用于支持或反驳该原子主张的有效证据。",
            }.get(verdict, "当前检索未获得可用于支持或反驳该原子主张的有效证据。")
            atomic.append(
                AtomicClaimDecision(
                    claim_id=claim.claim_id,
                    claim_text=claim_text_by_id[claim.claim_id],
                    logic_role=task.claim_logic_operator.value,
                    verdict=verdict,
                    citation_ids=list(dict.fromkeys(row.citation_id for row in claim_citations)),
                    missing_slots=result.evidence_quality.missing_slots,
                    reason=_claim_reason,
                )
            )

        warning_codes: list[str] = []
        for error in result.errors:
            warning_codes.append(error.code)
            warning = PublicWarning(
                code=error.code,
                message=error.reason,
                task_ids=[result.task_id],
                retryable=error.retryable,
            )
            public_warnings[(warning.code, warning.message)] = warning
        for raw_warning in web_task_metrics.get(result.task_id, {}).get("warnings", []):
            code = str(raw_warning).split(":", 1)[0]
            warning_codes.append(code)
            warning = PublicWarning(
                code=code, message=str(raw_warning), task_ids=[result.task_id], retryable=True
            )
            public_warnings[(warning.code, warning.message)] = warning

        public_claim_verdicts = {row.claim_id: row.verdict for row in atomic}
        logic_verdict = apply_claim_logic(
            list(public_claim_verdicts.values()), task.claim_logic_operator
        )
        verdict = result.verification.verdict.value
        if verdict in {"SUPPORTED", "REFUTED", "CONFLICT"} and logic_verdict != verdict:
            verdict = "INCONCLUSIVE"
        # If internal verdict is INCONCLUSIVE but atomic logic says CONFLICT,
        # upgrade to CONFLICT (V12 P0-9: any claim CONFLICT → overall CONFLICT).
        if verdict == "INCONCLUSIVE" and logic_verdict == "CONFLICT":
            verdict = "CONFLICT"
        unresolved_gap = bool(result.evidence_gap and not result.gap_resolved)
        decisive_and_refuted = (
            task.claim_logic_operator.value == "AND" and logic_verdict == "REFUTED"
        )
        if unresolved_gap and verdict in {"SUPPORTED", "REFUTED"} and not decisive_and_refuted:
            verdict = "INCONCLUSIVE"

        # Hard consistency rule: if final verdict is INCONCLUSIVE but atomic
        # claims have definitive verdicts, downgrade atomic claims to match.
        # This prevents downstream from seeing Task=INCONCLUSIVE while
        # AtomicClaim=SUPPORTED — they must be consistent.
        #
        # Scope: only apply to SINGLE tasks (where task==claim is 1:1) or
        # when internal gap is CONFLICTING_EVIDENCE (same claim has both
        # support and refute). For AND/OR tasks with non-conflicting gaps,
        # individual claims may legitimately have different verdicts.
        _force_downgrade = verdict == "INCONCLUSIVE" and (
            task.claim_logic_operator.value == "SINGLE"
            or result.evidence_gap == "CONFLICTING_EVIDENCE"
        )
        if _force_downgrade:
            _has_internal_conflict = result.evidence_gap == "CONFLICTING_EVIDENCE"
            _downgraded_atomic = []
            for acr in atomic:
                if acr.verdict in {"SUPPORTED", "REFUTED"}:
                    _dv = "CONFLICT" if _has_internal_conflict else "INCONCLUSIVE"
                    _dr = {
                        "CONFLICT": "同时存在支持和反驳证据，无法形成确定结论。",
                        "INCONCLUSIVE": "当前检索未获得可用于支持或反驳该原子主张的有效证据。",
                    }.get(_dv, acr.reason)
                    _downgraded_atomic.append(
                        AtomicClaimDecision(
                            claim_id=acr.claim_id,
                            claim_text=acr.claim_text,
                            logic_role=acr.logic_role,
                            verdict=_dv,
                            citation_ids=acr.citation_ids,
                            missing_slots=acr.missing_slots,
                            reason=_dr,
                        )
                    )
                else:
                    _downgraded_atomic.append(acr)
            atomic = _downgraded_atomic
        decisions.append(
            TaskDecision(
                task_id=result.task_id,
                item_id=result.item_id,
                node_id=result.node_id,
                hypothesis_id=result.hypothesis_id,
                line_type=result.line_type.value,
                target_text=result.target_text,
                run_status=result.execution_status.value,
                verdict=verdict,
                confidence=result.verification.confidence
                if verdict != "INCONCLUSIVE"
                else min(result.verification.confidence, 0.59),
                conclusion_summary=_conclusion_summary(
                    verdict, task.target_text, atomic, decisive_and_refuted
                ),
                citation_ids=list(dict.fromkeys([*supporting, *refuting, *supplementary])),
                supporting_citation_ids=list(dict.fromkeys(supporting)),
                refuting_citation_ids=list(dict.fromkeys(refuting)),
                supplementary_citation_ids=list(dict.fromkeys(supplementary)),
                atomic_claim_results=atomic,
                evidence_gap=EvidenceGap(
                    reason=(_gap_reason := result.evidence_gap)
                    if result.evidence_gap != "CONFLICTING_EVIDENCE"
                    or (len(supporting) + len(refuting) + len(supplementary)) > 0
                    else "INSUFFICIENT_EFFECTIVE_COUNT",
                    missing_slots=result.evidence_quality.missing_slots,
                    gap_retrieval_triggered=result.gap_retrieval_triggered,
                    resolved=result.gap_resolved,
                )
                if result.evidence_gap
                else None,
                warnings=list(dict.fromkeys(warning_codes)),
            )
        )

    structured_warnings = (
        diagnostic.get("flow_metrics", {}).get("structured_tool_calling", {}).get("warnings", [])
    )
    for raw in structured_warnings:
        warning = PublicWarning(code=str(raw).split(":", 1)[0], message=str(raw), task_ids=[])
        public_warnings[(warning.code, warning.message)] = warning

    completed = sum(row.run_status == "SUCCESS" for row in decisions)
    partial = sum(row.run_status == "PARTIAL" for row in decisions)
    errors = sum(row.run_status == "ERROR" for row in decisions)
    status = (
        "ERROR"
        if errors and not completed and not partial
        else "PARTIAL"
        if partial or errors
        else "SUCCESS"
    )
    return SearchAgentOutputState(
        request_id=public_input.request_id,
        document_id=public_input.document_id,
        paragraph_id=public_input.paragraph.paragraph_id,
        run_status=AgentRunStatus(
            status=status,
            completed_task_count=completed,
            partial_task_count=partial,
            error_task_count=errors,
        ),
        results=decisions,
        citations=list(citations.values()),
        warnings=list(public_warnings.values()),
        trace=TraceReference(trace_id=diagnostic.get("trace_id"), trace_url=None),
    )


def build_public_batch_outputs(
    public_inputs: list[SearchAgentInputState],
    diagnostic: dict[str, Any],
    config: EvidenceRetrievalConfig | None = None,
) -> list[SearchAgentOutputState]:
    """Map one internal Batch result back to stable per-paragraph public outputs."""

    config = config or EvidenceRetrievalConfig.from_env()
    request = to_internal_batch_request_many(public_inputs)
    tasks = build_retrieval_tasks(request)
    task_by_id = {task.task_id: task for task in tasks}
    return [
        build_public_output(
            value,
            diagnostic,
            config,
            task_by_id=task_by_id,
        )
        for value in public_inputs
    ]


__all__ = [
    "to_internal_batch_request",
    "to_internal_batch_request_many",
    "adapt_v11_input_to_legacy",
    "build_public_output",
    "build_public_batch_outputs",
]
