"""Deterministic evidence scoring and independence-aware aggregation."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlsplit

from .config import EvidenceRetrievalConfig
from .schemas import EvidenceItem, EvidenceQuality, EvidenceRelation, SourceType
from .schemas import canonical_url
from .retrievers.bm25_retriever import tokenize


def _authority(item: EvidenceItem) -> float:
    if item.source_type == SourceType.STRUCTURED:
        return 0.9
    if item.source_type == SourceType.KNOWLEDGE_BASE:
        return 0.75
    host = (urlsplit(item.source_ref.url or "").hostname or "").lower()
    if host.endswith((".gov.cn", ".gov", ".edu.cn", ".edu")):
        return 0.9
    return 0.55


def _source_key(item: EvidenceItem) -> str:
    if item.source_type == SourceType.WEB:
        return (urlsplit(item.source_ref.url or "").hostname or item.content_fingerprint).lower()
    if item.source_type == SourceType.KNOWLEDGE_BASE:
        return f"{item.source_ref.knowledge_id}:{item.source_ref.file_id}"
    return item.source_ref.dataset_id or item.source_ref.scenario_name or item.content_fingerprint


def _document_key(item: EvidenceItem) -> str:
    if item.source_type == SourceType.WEB:
        return canonical_url(item.source_ref.url or "") if item.source_ref.url else item.content_fingerprint
    if item.source_type == SourceType.KNOWLEDGE_BASE:
        return f"{item.source_ref.knowledge_id}:{item.source_ref.file_id}"
    return item.source_ref.query_execution_id or f"{item.source_ref.scenario_name}:{item.source_ref.query_params_hash or 'unknown-query'}"


def _content_tokens(item: EvidenceItem) -> set[str]:
    return set(tokenize(f"{item.title} {item.content}"))


def _near_duplicate(tokens: set[str], representatives: list[set[str]], threshold: float) -> bool:
    for other in representatives:
        union = tokens | other
        if union and len(tokens & other) / len(union) >= threshold:
            return True
    return False


def _freshness(item: EvidenceItem, config: EvidenceRetrievalConfig) -> float:
    # Source lifecycle time and observation period are different concepts.
    # Historical data years/content years never imply an old publication.
    raw = next((item.metadata.get(key) for key in (
        "source_published_at", "source_updated_at", "published_at",
        "publish_time", "updated_at", "update_time",
    ) if item.metadata.get(key)), None)
    if raw is None:
        return 0.5
    try:
        if isinstance(raw, (int, float)) or str(raw).isdigit() and len(str(raw)) == 4:
            moment = datetime(int(raw), 1, 1, tzinfo=timezone.utc)
        else:
            moment = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - moment).total_seconds() / 86400)
        return max(0.0, min(1.0, 0.5 ** (age_days / config.freshness_half_life_days)))
    except (ValueError, TypeError, OverflowError):
        return 0.5


def analyze_evidence_quality(items: list[EvidenceItem], required_slots: list[str], candidate_count: int, config: EvidenceRetrievalConfig) -> EvidenceQuality:
    valid = [x for x in items if x.relation != EvidenceRelation.NEUTRAL]
    independent_sources: set[str] = set()
    seen_sources: set[str] = set()
    seen_documents: set[str] = set()
    representative_tokens: list[set[str]] = []
    support = refute = supplement = 0.0
    covered: set[str] = set()
    direct = authoritative = 0
    for item in valid:
        item.scores.authority = _authority(item)
        item.scores.slot_coverage = len(set(item.covered_slots) & set(required_slots)) / max(1, len(required_slots)) if required_slots else 1.0
        item.scores.traceability = 1.0 if any(item.source_ref.model_dump(exclude_none=True).values()) else 0.0
        item.scores.freshness = _freshness(item, config)
        score = sum(config.quality_weights[name] * getattr(item.scores, name) for name in config.quality_weights)
        if item.snippet_only:
            score = min(score, config.snippet_only_score_cap)
        document_key = _document_key(item)
        tokens = _content_tokens(item)
        duplicate_document = document_key in seen_documents
        duplicate_content = _near_duplicate(tokens, representative_tokens, config.near_duplicate_threshold)
        source_key = _source_key(item)
        duplicate_source = source_key in seen_sources
        independence = 0.35 if duplicate_document or duplicate_content else (0.5 if duplicate_source else 1.0)
        seen_documents.add(document_key)
        seen_sources.add(source_key)
        if not duplicate_document and not duplicate_content:
            representative_tokens.append(tokens)
            independent_sources.add(source_key)
        item.scores.final = max(0.0, min(1.0, score))
        weight = item.scores.final * item.judge_confidence * independence
        if item.scores.directness >= config.min_directness_score and item.relation in {EvidenceRelation.SUPPORT, EvidenceRelation.REFUTE}:
            direct += 1
        if item.scores.authority >= config.authority_threshold:
            authoritative += 1
        covered.update(item.covered_slots)
        if item.relation == EvidenceRelation.SUPPORT:
            support += weight
        elif item.relation == EvidenceRelation.REFUTE:
            refute += weight
        elif item.relation == EvidenceRelation.SUPPLEMENT:
            supplement += weight
    coverage = len(set(required_slots) & covered) / max(1, len(required_slots)) if required_slots else (1.0 if valid else 0.0)
    conflict = min(support, refute)
    final = max(support, refute) / max(1.0, max(support, refute) + supplement)
    return EvidenceQuality(
        effective_evidence_count=len(valid), direct_evidence_count=direct,
        authoritative_evidence_count=authoritative,
        independent_source_count=len(independent_sources),
        independent_document_count=len(representative_tokens),
        claim_coverage_score=coverage, support_weight=support, refute_weight=refute,
        supplement_weight=supplement, conflict_score=conflict, final_evidence_score=min(1, final),
        noise_ratio=max(0, candidate_count - len(valid)) / max(1, candidate_count),
        missing_slots=[slot for slot in required_slots if slot not in covered],
        only_snippets=bool(valid) and all(x.snippet_only for x in valid),
    )
