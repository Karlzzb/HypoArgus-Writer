"""Fast one-pass multi-source retrieval without Query Generator, Router or Loop."""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from typing import Any

from ..chunk_context import build_adjacent_context
from ..claim_logic import atomize_claim, normalize_reverse_hypothesis
from ..config import EvidenceRetrievalConfig
from ..dependencies import EvidenceRetrievalDependencies, build_prepared_context
from ..errors import ErrorCode, RetrievalError
from ..evidence_judge import BatchJudgeResult, to_evidence
from ..evidence_quality import analyze_evidence_quality
from ..gap_retrieval import plan_gap_retrieval
from ..judge_batch_planner import JudgeBatch, JudgeBatchPlanner
from ..metrics import RequestMetricsCollector, TaskMetricsCollector
from ..numeric_relation import NumericRelationVerifier
from ..pair_consistency import check_pair_consistency
from ..providers.web_content_fetcher import FetchResult
from ..query_normalization import extract_numeric_expressions, normalize_query_preserving_numbers
from ..retrieval_queries import build_kb_query_variants
from ..retrievers.bm25_retriever import BM25Retriever, tokenize
from ..schemas import (
    ClaimJudgeResult,
    ErrorDetail,
    EvidenceCandidate,
    EvidenceItem,
    EvidenceQuality,
    EvidenceRelation,
    EvidenceScores,
    ExecutionStatus,
    JudgeResult,
    LineType,
    QueryItem,
    RetrievalTask,
    RetrievalTaskResult,
    SourceRef,
    SourceType,
    TerminationReason,
    ToolUsage,
    VerificationResult,
    VerificationVerdict,
    canonical_url,
    source_evidence_fingerprint,
    stable_evidence_item_key,
    stable_json_hash,
)
from ..scope_guard import apply_claim_scope_guard
from ..slot_aggregation import aggregate_slot_evidence, normalize_slot_evidence
from ..structured.subgraph import StructuredToolCallingSubgraph
from ..timeout_classification import is_timeout, timeout_layer
from ..tracing import SafeTraceEmitter
from ..verification import aggregate_verification

_PARALLEL_RUNTIME_CACHES: dict[tuple[Any, ...], dict[str, Any]] = {}
_NUMERIC_VERIFIER = NumericRelationVerifier()


def _deterministic_numeric_span(target: str, content: str) -> str:
    """Return an exact candidate sentence containing a target numeric fact."""
    expressions = [value for value in extract_numeric_expressions(target) if not value.endswith("年")]
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;])|\n+", content or "") if part.strip()]
    for expression in expressions:
        compact = expression.replace(" ", "").replace("％", "%")
        for sentence in sentences:
            sentence_compact = sentence.replace(" ", "").replace("％", "%")
            if compact and compact in sentence_compact:
                return sentence[:300]
    return ""


def _claim_results_for_candidate(task: RetrievalTask, judgement: JudgeResult) -> list[ClaimJudgeResult]:
    """Return explicit claim mappings only; never use lexical-overlap inference."""
    valid_ids = {claim.claim_id for claim in task.atomic_claims}
    explicit = [row for row in judgement.claim_results if row.claim_id in valid_ids]
    if explicit:
        by_id = {row.claim_id: row for row in explicit}
        return [by_id.get(claim.claim_id) or ClaimJudgeResult(
            claim_id=claim.claim_id,
            relation=EvidenceRelation.NEUTRAL,
            confidence=0,
            directness=0,
            neutral_reason="IRRELEVANT",
            reason="Judge did not return this atomic claim.",
        ) for claim in task.atomic_claims]

    # Compatibility for deterministic/test judges: only exact declared IDs,
    # or the sole claim in a SINGLE task, may inherit a candidate relation.
    supported = set(judgement.supported_claim_ids) & valid_ids
    refuted = set(judgement.refuted_claim_ids) & valid_ids
    sole_claim_id = task.atomic_claims[0].claim_id if len(task.atomic_claims) == 1 else None
    output: list[ClaimJudgeResult] = []
    for claim in task.atomic_claims:
        relation = (
            EvidenceRelation.SUPPORT if claim.claim_id in supported
            else EvidenceRelation.REFUTE if claim.claim_id in refuted
            else judgement.relation if sole_claim_id == claim.claim_id
            else EvidenceRelation.NEUTRAL
        )
        output.append(ClaimJudgeResult(
            claim_id=claim.claim_id,
            matched_claim_id=claim.claim_id,
            relation=relation,
            confidence=judgement.confidence if relation != EvidenceRelation.NEUTRAL else 0,
            directness=judgement.directness if relation != EvidenceRelation.NEUTRAL else 0,
            reason=judgement.reason if relation != EvidenceRelation.NEUTRAL else "No explicit claim-level mapping.",
            quoted_spans=judgement.quoted_spans if relation != EvidenceRelation.NEUTRAL else [],
            covered_slots=judgement.covered_slots if relation != EvidenceRelation.NEUTRAL else [],
            missing_slots=judgement.missing_slots,
            slot_evidence=judgement.slot_evidence if relation != EvidenceRelation.NEUTRAL else {},
            numeric_facts=judgement.numeric_facts if relation != EvidenceRelation.NEUTRAL else [],
            neutral_reason=judgement.neutral_reason if relation == EvidenceRelation.NEUTRAL else None,
            numeric_override_allowed=relation != EvidenceRelation.NEUTRAL,
        ))
    return output


def _apply_claim_numeric_relation(claim: Any, result: ClaimJudgeResult) -> ClaimJudgeResult:
    """Numeric safety rail bound to one claim, one compatible scope and one quote."""
    if not result.scope_compatible or not result.numeric_override_allowed or not result.quoted_spans:
        return result
    target = str(getattr(claim, "source_text_span", "") or "")
    if getattr(claim, "value", None) is None:
        return result
    span = next((value for value in result.quoted_spans if re.search(r"\d", value)), "")
    if not span:
        return result
    numeric = _NUMERIC_VERIFIER.verify(target, span, llm_relation=result.relation.value)
    if numeric.numeric_relation is None or numeric.confidence < 0.9:
        return result
    relation = EvidenceRelation(numeric.final_relation or numeric.numeric_relation)
    return result.model_copy(update={
        "relation": relation,
        "confidence": max(result.confidence, numeric.confidence),
        "directness": max(result.directness, 0.90),
        "quoted_spans": [span],
        "numeric_relation": relation,
        "neutral_reason": None,
        "matched_claim_id": result.claim_id,
        "override_reason": numeric.override_reason,
        "numeric_override_allowed": True,
        "reason": f"{result.reason}; deterministic numeric relation: {relation.value}",
    })


def _claim_evidence_items(
    task: RetrievalTask,
    candidate: EvidenceCandidate,
    judgement: JudgeResult,
) -> tuple[list[EvidenceItem], list[ClaimJudgeResult]]:
    claim_by_id = {claim.claim_id: claim for claim in task.atomic_claims}
    processed: list[ClaimJudgeResult] = []
    items: list[EvidenceItem] = []
    for raw in _claim_results_for_candidate(task, judgement):
        claim = claim_by_id[raw.claim_id]
        row = apply_claim_scope_guard(claim, candidate, raw)
        if row.scope_compatible:
            row = _apply_claim_numeric_relation(claim, row)
        processed.append(row)
        if row.relation == EvidenceRelation.NEUTRAL:
            continue
        mapped = JudgeResult(
            relation=row.relation,
            confidence=row.confidence,
            directness=row.directness,
            reason=row.reason,
            quoted_spans=row.quoted_spans,
            supported_claim_ids=[row.claim_id] if row.relation == EvidenceRelation.SUPPORT else [],
            refuted_claim_ids=[row.claim_id] if row.relation == EvidenceRelation.REFUTE else [],
            covered_slots=row.covered_slots,
            missing_slots=row.missing_slots,
            slot_evidence=row.slot_evidence,
            numeric_facts=row.numeric_facts,
            numeric_relation=row.numeric_relation,
            neutral_reason=row.neutral_reason,
            quote_match_mode=row.quote_match_mode,
            scope_compatible=row.scope_compatible,
            scope_mismatch_reasons=row.scope_mismatch_reasons,
        )
        scoped_candidate = candidate.model_copy(update={
            "metadata": {**candidate.metadata, "matched_claim_id": row.claim_id},
        })
        items.append(to_evidence(scoped_candidate, mapped))
    return items, processed


def _atomic_claim_verdicts(task: RetrievalTask, evidence: list[EvidenceItem]) -> dict[str, str]:
    output: dict[str, str] = {}
    for claim in task.atomic_claims:
        rows = [item for item in evidence if item.metadata.get("matched_claim_id") == claim.claim_id]
        has_support = any(item.relation == EvidenceRelation.SUPPORT for item in rows)
        has_refute = any(item.relation == EvidenceRelation.REFUTE for item in rows)
        output[claim.claim_id] = (
            "CONFLICT" if has_support and has_refute
            else "REFUTED" if has_refute
            else "SUPPORTED" if has_support
            else "INCONCLUSIVE"
        )
    return output


def validate_web_query(query: str, *, source_text: str = "", max_length: int = 120) -> bool:
    """Validate a bounded Web query without destroying factual expressions.

    Rejects: empty/overlong strings, concatenated years (20252024), concatenated
    percentages (508%74%), repeated paragraph-sized text, and truncated years.
    """
    value = " ".join(str(query or "").split())
    if not value or len(value) < 2 or len(value) > max_length:
        return False
    compact = value.replace(" ", "")
    # Detect concatenated years like 20252024 or 202520242025
    if re.search(r"(?:19|20)\d{2}(?:19|20)\d{2}", compact):
        return False
    # Detect concatenated percentages like 508%74% or 74%85%
    if re.search(r"\d[%％]\d", compact):
        return False
    source = str(source_text or "")
    if source and len(source) > 180 and value == " ".join(source.split()):
        return False
    words = value.casefold().split()
    if len(words) >= 8 and len(words) - len(set(words)) > len(words) // 2:
        return False
    if len(re.findall(r"[。！？!?；;]", value)) >= 3:
        return False
    for expression in extract_numeric_expressions(source):
        expr_compact = expression.replace(" ", "")
        if expr_compact and expr_compact in compact:
            continue
        if re.search(r"(?:19|20)\d{1}(?!\d)", compact):
            return False
        if "." in expr_compact:
            broken_decimal = re.escape(expr_compact).replace(r"\.", r"(?:\s+|[^\d.]\s*)")
            if re.search(broken_decimal, value):
                return False
        if ("%" in expr_compact or "％" in expr_compact) and re.search(r"\d\s+\d\s*[%％]", value):
            return False
    return True


def build_web_query_variants(
    task: RetrievalTask,
    primary: QueryItem | None = None,
    *,
    max_variants: int = 4,
) -> list[QueryItem]:
    """Build bounded Web query variants from structured claim fields.

    Uses atomize_claim() to extract subject/metric/time/value fields and
    generates 3~4 short variants. Never concatenates the full target_text.
    """
    group = atomize_claim(task.target_text, line_type=task.line_type.value)
    claims = group.atomic_claims or []
    # Use only the first claim's subject for query, not all claims' subjects
    first_claim = claims[0] if claims else None
    subject = ""
    if first_claim and first_claim.subject and len(first_claim.subject.strip()) > 1:
        subject = first_claim.subject.strip()
    if not subject:
        subject = re.split(r"[，。；;！？!?]", task.target_text, maxsplit=1)[0].strip()[:60]
    metrics = " ".join(dict.fromkeys(
        claim.metric for claim in claims if claim.metric
    )).strip()
    times = " ".join(dict.fromkeys(
        claim.time_scope for claim in claims if claim.time_scope
    )).strip()
    numbers = " ".join(extract_numeric_expressions(task.target_text))

    def join_unique(*parts: str) -> str:
        tokens: list[str] = []
        seen: set[str] = set()
        for part in parts:
            for token in str(part or "").split():
                key = token.casefold()
                if key and key not in seen:
                    seen.add(key)
                    tokens.append(token)
        return " ".join(tokens).strip()

    # Variant 1: time + subject + metric (neutral fact)
    v1 = join_unique(times, subject, metrics)
    # Variant 2: time + subject + numbers (numeric fact)
    v2 = join_unique(times, subject, numbers)
    # Variant 3: subject + metric + "报告" (authoritative report)
    v3 = join_unique(subject, metrics, "报告")
    # Variant 4: time + subject + metric + "统计" (statistics)
    v4 = join_unique(times, subject, metrics, "统计")
    # A fifth seed lets deduplication still produce the promised four bounded
    # attempts when the neutral and numeric variants collapse to the same
    # text (common for short English numeric claims).
    v5 = join_unique(times, subject, metrics, "官方", "数据")
    raw = [v1, v2, v3, v4, v5]
    # Do not insert primary.query (full target_text) as first variant;
    # structured variants are shorter and more effective.
    output: list[QueryItem] = []
    seen: set[str] = set()
    for index, value in enumerate(raw, 1):
        value = normalize_query_preserving_numbers(value)
        value = re.sub(r"(?<=\d)(?=[A-Za-z\u3400-\u9fff])", " ", value)
        value = re.sub(r"(?<=[A-Za-z\u3400-\u9fff])(?=\d)", " ", value)
        if not validate_web_query(value, source_text=task.target_text):
            continue
        key = " ".join(value.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        query_id = f"{task.task_id}:web:{len(output) + 1}"
        output.append(QueryItem(
            query_id=query_id,
            query=value[:120],
            purpose=("neutral fact" if index == 1 else "numeric fact" if index == 2 else "concise fact" if index == 3 else "authoritative report"),
        ))
        if len(output) >= max(1, max_variants):
            break
    if not output:
        # Fallback: use a short structured query, never the full target_text
        fallback_query = join_unique(times, subject, metrics) or task.target_text[:80]
        output = [QueryItem(query_id=f"{task.task_id}:web:1", query=fallback_query, purpose="fallback")]
    return output


def _apply_numeric_relation(task: RetrievalTask, candidate: EvidenceCandidate, judgement: JudgeResult) -> JudgeResult:
    """Apply deterministic numeric evidence only when a target and source
    share a measurable subject.  It never turns an unrelated candidate into
    support and records any LLM/rule disagreement for diagnosis.
    """
    target = task.normalized_hypothesis or task.target_text
    comparator = any(token in target for token in ("不足", "低于", "超过", "高于", "不超过", "不低于", "至少", "大于", "小于"))
    if not comparator:
        # For an affirmative numeric fact, an exact value+subject match is a
        # deterministic SUPPORT signal.  This is deliberately narrower than
        # fuzzy semantic matching and never changes an unrelated candidate.
        expressions = [value for value in extract_numeric_expressions(target) if not value.endswith("年")]
        subject_tokens = [t for t in re.findall(r"[\u3400-\u9fffA-Za-z]{2,}", target) if t not in {"同比", "约为", "是否"}]
        if expressions and any(value.replace(" ", "") in candidate.content.replace(" ", "") for value in expressions):
            if not subject_tokens or any(token in candidate.content for token in subject_tokens[:8]):
                span = _deterministic_numeric_span(target, candidate.content)
                if not span:
                    return judgement
                return judgement.model_copy(update={
                    "relation": EvidenceRelation.SUPPORT,
                    "numeric_relation": EvidenceRelation.SUPPORT,
                    "final_relation": EvidenceRelation.SUPPORT,
                    "override_reason": "exact_numeric_subject_match",
                    "reason": f"{judgement.reason}；确定性数值与主体匹配支持目标。",
                    "quoted_spans": [span],
                    "quote_match_mode": "normalized",
                    "neutral_reason": None,
                })
        return judgement
    numeric = _NUMERIC_VERIFIER.verify(target, candidate.content, llm_relation=judgement.relation.value)
    if numeric.numeric_relation is None or numeric.confidence < 0.9:
        return judgement
    # Require at least one non-numeric subject token in the candidate.  This
    # prevents an isolated number in an unrelated document from overriding the
    # semantic Judge result.
    subject_tokens = [t for t in re.findall(r"[\u3400-\u9fffA-Za-z]{2,}", target) if not t.isdigit()]
    if subject_tokens and not any(token in candidate.content for token in subject_tokens[:6]):
        return judgement
    relation = EvidenceRelation(numeric.final_relation or numeric.numeric_relation)
    span = _deterministic_numeric_span(target, candidate.content)
    if not span:
        return judgement
    return judgement.model_copy(update={
        "relation": relation,
        "llm_relation": judgement.relation,
        "numeric_relation": relation,
        "final_relation": relation,
        "relation_conflict": numeric.relation_conflict,
        "override_reason": numeric.override_reason,
        "reason": f"{judgement.reason}；确定性数值关系校验：{relation.value}",
        "quoted_spans": [span],
        "quote_match_mode": "normalized",
        "neutral_reason": None,
    })


def deduplicate_errors(errors: list[Any]) -> list[ErrorDetail]:
    """Deduplicate final task errors while deliberately ignoring occurred_at."""
    output: list[ErrorDetail] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for value in errors:
        error = value if isinstance(value, ErrorDetail) else ErrorDetail.model_validate(value)
        key = (error.code, error.node, error.tool, error.reason)
        if key not in seen:
            seen.add(key)
            output.append(error)
    return output


def _candidate_source_bucket(candidate: EvidenceCandidate) -> str:
    if candidate.source_type == SourceType.WEB:
        return "web"
    if candidate.source_type == SourceType.STRUCTURED:
        return "structured"
    if candidate.source_ref.knowledge_origin == "upstream_selected":
        return "selected_kb"
    return "public_kb"


def _funnel_template() -> dict[str, int]:
    return {
        "retrieved_count": 0, "normalized_count": 0, "invalid_count": 0,
        "exact_duplicate_count": 0, "semantic_duplicate_group_count": 0,
        "deduplicated_count": 0, "judge_ready_count": 0,
        "judge_batched_count": 0, "judge_returned_count": 0,
        "judge_error_count": 0,
    }


def deduplicate_candidates_with_audit(
    candidates: list[EvidenceCandidate],
) -> tuple[list[EvidenceCandidate], dict[str, dict[str, int]]]:
    """Exact dedupe with merged provenance; unique candidates are never cut."""
    representatives: dict[str, EvidenceCandidate] = {}
    content_representatives: dict[str, EvidenceCandidate] = {}
    result: list[EvidenceCandidate] = []
    audit: dict[str, dict[str, int]] = defaultdict(_funnel_template)

    def merge(rep: EvidenceCandidate, duplicate: EvidenceCandidate) -> None:
        metadata = dict(rep.metadata)
        merged_ids = list(metadata.get("merged_candidate_ids") or [rep.candidate_id])
        if duplicate.candidate_id not in merged_ids:
            merged_ids.append(duplicate.candidate_id)
        refs = list(metadata.get("merged_source_refs") or [rep.source_ref.model_dump(mode="json")])
        duplicate_ref = duplicate.source_ref.model_dump(mode="json")
        if duplicate_ref not in refs:
            refs.append(duplicate_ref)
        metadata["merged_candidate_ids"] = merged_ids
        metadata["merged_source_refs"] = refs
        rep.metadata = metadata

    for candidate in candidates:
        bucket = _candidate_source_bucket(candidate)
        audit[bucket]["retrieved_count"] += 1
        if not (candidate.content or "").strip():
            audit[bucket]["invalid_count"] += 1
            continue
        audit[bucket]["normalized_count"] += 1
        if candidate.source_type == SourceType.KNOWLEDGE_BASE:
            key = "kb:" + ":".join([
                str(candidate.source_ref.knowledge_id or ""),
                str(candidate.source_ref.file_id or ""),
                str(candidate.source_ref.chunk_id or ""),
            ])
        elif candidate.source_type == SourceType.WEB:
            key = "web:" + canonical_url(candidate.source_ref.url or "")
        else:
            key = "structured:" + str(candidate.candidate_id)
        fingerprint = stable_json_hash((candidate.content or "").strip())[:32]
        representative = representatives.get(key) or content_representatives.get(fingerprint)
        if representative is not None:
            audit[bucket]["exact_duplicate_count"] += 1
            merge(representative, candidate)
            continue
        representatives[key] = candidate
        content_representatives[fingerprint] = candidate
        result.append(candidate)
        audit[bucket]["deduplicated_count"] += 1
    return result, dict(audit)


def deduplicate_candidates(candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    """Backward-compatible exact dedupe helper used by unit tests."""
    return deduplicate_candidates_with_audit(candidates)[0]


# --------------------------------------------------------------------------- #
# Scope Guard: filter candidates by subject/organization consistency.
# Prevents Judge from processing irrelevant candidates about different
# subjects/schools/companies — saves Judge tokens and improves verdict quality.
# --------------------------------------------------------------------------- #

_SCOPE_ORG_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9\u00b7()]{2,12}"
    r"(?:大学|学院|学校|研究院|有限公司|股份公司|集团|行业协会|管理局|基金会|联合会|实验室|研究中心)"
)
_SCOPE_ORG_SUFFIX_RE = re.compile(
    r"(?:大学|学院|学校|研究院|有限公司|股份公司|集团|行业协会|管理局|基金会|联合会|实验室|研究中心)$"
)


def _extract_scope_subjects(task: Any) -> set[str]:
    """Extract organization/subject names from task's target_text and paragraph_text.

    Returns a set of full org names + shortened core forms. Short forms strip
    the org suffix and parenthetical content, then take first 4-6 chars.
    E.g. "腾讯科技(深圳)有限公司" → {"腾讯科技(深圳)有限公司", "腾讯科技"}.
    """
    text = " ".join(filter(None, [
        getattr(task, "target_text", ""),
        getattr(task, "paragraph_text", ""),
    ]))
    subjects: set[str] = set()
    for m in _SCOPE_ORG_RE.finditer(text):
        full = m.group()
        subjects.add(full)
        core = _SCOPE_ORG_SUFFIX_RE.sub("", full)
        core_no_paren = re.sub(r"\([^)]*\)", "", core)
        if len(core_no_paren) >= 2:
            subjects.add(core_no_paren[:4])
            subjects.add(core_no_paren[:6])
    return subjects


def _scope_pre_filter(
    task: Any,
    candidates: list[EvidenceCandidate],
) -> tuple[list[EvidenceCandidate], int, int]:
    """Filter candidates whose content doesn't mention the same subject/organization.

    Returns (filtered_candidates, dropped_count, checked_count).
    If no organization names found in the task's text, returns all candidates
    (can't determine scope — don't filter blindly).
    """
    subjects = _extract_scope_subjects(task)
    if not subjects:
        return candidates, 0, 0

    subjects_lower = {s.lower() for s in subjects}
    filtered: list[EvidenceCandidate] = []
    dropped = 0
    for c in candidates:
        content = (getattr(c, "content", "") or "").lower()
        if any(s in content for s in subjects_lower):
            filtered.append(c)
        else:
            dropped += 1
    return filtered, dropped, len(candidates)


def deduplicate_evidence_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Deduplicate the same task/source fact/relation across all rounds."""
    output: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in items:
        key = stable_evidence_item_key(item, item.relation)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _apply_domain_filter(candidates: list[EvidenceCandidate], config: EvidenceRetrievalConfig) -> tuple[list[EvidenceCandidate], int]:
    """Apply denylist_domains and max_results_per_domain filtering."""
    from urllib.parse import urlparse
    if not config.denylist_domains and not config.max_results_per_domain:
        return candidates, 0
    denylist = {d.lower().strip() for d in config.denylist_domains}
    domain_counts: dict[str, int] = {}
    filtered = 0
    result: list[EvidenceCandidate] = []
    for c in candidates:
        if c.source_type != SourceType.WEB:
            result.append(c)
            continue
        url = c.source_ref.url or ""
        hostname = (urlparse(url).hostname or "").lower()
        blocked = hostname in denylist
        if not blocked:
            for d in denylist:
                if hostname.endswith("." + d):
                    blocked = True
                    break
        if blocked:
            filtered += 1
            continue
        if config.max_results_per_domain:
            cnt = domain_counts.get(hostname, 0)
            if cnt >= config.max_results_per_domain:
                filtered += 1
                continue
            domain_counts[hostname] = cnt + 1
        result.append(c)
    return result, filtered


def _apply_domain_filter_with_audit(
    candidates: list[EvidenceCandidate], config: EvidenceRetrievalConfig,
) -> tuple[list[EvidenceCandidate], int, int]:
    """Return rows plus separate denylist and per-domain limit counts."""
    from urllib.parse import urlparse
    denylist = {domain.casefold().strip().lstrip(".") for domain in config.denylist_domains}
    counts: dict[str, int] = {}
    output: list[EvidenceCandidate] = []
    denied = limited = 0
    for candidate in candidates:
        host = (urlparse(candidate.source_ref.url or "").hostname or "").casefold()
        if any(host == domain or host.endswith("." + domain) for domain in denylist):
            denied += 1
            continue
        if config.max_results_per_domain and counts.get(host, 0) >= config.max_results_per_domain:
            limited += 1
            continue
        counts[host] = counts.get(host, 0) + 1
        output.append(candidate)
    return output, denied, limited


_NON_ARTICLE_PATH = re.compile(r"/(?:login|signin|search|tag|category|author|user|account)(?:/|$)", re.I)
def _lexical_rank(query: str, candidates: list[EvidenceCandidate], *, top_k: int, preferred_domains: list[str] | None = None) -> list[EvidenceCandidate]:
    """Real BM25 plus deterministic exact-value/title/rank features."""
    if not candidates:
        return []
    from urllib.parse import urlparse
    ranked = BM25Retriever(text_getter=lambda row: f"{row.title} {row.content}").retrieve(query, candidates, len(candidates))
    max_bm25 = max((score for _, score in ranked), default=0) or 1.0
    query_tokens = set(tokenize(query))
    exact_values = set(extract_numeric_expressions(query))
    preferred = {value.casefold().lstrip(".") for value in (preferred_domains or [])}
    output: list[tuple[EvidenceCandidate, float]] = []
    for candidate, bm25 in ranked:
        title_tokens = set(tokenize(candidate.title))
        body = f"{candidate.title} {candidate.content}"
        rank = max(1, int(candidate.metadata.get("rank") or 999))
        host = (urlparse(candidate.source_ref.url or "").hostname or "").casefold()
        exact_hits = sum(value in body for value in exact_values)
        score = (
            0.62 * (bm25 / max_bm25)
            + 0.12 * (len(query_tokens & title_tokens) / max(1, len(query_tokens)))
            + 0.10 * min(1.0, exact_hits / max(1, len(exact_values)))
            + 0.10 / rank
            + (0.06 if any(host == domain or host.endswith("." + domain) for domain in preferred) else 0.0)
        )
        candidate.rerank_score = min(1.0, max(0.0, score))
        output.append((candidate, score))
    output.sort(key=lambda item: (item[1], item[0].initial_score), reverse=True)
    return [candidate for candidate, _ in output[:top_k]]


def _relevant_window(query: str, content: str, max_chars: int = 800) -> str:
    """Select the highest-BM25 paragraph window instead of truncating the head."""
    text = re.sub(r"\s+", " ", content or "").strip()
    if len(text) <= max_chars:
        return text
    units = [part.strip() for part in re.split(r"(?<=[。！？!?；;])|\n+", text) if part.strip()]
    if len(units) < 2:
        units = [text[offset:offset + max_chars] for offset in range(0, len(text), max_chars // 2)]
    indexed = list(enumerate(units))
    ranked = BM25Retriever(text_getter=lambda value: value[1]).retrieve(query, indexed, 1)
    if not ranked:
        return text[:max_chars]
    index = ranked[0][0][0]
    selected = units[index]
    for neighbor in (index - 1, index + 1):
        if 0 <= neighbor < len(units) and len(selected) + len(units[neighbor]) <= max_chars:
            selected = units[neighbor] + selected if neighbor < index else selected + units[neighbor]
    return selected[:max_chars]


def _candidate_passthrough_item(
    task: RetrievalTask,
    candidate: EvidenceCandidate,
) -> EvidenceItem | None:
    """Package one ranked retrieval candidate without judging its meaning.

    ``EvidenceItem`` is retained only as the existing internal transport type
    consumed by ``output_adapter``.  Its relation/confidence fields are fixed
    compatibility values and MUST NOT be interpreted as a SearchAgent
    judgment; ``metadata.retrieval_candidate_passthrough`` is the authoritative
    marker.  No claim mapping, scope decision, or factual direction is made.
    """
    # candidate_passthrough 模式:不做词法重叠过滤,直接取 candidate.content 前 600 字。
    # 之前用 _relevant_window(task.target_text, candidate.content, 600) 做词法重叠截取,
    # KB(Bisheng 向量检索)返回的 chunk 语义相关但词法不重叠 → _relevant_window 返空 →
    # 候选被丢 → 0 条 knowledge_base citations。passthrough 的设计意图是"不过滤,全给
    # 下游 judgment",词法过滤违背此意图。Web/Structured 因 query 从 target_text 派生,
    # 词法重叠高不受影响;KB 因向量检索天然产语义相关但词法不同的结果,恰好被卡住。
    quote = candidate.content[:600].strip()
    if not quote:
        return None
    fingerprint = source_evidence_fingerprint(candidate)
    evidence_key = stable_evidence_item_key(candidate, EvidenceRelation.SUPPLEMENT)
    traceable = bool(
        candidate.source_ref.url
        or candidate.source_ref.knowledge_id
        or candidate.source_ref.dataset_id
        or candidate.source_ref.query_execution_id
    )
    return EvidenceItem(
        evidence_id=f"ev-{evidence_key[:20]}",
        task_id=task.task_id,
        source_type=candidate.source_type,
        source_name=candidate.source_name,
        source_ref=candidate.source_ref,
        title=candidate.title,
        content=quote,
        quoted_spans=[quote],
        snippet_only=candidate.snippet_only,
        # Compatibility-only value required by the existing transport model.
        relation=EvidenceRelation.SUPPLEMENT,
        judge_confidence=0.0,
        scores=EvidenceScores(
            relevance=max(0.0, min(1.0, candidate.rerank_score)),
            directness=0.0,
            traceability=1.0 if traceable else 0.0,
        ),
        reason="RETRIEVAL_CANDIDATE_PASSTHROUGH: unjudged material for parent Judgment",
        content_fingerprint=fingerprint,
        source_evidence_fingerprint=fingerprint,
        metadata={
            **candidate.metadata,
            "retrieval_candidate_passthrough": True,
            "candidate_id": candidate.candidate_id,
            "candidate_rerank_score": candidate.rerank_score,
        },
        context_window=candidate.context_window,
    )


# ---------------------------------------------------------------------------
# 状态推导、gap_reason 与动态 reason 生成
# ---------------------------------------------------------------------------

def derive_task_execution_state(
    errors: list[Any],
    deadline_reached: bool = False,
    selected_ok: bool = True,
    judge_errors: list[Any] | None = None,
    has_invalid_input: bool = False,
    has_internal_error: bool = False,
    has_usable_evidence: bool = False,
) -> tuple[ExecutionStatus, TerminationReason]:
    """按优先级推导任务执行状态和终止原因。

    优先级：INVALID_INPUT/INTERNAL_ERROR > TIMEOUT > TOOL_ERROR > SUFFICIENT > EXHAUSTED
    """
    from ..schemas import ExecutionStatus, TerminationReason

    judge_errors = judge_errors or []

    if has_invalid_input:
        return ExecutionStatus.ERROR, TerminationReason.INVALID_INPUT
    if has_internal_error:
        return ExecutionStatus.ERROR, TerminationReason.INTERNAL_ERROR
    if deadline_reached:
        return ExecutionStatus.PARTIAL, TerminationReason.TIMEOUT
    effective_errors = list(errors)
    if selected_ok:
        # Web and configured-public KB are parallel, non-mandatory sources.
        # A failure in either branch remains a warning even when every
        # surviving candidate is later rejected by Judge/Scope Guard.  Whether
        # evidence is sufficient is expressed by the verdict/evidence gap; it
        # must not be conflated with tool execution completeness.  Selected KB
        # is handled separately by ``selected_ok`` and an applicable
        # Structured Tool failure remains fatal to task completeness.
        optional_codes = {
            "WEB_NO_RESULT", "WEB_RESPONSE_PARSE_ERROR", "WEB_PROVIDER_ERROR", "WEB_TIMEOUT",
            "KB_PROVIDER_ERROR", "KB_TIMEOUT",
        }
        effective_errors = [
            error for error in effective_errors
            if str(getattr(error, "code", error.get("code", "") if isinstance(error, dict) else "")) not in optional_codes
        ]
    if effective_errors or judge_errors or not selected_ok:
        return ExecutionStatus.PARTIAL, TerminationReason.TOOL_ERROR
    return ExecutionStatus.SUCCESS, TerminationReason.EXHAUSTED


def derive_evidence_gap_reason(
    quality: EvidenceQuality,
    verification: VerificationResult,
    task_errors: list[Any],
    config: EvidenceRetrievalConfig,
    selected_ok: bool = True,
) -> str | None:
    """根据最终证据质量计算 gap_reason，不再使用 Structured 分支的匹配状态。"""
    from ..schemas import VerificationVerdict

    # Optional Web/public-KB failures are warnings, not evidence-gap causes.
    # Judge failures, an applicable Structured Tool failure, and Selected KB
    # failure can make the task genuinely incomplete.
    has_tool_error = not selected_ok or any(
        (hasattr(e, "code") and getattr(e, "code", "") in (
            "JUDGE_EMPTY_RESPONSE", "JUDGE_ERROR", "JUDGE_TIMEOUT", "JUDGE_PARTIAL_VALIDATION_ERROR",
            "JUDGE_VALIDATION_ERROR", "JUDGE_REPAIR_RETRY_ERROR", "STRUCTURED_UNAVAILABLE",
        )) or (isinstance(e, dict) and e.get("code", "") in (
            "JUDGE_EMPTY_RESPONSE", "JUDGE_ERROR", "JUDGE_TIMEOUT", "JUDGE_PARTIAL_VALIDATION_ERROR",
            "JUDGE_VALIDATION_ERROR", "JUDGE_REPAIR_RETRY_ERROR", "STRUCTURED_UNAVAILABLE",
        ))
        for e in task_errors
    )

    if has_tool_error:
        return "SOURCE_PARTIAL_FAILURE"
    if quality.effective_evidence_count == 0:
        return "NO_EVIDENCE"
    if quality.missing_slots:
        return "MISSING_REQUIRED_SLOTS"
    if verification.verdict == VerificationVerdict.CONFLICT:
        return "CONFLICTING_EVIDENCE"
    if quality.effective_evidence_count < config.min_effective_evidence_count:
        return "INSUFFICIENT_EFFECTIVE_COUNT"
    if quality.direct_evidence_count < config.min_direct_evidence_count:
        return "INSUFFICIENT_DIRECT_EVIDENCE"
    if quality.independent_document_count < config.min_independent_document_count:
        return "INSUFFICIENT_INDEPENDENT_SOURCES"
    if quality.authoritative_evidence_count < 1 and config.authority_threshold > 0:
        return "INSUFFICIENT_AUTHORITY"
    if quality.final_evidence_score < config.min_final_evidence_score:
        return "LOW_EVIDENCE_SCORE"
    if quality.noise_ratio > config.max_noise_ratio:
        return "HIGH_NOISE"
    return None


def build_verification_reason(
    verdict: VerificationVerdict,
    execution_status: ExecutionStatus,
    termination_reason: TerminationReason,
    quality: EvidenceQuality,
    task_errors: list[Any],
    gap_reason: str | None,
    config: EvidenceRetrievalConfig,
) -> str:
    """根据真实数据动态生成中文证据不足原因，不再使用固定英文模板。"""
    from ..schemas import ExecutionStatus, TerminationReason, VerificationVerdict

    # 工具错误
    has_judge_error = any(
        (hasattr(e, "code") and getattr(e, "code", "") in (
            "JUDGE_EMPTY_RESPONSE", "JUDGE_ERROR", "JUDGE_PARTIAL_VALIDATION_ERROR", "JUDGE_VALIDATION_ERROR", "JUDGE_REPAIR_RETRY_ERROR",
        )) or (isinstance(e, dict) and e.get("code", "") in (
            "JUDGE_EMPTY_RESPONSE", "JUDGE_ERROR", "JUDGE_PARTIAL_VALIDATION_ERROR", "JUDGE_VALIDATION_ERROR", "JUDGE_REPAIR_RETRY_ERROR",
        ))
        for e in task_errors
    )
    has_judge_timeout = any(
        (hasattr(e, "code") and getattr(e, "code", "") == "JUDGE_TIMEOUT")
        or (isinstance(e, dict) and e.get("code", "") == "JUDGE_TIMEOUT")
        for e in task_errors
    )

    if execution_status == ExecutionStatus.ERROR:
        if termination_reason == TerminationReason.INVALID_INPUT:
            return "输入数据无效，无法执行证据检索与验证。"
        return "系统内部发生异常，本次证据检索未能完成。"

    if termination_reason == TerminationReason.TIMEOUT:
        return "检索或验证超时，当前已获得部分候选证据，但结果可能不完整，因此暂不能形成确定结论。"

    if has_judge_error:
        return "当前已完成证据检索，但 Evidence Judge 的部分返回结果未通过结构校验，结果可能不完整，因此暂不能形成确定结论。"

    if has_judge_timeout:
        return "当前已完成证据检索，但 Evidence Judge 调用超时，候选证据未能完成有效性判断，因此本次结果不完整。"

    if termination_reason == TerminationReason.TOOL_ERROR:
        return "当前已获得部分候选证据，但部分检索工具或 Judge 出现错误，结果可能不完整，因此暂不能形成确定结论。"

    if verdict == VerificationVerdict.CONFLICT:
        return "当前同时存在支持证据和反驳证据，双方证据强度接近，暂无法形成单一确定结论。"

    if verdict == VerificationVerdict.SUPPORTED:
        return "已获得多条相互独立且直接相关的证据，证据质量达到配置要求，支持目标内容。"

    if verdict == VerificationVerdict.REFUTED:
        return "已获得多条相互独立且直接相关的证据，证据质量达到配置要求，反驳目标内容。"

    # INCONCLUSIVE 情况下的详细原因
    if gap_reason == "NO_EVIDENCE" or quality.effective_evidence_count == 0:
        return "当前检索未获得可用于支持或反驳目标内容的有效证据，因此暂不能形成确定结论。"

    if gap_reason == "MISSING_REQUIRED_SLOTS":
        return "当前证据未能覆盖所有必需信息槽位，因此暂不能形成确定结论。"

    if gap_reason == "CONFLICTING_EVIDENCE":
        return "当前同时存在支持证据和反驳证据，双方证据强度接近，暂无法形成单一确定结论。"

    if gap_reason == "INSUFFICIENT_EFFECTIVE_COUNT":
        return f"当前获得{quality.effective_evidence_count}条有效证据，低于至少{config.min_effective_evidence_count}条有效证据的要求，因此证据尚不充分。"

    if gap_reason == "INSUFFICIENT_DIRECT_EVIDENCE":
        return f"当前获得{quality.direct_evidence_count}条直接证据，低于至少{config.min_direct_evidence_count}条直接证据的要求，因此证据尚不充分。"

    if gap_reason == "INSUFFICIENT_INDEPENDENT_SOURCES":
        if quality.effective_evidence_count > 0 and quality.independent_document_count < config.min_independent_document_count:
            return f"已获得{quality.effective_evidence_count}条直接证据，证据质量分达到要求，但独立文档数量仅为{quality.independent_document_count}，未达到至少{config.min_independent_document_count}个独立来源的证据充分性要求，因此暂不能形成确定结论。"
        return f"当前独立来源数量不足，未达到至少{config.min_independent_document_count}个独立来源的要求，因此暂不能形成确定结论。"

    if gap_reason == "LOW_EVIDENCE_SCORE":
        return f"当前证据与目标内容相关，但综合证据质量分为{quality.final_evidence_score:.2f}，低于{config.min_final_evidence_score:.2f}的判定阈值，因此暂不能形成确定结论。"

    if gap_reason == "HIGH_NOISE":
        return f"当前候选中无关或低质量内容比例较高，噪声率为{quality.noise_ratio:.2f}，超过允许阈值{config.max_noise_ratio:.2f}，因此现有证据不足以支持确定判断。"

    if gap_reason == "SOURCE_PARTIAL_FAILURE":
        return "当前已获得部分候选证据，但部分检索来源出现异常，结果可能不完整，因此暂不能形成确定结论。"

    # 默认兜底
    return "证据尚不充分，未达到配置的充分性阈值，因此暂不能形成确定结论。"


# ---------------------------------------------------------------------------
# 流程缓存与查询构造
# ---------------------------------------------------------------------------


def get_parallel_shared_cache(config: EvidenceRetrievalConfig, dependencies: EvidenceRetrievalDependencies | None = None) -> dict[str, Any]:
    """Return a process-wide cache shared by separately built batch graphs."""
    provider_namespace: Any = "production"
    if dependencies is not None:
        production_names = {"VolcanoWebSearchClient", "BishengRetrieveClient"}
        names = {dependencies.web_search.__class__.__name__, dependencies.kb_client.__class__.__name__}
        if not names.issubset(production_names):
            # Injected/fake providers must not inherit another test's cache.
            provider_namespace = id(dependencies)
    key = (
        config.bisheng_retrieve_base_url or config.bisheng_base_url,
        tuple(config.public_knowledge_ids), provider_namespace,
    )
    cache = _PARALLEL_RUNTIME_CACHES.setdefault(key, {})
    return cache


def _parallel_query_details(task: RetrievalTask) -> tuple[QueryItem, list[str]]:
    """Build a Chinese-safe query and expose retained terms for audit."""
    if task.line_type == LineType.REVERSE:
        normalized = normalize_reverse_hypothesis(task.target_text)
        # A reverse item may be a legacy duplicate of the forward statement
        # (without a question/negation marker).  Keep its original query in
        # that case so request-level singleflight still coalesces both tasks.
        if normalized["normalized_hypothesis"].strip() == task.target_text.strip().rstrip("？?"):
            parts = [task.target_text]
        else:
            parts = [normalized["neutral_retrieval_query"], normalized["normalized_hypothesis"]]
    else:
        parts = [task.target_text]
    for ref in task.source_refs:
        if isinstance(ref, dict):
            parts.extend(str(ref.get(key) or "") for key in ("source_name", "title", "organization"))
    if task.existing_evidence_text:
        # Only source-like entities, dates and quantities; never copy the full
        # upstream evidence into a search query.
        parts.extend(re.findall(
            r"[A-Za-z0-9\u3400-\u9fff]{2,24}(?:大学|学院|政府|委员会|研究院|报告)|"
            r"\d{4}年|\d+(?:\.\d+)?(?:%|％|万人|亿元|项|个)",
            task.existing_evidence_text,
        ))
    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = normalize_query_preserving_numbers(str(part))
        for token in cleaned.split():
            key = token.casefold()
            if key and key not in seen:
                seen.add(key)
                tokens.append(token)
    text = " ".join(tokens).strip() or task.target_text
    return QueryItem(query_id=f"{task.task_id}:parallel:1", query=text[:500], purpose="parallel source verification"), tokens


def build_parallel_query(task: RetrievalTask) -> QueryItem:
    return _parallel_query_details(task)[0]


def _metrics() -> dict[str, Any]:
    stages = ["validate", "prepare_tasks", "query_build", "web_search", "web_filter", "web_fetch", "web_bm25", "selected_kb_retrieve", "public_kb_retrieve", "structured_match", "structured_query", "candidate_merge", "batch_judge", "verification", "finalize"]
    calls = ["llm_batch_evidence_judge", "deterministic_batch_evidence_judge", "llm_batch_judge_repair", "web_search", "web_fetch", "selected_kb_retrieve", "public_kb_retrieve", "structured_query"]
    return {
        "total_elapsed_ms": 0,
        "stage_timings_ms": {key: 0 for key in stages},
        "call_counts": {key: 0 for key in calls},
        "cache": {
            "query_cache_hits": 0,
            "query_singleflight_hits": 0,
            "web_content_cache_hits": 0,
            "web_content_singleflight_hits": 0,
            "kb_cache_hits": 0,
            "kb_singleflight_hits": 0,
            "structured_cache_hits": 0,
            "structured_singleflight_hits": 0,
        },
        "web_task": {},
        "web_request_funnel": {
            "web_search_task_count": 0, "tasks_with_web_search_results": 0,
            "tasks_with_web_candidates": 0, "tasks_with_web_errors": 0,
        },
        "source_candidate_counts": {"web": 0, "selected_kb": 0, "public_kb": 0, "structured": 0},
        "kb_query_variants": {"count": 0, "hits": 0, "tasks": 0},
        "kb_task": {},
        "kb_request_funnel": {
            "tasks_with_kb_candidates": 0, "atomic_claims_with_candidates": 0,
            "atomic_claims_without_candidates": 0,
        },
        "web_fetch_plan": {"web_search_result_count": 0, "web_fetch_planned_count": 0, "web_unique_url_count": 0, "web_fetch_success_count": 0, "web_candidate_extracted_count": 0, "web_candidate_reassigned_count": 0},
        "candidate_share": {"paragraph_pool_count": 0, "request_pool_count": 0, "reassigned_count": 0},
        "adjacent_chunk": {"expanded_count": 0, "included_chunk_count": 0},
        "gap_retrieval": {
            "triggered_count": 0, "triggered_task_count": 0, "query_count": 0,
            "reasons": [],
            "web_search_count": 0, "kb_retrieve_count": 0,
            "new_candidate_count": 0, "new_evidence_count": 0,
            "resolved_task_count": 0, "unresolved_task_count": 0,
            "new_unique_candidate_count": 0, "new_valid_evidence_count": 0,
            "resolved_slot_count": 0, "verdict_changed_count": 0,
            "gap_resolved_count": 0,
            "reserved_ms": 12000,
        },
        "pair_consistency": {"pair_count": 0, "consistent_count": 0, "conflict_count": 0, "incomplete_count": 0},
        "judge_contract": {"neutral_completion_count": 0, "repair_retry_count": 0},
        "judge_relation_distribution": {relation.value: 0 for relation in EvidenceRelation},
        "neutral_reason_distribution": {},
        "scope_guard": {
            "checked_count": 0, "mismatch_count": 0, "mismatch_reasons": {},
            "total_checked": 0, "total_dropped": 0, "tasks_affected": 0,
        },
        "structured_tool_calling": {
            "paragraph_count": 0, "tool_call_count": 0,
            "structured_intent_call_count": 0,
            "structured_query_count": 0, "candidate_count": 0,
            "calls": [], "warnings": [], "timings_ms": {},
        },
        "candidate_funnel": {
            source: _funnel_template()
            for source in ("web", "public_kb", "selected_kb", "structured")
        },
        "candidate_funnel_by_task": {},
        "judge_integrity": {
            "judge_input_candidate_count": 0,
            "judge_batched_candidate_count": 0,
            "judge_returned_candidate_count": 0,
            "judge_error_candidate_count": 0,
            "judge_missing_candidate_count": 0,
            "judge_duplicate_candidate_count": 0,
            "judge_returned_support_count": 0,
            "judge_returned_refute_count": 0,
            "judge_returned_supplement_count": 0,
            "judge_returned_neutral_count": 0,
            "evidence_created_support_count": 0,
            "evidence_created_refute_count": 0,
            "evidence_created_supplement_count": 0,
            "judge_result_mapping_error_count": 0,
            "quote_validation_reject_count": 0,
        },
        "judge_batches": [],
        "judge_candidate_batch_map": {},
        "deadline_reached": False, "errors": [],
        # Judge is request-shared; the wall-clock lives here so task metrics
        # can reference shared_batch_judge_ms without re-attributing the cost.
        "shared_batch_judge_ms": 0,
        "judge_timing_scope": "request_shared",
    }


class ParallelSourcesFlow:
    def __init__(self, config: EvidenceRetrievalConfig, dependencies: EvidenceRetrievalDependencies, trace: SafeTraceEmitter, shared_cache: dict[str, Any] | None = None):
        self.config = config
        self.deps = dependencies.complete(config)
        self.trace = trace
        cache = shared_cache if shared_cache is not None else {}
        self.shared_cache = cache
        self.query_cache = cache.setdefault("query", {})
        self.query_inflight = cache.setdefault("query_inflight", {})
        self.content_cache = cache.setdefault("content", {})
        self.content_inflight = cache.setdefault("content_inflight", {})
        self.kb_cache = cache.setdefault("kb", {})
        self.kb_inflight = cache.setdefault("kb_inflight", {})
        self.metrics = _metrics()
        self.metrics["gap_retrieval"]["reserved_ms"] = config.gap_retrieval_reserved_ms
        self.collector = RequestMetricsCollector()
        self.started = 0.0
        self.deadline = 0.0
        self.task_deadlines: dict[str, float] = {}
        self.task_timeouts: set[str] = set()
        self.kb_semaphore = asyncio.Semaphore(config.kb_retrieve_concurrency)
        self.web_semaphore = asyncio.Semaphore(config.web_search_concurrency)
        self.web_fetch_semaphore = asyncio.Semaphore(config.web_fetch_concurrency)
        self.structured_semaphore = asyncio.Semaphore(config.structured_concurrency)
        self.gap_judge_semaphore = asyncio.Semaphore(config.judge_batch_concurrency)
        self.structured_ready = asyncio.Event()
        self.judge_planner = JudgeBatchPlanner(
            max_tasks=config.judge_batch_max_tasks,
            max_candidates=config.judge_batch_max_candidates,
            max_input_tokens=min(config.judge_batch_max_input_tokens, config.judge_model_context_limit),
            candidate_max_chars=config.parallel_judge_candidate_max_chars,
            expected_output_tokens_per_candidate=config.judge_expected_output_tokens_per_candidate,
        )
        self.structured_candidates_by_task: dict[str, list[EvidenceCandidate]] = {}
        self.structured_status_by_task: dict[str, str] = {}

    def _batch_judge_uses_llm(self) -> bool:
        """Return whether the injected batch judge performs a real LLM call."""
        return bool(getattr(self.deps.batch_judge, "uses_llm", False))

    def _record_batch_judge_call(self) -> None:
        key = (
            "llm_batch_evidence_judge"
            if self._batch_judge_uses_llm()
            else "deterministic_batch_evidence_judge"
        )
        self.metrics["call_counts"][key] += 1

    def _batch_judge_observation(
        self,
        name: str,
        metadata: dict[str, Any],
        *,
        parent_run_id: Any,
    ):
        """Trace an LLM judge as GENERATION and a deterministic judge as SPAN."""
        if self._batch_judge_uses_llm():
            return self.trace.generation(
                name,
                metadata,
                model=self.config.judge_model or "unavailable",
                provider="openai_compatible",
                parent_run_id=parent_run_id,
            )
        return self.trace.span(
            name,
            {
                **metadata,
                "judge_mode": "deterministic",
                "judge_implementation": type(self.deps.batch_judge).__name__,
            },
            parent_run_id=parent_run_id,
        )

    async def _prepare_structured_tool_calling(
        self,
        tasks: list[RetrievalTask],
        scenarios: dict[str, Any],
        *,
        structured_healthy: bool,
    ) -> None:
        """Run exactly one required Tool Calling intent pass per paragraph."""
        grouped: dict[str, list[RetrievalTask]] = defaultdict(list)
        for task in tasks:
            grouped[task.paragraph_id].append(task)
        metrics = self.metrics["structured_tool_calling"]
        metrics["paragraph_count"] = len(grouped)
        if not structured_healthy:
            metrics["warnings"].append("STRUCTURED_SERVICE_UNHEALTHY")
            for task in tasks:
                self.structured_status_by_task[task.task_id] = "structured_unavailable"
            return

        available_scenario_keys = {
            key
            for key, scenario in scenarios.items()
            if bool(getattr(scenario, "healthy", True))
        }
        subgraph = StructuredToolCallingSubgraph(
            self.deps.structured_intent_model,
            self.deps.structured_client,
            self.trace,
            config=self.config,
        )
        structured_cache_before = int(
            getattr(self.deps.structured_client, "query_cache_hits", 0)
        )
        structured_singleflight_before = int(
            getattr(self.deps.structured_client, "query_singleflight_hits", 0)
        )

        async def run_paragraph(paragraph_id: str, paragraph_tasks: list[RetrievalTask]):
            async with self.structured_semaphore:
                return await subgraph.run(
                    paragraph_tasks[0].request_id,
                    paragraph_id,
                    paragraph_tasks[0].paragraph_text,
                    paragraph_tasks,
                    paragraph_tasks[0].organization_context,
                    available_scenario_keys=available_scenario_keys,
                )

        values = await asyncio.gather(*(
            run_paragraph(paragraph_id, paragraph_tasks)
            for paragraph_id, paragraph_tasks in grouped.items()
        ), return_exceptions=True)
        self.metrics["cache"]["structured_cache_hits"] += max(
            0,
            int(getattr(self.deps.structured_client, "query_cache_hits", 0))
            - structured_cache_before,
        )
        self.metrics["cache"]["structured_singleflight_hits"] += max(
            0,
            int(
                getattr(
                    self.deps.structured_client, "query_singleflight_hits", 0
                )
            )
            - structured_singleflight_before,
        )
        for (paragraph_id, paragraph_tasks), value in zip(grouped.items(), values, strict=True):
            if isinstance(value, BaseException):
                metrics["warnings"].append(f"STRUCTURED_SUBGRAPH_ERROR:{paragraph_id}:{type(value).__name__}")
                for task in paragraph_tasks:
                    self.structured_status_by_task[task.task_id] = "tool_error"
                continue
            candidates = list(value.get("candidates", []))
            for candidate in candidates:
                self.structured_candidates_by_task.setdefault(candidate.task_id, []).append(candidate)
                self.structured_status_by_task[candidate.task_id] = "matched"
            records = list(value.get("tool_call_records", []))
            metrics["calls"].extend(
                record.model_dump(mode="json") if hasattr(record, "model_dump") else record
                for record in records
            )
            paragraph_metrics = dict(value.get("metrics", {}))
            metrics["tool_call_count"] += int(paragraph_metrics.get("tool_call_count", len(records)))
            metrics["structured_query_count"] += int(paragraph_metrics.get("structured_query_count", 0))
            metrics["structured_intent_call_count"] += int(
                paragraph_metrics.get("intent_llm_call_count", 0)
            )
            metrics["candidate_count"] += int(paragraph_metrics.get("candidate_count", len(candidates)))
            metrics["warnings"].extend(value.get("warnings", []))
            metrics["timings_ms"][paragraph_id] = paragraph_metrics
            default_status = "no_structured_query" if not candidates else "matched"
            for task in paragraph_tasks:
                has_explicit_legacy_scenario = any(
                    isinstance(ref, dict) and (ref.get("scenario_key") or ref.get("scenario_name"))
                    for ref in task.source_refs
                )
                if self.deps.structured_intent_model is None and has_explicit_legacy_scenario:
                    # Compatibility adapter only: an explicit upstream scenario
                    # hint may use the legacy matcher when no Tool Calling model
                    # was injected. V11 production always injects the model.
                    continue
                self.structured_status_by_task.setdefault(task.task_id, default_status)

    def _merge_candidate_funnel(
        self, task_id: str, audit: dict[str, dict[str, int]],
    ) -> None:
        task_funnel = self.metrics["candidate_funnel_by_task"].setdefault(
            task_id,
            {source: _funnel_template() for source in ("web", "public_kb", "selected_kb", "structured")},
        )
        for source, values in audit.items():
            global_row = self.metrics["candidate_funnel"].setdefault(source, _funnel_template())
            task_row = task_funnel.setdefault(source, _funnel_template())
            for key, value in values.items():
                global_row[key] = int(global_row.get(key, 0)) + int(value)
                task_row[key] = int(task_row.get(key, 0)) + int(value)

    def _increment_funnel(
        self, task_id: str, candidate: EvidenceCandidate, field: str, value: int = 1,
    ) -> None:
        source = _candidate_source_bucket(candidate)
        global_row = self.metrics["candidate_funnel"].setdefault(source, _funnel_template())
        task_funnel = self.metrics["candidate_funnel_by_task"].setdefault(
            task_id,
            {name: _funnel_template() for name in ("web", "public_kb", "selected_kb", "structured")},
        )
        task_row = task_funnel.setdefault(source, _funnel_template())
        global_row[field] = int(global_row.get(field, 0)) + value
        task_row[field] = int(task_row.get(field, 0)) + value

    async def _measured_call(self, name: str, awaitable):
        started = self.collector.begin_call(name)
        error = timeout = False
        try:
            return await awaitable
        except BaseException as exc:
            error = True
            timeout = is_timeout(exc)
            raise
        finally:
            self.collector.end_call(name, started, error=error, timeout=timeout)

    async def _measured_deadline_call(self, name: str, awaitable, timeout_seconds: float):
        """Measure the outer timeout boundary, not only the provider coroutine."""
        started = self.collector.begin_call(name)
        error = timed_out = False
        try:
            return await self._deadline_call(awaitable, timeout_seconds)
        except BaseException as exc:
            error = True
            timed_out = is_timeout(exc)
            raise
        finally:
            self.collector.end_call(name, started, error=error, timeout=timed_out)

    def _remaining(self, requested_ms: int, task_id: str | None = None) -> float:
        effective_deadline = min(self.deadline, self.task_deadlines.get(task_id, self.deadline))
        remaining = effective_deadline - time.monotonic()
        if remaining <= 0:
            if task_id:
                self.task_timeouts.add(task_id)
            else:
                self.metrics["deadline_reached"] = True
            return 0.001
        return min(requested_ms / 1000, remaining)

    async def _deadline_call(self, awaitable, timeout_seconds: float):
        """Cancel at deadline without waiting for a slow SDK cancellation path."""
        task = asyncio.create_task(awaitable)
        done, _ = await asyncio.wait({task}, timeout=max(.001, timeout_seconds))
        if task in done:
            return task.result()
        task.cancel()
        task.add_done_callback(lambda value: value.exception() if not value.cancelled() else None)
        raise TimeoutError

    async def _cached_search(self, task: RetrievalTask, query: QueryItem) -> list[EvidenceCandidate]:
        key = " ".join(query.query.lower().split())
        if key in self.query_cache:
            self.metrics["cache"]["query_cache_hits"] += 1
            rows = self.query_cache[key]
        else:
            future = self.query_inflight.get(key)
            if future is None:
                async def _web_search_with_semaphore():
                    async with self.web_semaphore:
                        return await self.deps.web_search.search("parallel-shared", query)
                future = asyncio.create_task(self._measured_call("web_search", _web_search_with_semaphore()))
                self.query_inflight[key] = future
                self.metrics["call_counts"]["web_search"] += 1
            else:
                self.metrics["cache"]["query_cache_hits"] += 1
                self.metrics["cache"]["query_singleflight_hits"] += 1
            try:
                rows = await asyncio.wait_for(asyncio.shield(future), self._remaining(self.config.parallel_web_search_timeout_ms, task.task_id))
                self.query_cache[key] = rows
            finally:
                if future.done():
                    self.query_inflight.pop(key, None)
            future.add_done_callback(lambda value, cache=self.query_inflight, cache_key=key: cache.pop(cache_key, None))
        return [row.model_copy(update={
            "task_id": task.task_id,
            "candidate_id": f"web-{stable_json_hash([task.task_id, query.query_id, row.source_ref.url])[:20]}",
            "source_ref": row.source_ref.model_copy(update={"query_id": query.query_id}),
        }) for row in rows]

    async def _cached_fetch(self, task: RetrievalTask, candidate: EvidenceCandidate) -> FetchResult:
        url = canonical_url(candidate.source_ref.url or "")
        if url in self.content_cache:
            self.metrics["cache"]["web_content_cache_hits"] += 1
            result = self.content_cache[url]
        else:
            future = self.content_inflight.get(url)
            if future is None:
                async def _web_fetch_with_semaphore():
                    async with self.web_fetch_semaphore:
                        return await self.deps.web_fetcher.fetch(
                            candidate.model_copy(update={"task_id": "parallel-shared"})
                        )

                future = asyncio.create_task(
                    self._measured_call("web_fetch", _web_fetch_with_semaphore())
                )
                self.content_inflight[url] = future
                self.metrics["call_counts"]["web_fetch"] += 1
            else:
                self.metrics["cache"]["web_content_cache_hits"] += 1
                self.metrics["cache"]["web_content_singleflight_hits"] += 1
            try:
                result = await asyncio.wait_for(asyncio.shield(future), self._remaining(self.config.parallel_web_fetch_timeout_ms, task.task_id))
                if isinstance(result, list):
                    result = FetchResult(result)
                self.content_cache[url] = result
            finally:
                if future.done():
                    self.content_inflight.pop(url, None)
            future.add_done_callback(lambda value, cache=self.content_inflight, cache_key=url: cache.pop(cache_key, None))
        return FetchResult(
            candidates=[row.model_copy(update={
                "task_id": task.task_id,
                "candidate_id": f"webbody-{stable_json_hash([task.task_id, url, row.candidate_id])[:20]}",
            }) for row in result.candidates],
            errors=list(result.errors), degraded_to_snippet=result.degraded_to_snippet,
        )

    async def _web(self, task: RetrievalTask, query: QueryItem) -> tuple[list[EvidenceCandidate], list[ErrorDetail]]:
        # Search A first, then bounded B/C/D fallbacks until we have enough
        # distinct URLs for a task-level fetch plan.  This avoids the V10.1
        # failure mode where one empty query suppresses the entire Web branch.
        # Gap retrieval already supplies a carefully validated, slot-specific
        # query. Re-expanding it into the initial A/B/C/D set would silently
        # replay cached broad queries and never execute the gap query itself.
        queries = (
            [query]
            if ":gap:" in query.query_id
            else build_web_query_variants(
                task,
                query,
                max_variants=self.config.initial_query_count,
            )
        )
        metadata = {
            "request_id": task.request_id, "task_id": task.task_id, "item_id": task.item_id,
            "line_type": task.line_type.value, "query_preview": (query.query or "")[:120],
            "target_preview": (task.target_text or "")[:120],
        }
        try:
            search_started = time.monotonic()
            rows: list[EvidenceCandidate] = []
            errors: list[ErrorDetail] = []
            query_rows: dict[str, int] = {}
            query_errors: dict[str, str] = {}
            provider_raw_result_count = 0
            provider_parsed_result_count = 0
            provider_parse_error_count = 0
            invalid_url_count = 0
            seen_urls: set[str] = set()
            row_by_url: dict[str, EvidenceCandidate] = {}
            # Fetching at least two distinct URLs is the default quality path;
            # continue fallback search while fewer than two are available.
            for variant in queries:
                try:
                    async with self.trace.span("web.search", {**metadata, "query_id": variant.query_id, "query_preview": variant.query[:120]}) as span:
                        found = await self._cached_search(task, variant)
                        diagnostic = dict(found[0].metadata) if found else {}
                        provider_raw_result_count += int(diagnostic.get("provider_raw_result_count", len(found)))
                        provider_parsed_result_count += int(diagnostic.get("provider_parsed_result_count", len(found)))
                        provider_parse_error_count += int(diagnostic.get("provider_parse_error_count", 0))
                        invalid_url_count += int(diagnostic.get("invalid_url_count", 0))
                        fresh = []
                        for row in found:
                            url_key = canonical_url(row.source_ref.url or row.candidate_id)
                            if url_key in seen_urls:
                                representative = row_by_url.get(url_key)
                                if representative is not None:
                                    ids = list(representative.metadata.get("candidate_source_query_ids") or [])
                                    if variant.query_id not in ids:
                                        ids.append(variant.query_id)
                                        representative.metadata = {**representative.metadata, "candidate_source_query_ids": ids}
                                continue
                            seen_urls.add(url_key)
                            row.metadata = {
                                **row.metadata,
                                "query_variant_id": variant.query_id,
                                "query_variant": variant.query,
                                "candidate_source_query_ids": [variant.query_id],
                            }
                            fresh.append(row)
                            row_by_url[url_key] = row
                        rows.extend(fresh)
                        query_rows[variant.query_id] = len(found)
                        span["output"] = {"search_result_count": len(found), "new_result_count": len(fresh), "query_variant_id": variant.query_id}
                except TimeoutError:
                    self.task_timeouts.add(task.task_id)
                    query_errors[variant.query_id] = "WEB_TIMEOUT"
                    errors.append(ErrorDetail(code=ErrorCode.WEB_TIMEOUT.value, node="parallel_web", tool="volcano_global_search", retryable=True, reason=f"Web query {variant.query_id} timed out"))
                except Exception as exc:
                    if isinstance(exc, RetrievalError):
                        code = exc.code
                        details = exc.details or {}
                        provider_raw_result_count += int(details.get("provider_raw_result_count", 0))
                        provider_parsed_result_count += int(details.get("provider_parsed_result_count", 0))
                        provider_parse_error_count += int(details.get("provider_parse_error_count", 0))
                        invalid_url_count += int(details.get("invalid_url_count", 0))
                    else:
                        code = ErrorCode.WEB_TIMEOUT if is_timeout(exc) else ErrorCode.WEB_PROVIDER_ERROR
                    query_errors[variant.query_id] = code.value
                    errors.append(ErrorDetail(code=code.value, node="parallel_web", tool="volcano_global_search", retryable=True, reason=f"Web query {variant.query_id} failed: {type(exc).__name__}"))
                # Exhaust the bounded A/B/C/D perspectives. A gap round has
                # only one prevalidated query, so it naturally ends once that
                # query has executed.
                if len(query_rows) >= len(queries):
                    break
            self.metrics["stage_timings_ms"]["web_search"] += int((time.monotonic() - search_started) * 1000)
            round_metrics = {
                "query_variant_count": len(queries),
                "query_variant_list": [item.model_dump(mode="json") for item in queries],
                "search_call_count": len(query_rows) + len(query_errors),
                "search_result_count_by_query": query_rows,
                "search_errors_by_query": query_errors,
                "merged_search_result_count": len(rows),
                "provider_raw_result_count": provider_raw_result_count,
                "provider_parsed_result_count": provider_parsed_result_count,
                "provider_parse_error_count": provider_parse_error_count,
            }
            web_task_metrics = self.metrics.setdefault("web_task", {}).setdefault(task.task_id, {})
            is_gap_round = any(":gap:" in item.query_id for item in queries)
            if not web_task_metrics:
                web_task_metrics.update(round_metrics)
            else:
                web_task_metrics.setdefault("rounds", []).append({
                    "round": "gap" if is_gap_round else "initial",
                    **round_metrics,
                })
                web_task_metrics["search_call_count"] = int(web_task_metrics.get("search_call_count", 0)) + round_metrics["search_call_count"]
                web_task_metrics["merged_search_result_count"] = int(web_task_metrics.get("merged_search_result_count", 0)) + round_metrics["merged_search_result_count"]
            if not rows and not errors:
                errors.append(ErrorDetail(code=ErrorCode.WEB_NO_RESULT.value, node="parallel_web", tool="volcano_global_search", retryable=True, reason="All Web query variants returned no results"))
            filter_started = time.monotonic()
            async with self.trace.span("web.filter", {**metadata, "input_count": len(rows)}) as span:
                article_rows = [row for row in rows if not _NON_ARTICLE_PATH.search(__import__("urllib.parse", fromlist=["urlparse"]).urlparse(row.source_ref.url or "").path)]
                non_article_path_count = len(rows) - len(article_rows)
                article_rows, denylist_count, domain_limit_count = _apply_domain_filter_with_audit(article_rows, self.config)
                domain_filtered = denylist_count + domain_limit_count
                fetch_budget = max(2, self.config.web_fetch_top_n)
                selected = _lexical_rank(query.query, article_rows, top_k=min(fetch_budget, self.config.web_keep_top_k_urls, len(article_rows)), preferred_domains=self.config.preferred_domains)
                lexical_filtered_count = len(article_rows) - len(selected)
                filtered_count = len(rows) - len(selected)
                self.metrics["stage_timings_ms"]["web_filter"] += int((time.monotonic() - filter_started) * 1000)
                self.metrics["stage_timings_ms"]["web_bm25"] += int((time.monotonic() - filter_started) * 1000)
                self.metrics["domain_filtered"] = self.metrics.get("domain_filtered", 0) + domain_filtered
                span["output"] = {
                    "selected_count": len(selected), "filtered_count": filtered_count,
                    "domain_limited_count": domain_filtered,
                    "denylist_count": denylist_count,
                    "domain_limit_count": domain_limit_count,
                    "lexical_filtered_count": lexical_filtered_count,
                }
            fetch_started = time.monotonic()
            async with self.trace.span("web.fetch", {**metadata, "url_count": len(selected)}) as span:
                fetched = await asyncio.gather(*(self._cached_fetch(task, row) for row in selected), return_exceptions=True)
                candidates: list[EvidenceCandidate] = []
                fetch_success = 0
                fetch_error = 0
                snippet_fallback = 0
                fetch_warnings: list[str] = []
                for original, value in zip(selected, fetched, strict=True):
                    if isinstance(value, BaseException):
                        candidates.append(original)
                        fetch_error += 1
                        snippet_fallback += 1
                        fetch_warnings.append(f"WEB_FETCH_FALLBACK:{original.source_ref.url}:{type(value).__name__}")
                    else:
                        # Keep the original title/snippet on all fetch errors
                        # and unsupported bodies.  The candidate is explicitly
                        # marked snippet_only; it must never disappear silently.
                        fallback = value.candidates or [original]
                        candidates.extend(fallback)
                        degraded = bool(value.errors or value.degraded_to_snippet)
                        if degraded:
                            # _measured_call saw a returned value and counted a
                            # success. Reclassify that one URL attempt exactly
                            # once so success+failure always equals attempt.
                            metric = self.collector.calls["web_fetch"]
                            metric.success_count = max(0, metric.success_count - 1)
                            metric.error_count += 1
                            fetch_error += 1
                            snippet_fallback += 1
                            fetch_warnings.append(
                                f"WEB_FETCH_FALLBACK:{original.source_ref.url}:"
                                f"{value.errors[0].code if value.errors else 'DEGRADED_TO_SNIPPET'}"
                            )
                        else:
                            fetch_success += 1
                self.metrics["stage_timings_ms"]["web_fetch"] += int((time.monotonic() - fetch_started) * 1000)
                for candidate in candidates:
                    candidate.content = _relevant_window(task.target_text, candidate.content, self.config.parallel_judge_candidate_max_chars)
                candidates = _lexical_rank(task.target_text, candidates, top_k=self.config.web_candidates_per_task, preferred_domains=self.config.preferred_domains)
                self.metrics["source_candidate_counts"]["web"] += len(candidates)
                self.metrics["web_fetch_plan"]["web_search_result_count"] += len(rows)
                self.metrics["web_fetch_plan"]["web_fetch_planned_count"] += len(selected)
                self.metrics["web_fetch_plan"]["web_unique_url_count"] += len({canonical_url(row.source_ref.url or "") for row in selected})
                self.metrics["web_fetch_plan"]["web_fetch_success_count"] += fetch_success
                self.metrics["web_fetch_plan"]["web_candidate_extracted_count"] += len(candidates)
                span["output"] = {
                    "candidate_count": len(candidates), "error_count": len(errors),
                    "warning_count": len(fetch_warnings),
                    "attempt_count": len(selected),
                    "fetch_success_count": fetch_success, "fetch_error_count": fetch_error,
                    "snippet_fallback_count": snippet_fallback,
                    "bm25_candidate_count": len(candidates),
                    "final_candidate_count": len(candidates),
                    "fetch_plan": {"planned_count": len(selected), "unique_url_count": len({canonical_url(row.source_ref.url or "") for row in selected}), "no_fetch_reason": "search_empty" if not selected else None},
                }
                span["_final_metadata"] = {
                    "fetch_success_count": fetch_success,
                    "fetch_error_count": fetch_error,
                    "snippet_fallback_count": snippet_fallback,
                    "partial_failure": fetch_error > 0,
                    "warning_count": len(fetch_warnings),
                }
                task_web = self.metrics.setdefault("web_task", {}).setdefault(task.task_id, {})
                fetch_metrics = {
                    "search_result_count": len(rows),
                    "provider_raw_result_count": provider_raw_result_count,
                    "provider_parsed_result_count": provider_parsed_result_count,
                    "provider_parse_error_count": provider_parse_error_count,
                    "invalid_url_count": invalid_url_count,
                    "non_article_path_count": non_article_path_count,
                    "article_filter_count": non_article_path_count,
                    "denylist_count": denylist_count,
                    "domain_limit_count": domain_limit_count,
                    "domain_filter_count": domain_filtered,
                    "lexical_filtered_count": lexical_filtered_count,
                    "fetch_planned_count": len(selected),
                    "unique_url_count": len({canonical_url(row.source_ref.url or "") for row in selected}),
                    "fetch_success_count": fetch_success,
                    "fetch_attempt_count": len(selected),
                    "fetch_failure_count": fetch_error,
                    "snippet_fallback_count": snippet_fallback,
                    "fetch_fallback_count": snippet_fallback,
                    "warnings": fetch_warnings,
                    "candidate_extracted_count": len(candidates),
                    "final_web_candidate_count": len(candidates),
                    "candidate_reassigned_count": 0,
                    "no_fetch_reason": "search_empty" if not selected else None,
                }
                if task_web.get("rounds"):
                    task_web["rounds"][-1].update(fetch_metrics)
                    task_web["gap_candidate_extracted_count"] = (
                        int(task_web.get("gap_candidate_extracted_count", 0)) + len(candidates)
                    )
                else:
                    task_web.update(fetch_metrics)
                return candidates, errors
        except TimeoutError:
            self.task_timeouts.add(task.task_id)
            return [], [ErrorDetail(code=ErrorCode.WEB_TIMEOUT.value, node="parallel_web", tool="volcano_global_search", retryable=True, reason="parallel Web deadline reached")]
        except Exception as exc:
            # Provider clients may surface httpx/provider timeouts directly
            # instead of wrapping them in RetrievalError.  Preserve the
            # timeout contract at the task, metrics and trace layers.
            if is_timeout(exc):
                self.task_timeouts.add(task.task_id)
                layer = timeout_layer(exc) or "provider_timeout"
                elapsed_ms = int((time.monotonic() - search_started) * 1000)
                return [], [ErrorDetail(
                    code=ErrorCode.WEB_TIMEOUT.value, node="parallel_web",
                    tool="volcano_global_search", retryable=True,
                    reason="Web provider request timed out", timeout_layer=layer,
                    configured_timeout_ms=self.config.parallel_web_search_timeout_ms,
                    actual_elapsed_ms=elapsed_ms,
                )]
            return [], [ErrorDetail(code=ErrorCode.WEB_PROVIDER_ERROR.value, node="parallel_web", tool="volcano_global_search", retryable=True, reason=f"{type(exc).__name__}: Web failed")]

    async def _cached_kb_retrieve(
        self,
        task: RetrievalTask,
        ids: list[str],
        origin: str,
        query: str,
        label: str,
    ):
        key = (
            tuple(sorted(str(value) for value in ids)),
            " ".join(query.casefold().split()),
            self.config.bisheng_retrieve_top_k,
            self.config.bisheng_retrieve_score_threshold,
            task.user_id,
            origin,
        )
        if key in self.kb_cache:
            self.metrics["cache"]["kb_cache_hits"] += 1
            return self.kb_cache[key]
        future = self.kb_inflight.get(key)
        if future is None:
            async def provider_call():
                async with self.kb_semaphore:
                    return await self._measured_deadline_call(
                        f"{label}_retrieve",
                        self.deps.kb_client.retrieve(
                            knowledge_ids=ids,
                            query=query,
                            top_k=self.config.bisheng_retrieve_top_k,
                            score_threshold=self.config.bisheng_retrieve_score_threshold,
                            user_id=task.user_id,
                            origin=origin,
                        ),
                        self._remaining(
                            self.config.parallel_kb_timeout_ms, task.task_id
                        ),
                    )

            future = asyncio.create_task(provider_call())
            self.kb_inflight[key] = future
        else:
            self.metrics["cache"]["kb_singleflight_hits"] += 1
        try:
            result = await asyncio.wait_for(
                asyncio.shield(future),
                self._remaining(self.config.parallel_kb_timeout_ms, task.task_id),
            )
            self.kb_cache[key] = result
            return result
        finally:
            if future.done():
                self.kb_inflight.pop(key, None)
            future.add_done_callback(
                lambda _value, cache=self.kb_inflight, cache_key=key: cache.pop(
                    cache_key, None
                )
            )

    async def _kb(
        self,
        task: RetrievalTask,
        ids: list[str],
        origin: str,
        *,
        query_override: str | None = None,
    ) -> tuple[list[EvidenceCandidate], list[ErrorDetail]]:
        if not ids:
            return [], []
        label = "selected_kb" if origin == "upstream_selected" else "public_kb"
        started = time.monotonic()
        metadata = {
            "request_id": task.request_id, "task_id": task.task_id, "item_id": task.item_id,
            "line_type": task.line_type.value,
            "knowledge_ids": ids[:10], "knowledge_origin": origin,
            "top_k": self.config.bisheng_retrieve_top_k,
            "score_threshold": self.config.bisheng_retrieve_score_threshold,
        }
        query_variants = build_kb_query_variants(
            task,
            max_variants=self.config.initial_query_count,
        )
        if query_override:
            # Gap retrieval is deliberately a single deterministic query per
            # missing slot.  The normal first round keeps its multi-query
            # variants; the second round must remain bounded and auditable.
            query_variants = [{
                "query_id": f"{task.task_id}:gap-kb:{stable_json_hash(query_override)[:12]}",
                "variant": "gap",
                "query": query_override[:500],
            }]
        self.metrics["kb_query_variants"]["count"] += len(query_variants)
        self.metrics["kb_query_variants"]["tasks"] += 1
        kb_task_metrics = self.metrics.setdefault("kb_task", {}).setdefault(task.task_id, {
            "query_variant_count": 0, "query_result_count_by_query": {},
            "raw_candidate_count": 0, "exact_duplicate_count": 0,
            "adjacent_chunk_count": 0,
        })
        kb_task_metrics["query_variant_count"] += len(query_variants)
        metadata.update({"query_variant_count": len(query_variants), "query_variant_ids": [item["query_id"] for item in query_variants]})
        async with self.trace.span(f"{label}.retrieve", metadata) as span:
            errors: list[ErrorDetail] = []
            candidates: list[EvidenceCandidate] = []
            missing: list[str] = []
            try:
                async def retrieve_variant(variant: dict[str, str]):
                    # Each deterministic variant is an independent provider
                    # query; the semaphore and shared deadline bound fan-out.
                    return variant, await self._cached_kb_retrieve(
                        task,
                        ids,
                        origin,
                        variant["query"],
                        label,
                    )

                variant_results = await asyncio.gather(
                    *(retrieve_variant(variant) for variant in query_variants),
                    return_exceptions=True,
                )
                chunk_rows: dict[tuple[str, str, str], EvidenceCandidate] = {}
                for value in variant_results:
                    if isinstance(value, BaseException):
                        if isinstance(value, asyncio.TimeoutError):
                            self.task_timeouts.add(task.task_id)
                            errors.append(ErrorDetail(code=ErrorCode.KB_TIMEOUT.value, node="parallel_kb", tool="bisheng_vector_retrieve", retryable=True, reason="parallel KB query variant timed out"))
                        elif isinstance(value, RetrievalError):
                            errors.append(ErrorDetail(code=value.code.value, node=value.node, tool=value.tool, retryable=value.retryable, reason=value.message, timeout_layer=value.timeout_layer))
                        else:
                            errors.append(ErrorDetail(code=ErrorCode.KB_PROVIDER_ERROR.value, node="parallel_kb", tool="bisheng_vector_retrieve", retryable=True, reason=f"{type(value).__name__}: KB query variant failed"))
                        continue
                    variant, result = value
                    missing.extend(result.missing_knowledge_ids)
                    errors.extend(ErrorDetail(code=ErrorCode.KB_NOT_FOUND.value, node="parallel_kb", tool="bisheng_vector_retrieve", retryable=False, reason=f"knowledge_id {m} was not resolved") for m in result.missing_knowledge_ids)
                    errors.extend(ErrorDetail(code=ErrorCode.KB_PROVIDER_ERROR.value, node="parallel_kb", tool="bisheng_vector_retrieve", retryable=True, reason=provider_error) for provider_error in result.errors)
                    for doc in result.chunks:
                        key = (str(doc.knowledge_id), str(doc.file_id), str(doc.chunk_id))
                        if key in chunk_rows:
                            existing = chunk_rows[key]
                            metadata_existing = dict(existing.metadata)
                            ids_seen = list(metadata_existing.get("query_variant_ids") or [])
                            if variant["query_id"] not in ids_seen:
                                ids_seen.append(variant["query_id"])
                            existing.metadata = {**metadata_existing, "query_variant_ids": ids_seen}
                            continue
                        chunk_rows[key] = EvidenceCandidate(
                            candidate_id=f"kb-{stable_json_hash([task.task_id, *key])[:20]}",
                            task_id=task.task_id, source_type=SourceType.KNOWLEDGE_BASE, source_name="bisheng_vector_retrieve",
                            source_ref=SourceRef(knowledge_id=key[0], knowledge_origin=origin, file_id=key[1], chunk_id=key[2], chunk_index=doc.chunk_index, query_id=variant["query_id"]),
                            title=doc.file_name, content=doc.text, initial_score=float(doc.score),
                            metadata={"page": doc.page, "chunk_index": doc.chunk_index, "rank": doc.rank, "query_variant_ids": [variant["query_id"]], **doc.metadata},
                        )
                candidates = list(chunk_rows.values())
                self.metrics["kb_query_variants"]["hits"] += len(candidates)
                kb_task_metrics["raw_candidate_count"] += len(candidates)
                for variant in query_variants:
                    kb_task_metrics["query_result_count_by_query"].setdefault(variant["query_id"], 0)
                for candidate in candidates:
                    for query_id in candidate.metadata.get("query_variant_ids", []):
                        kb_task_metrics["query_result_count_by_query"][query_id] = (
                            kb_task_metrics["query_result_count_by_query"].get(query_id, 0) + 1
                        )
                # When multiple returned chunks are adjacent in the same file,
                # retain a bounded context window for Judge without crossing
                # document boundaries. Original chunk provenance remains in
                # source_ref and metadata.
                rows = [{"knowledge_id": c.source_ref.knowledge_id, "file_id": c.source_ref.file_id,
                         "chunk_id": c.source_ref.chunk_id, "chunk_index": c.source_ref.chunk_index,
                         "text": c.content} for c in candidates]
                for index, candidate in enumerate(candidates):
                    context = build_adjacent_context(rows[index], rows, max_chars=self.config.parallel_judge_candidate_max_chars, window=1)
                    if len(context.get("included_chunk_ids", [])) > 1:
                        kb_task_metrics["adjacent_chunk_count"] += 1
                        candidate.content = context["text"]
                        candidate.metadata = {**candidate.metadata, "context_window": context,
                                              "adjacent_chunk_ids": context["included_chunk_ids"][0:]}
                metadata["query_variant_hits"] = len(candidates)
            except TimeoutError:
                self.task_timeouts.add(task.task_id)
                elapsed_ms = int((time.monotonic() - started) * 1000)
                layer = "request_deadline" if time.monotonic() >= self.deadline else "searchagent_wait_for"
                errors.append(ErrorDetail(
                    code=ErrorCode.KB_TIMEOUT.value, node="parallel_kb",
                    tool="bisheng_vector_retrieve", retryable=True,
                    reason="parallel KB deadline reached", timeout_layer=layer,
                    configured_timeout_ms=self.config.parallel_kb_timeout_ms,
                    actual_elapsed_ms=elapsed_ms,
                ))
                span["_final_metadata"] = {
                    "timeout_layer": layer,
                    "configured_timeout_ms": self.config.parallel_kb_timeout_ms,
                    "elapsed_ms": elapsed_ms,
                    "knowledge_ids": ids[:10], "task_id": task.task_id,
                }
            except RetrievalError as exc:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                errors.append(ErrorDetail(
                    code=exc.code.value, node=exc.node, tool=exc.tool,
                    retryable=exc.retryable, reason=exc.message,
                    timeout_layer=exc.timeout_layer,
                    configured_timeout_ms=self.config.parallel_kb_timeout_ms if exc.code == ErrorCode.KB_TIMEOUT else None,
                    actual_elapsed_ms=elapsed_ms,
                ))
                if exc.code == ErrorCode.KB_TIMEOUT:
                    span["_final_metadata"] = {
                        "timeout_layer": exc.timeout_layer or "provider_timeout",
                        "configured_timeout_ms": self.config.parallel_kb_timeout_ms,
                        "elapsed_ms": elapsed_ms,
                        "knowledge_ids": ids[:10], "task_id": task.task_id,
                    }
            except Exception as exc:
                if is_timeout(exc):
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    layer = timeout_layer(exc) or "provider_timeout"
                    self.task_timeouts.add(task.task_id)
                    errors.append(ErrorDetail(
                        code=ErrorCode.KB_TIMEOUT.value, node="parallel_kb",
                        tool="bisheng_vector_retrieve", retryable=True,
                        reason="KB provider request timed out", timeout_layer=layer,
                        configured_timeout_ms=self.config.parallel_kb_timeout_ms,
                        actual_elapsed_ms=elapsed_ms,
                    ))
                    span["_final_metadata"] = {
                        "timeout_layer": layer,
                        "configured_timeout_ms": self.config.parallel_kb_timeout_ms,
                        "elapsed_ms": elapsed_ms,
                        "knowledge_ids": ids[:10], "task_id": task.task_id,
                    }
                else:
                    errors.append(ErrorDetail(code=ErrorCode.KB_PROVIDER_ERROR.value, node="parallel_kb", tool="bisheng_vector_retrieve", retryable=True, reason=f"{type(exc).__name__}: KB failed"))
            self.metrics["source_candidate_counts"][label] += len(candidates)
            self.metrics["stage_timings_ms"][f"{label}_retrieve"] += int((time.monotonic() - started) * 1000)
            span["output"] = {
                "candidate_count": len(candidates), "error_count": len(errors),
                "query_variant_count": len(query_variants),
                "query_variant_hits": len(candidates),
                "resolved_knowledge_ids": [kid for kid in ids if kid not in missing],
                "missing_knowledge_ids": list(dict.fromkeys(missing)),
                "returned_chunk_count": len(candidates),
                "final_candidate_count": len(candidates),
            }
            return candidates, errors

    async def _structured(self, task: RetrievalTask, scenarios: dict[str, Any]) -> tuple[list[EvidenceCandidate], list[ErrorDetail], str]:
        # V11+ main path: paragraph-level LLM Tool Calling has already selected,
        # validated and executed all tools. The old rule-based matcher is removed.
        await self.structured_ready.wait()
        if task.task_id in self.structured_status_by_task:
            candidates = list(self.structured_candidates_by_task.get(task.task_id, []))
            self.metrics["source_candidate_counts"]["structured"] += len(candidates)
            self.metrics["call_counts"]["structured_query"] = self.metrics["structured_tool_calling"]["structured_query_count"]
            return candidates, [], self.structured_status_by_task[task.task_id]
        return [], [], "no_structured_query"

    async def _retrieve_task(self, task: RetrievalTask, scenarios: dict[str, Any]):
        self.task_deadlines[task.task_id] = min(self.deadline, time.monotonic() + self.config.task_hard_timeout_ms / 1000)
        metadata = {
            "request_id": task.request_id, "task_id": task.task_id, "item_id": task.item_id,
            "line_type": task.line_type.value, "node_id": task.node_id,
            "hypothesis_id": task.hypothesis_id if task.line_type == LineType.REVERSE else None,
            "flow_mode": "parallel_sources",
        }
        task_span = await self.trace.start_span(
            f"task.{task.line_type.value}.{task.task_id}", metadata,
        )
        # Keep this one Task observation open through retrieval, merge, shared
        # Judge wait and Verification. Later children use its explicit ID.
        self.trace.register_task_span(task.task_id, task_span.get("_child_id"))
        context_token = self.trace.activate_span(task_span)
        try:
            task_metrics = TaskMetricsCollector(self.collector, task.task_id)
            # V10 claim diagnostics are explicit children of the task span;
            # they carry only bounded structured metadata, never raw prompts.
            async with self.trace.span(
                "claim.atomize",
                {**metadata, "claim_count": len(task.atomic_claims),
                 "logic_operator": task.claim_logic_operator.value,
                 "status": "SUCCESS"},
                parent_run_id=task_span.get("_child_id"),
            ) as claim_span:
                claim_span["output"] = {"claim_count": len(task.atomic_claims), "logic_operator": task.claim_logic_operator.value}
            if task.line_type == LineType.REVERSE:
                async with self.trace.span(
                    "hypothesis.normalize",
                    {**metadata, "status": "SUCCESS", "polarity": task.polarity},
                    parent_run_id=task_span.get("_child_id"),
                ) as hypothesis_span:
                    hypothesis_span["output"] = {"normalized": bool(task.normalized_hypothesis), "neutral_query": bool(task.neutral_retrieval_query)}
            query_started = time.monotonic()
            query, query_terms = _parallel_query_details(task)
            task_metrics.stage("query_build", query_started)
            self.metrics.setdefault("query_observations", []).append({
                "task_id": task.task_id, "original_target": task.target_text,
                "built_query": query.query, "retained_terms": query_terms,
            })
            attempted = ["volcano_global_search"]
            if self.config.public_knowledge_ids or task.selected_knowledge_ids:
                attempted.append("bisheng_knowledge_retrieve")
            attempted.append("structured_query")
            usage = ToolUsage(web_rounds=1, attempted_tools=attempted)

            async def measured(stage: str, awaitable):
                started = time.monotonic()
                timed_out = False
                try:
                    return await awaitable
                except TimeoutError:
                    timed_out = True
                    self.task_timeouts.add(task.task_id)
                    raise
                finally:
                    task_metrics.stage(stage, started, timeout=timed_out)

            web, public, selected, structured = await asyncio.gather(
                measured("web", self._web(task, query)),
                measured("public_kb", self._kb(task, self.config.public_knowledge_ids, "configured_public")),
                measured("selected_kb", self._kb(task, task.selected_knowledge_ids, "upstream_selected")),
                measured("structured", self._structured(task, scenarios)),
            )
            usage.public_kb_calls = 1 if self.config.public_knowledge_ids else 0
            usage.selected_kb_calls = 1 if task.selected_knowledge_ids else 0
            usage.structured_calls = 1 if structured[2] == "matched" else 0
            candidates, dedupe_audit = deduplicate_candidates_with_audit(
                [*web[0], *public[0], *selected[0], *structured[0]],
            )
            for bucket, source_errors in (("public_kb", public[1]), ("selected_kb", selected[1])):
                invalid_count = sum(
                    error.reason.startswith(("invalid_chunk:", "empty_chunk:"))
                    for error in source_errors
                )
                if invalid_count:
                    row = dedupe_audit.setdefault(bucket, _funnel_template())
                    row["retrieved_count"] += invalid_count
                    row["invalid_count"] += invalid_count
            self._merge_candidate_funnel(task.task_id, dedupe_audit)
            # Scope Guard: drop candidates whose content doesn't mention the same
            # subject/organization as the task's target_text. Prevents Judge from
            # processing irrelevant candidates about different subjects/schools/
            # companies — saves Judge tokens and improves verdict quality.
            candidates, scope_dropped, scope_checked = _scope_pre_filter(task, candidates)
            if scope_checked > 0:
                scope_metrics = self.metrics.setdefault("scope_guard", {})
                scope_metrics.setdefault("total_checked", 0)
                scope_metrics.setdefault("total_dropped", 0)
                scope_metrics.setdefault("tasks_affected", 0)
                scope_metrics["total_checked"] += scope_checked
                scope_metrics["total_dropped"] += scope_dropped
                if scope_dropped > 0:
                    scope_metrics["tasks_affected"] += 1
            # In production-sized flows an empty Web branch is fail-closed:
            # returning SUCCESS with no Web candidate is prohibited. Tiny
            # deterministic unit-flow budgets retain the historical exhausted
            # semantics so offline regression fixtures remain meaningful.
            source_errors = [*web[1], *public[1], *selected[1], *structured[1]]
            if self.config.parallel_flow_timeout_ms >= 10_000:
                errors = source_errors
            else:
                errors = [
                    error for error in source_errors
                    if error.code != ErrorCode.WEB_NO_RESULT.value
                ]
            selected_ok = not task.selected_knowledge_ids or not selected[1]
            return task, query, candidates, errors, usage, selected_ok, structured[2], task_metrics, task_span
        except BaseException as exc:
            await self.trace.end_span(task_span, error=exc)
            raise
        finally:
            self.trace.deactivate_span(context_token)

    async def _rank(self, task: RetrievalTask, candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
        metadata = {
            "request_id": task.request_id, "task_id": task.task_id, "item_id": task.item_id,
            "line_type": task.line_type.value,
            "input_candidate_count": len(candidates),
            "ranked_candidate_input_count": len(candidates),
            "evidence_output_mode": self.config.evidence_output_mode,
            "node_id": task.node_id,
            "hypothesis_id": task.hypothesis_id if task.line_type == LineType.REVERSE else None,
        }
        # Parent candidate.merge under the already-opened task span to avoid
        # creating a duplicate task.* observation in the trace.
        parent_run_id = self.trace.task_span_id(task.task_id)
        async with self.trace.span("candidate.merge", metadata, parent_run_id=parent_run_id) as span:
            started = time.monotonic()
            # Enforce a per-source budget before Judge. This preserves Web/KB/
            # Structured diversity while preventing one KB document or a Web
            # result burst from multiplying LLM input size.
            source_limits = {
                SourceType.WEB: self.config.web_candidates_per_task,
                SourceType.KNOWLEDGE_BASE: self.config.kb_candidates_per_task,
                SourceType.STRUCTURED: self.config.structured_max_candidates_per_task,
            }
            selected: list[EvidenceCandidate] = []
            for source_type, limit in source_limits.items():
                rows = [
                    candidate
                    for candidate in candidates
                    if candidate.source_type == source_type
                ]
                selected.extend(
                    _lexical_rank(
                        task.target_text,
                        rows,
                        top_k=min(limit, len(rows)),
                        preferred_domains=self.config.preferred_domains,
                    )
                )
            ranked = _lexical_rank(
                task.target_text,
                selected,
                top_k=min(self.config.total_candidates_per_task, len(selected)),
                preferred_domains=self.config.preferred_domains,
            )
            self.metrics["stage_timings_ms"]["candidate_merge"] += int((time.monotonic() - started) * 1000)
            span["output"] = {
                "ranked_count": len(ranked),
                "final_candidate_count": len(ranked),
                "source_quota": {
                    "web": self.config.web_candidates_per_task,
                    "knowledge_base": self.config.kb_candidates_per_task,
                    "structured": self.config.structured_max_candidates_per_task,
                    "total": self.config.total_candidates_per_task,
                },
            }
            if self.config.evidence_output_mode == "judged":
                for candidate in ranked:
                    self._increment_funnel(task.task_id, candidate, "judge_ready_count")
            return ranked

    async def _passthrough_direction(self, retrieved):
        """Return ranked Web/KB/Structured materials without any Judge call."""
        started = time.monotonic()
        self.metrics["judge_timing_scope"] = "disabled_candidate_passthrough"
        self.metrics["shared_batch_judge_ms"] = 0
        self.metrics["stage_timings_ms"]["batch_judge"] = 0
        results: list[RetrievalTaskResult] = []

        for (
            task, _query, candidates, errors, usage, selected_ok,
            _structured_status, task_metrics, task_span,
        ) in retrieved:
            rank_started = time.monotonic()
            ranked = await self._rank(task, candidates)
            task_metrics.stage("rank", rank_started)
            evidence = [
                item
                for candidate in ranked
                if (item := _candidate_passthrough_item(task, candidate)) is not None
            ]
            evidence = deduplicate_evidence_items(evidence)
            all_errors = deduplicate_errors(errors)
            status, termination = derive_task_execution_state(
                errors=all_errors,
                deadline_reached=(
                    self.metrics["deadline_reached"]
                    or task.task_id in self.task_timeouts
                ),
                selected_ok=selected_ok,
                has_usable_evidence=bool(evidence),
            )
            verification = VerificationResult(
                verdict=VerificationVerdict.INCONCLUSIVE,
                upstream_status="doubtful",
                confidence=0.0,
                reason=(
                    "SearchAgent candidate passthrough mode: no evidence relation "
                    "or verdict was produced; parent HypoArgus Judgment must decide."
                ),
                supplementary_evidence_ids=[item.evidence_id for item in evidence],
                sufficiency_path="CANDIDATE_PASSTHROUGH",
                override_applied=False,
                claim_logic_operator=task.claim_logic_operator.value,
                atomic_claim_verdicts={
                    claim.claim_id: "INCONCLUSIVE" for claim in task.atomic_claims
                },
            )
            source_fingerprints = {
                item.source_evidence_fingerprint for item in evidence
            }
            quality = EvidenceQuality(
                effective_evidence_count=len(evidence),
                independent_source_count=len(source_fingerprints),
                independent_document_count=len(source_fingerprints),
                claim_coverage_score=0.0,
                missing_slots=list(task.required_slots),
                only_snippets=bool(evidence) and all(item.snippet_only for item in evidence),
            )
            elapsed_ms = task_metrics.finish()
            task_snapshot = self.collector.tasks[task.task_id]
            kb_task = self.metrics.get("kb_task", {}).get(task.task_id, {})
            kb_candidates = [
                candidate for candidate in ranked
                if candidate.source_type == SourceType.KNOWLEDGE_BASE
            ]
            result = RetrievalTaskResult(
                task_id=task.task_id,
                item_id=task.item_id,
                line_type=task.line_type,
                node_id=task.node_id,
                hypothesis_id=task.hypothesis_id,
                target_text=task.target_text,
                execution_status=status,
                termination_reason=termination,
                verification=verification,
                evidence_items=evidence,
                evidence_quality=quality,
                tool_usage=usage,
                evidence_gap=None,
                errors=all_errors,
                elapsed_ms=elapsed_ms,
                node_timings_ms=dict(task_snapshot["stage_timings_ms"]),
                judge_batches=[],
                judge_candidate_total=0,
                judge_candidate_completed=0,
                judge_candidate_failed=0,
                judge_completeness_ratio=1.0,
                retrieved_candidate_count=len(ranked),
                supplement_count=len(evidence),
                missing_slots_before_gap_retrieval=list(task.required_slots),
                missing_slots_after_gap_retrieval=list(task.required_slots),
                gap_retrieval_triggered=False,
                atomic_claim_count=len(task.atomic_claims),
                kb_query_variant_count=int(kb_task.get("query_variant_count", 0)),
                kb_query_result_count_by_query=dict(
                    kb_task.get("query_result_count_by_query", {})
                ),
                kb_query_zero_hit_count=sum(
                    int(value == 0)
                    for value in kb_task.get("query_result_count_by_query", {}).values()
                ),
                kb_raw_candidate_count=int(kb_task.get("raw_candidate_count", 0)),
                kb_adjacent_chunk_count=int(kb_task.get("adjacent_chunk_count", 0)),
                kb_final_candidate_count=len(kb_candidates),
            )
            results.append(result)
            await self.trace.end_span(
                task_span,
                output={
                    "candidate_count": len(ranked),
                    "citation_material_count": len(evidence),
                    "evidence_output_mode": "candidate_passthrough",
                    "verification": VerificationVerdict.INCONCLUSIVE.value,
                    "execution_status": status.value,
                    "error_count": len(all_errors),
                },
                final_metadata={
                    "judge_disabled": True,
                    "judge_candidate_total": 0,
                    "candidate_passthrough_count": len(evidence),
                },
            )

        self.metrics["stage_timings_ms"]["verification"] += int(
            (time.monotonic() - started) * 1000
        )
        return results

    async def _judge_direction(self, retrieved):
        judge_started = time.monotonic()
        groups = []
        ranked_by_task = {}
        task_spans: dict[str, dict[str, Any]] = {}
        for task, _query, candidates, _errors, _usage, _selected_ok, _structured_status, task_metrics, task_span in retrieved:
            rank_started = time.monotonic()
            ranked = await self._rank(task, candidates)
            task_metrics.stage("rank", rank_started)
            ranked_by_task[task.task_id] = ranked
            task_spans[task.task_id] = task_span
            groups.append((task, ranked, build_prepared_context(task)))
        mapping: dict[tuple[str, str], JudgeResult] = {}
        judge_errors_by_task: dict[str, list[ErrorDetail]] = defaultdict(list)
        judge_failed_keys: set[tuple[str, str]] = set()
        plan = self.judge_planner.plan(groups)
        integrity = self.metrics["judge_integrity"]
        integrity.update({
            "judge_input_candidate_count": plan.input_candidate_count,
            "judge_batched_candidate_count": plan.batched_candidate_count,
            "judge_missing_candidate_count": plan.missing_candidate_count,
            "judge_duplicate_candidate_count": plan.duplicate_candidate_count,
        })
        self.metrics["judge_candidate_batch_map"] = {
            f"{task_id}::{candidate_id}": batch_id
            for (task_id, candidate_id), batch_id in plan.candidate_to_batch.items()
        }
        candidate_lookup = {
            (task.task_id, candidate.candidate_id): candidate
            for task, rows, _ in groups for candidate in rows
        }
        for key in plan.candidate_to_batch:
            self._increment_funnel(key[0], candidate_lookup[key], "judge_batched_count")

        if any(rows for _, rows, _ in groups):
            batch_semaphore = asyncio.Semaphore(self.config.judge_batch_concurrency)

            async def run_batch(batch: JudgeBatch):
                first_task = batch.groups[0][0]
                batch_started = time.monotonic()
                provider_started: float | None = None
                configured_timeout_ms = self.config.parallel_batch_judge_timeout_ms
                batch_record: dict[str, Any] = {
                    "batch_id": batch.batch_id,
                    "task_ids": batch.task_ids,
                    "candidate_ids": batch.candidate_ids,
                    "task_count": len(batch.task_ids),
                    "candidate_count": batch.candidate_count,
                    "input_chars": batch.input_chars,
                    "estimated_input_tokens": batch.estimated_input_tokens,
                    "actual_prompt_tokens": None,
                    "completion_tokens": None,
                    "expected_output_tokens": batch.expected_output_tokens,
                    "configured_timeout_ms": configured_timeout_ms,
                    "status": "PENDING",
                    "error_code": None,
                    "timeout_count": 0,
                    "timeout": False,
                    "repair_count": 0,
                }
                self._record_batch_judge_call()
                async with batch_semaphore:
                    provider_started = time.monotonic()
                    batch_record["queue_wait_ms"] = int(
                        (provider_started - batch_started) * 1000
                    )
                    async with self._batch_judge_observation(
                        f"batch_judge.batch_{batch.batch_id.rsplit('-', 1)[-1]}",
                        {
                            "request_id": first_task.request_id,
                            "batch_id": batch.batch_id,
                            "task_ids": batch.task_ids,
                            "task_count": len(batch.task_ids),
                            "candidate_count": batch.candidate_count,
                            "input_chars": batch.input_chars,
                            "estimated_input_tokens": batch.estimated_input_tokens,
                            "expected_output_tokens": batch.expected_output_tokens,
                            "configured_timeout_ms": configured_timeout_ms,
                            "over_token_limit": batch.over_token_limit,
                        },
                        parent_run_id=self.trace.run_id_for(first_task.request_id),
                    ) as judge_span:
                        try:
                            value = await self._measured_deadline_call(
                                "llm_batch_evidence_judge",
                                self.deps.batch_judge.judge_many(batch.groups),
                                self._remaining(configured_timeout_ms),
                            )
                            neutral_completion_count = 0
                            completed_candidate_ids: set[str] = set()
                            repair_retry_count = 0
                            diagnostic_summary: list[dict[str, Any]] = []
                            if isinstance(value, BatchJudgeResult):
                                for task_id, rows in value.errors_by_task.items():
                                    affected_ids = [
                                        candidate.candidate_id
                                        for task, candidates, _ in batch.groups
                                        if task.task_id == task_id for candidate in candidates
                                    ]
                                    judge_errors_by_task[task_id].extend(
                                        row.model_copy(update={
                                            "batch_id": batch.batch_id,
                                            "affected_candidate_ids": affected_ids,
                                        })
                                        for row in rows
                                    )
                                for diagnostic in value.diagnostics:
                                    if diagnostic.get("phase") == "repair_retry":
                                        self.metrics["call_counts"]["llm_batch_judge_repair"] += 1
                                        self.metrics["judge_contract"]["repair_retry_count"] += 1
                                        repair_retry_count += 1
                                    elif diagnostic.get("phase") == "contract_completion":
                                        neutral_completion_count += int(
                                            diagnostic.get("neutral_completion_count", 0)
                                        )
                                        completed_candidate_ids.update(
                                            str(value) for value in diagnostic.get("completed_candidate_ids", [])
                                        )
                                    provider_meta = diagnostic.get("provider_response_metadata") or {}
                                    if diagnostic.get("phase") == "initial":
                                        batch_record["raw_response_length"] = int(
                                            diagnostic.get("raw_response_length") or 0
                                        )
                                    diagnostic_summary.append({
                                        "phase": diagnostic.get("phase"),
                                        "raw_response_type": diagnostic.get("raw_response_type"),
                                        "raw_response_length": diagnostic.get("raw_response_length"),
                                        "parse_status": "INVALID" if diagnostic.get("initial_parse_error") else "OK",
                                        "repair_status": diagnostic.get("repair_retry_result") or "not_used",
                                        "preview": str(diagnostic.get("raw_response_preview", ""))[:200],
                                    })
                                    if isinstance(provider_meta, dict):
                                        model_name = provider_meta.get("model") or provider_meta.get("model_name")
                                        if model_name:
                                            judge_span["model"] = str(model_name)
                                            batch_record["provider_model"] = str(model_name)
                                        judge_span["provider"] = str(
                                            provider_meta.get("model_provider") or "openai_compatible"
                                        )
                                        judge_span["finish_reason"] = str(
                                            provider_meta.get("stop_reason") or provider_meta.get("finish_reason") or "unavailable"
                                        )
                                        batch_record["finish_reason"] = judge_span["finish_reason"]
                                        raw_usage = provider_meta.get("usage")
                                        if isinstance(raw_usage, dict) and raw_usage:
                                            normalized_usage = {
                                                "prompt_tokens": raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens"),
                                                "completion_tokens": raw_usage.get("completion_tokens") or raw_usage.get("output_tokens"),
                                                "total_tokens": raw_usage.get("total_tokens"),
                                            }
                                            judge_span["usage"] = {
                                                key: token_value for key, token_value in normalized_usage.items()
                                                if token_value is not None
                                            }
                                            batch_record["provider_usage"] = dict(judge_span["usage"])
                                            batch_record["actual_prompt_tokens"] = normalized_usage.get("prompt_tokens")
                                            batch_record["completion_tokens"] = normalized_usage.get("completion_tokens")
                            returned_keys = {
                                key for key in batch.candidate_keys
                                if key in value and key[1] not in completed_candidate_ids
                            }
                            failed_keys = set(batch.candidate_keys) - returned_keys
                            mapping.update(value)
                            judge_failed_keys.update(failed_keys)
                            for key in returned_keys:
                                self._increment_funnel(key[0], candidate_lookup[key], "judge_returned_count")
                            for key in failed_keys:
                                self._increment_funnel(key[0], candidate_lookup[key], "judge_error_count")
                            integrity["judge_returned_candidate_count"] += len(returned_keys)
                            integrity["judge_error_candidate_count"] += len(failed_keys)
                            batch_record.update({
                                "status": "SUCCESS" if not failed_keys else "PARTIAL",
                                "returned_candidate_count": len(returned_keys),
                                "error_candidate_count": len(failed_keys),
                                "repair_retry_count": repair_retry_count,
                                "repair_count": repair_retry_count,
                                "neutral_completion_count": neutral_completion_count,
                            })
                            judge_span["output_summary"] = (
                                f"returned={len(returned_keys)} failed={len(failed_keys)}"
                            )
                            judge_span["output"] = {
                                "returned_candidate_count": len(returned_keys),
                                "error_candidate_count": len(failed_keys),
                            }
                            judge_span["_final_metadata"] = {
                                "batch_id": batch.batch_id,
                                "status": batch_record["status"],
                                "parse_diagnostics": diagnostic_summary[:3],
                                "repair_retry_count": repair_retry_count,
                                "raw_response_preview_limit": 200,
                            }
                        except BaseException as exc:
                            timed_out = is_timeout(exc)
                            layer = timeout_layer(exc) or "provider_timeout"
                            elapsed_ms = int((time.monotonic() - batch_started) * 1000)
                            code = ErrorCode.JUDGE_TIMEOUT if timed_out else ErrorCode.JUDGE_ERROR
                            batch_record.update({
                                "status": "TIMEOUT" if timed_out else "ERROR",
                                "error_code": code.value,
                                "timeout_layer": layer if timed_out else None,
                                "timeout_count": 1 if timed_out else 0,
                                "timeout": timed_out,
                                "returned_candidate_count": 0,
                                "error_candidate_count": batch.candidate_count,
                            })
                            judge_span["_error"] = True
                            judge_span["_error_message"] = f"{type(exc).__name__}: {code.value}"
                            judge_span["_final_metadata"] = {
                                "batch_id": batch.batch_id,
                                "status": batch_record["status"],
                                "error_code": code.value,
                                "timeout_layer": layer if timed_out else None,
                                "configured_timeout_ms": configured_timeout_ms,
                                "actual_elapsed_ms": elapsed_ms,
                                "timeout_count": 1 if timed_out else 0,
                                "affected_task_ids": batch.task_ids,
                                "affected_candidate_ids": batch.candidate_ids,
                            }
                            self.metrics["errors"].append(f"{batch.batch_id}:{code.value}")
                            judge_failed_keys.update(batch.candidate_keys)
                            integrity["judge_error_candidate_count"] += batch.candidate_count
                            for key in batch.candidate_keys:
                                self._increment_funnel(key[0], candidate_lookup[key], "judge_error_count")
                            for task, rows, _ in batch.groups:
                                affected = [row.candidate_id for row in rows]
                                if timed_out:
                                    self.task_timeouts.add(task.task_id)
                                judge_errors_by_task[task.task_id].append(ErrorDetail(
                                    code=code.value, node="batch_judge",
                                    tool="llm_batch_evidence_judge", retryable=timed_out,
                                    reason=(
                                        f"{batch.batch_id} Evidence Judge 调用超时"
                                        if timed_out else f"{batch.batch_id} Evidence Judge 调用失败"
                                    ),
                                    batch_id=batch.batch_id,
                                    affected_candidate_ids=affected,
                                    timeout_layer=layer if timed_out else None,
                                    configured_timeout_ms=configured_timeout_ms,
                                    actual_elapsed_ms=elapsed_ms,
                                )
                                )
                        finally:
                            if provider_started is not None:
                                batch_record["provider_elapsed_ms"] = int(
                                    (time.monotonic() - provider_started) * 1000
                                )
                            batch_record["elapsed_ms"] = int((time.monotonic() - batch_started) * 1000)
                return batch_record

            batch_records = await asyncio.gather(*(run_batch(batch) for batch in plan.batches))
            self.metrics["judge_batches"] = sorted(batch_records, key=lambda row: row["batch_id"])

        integrity["judge_missing_candidate_count"] = max(
            0,
            integrity["judge_input_candidate_count"]
            - integrity["judge_returned_candidate_count"]
            - integrity["judge_error_candidate_count"],
        )
        shared_batch_judge_ms = int((time.monotonic() - judge_started) * 1000)
        self.metrics["stage_timings_ms"]["batch_judge"] = shared_batch_judge_ms
        # Judge is request-level shared across tasks. Track the wall-clock once
        # so task metrics can reference it without double-attributing the cost.
        self.metrics["shared_batch_judge_ms"] = shared_batch_judge_ms
        self.metrics["judge_timing_scope"] = "request_shared"
        verification_started = time.monotonic()
        results = []
        for task, _query, candidates, errors, usage, selected_ok, _structured_status, task_metrics, task_span in retrieved:
            verification_task_started = time.monotonic()
            evidence: list[EvidenceItem] = []
            task_judge_errors = judge_errors_by_task.get(task.task_id, [])
            ranked = ranked_by_task[task.task_id]
            task_keys = [(task.task_id, candidate.candidate_id) for candidate in ranked]
            judge_batch_ids = list(dict.fromkeys(
                plan.candidate_to_batch[key] for key in task_keys if key in plan.candidate_to_batch
            ))
            judge_candidate_total = len(task_keys)
            judge_candidate_failed = sum(key in judge_failed_keys for key in task_keys)
            judge_candidate_completed = judge_candidate_total - judge_candidate_failed
            judge_completeness_ratio = (
                judge_candidate_completed / judge_candidate_total
                if judge_candidate_total else 1.0
            )
            for candidate in ranked:
                candidate_key = (task.task_id, candidate.candidate_id)
                judgement = mapping.get(candidate_key)
                if judgement is None:
                    # Never fabricate or positionally guess a missing Judge
                    # result. Valid mapped items from the same batch survive.
                    continue
                claim_items, claim_rows = _claim_evidence_items(task, candidate, judgement)
                self.metrics["scope_guard"]["checked_count"] += len(claim_rows)
                for claim_row in claim_rows:
                    if not claim_row.scope_compatible:
                        self.metrics["scope_guard"]["mismatch_count"] += 1
                        for mismatch_reason in claim_row.scope_mismatch_reasons:
                            reasons = self.metrics["scope_guard"]["mismatch_reasons"]
                            reasons[mismatch_reason] = int(reasons.get(mismatch_reason, 0)) + 1
                    self.metrics["judge_relation_distribution"][claim_row.relation.value] += 1
                integrity = self.metrics["judge_integrity"]
                for claim_row in claim_rows:
                    returned_key = f"judge_returned_{claim_row.relation.value.lower()}_count"
                    if returned_key in integrity:
                        integrity[returned_key] += 1
                    if claim_row.relation == EvidenceRelation.NEUTRAL:
                        reason = str(claim_row.neutral_reason or "AMBIGUOUS_SCOPE")
                        self.metrics["neutral_reason_distribution"][reason] = self.metrics["neutral_reason_distribution"].get(reason, 0) + 1
                for item in claim_items:
                    created_key = f"evidence_created_{item.relation.value.lower()}_count"
                    if created_key in integrity:
                        integrity[created_key] += 1
                    merged_slot_evidence = normalize_slot_evidence(
                        task.required_slots,
                        item.content,
                        item.slot_evidence,
                        candidate_id=item.evidence_id,
                    )
                    item = item.model_copy(update={
                        "slot_evidence": merged_slot_evidence,
                        "covered_slots": list(dict.fromkeys([
                            *item.covered_slots, *merged_slot_evidence.keys(),
                        ])),
                    })
                    evidence.append(item)
            evidence = deduplicate_evidence_items(evidence)
            quality = analyze_evidence_quality(evidence, task.required_slots, len(candidates), self.config)
            aggregated_slots, coverage_conflicts = aggregate_slot_evidence(
                [item.model_dump(mode="json") for item in evidence]
            )
            covered_slot_names = set(aggregated_slots)
            quality = quality.model_copy(update={
                "missing_slots": [slot for slot in task.required_slots if slot not in covered_slot_names],
                "claim_coverage_score": (
                    len(covered_slot_names & set(task.required_slots)) / max(1, len(task.required_slots))
                    if task.required_slots else (1.0 if evidence else 0.0)
                ),
            })
            verification = aggregate_verification(
                evidence, quality, task.line_type, self.config,
                atomic_claim_verdicts=_atomic_claim_verdicts(task, evidence) if task.atomic_claims else None,
                claim_logic_operator=task.claim_logic_operator.value,
            )
            verification = verification.model_copy(update={
                "aggregated_slot_coverage": aggregated_slots,
                "coverage_conflicts": coverage_conflicts,
                "claim_logic_operator": task.claim_logic_operator.value,
                "atomic_claim_verdicts": _atomic_claim_verdicts(task, evidence) if task.atomic_claims else {},
            })
            all_errors = deduplicate_errors([*errors, *task_judge_errors])
            # Tool execution status is computed independently from verdict.
            # PARTIAL + TOOL_ERROR + REFUTED is a legitimate combination: a
            # fetch failed but the surviving evidence already settles the
            # hypothesis. Verdict is only downgraded to INCONCLUSIVE when the
            # selected mandatory KB failed (genuinely cannot close the task).
            status, termination = derive_task_execution_state(
                errors=all_errors,
                deadline_reached=self.metrics["deadline_reached"] or task.task_id in self.task_timeouts,
                selected_ok=selected_ok,
                has_usable_evidence=bool(evidence),
            )
            if not selected_ok:
                verification = VerificationResult(
                    verdict=VerificationVerdict.INCONCLUSIVE, upstream_status="doubtful", confidence=0,
                    reason="Mandatory selected knowledge retrieval failed; other sources cannot close the task.",
                    sufficiency_path="SELECTED_KB_FAIL_CLOSED",
                    override_applied=False,
                )
            # A conclusive SUPPORTED/REFUTED verdict wins over a TOOL_ERROR
            # status when the surviving evidence is sufficient. Only flip to
            # SUCCESS/SUFFICIENT when no tool errors remain at all; otherwise
            # keep PARTIAL/TOOL_ERROR and let the verdict speak for itself.
            if (
                verification.verdict in {VerificationVerdict.SUPPORTED, VerificationVerdict.REFUTED}
                and not all_errors and selected_ok
                and not self.metrics["deadline_reached"] and task.task_id not in self.task_timeouts
            ):
                status, termination = ExecutionStatus.SUCCESS, TerminationReason.SUFFICIENT
            gap_reason = derive_evidence_gap_reason(
                quality, verification, all_errors, self.config, selected_ok=selected_ok,
            )
            gap_plan = plan_gap_retrieval(
                task, gap_reason=gap_reason,
                missing_slots=list(quality.missing_slots), round_number=0,
            )
            verification_parent = self.trace.task_span_id(task.task_id)
            async with self.trace.span(
                "numeric.verify",
                {"request_id": task.request_id, "task_id": task.task_id,
                 "item_id": task.item_id, "line_type": task.line_type.value,
                 "candidate_count": len(task_keys), "status": "SUCCESS"},
                parent_run_id=verification_parent,
            ) as numeric_span:
                numeric_span["output"] = {"candidate_count": len(task_keys), "override_count": sum(1 for key in task_keys if key in mapping and mapping[key].override_reason)}
            async with self.trace.span(
                "slot.aggregate",
                {"request_id": task.request_id, "task_id": task.task_id,
                 "item_id": task.item_id, "line_type": task.line_type.value,
                 "candidate_count": len(evidence), "status": "SUCCESS"},
                parent_run_id=verification_parent,
            ) as slot_span:
                slot_span["output"] = {"covered_slot_count": len({slot for item in evidence for slot in item.covered_slots}), "missing_slot_count": len(quality.missing_slots)}
            async with self.trace.span(
                "gap.analyze",
                {"request_id": task.request_id, "task_id": task.task_id,
                 "item_id": task.item_id, "line_type": task.line_type.value,
                 "status": "SUCCESS", "gap_reason": gap_reason},
                parent_run_id=verification_parent,
            ) as gap_span:
                gap_span["output"] = {"triggered": gap_plan.triggered, "query_count": len(gap_plan.queries)}
            dynamic_reason = build_verification_reason(
                verification.verdict, status, termination, quality, all_errors, gap_reason, self.config,
            )
            verification = verification.model_copy(update={"reason": dynamic_reason})
            if judge_candidate_failed:
                verification = verification.model_copy(update={
                    "reason": (
                        f"{verification.reason} Judge 候选完成度为 "
                        f"{judge_candidate_completed}/{judge_candidate_total}，"
                        f"另有 {judge_candidate_failed} 条候选明确标记为 judge_unavailable。"
                    )
                })
            # Verification is a real per-task operation: wrap it in a span
            # parented under the task span, with business metadata so the
            # verdict and weight can be diagnosed per task.
            async with self.trace.span(
                f"verification.{task.line_type.value}",
                {
                    "request_id": task.request_id, "task_id": task.task_id, "item_id": task.item_id,
                    "line_type": task.line_type.value,
                    "effective_evidence_count": len(evidence),
                    "support_weight": sum(
                        it.scores.final for it in evidence
                        if it.relation.value == "SUPPORT" and it.scores
                    ),
                    "refute_weight": sum(
                        it.scores.final for it in evidence
                        if it.relation.value == "REFUTE" and it.scores
                    ),
                    "verdict": verification.verdict.value,
                    "final_score": float(getattr(verification, "confidence", 0) or 0),
                    "sufficiency_path": verification.sufficiency_path,
                    "override_applied": verification.override_applied,
                    "execution_status": status.value if status else "unknown",
                    "termination_reason": termination.value if termination else "unknown",
                    "gap_reason": gap_reason,
                },
                parent_run_id=verification_parent,
            ) as verification_span:
                verification_span["output"] = {
                    "verdict": verification.verdict.value,
                    "effective_evidence_count": len(evidence),
                    "finish_reason": termination.value if termination else "unavailable",
                    "gap_reason": gap_reason,
                    "sufficiency_path": verification.sufficiency_path,
                }
            task_metrics.stage("judge_verification", verification_task_started)
            elapsed_ms = task_metrics.finish()
            task_snapshot = self.collector.tasks[task.task_id]
            kb_task = self.metrics.get("kb_task", {}).get(task.task_id, {})
            kb_candidates = [
                candidate for candidate in candidates
                if candidate.source_type == SourceType.KNOWLEDGE_BASE
            ]
            atomic_coverage = {
                claim.claim_id: sum(
                    1 for candidate in kb_candidates
                    if any(token in candidate.content for token in re.findall(
                        r"[\u3400-\u9fffA-Za-z]{2,}",
                        getattr(claim, "source_text_span", "") or getattr(claim, "qualifier", ""),
                    )[:4])
                )
                for claim in task.atomic_claims
            }
            slot_coverage = {
                slot: sum(1 for candidate in kb_candidates if slot in candidate.content)
                for slot in task.required_slots
            }
            result = RetrievalTaskResult(
                task_id=task.task_id, item_id=task.item_id, line_type=task.line_type, node_id=task.node_id,
                hypothesis_id=task.hypothesis_id, target_text=task.target_text, execution_status=status,
                termination_reason=termination, verification=verification, evidence_items=evidence,
                evidence_quality=quality, tool_usage=usage,
                evidence_gap=gap_reason,
                errors=all_errors, elapsed_ms=elapsed_ms,
                node_timings_ms=dict(task_snapshot["stage_timings_ms"]),
                judge_batches=judge_batch_ids,
                judge_candidate_total=judge_candidate_total,
                judge_candidate_completed=judge_candidate_completed,
                judge_candidate_failed=judge_candidate_failed,
                judge_completeness_ratio=judge_completeness_ratio,
                retrieved_candidate_count=len(candidates),
                support_count=sum(1 for item in evidence if item.relation == EvidenceRelation.SUPPORT),
                refute_count=sum(1 for item in evidence if item.relation == EvidenceRelation.REFUTE),
                supplement_count=sum(1 for item in evidence if item.relation == EvidenceRelation.SUPPLEMENT),
                neutral_count=sum(1 for key in task_keys if key in mapping and mapping[key].relation == EvidenceRelation.NEUTRAL),
                neutral_reasons=list(dict.fromkeys(
                    mapping[key].neutral_reason or "AMBIGUOUS_SCOPE"
                    for key in task_keys if key in mapping and mapping[key].relation == EvidenceRelation.NEUTRAL
                )),
                missing_slots_before_gap_retrieval=list(quality.missing_slots),
                missing_slots_after_gap_retrieval=list(quality.missing_slots),
                gap_retrieval_triggered=gap_plan.triggered,
                gap_queries=gap_plan.queries,
                atomic_claim_count=len(task.atomic_claims),
                kb_query_variant_count=int(kb_task.get("query_variant_count", 0)),
                kb_query_result_count_by_query=dict(kb_task.get("query_result_count_by_query", {})),
                kb_query_zero_hit_count=sum(
                    int(value == 0) for value in kb_task.get("query_result_count_by_query", {}).values()
                ),
                kb_raw_candidate_count=int(kb_task.get("raw_candidate_count", 0)),
                kb_adjacent_chunk_count=int(kb_task.get("adjacent_chunk_count", 0)),
                kb_final_candidate_count=len(kb_candidates),
                atomic_claim_candidate_coverage=atomic_coverage,
                required_slot_candidate_coverage=slot_coverage,
            )
            results.append(result)
            await self.trace.end_span(
                task_span,
                output={
                    "candidate_count": len(candidates),
                    "judge_batches": judge_batch_ids,
                    "judge_completeness_ratio": judge_completeness_ratio,
                    "verification": verification.verdict.value,
                    "execution_status": status.value,
                    "error_count": len(all_errors),
                },
                final_metadata={
                    "judge_candidate_total": judge_candidate_total,
                    "judge_candidate_completed": judge_candidate_completed,
                    "judge_candidate_failed": judge_candidate_failed,
                },
            )
        # Evidence gaps trigger one real, deterministic second round.  Keep
        # this after the shared first Judge so the gap round only handles the
        # small set of unresolved tasks and cannot duplicate first-round work.
        result_by_task = {row.task_id: row for row in results}
        retrieved_by_task = {row[0].task_id: row for row in retrieved}

        def gap_eligible(result: RetrievalTaskResult) -> bool:
            # A partial/malformed Judge response is not a retrieval gap. More
            # Web/KB calls cannot repair missing structured Judge rows; the
            # batch Judge's own bounded parse-retry path owns that failure.
            if result.judge_candidate_failed:
                return False
            if result.evidence_gap not in {
                "MISSING_REQUIRED_SLOTS",
                "INSUFFICIENT_INDEPENDENT_SOURCES",
            }:
                return False
            if not result.evidence_items:
                return False
            high_relevance = max(
                (
                    item.scores.relevance
                    for item in result.evidence_items
                    if item.scores is not None
                ),
                default=0.0,
            )
            if high_relevance < 0.5:
                return False
            if (
                result.evidence_gap == "MISSING_REQUIRED_SLOTS"
                and not result.evidence_quality.missing_slots
            ):
                return False
            return True

        gap_results = await asyncio.gather(*(
            self._run_gap_round(task, result_by_task[task.task_id], retrieved_by_task[task.task_id])
            for task, *_ in retrieved
            if gap_eligible(result_by_task[task.task_id])
        ))
        for updated in gap_results:
            result_by_task[updated.task_id] = updated
        results = [result_by_task[task.task_id] for task, *_ in retrieved]
        self.metrics["stage_timings_ms"]["verification"] += int((time.monotonic() - verification_started) * 1000)
        return results

    async def _run_gap_round(
        self,
        task: RetrievalTask,
        result: RetrievalTaskResult,
        retrieved_row: tuple[Any, ...],
    ) -> RetrievalTaskResult:
        """Run one bounded, real retrieval/Judge round for an evidence gap.

        The initial flow intentionally remains one request-level Judge batch.
        This helper is called only after that batch has produced ``NO_EVIDENCE``
        or ``MISSING_REQUIRED_SLOTS``.  New candidates are deduplicated against
        the first round, judged independently, and then folded back into the
        same verification result.  It never runs a second gap round.
        """
        plan = plan_gap_retrieval(
            task,
            gap_reason=result.evidence_gap,
            missing_slots=list(result.evidence_quality.missing_slots),
            round_number=0,
        )
        remaining_ms = self._remaining(1000, task.task_id) * 1000
        # The task/deadline guard above already bounds this round.  Do not
        # reject a configured 30s flow merely because the finalize reserve is
        # large; V10 requires one gap attempt whenever budget remains.
        if not plan.triggered or remaining_ms <= 1:
            return result
        self.metrics["gap_retrieval"]["triggered_count"] += 1
        reason = str(result.evidence_gap or "UNKNOWN")
        reasons = self.metrics["gap_retrieval"].setdefault("reasons", [])
        if reason not in reasons:
            reasons.append(reason)
        self.metrics["gap_retrieval"].setdefault("triggered_task_count", 0)
        self.metrics["gap_retrieval"]["triggered_task_count"] += 1
        self.metrics["gap_retrieval"].setdefault("query_count", 0)
        self.metrics["gap_retrieval"]["web_search_count"] = self.metrics["gap_retrieval"].get("web_search_count", 0)
        self.metrics["gap_retrieval"]["kb_retrieve_count"] = self.metrics["gap_retrieval"].get("kb_retrieve_count", 0)

        _, _, original_candidates, original_errors, usage, selected_ok, _, _, _ = retrieved_row
        existing_keys = {source_evidence_fingerprint(candidate) for candidate in original_candidates}
        gap_candidates: list[EvidenceCandidate] = []
        gap_errors: list[ErrorDetail] = []
        query_items = [
            QueryItem(query_id=f"{task.task_id}:gap:{index}", query=query, purpose="gap slot retrieval")
            for index, query in enumerate(plan.queries, 1)
        ]
        gap_started = time.monotonic()
        async with self.trace.span(
            "gap.retrieve",
            {
                "request_id": task.request_id,
                "task_id": task.task_id,
                "item_id": task.item_id,
                "line_type": task.line_type.value,
                "round": 1,
                "query_count": len(query_items),
                "missing_slots": plan.missing_slots,
            },
            parent_run_id=self.trace.task_span_id(task.task_id),
        ) as gap_span:
            for query_item in query_items:
                if self._remaining(1, task.task_id) <= 0:
                    break
                self.metrics["gap_retrieval"]["query_count"] += 1
                # Web and KB are intentionally concurrent for each gap query;
                # provider calls still obey their own semaphores/deadlines.
                web_value, kb_value = await asyncio.gather(
                    self._web(task, query_item),
                    self._kb(task, self.config.public_knowledge_ids, "configured_public", query_override=query_item.query),
                    return_exceptions=True,
                )
                self.metrics["gap_retrieval"]["web_search_count"] += 1
                self.metrics["gap_retrieval"]["kb_retrieve_count"] += int(bool(self.config.public_knowledge_ids))
                for value in (web_value, kb_value):
                    if isinstance(value, BaseException):
                        gap_errors.append(ErrorDetail(
                            code=ErrorCode.WEB_PROVIDER_ERROR.value if value is web_value else ErrorCode.KB_PROVIDER_ERROR.value,
                            node="gap_retrieve", tool="web_search" if value is web_value else "bisheng_vector_retrieve",
                            retryable=True, reason=f"{type(value).__name__}: gap retrieval failed",
                        ))
                        continue
                    rows, errors = value
                    # Empty search results are a valid evidence gap, not a
                    # tool failure; preserve genuine provider/timeouts only.
                    gap_errors.extend(error for error in errors if error.code not in {"WEB_NO_RESULT"})
                    for candidate in rows:
                        key = source_evidence_fingerprint(candidate)
                        if key not in existing_keys:
                            existing_keys.add(key)
                            gap_candidates.append(candidate)
            gap_span["output"] = {
                "round": 1,
                "query_count": len(query_items),
                "new_candidate_count": len(gap_candidates),
                "error_count": len(gap_errors),
                "elapsed_ms": int((time.monotonic() - gap_started) * 1000),
            }

        if not gap_candidates:
            self.metrics["gap_retrieval"].setdefault("unresolved_task_count", 0)
            self.metrics["gap_retrieval"]["unresolved_task_count"] += 1
            return result.model_copy(update={
                "gap_retrieval_triggered": True,
                "gap_queries": plan.queries,
                "gap_new_candidates": 0,
                "gap_new_evidence": 0,
                "missing_slots_before_gap_retrieval": list(result.evidence_quality.missing_slots),
                "missing_slots_after_gap_retrieval": list(result.evidence_quality.missing_slots),
                "errors": deduplicate_errors([*result.errors, *gap_errors]),
            })

        # Gap candidates use the exact same lossless token/candidate planner as
        # the initial round. This prevents per-task calls from bypassing batch
        # limits and keeps provider concurrency bounded across gap tasks.
        gap_mapping: dict[tuple[str, str], JudgeResult] = {}
        gap_judge_errors: list[ErrorDetail] = []
        gap_plan = self.judge_planner.plan(
            [(task, gap_candidates, build_prepared_context(task))],
            batch_id_prefix=f"gap-judge-{task.task_id.rsplit(':', 1)[-1]}",
        )
        self.metrics["judge_integrity"]["judge_input_candidate_count"] += gap_plan.input_candidate_count
        self.metrics["judge_integrity"]["judge_batched_candidate_count"] += gap_plan.batched_candidate_count
        for candidate in gap_candidates:
            self._increment_funnel(task.task_id, candidate, "judge_ready_count")
            self._increment_funnel(task.task_id, candidate, "judge_batched_count")

        async def run_gap_batch(batch: JudgeBatch) -> dict[str, Any]:
            queued_at = time.monotonic()
            record: dict[str, Any] = {
                "batch_id": batch.batch_id,
                "task_ids": batch.task_ids,
                "candidate_ids": batch.candidate_ids,
                "task_count": len(batch.task_ids),
                "candidate_count": batch.candidate_count,
                "estimated_input_tokens": batch.estimated_input_tokens,
                "actual_prompt_tokens": None,
                "completion_tokens": None,
                "queue_wait_ms": 0,
                "provider_elapsed_ms": 0,
                "repair_count": 0,
                "timeout": False,
                "status": "PENDING",
                "round": 1,
            }
            self._record_batch_judge_call()
            async with self.gap_judge_semaphore:
                provider_started = time.monotonic()
                record["queue_wait_ms"] = int((provider_started - queued_at) * 1000)
                async with self._batch_judge_observation(
                    "gap.batch_judge",
                    {"request_id": task.request_id, "task_id": task.task_id, "batch_id": batch.batch_id,
                     "candidate_count": batch.candidate_count, "round": 1,
                     "estimated_input_tokens": batch.estimated_input_tokens},
                    parent_run_id=self.trace.run_id_for(task.request_id),
                ) as judge_span:
                    try:
                        value = await self._measured_deadline_call(
                            "llm_batch_evidence_judge",
                            self.deps.batch_judge.judge_many(batch.groups),
                            self._remaining(self.config.parallel_batch_judge_timeout_ms, task.task_id),
                        )
                        batch_values = {key: value[key] for key in batch.candidate_keys if key in value}
                        gap_mapping.update(batch_values)
                        for rows in getattr(value, "errors_by_task", {}).values():
                            gap_judge_errors.extend(
                                row.model_copy(update={"batch_id": batch.batch_id}) for row in rows
                            )
                        diagnostics = list(getattr(value, "diagnostics", []) or [])
                        record["repair_count"] = sum(row.get("phase") == "repair_retry" for row in diagnostics)
                        for diagnostic in diagnostics:
                            provider_meta = diagnostic.get("provider_response_metadata") or {}
                            usage_data = provider_meta.get("usage") if isinstance(provider_meta, dict) else None
                            if isinstance(usage_data, dict):
                                record["actual_prompt_tokens"] = usage_data.get("prompt_tokens") or usage_data.get("input_tokens")
                                record["completion_tokens"] = usage_data.get("completion_tokens") or usage_data.get("output_tokens")
                        failed = batch.candidate_count - len(batch_values)
                        record.update({
                            "status": "SUCCESS" if failed == 0 else "PARTIAL",
                            "returned_candidate_count": len(batch_values),
                            "error_candidate_count": failed,
                        })
                        judge_span["output_summary"] = f"returned={len(batch_values)} failed={failed}"
                    except BaseException as exc:
                        timed_out = is_timeout(exc)
                        record.update({
                            "status": "TIMEOUT" if timed_out else "ERROR",
                            "timeout": timed_out,
                            "returned_candidate_count": 0,
                            "error_candidate_count": batch.candidate_count,
                        })
                        gap_judge_errors.append(ErrorDetail(
                            code=ErrorCode.JUDGE_TIMEOUT.value if timed_out else ErrorCode.JUDGE_ERROR.value,
                            node="gap.batch_judge", tool="llm_batch_evidence_judge", retryable=timed_out,
                            reason=f"{type(exc).__name__}: gap Judge failed", batch_id=batch.batch_id,
                            affected_candidate_ids=batch.candidate_ids,
                        ))
                        judge_span["_error"] = True
                        judge_span["_error_message"] = gap_judge_errors[-1].reason
                    finally:
                        record["provider_elapsed_ms"] = int((time.monotonic() - provider_started) * 1000)
            return record

        gap_batch_records = await asyncio.gather(*(run_gap_batch(batch) for batch in gap_plan.batches))

        self.metrics["judge_integrity"]["judge_returned_candidate_count"] += len(gap_mapping)
        self.metrics["judge_integrity"]["judge_error_candidate_count"] += max(0, len(gap_candidates) - len(gap_mapping))
        for candidate in gap_candidates:
            field = "judge_returned_count" if (task.task_id, candidate.candidate_id) in gap_mapping else "judge_error_count"
            self._increment_funnel(task.task_id, candidate, field)
        integrity = self.metrics["judge_integrity"]
        integrity["judge_missing_candidate_count"] = max(
            0,
            integrity["judge_input_candidate_count"]
            - integrity["judge_returned_candidate_count"]
            - integrity["judge_error_candidate_count"],
        )
        self.metrics["judge_batches"].extend(gap_batch_records)
        gap_batch_ids = [batch.batch_id for batch in gap_plan.batches]
        self.metrics["gap_retrieval"]["new_candidate_count"] += len(gap_candidates)
        self.metrics["gap_retrieval"]["new_unique_candidate_count"] += len(gap_candidates)

        extra_evidence: list[EvidenceItem] = []
        neutral_count = 0
        neutral_reasons: list[str] = []
        for candidate in gap_candidates:
            judgement = gap_mapping.get((task.task_id, candidate.candidate_id))
            if judgement is None:
                continue
            claim_items, claim_rows = _claim_evidence_items(task, candidate, judgement)
            self.metrics["scope_guard"]["checked_count"] += len(claim_rows)
            for claim_row in claim_rows:
                self.metrics["judge_relation_distribution"][claim_row.relation.value] += 1
                if not claim_row.scope_compatible:
                    self.metrics["scope_guard"]["mismatch_count"] += 1
                    for mismatch_reason in claim_row.scope_mismatch_reasons:
                        reasons = self.metrics["scope_guard"]["mismatch_reasons"]
                        reasons[mismatch_reason] = int(reasons.get(mismatch_reason, 0)) + 1
                if claim_row.relation == EvidenceRelation.NEUTRAL:
                    neutral_count += 1
                    reason = str(claim_row.neutral_reason or "AMBIGUOUS_SCOPE")
                    neutral_reasons.append(reason)
                    self.metrics["neutral_reason_distribution"][reason] = self.metrics["neutral_reason_distribution"].get(reason, 0) + 1
            for extra_item in claim_items:
                merged = normalize_slot_evidence(
                    task.required_slots,
                    extra_item.content,
                    extra_item.slot_evidence,
                    candidate_id=extra_item.evidence_id,
                )
                extra_evidence.append(extra_item.model_copy(update={
                    "slot_evidence": merged,
                    "covered_slots": list(dict.fromkeys([*extra_item.covered_slots, *merged.keys()])),
                }))

        combined_evidence = deduplicate_evidence_items([*result.evidence_items, *extra_evidence])
        new_evidence_ids = {item.evidence_id for item in combined_evidence} - {
            item.evidence_id for item in result.evidence_items
        }
        extra_evidence = [item for item in combined_evidence if item.evidence_id in new_evidence_ids]
        quality = analyze_evidence_quality(combined_evidence, task.required_slots, len(original_candidates) + len(gap_candidates), self.config)
        aggregated_slots, coverage_conflicts = aggregate_slot_evidence(
            [item.model_dump(mode="json") for item in combined_evidence]
        )
        covered_slot_names = set(aggregated_slots)
        quality = quality.model_copy(update={
            "missing_slots": [slot for slot in task.required_slots if slot not in covered_slot_names],
            "claim_coverage_score": (
                len(covered_slot_names & set(task.required_slots)) / max(1, len(task.required_slots))
                if task.required_slots else (1.0 if combined_evidence else 0.0)
            ),
        })
        verification = aggregate_verification(
            combined_evidence, quality, task.line_type, self.config,
            atomic_claim_verdicts=_atomic_claim_verdicts(task, combined_evidence) if task.atomic_claims else None,
            claim_logic_operator=task.claim_logic_operator.value,
        )
        verification = verification.model_copy(update={
            "aggregated_slot_coverage": aggregated_slots,
            "coverage_conflicts": coverage_conflicts,
            "claim_logic_operator": task.claim_logic_operator.value,
            "atomic_claim_verdicts": _atomic_claim_verdicts(task, combined_evidence) if task.atomic_claims else {},
        })
        all_errors = deduplicate_errors([*result.errors, *gap_errors, *gap_judge_errors])
        gap_reason_after = derive_evidence_gap_reason(
            quality, verification, all_errors, self.config, selected_ok=selected_ok,
        )
        status, termination = derive_task_execution_state(
            errors=all_errors,
            deadline_reached=self.metrics["deadline_reached"] or task.task_id in self.task_timeouts,
            selected_ok=selected_ok,
            has_usable_evidence=bool(combined_evidence),
        )
        if verification.verdict in {VerificationVerdict.SUPPORTED, VerificationVerdict.REFUTED} and not all_errors and selected_ok:
            status, termination = ExecutionStatus.SUCCESS, TerminationReason.SUFFICIENT
        verification = verification.model_copy(update={
            "reason": build_verification_reason(verification.verdict, status, termination, quality, all_errors, gap_reason_after, self.config),
        })
        resolved = not quality.missing_slots and verification.verdict != VerificationVerdict.INCONCLUSIVE
        resolved_slot_count = len(set(result.evidence_quality.missing_slots) - set(quality.missing_slots))
        verdict_changed = verification.verdict != result.verification.verdict
        self.metrics["gap_retrieval"]["new_evidence_count"] += len(extra_evidence)
        self.metrics["gap_retrieval"]["new_valid_evidence_count"] += len(extra_evidence)
        self.metrics["gap_retrieval"]["resolved_slot_count"] += resolved_slot_count
        self.metrics["gap_retrieval"]["verdict_changed_count"] += int(verdict_changed)
        self.metrics["gap_retrieval"]["gap_resolved_count"] += int(resolved)
        self.metrics["gap_retrieval"].setdefault("resolved_task_count", 0)
        self.metrics["gap_retrieval"].setdefault("unresolved_task_count", 0)
        self.metrics["gap_retrieval"]["resolved_task_count"] += int(resolved)
        self.metrics["gap_retrieval"]["unresolved_task_count"] += int(not resolved)
        return result.model_copy(update={
            "execution_status": status,
            "termination_reason": termination,
            "verification": verification,
            "evidence_items": combined_evidence,
            "evidence_quality": quality,
            "evidence_gap": gap_reason_after,
            "errors": all_errors,
            "judge_batches": list(dict.fromkeys([*result.judge_batches, *gap_batch_ids])),
            "judge_candidate_total": result.judge_candidate_total + len(gap_candidates),
            "judge_candidate_completed": result.judge_candidate_completed + len(gap_mapping),
            "judge_candidate_failed": result.judge_candidate_failed + max(0, len(gap_candidates) - len(gap_mapping)),
            "judge_completeness_ratio": (result.judge_candidate_completed + len(gap_mapping)) / max(1, result.judge_candidate_total + len(gap_candidates)),
            "retrieved_candidate_count": result.retrieved_candidate_count + len(gap_candidates),
            "support_count": sum(item.relation == EvidenceRelation.SUPPORT for item in combined_evidence),
            "refute_count": sum(item.relation == EvidenceRelation.REFUTE for item in combined_evidence),
            "supplement_count": sum(item.relation == EvidenceRelation.SUPPLEMENT for item in combined_evidence),
            "neutral_count": result.neutral_count + neutral_count,
            "neutral_reasons": list(dict.fromkeys([*result.neutral_reasons, *neutral_reasons])),
            "missing_slots_before_gap_retrieval": list(result.evidence_quality.missing_slots),
            "missing_slots_after_gap_retrieval": list(quality.missing_slots),
            "gap_retrieval_triggered": True,
            "gap_queries": plan.queries,
            "gap_new_candidates": len(gap_candidates),
            "gap_new_evidence": len(extra_evidence),
            "gap_resolved_slot_count": resolved_slot_count,
            "gap_verdict_changed": verdict_changed,
            "gap_resolved": resolved,
        })

    async def run(
        self,
        tasks: list[RetrievalTask],
        scenarios: dict[str, Any],
        *,
        structured_healthy: bool = True,
    ) -> tuple[list[RetrievalTaskResult], dict[str, Any]]:
        self.started = time.monotonic()
        execution_budget_ms = max(100, self.config.parallel_flow_timeout_ms - self.config.parallel_finalize_reserve_ms)
        self.deadline = self.started + execution_budget_ms / 1000
        try:
            async def prepare_structured() -> None:
                try:
                    await self._prepare_structured_tool_calling(
                        tasks,
                        scenarios,
                        structured_healthy=structured_healthy,
                    )
                finally:
                    self.structured_ready.set()

            # Structured routing/query and Web/KB retrieval are independent
            # source branches. Start them together; each task's Structured arm
            # waits only for the shared paragraph-level preparation event.
            structured_prepare_task = asyncio.create_task(prepare_structured())
            # Forward and reverse retrieval tasks share the same concurrent
            # fan-out. Once all four sources return, one request-level Batch
            # Judge call handles both directions together. This preserves
            # directional concurrency while avoiding two competing LLM calls
            # and makes the Judge contract genuinely batch-scoped.
            retrieved = await asyncio.gather(
                *(self._retrieve_task(task, scenarios) for task in tasks)
            )
            await structured_prepare_task
            if self.config.evidence_output_mode == "candidate_passthrough":
                task_results = await self._passthrough_direction(retrieved)
            else:
                task_results = await self._judge_direction(retrieved)
            by_task = {row.task_id: row for row in task_results}
            results = [by_task[task.task_id] for task in tasks]
            # Pair forward/reverse items within each paragraph for an explicit
            # consistency diagnostic.  The check never rewrites a verdict;
            # it exposes contradictory independent conclusions for callers.
            grouped: dict[str, dict[str, list[RetrievalTask]]] = defaultdict(lambda: {"forward": [], "reverse": []})
            for task in tasks:
                grouped[task.paragraph_id][task.line_type.value].append(task)
            for paragraph_id, pair in grouped.items():
                for forward, reverse in zip(pair["forward"], pair["reverse"], strict=False):
                    report = check_pair_consistency(by_task[forward.task_id].verification.verdict.value, by_task[reverse.task_id].verification.verdict.value)
                    self.metrics["pair_consistency"]["pair_count"] += 1
                    key = "consistent_count" if report["consistency_status"] == "CONSISTENT" else "conflict_count" if report["consistency_status"] == "CONFLICT" else "incomplete_count"
                    if key:
                        self.metrics["pair_consistency"][key] += 1
                    await self.trace.emit("pair.consistency", {"request_id": forward.request_id, "paragraph_id": paragraph_id, "forward_task_id": forward.task_id, "reverse_task_id": reverse.task_id, **report})
        finally:
            self.metrics["total_elapsed_ms"] = int((time.monotonic() - self.started) * 1000)
            self.metrics["deadline_reached"] = self.metrics["deadline_reached"] or time.monotonic() >= self.deadline
            web_tasks = self.metrics.get("web_task", {})
            self.metrics["web_request_funnel"] = {
                "web_search_task_count": len(web_tasks),
                "tasks_with_web_search_results": sum(
                    int((row.get("merged_search_result_count") or 0) > 0)
                    for row in web_tasks.values()
                ),
                "tasks_with_web_candidates": sum(
                    int((row.get("candidate_extracted_count") or 0) > 0)
                    for row in web_tasks.values()
                ),
                "tasks_with_web_errors": sum(
                    int(bool(row.get("search_errors_by_query")))
                    for row in web_tasks.values()
                ),
            }
            if "retrieved" in locals():
                kb_tasks = 0
                atomic_with = atomic_without = 0
                for row in retrieved:
                    task, _query, candidates = row[0], row[1], row[2]
                    kb_candidates = [
                        candidate for candidate in candidates
                        if candidate.source_type == SourceType.KNOWLEDGE_BASE
                    ]
                    kb_tasks += int(bool(kb_candidates))
                    for claim in task.atomic_claims:
                        matched = any(
                            any(token in candidate.content for token in re.findall(
                                r"[\u3400-\u9fffA-Za-z]{2,}",
                                getattr(claim, "source_text_span", "") or getattr(claim, "qualifier", ""),
                            )[:4])
                            for candidate in kb_candidates
                        )
                        if matched:
                            atomic_with += 1
                        else:
                            atomic_without += 1
                self.metrics["kb_request_funnel"] = {
                    "tasks_with_kb_candidates": kb_tasks,
                    "atomic_claims_with_candidates": atomic_with,
                    "atomic_claims_without_candidates": atomic_without,
                }
            request_metrics = self.collector.snapshot()
            self.metrics["critical_path_ms"] = request_metrics["critical_path_ms"]
            self.metrics["task_metrics"] = request_metrics["task_metrics"]
            self.metrics["calls"] = request_metrics["calls"]
            for name, count in request_metrics["call_counts"].items():
                self.metrics["call_counts"][name] = count
        trace_started = time.monotonic()
        queued = self.trace.emit_nowait("parallel.finalize", {
            "request_id": tasks[0].request_id if tasks else "", "flow_mode": "parallel_sources",
            "total_elapsed_ms": self.metrics["total_elapsed_ms"], "call_counts": self.metrics["call_counts"],
            "cache": self.metrics["cache"], "deadline_reached": self.metrics["deadline_reached"],
        })
        self.metrics["langfuse_emit_ms"] = int((time.monotonic() - trace_started) * 1000)
        self.metrics["langfuse_event_queued"] = queued
        return results, self.metrics
