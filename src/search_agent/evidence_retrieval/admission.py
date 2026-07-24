"""证据准入的稳定、可审计结论。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .config import EvidenceRetrievalConfig
from .schemas import EvidenceItem, EvidenceRelation


class AdmissionReason(StrEnum):
    ADMITTED = "ADMITTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOW_DIRECTNESS = "LOW_DIRECTNESS"
    RELATION = "RELATION"
    SCOPE_OR_QUOTE = "SCOPE_OR_QUOTE"
    INCOMPLETE_FACT = "INCOMPLETE_FACT"
    DUPLICATE = "DUPLICATE"
    OTHER_POLICY = "OTHER_POLICY"


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reasons: tuple[AdmissionReason, ...]

    @property
    def primary_reason(self) -> AdmissionReason:
        return self.reasons[0]


_BLOCKING_SCOPE_REASONS = {
    "WRONG_YEAR", "WRONG_TIME_SCOPE", "WRONG_REGION", "WRONG_REGION_SCOPE",
    "WRONG_SUBJECT", "WRONG_ENTITY", "WRONG_SUBJECT_SCOPE", "WRONG_METRIC",
    "WRONG_METRIC_SCOPE", "WRONG_MARKET_SCOPE", "WRONG_STATISTICAL_SCOPE",
    "INCOMPARABLE_UNIT", "QUOTE_NOT_FOUND",
}
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_MEASUREMENT = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:万亿|千亿|百亿|亿|万|千)?\s*(?:美元|人民币|元|台|人|家|个|%|％|个百分点|倍)(?![\w%％])", re.I)
_METRICS = ("市场规模", "出货量", "同比", "环比", "增长率", "复合增长", "CAGR", "份额", "占比", "保有量", "招聘", "接收人数", "实习生人数", "薪资", "工资", "salary")


def complete_fact_sentence(value: str, claim_text: str = "") -> bool:
    text = " ".join(str(value or "").split()).strip()
    if len(text) < 8 or not re.search(r"[\u3400-\u9fffA-Za-z]", text):
        return False
    if re.search(r"(?:达到|约为|预计|截至|因为|以及|并且|为|到|至|and|or|because)\s*$", text, re.I):
        return False
    if re.match(r"^(?:同比|环比|其中|此外|同时|并且|以及|而且|该比例|该数值)", text, re.I):
        return False
    claim = " ".join(str(claim_text or "").split()).strip()
    if not claim:
        return True
    if set(_YEAR.findall(claim)) - set(_YEAR.findall(text)):
        return False
    if _MEASUREMENT.search(claim) and not _MEASUREMENT.search(text):
        return False
    return not any(metric.casefold() in claim.casefold() and metric.casefold() not in text.casefold() for metric in _METRICS)


def decide_admission(item: EvidenceItem, claim_text_by_id: dict[str, str], config: EvidenceRetrievalConfig) -> AdmissionDecision:
    """按固定优先级返回首个及全部适用阻断原因。"""
    content = " ".join(span.strip() for span in item.quoted_spans if span.strip())[:600]
    if item.metadata.get("retrieval_candidate_passthrough") is True:
        return AdmissionDecision(bool(content), (AdmissionReason.ADMITTED,) if content else (AdmissionReason.INCOMPLETE_FACT,))
    reasons: list[AdmissionReason] = []
    if item.relation == EvidenceRelation.NEUTRAL:
        reasons.append(AdmissionReason.RELATION)
    if not content or not complete_fact_sentence(content):
        reasons.append(AdmissionReason.INCOMPLETE_FACT)
    if not item.scope_compatible or any(value in _BLOCKING_SCOPE_REASONS for value in item.scope_mismatch_reasons):
        reasons.append(AdmissionReason.SCOPE_OR_QUOTE)
    if re.search(r"无法直接(?:支持|确认|反驳|否定)|cannot directly (?:support|refute)", item.reason or "", re.I):
        reasons.append(AdmissionReason.OTHER_POLICY)
    mapped_ids = list(item.metadata.get("supported_claim_ids") or []) if item.relation == EvidenceRelation.SUPPORT else list(item.metadata.get("refuted_claim_ids") or []) if item.relation == EvidenceRelation.REFUTE else []
    if item.relation in {EvidenceRelation.SUPPORT, EvidenceRelation.REFUTE}:
        if not mapped_ids or any(value not in claim_text_by_id for value in mapped_ids):
            reasons.append(AdmissionReason.RELATION)
        if item.judge_confidence < config.public_citation_min_confidence:
            reasons.append(AdmissionReason.LOW_CONFIDENCE)
        if item.scores.directness < config.public_citation_min_directness:
            reasons.append(AdmissionReason.LOW_DIRECTNESS)
        if any(not complete_fact_sentence(content, claim_text_by_id[key]) for key in mapped_ids if key in claim_text_by_id):
            reasons.append(AdmissionReason.INCOMPLETE_FACT)
    elif item.relation == EvidenceRelation.SUPPLEMENT:
        if str(item.metadata.get("matched_claim_id") or "") not in claim_text_by_id:
            reasons.append(AdmissionReason.RELATION)
        if item.judge_confidence < config.public_supplement_min_confidence:
            reasons.append(AdmissionReason.LOW_CONFIDENCE)
    elif item.relation != EvidenceRelation.NEUTRAL:
        reasons.append(AdmissionReason.RELATION)
    ordered = tuple(dict.fromkeys(reasons))
    return AdmissionDecision(not ordered, (AdmissionReason.ADMITTED,) if not ordered else ordered)
