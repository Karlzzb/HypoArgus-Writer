"""Deterministic claim-level scope compatibility guard."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import (
    ClaimJudgeResult, EvidenceCandidate, EvidenceRelation, JudgeResult,
    NeutralReason, RetrievalTask,
)


@dataclass(slots=True)
class ScopeCompatibility:
    compatible: bool
    mismatch_reasons: list[str] = field(default_factory=list)
    target_scope: dict[str, list[str]] = field(default_factory=dict)
    evidence_scope: dict[str, list[str]] = field(default_factory=dict)


_REGIONS = {
    "GLOBAL": ("全球", "全世界", "worldwide", "global"),
    "CHINA": ("中国大陆", "中国市场", "中国", "china"),
    "US": ("美国", "united states", "u.s."),
    "EUROPE": ("欧洲", "欧盟", "europe"),
    "APAC": ("亚太", "asia-pacific", "apac"),
    "JAPAN": ("日本", "japan"),
    "KOREA": ("韩国", "korea"),
    "INDIA": ("印度", "india"),
}
_SUBJECTS = {
    "EMBODIED_INTELLIGENCE": ("具身智能", "embodied intelligence"),
    "HUMANOID_ROBOT": ("人形机器人", "人型机器人", "humanoid robot"),
    "ROBOT_INDUSTRY": ("机器人行业", "机器人产业", "robotics industry"),
    "CORE_COMPONENT": ("核心零部件", "关键零部件", "核心部件", "component market"),
    "COMPLETE_MACHINE": ("整机市场", "整机", "complete robot"),
}
_METRICS = {
    "SHIPMENT": ("出货量", "出货", "shipment"),
    "INVENTORY": ("保有量", "存量", "installed base"),
    "SALES_VOLUME": ("销量", "销售量", "sales volume"),
    "MARKET_SIZE": ("市场规模", "market size"),
    "ENTERPRISE_COUNT": ("企业数量", "企业数", "公司数量"),
    "FINANCING": ("融资规模", "融资金额", "financing"),
    "PRICE": ("产品价格", "售价", "单价", "price"),
    "GROWTH": ("同比增长", "增长率", "增速", "growth rate"),
    "SHARE": ("市场份额", "企业份额", "出货量份额", "占比", "share"),
    "RECRUITMENT_PLAN": ("招聘计划", "计划招聘", "recruitment plan"),
    "ACTUAL_INTERN_COUNT": ("实际接收", "实习生人数", "intern count"),
    "SALARY": ("薪资", "工资", "salary"),
}


def _labels(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    folded = f" {str(text or '').casefold()} "
    return [name for name, terms in mapping.items() if any(term.casefold() in folded for term in terms)]


def _years(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(text or ""))))


def _unit_kinds(text: str) -> list[str]:
    output: list[str] = []
    value = text or ""
    if re.search(r"\d[\d,.]*(?:\.\d+)?\s*(?:万|亿|千|百)?\s*(?:台|套|辆|架|艘|件)", value):
        output.append("COUNT_DEVICE")
    if re.search(r"\d[\d,.]*(?:\.\d+)?\s*(?:万|亿|千|百)?\s*(?:家|所)", value):
        output.append("COUNT_ENTITY")
    if re.search(r"\d[\d,.]*(?:\.\d+)?\s*(?:万|亿|千|百)?\s*人", value):
        output.append("COUNT_PERSON")
    if re.search(r"\d[\d,.]*(?:\.\d+)?\s*(?:万|亿|千|百)?\s*(?:个|座|只)", value):
        output.append("COUNT_GENERIC")
    if re.search(r"\d[\d,.]*(?:\.\d+)?\s*(?:万亿|亿|万)?(?:美元|人民币|元|欧元|日元|港元)", value):
        output.append("CURRENCY")
    if re.search(r"%|％|百分之", value):
        output.append("PERCENTAGE")
    return output


def _currency_kinds(text: str) -> list[str]:
    value = str(text or "")
    output: list[str] = []
    for name, pattern in (
        ("USD", r"美元|USD|US\$"),
        ("EUR", r"欧元|EUR|€"),
        ("JPY", r"日元|JPY|¥(?=\s*\d)"),
        ("HKD", r"港元|HKD|HK\$"),
        ("CNY", r"人民币|CNY|RMB|(?<!美)(?<!欧)(?<!日)(?<!港)元"),
    ):
        if re.search(pattern, value, re.I):
            output.append(name)
    return output


def extract_scope(text: str) -> dict[str, list[str]]:
    value = str(text or "")
    return {
        "years": _years(value),
        "regions": _labels(value, _REGIONS),
        "subjects": _labels(value, _SUBJECTS),
        "metrics": _labels(value, _METRICS),
        "unit_kinds": _unit_kinds(value),
        "currencies": _currency_kinds(value),
        "forecast_years": _years(value) if re.search(r"预测|预计|预期|forecast", value, re.I) else [],
        "statistical_scope": (
            ["ORGANIZATION"] if re.search(r"本校|我校|本单位|本企业", value)
            else ["PLATFORM"] if re.search(r"全平台|全行业|全国院校|所有院校", value)
            else []
        ),
    }


def check_scope_compatibility(target_text: str, evidence_text: str) -> ScopeCompatibility:
    target, evidence = extract_scope(target_text), extract_scope(evidence_text)
    reasons: list[str] = []
    if target["years"] and evidence["years"] and not set(target["years"]) & set(evidence["years"]):
        reasons.append("WRONG_TIME_SCOPE")
    if target["forecast_years"] and evidence["forecast_years"] and not set(target["forecast_years"]) & set(evidence["forecast_years"]):
        reasons.append("WRONG_FORECAST_YEAR")
    for field_name, reason in (
        ("regions", "WRONG_REGION_SCOPE"),
        ("subjects", "WRONG_SUBJECT_SCOPE"),
        ("metrics", "WRONG_METRIC_SCOPE"),
        ("unit_kinds", "INCOMPARABLE_UNIT"),
        ("currencies", "INCOMPARABLE_UNIT"),
        ("statistical_scope", "WRONG_STATISTICAL_SCOPE"),
    ):
        if target[field_name] and evidence[field_name] and not set(target[field_name]) & set(evidence[field_name]):
            reasons.append(reason)
    return ScopeCompatibility(
        compatible=not reasons,
        mismatch_reasons=list(dict.fromkeys(reasons)),
        target_scope=target,
        evidence_scope=evidence,
    )


def _neutral_reason(reasons: list[str]) -> NeutralReason:
    mapping = {
        "WRONG_TIME_SCOPE": NeutralReason.WRONG_YEAR,
        "WRONG_FORECAST_YEAR": NeutralReason.WRONG_YEAR,
        "WRONG_REGION_SCOPE": NeutralReason.WRONG_REGION,
        "WRONG_SUBJECT_SCOPE": NeutralReason.WRONG_ENTITY,
        "WRONG_METRIC_SCOPE": NeutralReason.WRONG_METRIC,
        "WRONG_STATISTICAL_SCOPE": NeutralReason.WRONG_STATISTICAL_SCOPE,
        "INCOMPARABLE_UNIT": NeutralReason.UNIT_MISMATCH,
    }
    return next((mapping[row] for row in reasons if row in mapping), NeutralReason.IRRELEVANT)


def apply_claim_scope_guard(
    claim: object,
    candidate: EvidenceCandidate,
    judgment: ClaimJudgeResult,
) -> ClaimJudgeResult:
    claim_text = str(
        getattr(claim, "source_text_span", None)
        or getattr(claim, "qualifier", None)
        or getattr(claim, "subject", None)
        or ""
    )
    evidence_span = " ".join(value for value in judgment.quoted_spans if value).strip()
    report = check_scope_compatibility(claim_text, evidence_span or candidate.content)
    explicit = {
        NeutralReason.WRONG_ENTITY: "WRONG_SUBJECT_SCOPE",
        NeutralReason.WRONG_YEAR: "WRONG_TIME_SCOPE",
        NeutralReason.WRONG_REGION: "WRONG_REGION_SCOPE",
        NeutralReason.WRONG_METRIC: "WRONG_METRIC_SCOPE",
        NeutralReason.WRONG_MARKET_SCOPE: "WRONG_REGION_SCOPE",
        NeutralReason.WRONG_STATISTICAL_SCOPE: "WRONG_STATISTICAL_SCOPE",
        NeutralReason.UNIT_MISMATCH: "INCOMPARABLE_UNIT",
    }.get(judgment.neutral_reason)
    if explicit and explicit not in report.mismatch_reasons:
        report.mismatch_reasons.append(explicit)
        report.compatible = False
    reasons = list(dict.fromkeys(report.mismatch_reasons))
    if report.compatible:
        return judgment.model_copy(update={
            "scope_compatible": True,
            "scope_mismatch_reasons": [],
            "matched_claim_id": judgment.claim_id,
            "numeric_override_allowed": True,
        })
    relation = judgment.relation
    if relation in {EvidenceRelation.SUPPORT, EvidenceRelation.REFUTE}:
        relation = EvidenceRelation.NEUTRAL
    return judgment.model_copy(update={
        "relation": relation,
        "confidence": 0.0 if relation == EvidenceRelation.NEUTRAL else min(judgment.confidence, 0.39),
        "directness": 0.0 if relation == EvidenceRelation.NEUTRAL else min(judgment.directness, 0.39),
        "quoted_spans": [] if relation == EvidenceRelation.NEUTRAL else judgment.quoted_spans,
        "scope_compatible": False,
        "scope_mismatch_reasons": reasons,
        "neutral_reason": _neutral_reason(reasons),
        "reason": "SCOPE_COMPATIBILITY_GUARD: " + ", ".join(reasons),
        "matched_claim_id": judgment.claim_id,
        "numeric_override_allowed": False,
    })


def apply_scope_guard(
    task: RetrievalTask,
    candidate: EvidenceCandidate,
    judgment: JudgeResult,
    *,
    atomic_claims: list | None = None,
) -> JudgeResult:
    """Backward-compatible candidate guard; V12 decisions use the claim API."""
    claims = atomic_claims or []
    target = (
        getattr(claims[0], "source_text_span", "")
        if len(claims) == 1 else task.target_text
    )
    report = check_scope_compatibility(target, candidate.content)
    if report.compatible:
        return judgment.model_copy(update={"scope_compatible": True, "scope_mismatch_reasons": []})
    relation = judgment.relation
    if relation in {EvidenceRelation.SUPPORT, EvidenceRelation.REFUTE}:
        relation = EvidenceRelation.NEUTRAL
    return judgment.model_copy(update={
        "relation": relation,
        "final_relation": relation,
        "confidence": 0.0 if relation == EvidenceRelation.NEUTRAL else min(judgment.confidence, 0.39),
        "directness": 0.0 if relation == EvidenceRelation.NEUTRAL else min(judgment.directness, 0.39),
        "quoted_spans": [] if relation == EvidenceRelation.NEUTRAL else judgment.quoted_spans,
        "neutral_reason": _neutral_reason(report.mismatch_reasons),
        "override_reason": "SCOPE_COMPATIBILITY_GUARD",
        "scope_compatible": False,
        "scope_mismatch_reasons": report.mismatch_reasons,
    })


__all__ = [
    "ScopeCompatibility", "apply_claim_scope_guard", "apply_scope_guard",
    "check_scope_compatibility", "extract_scope",
]
