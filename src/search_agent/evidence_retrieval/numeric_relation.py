"""Claim-bound deterministic numeric relation safety rail."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class NumericRelationResult:
    numeric_relation: str | None
    relation_conflict: bool = False
    final_relation: str | None = None
    override_reason: str | None = None
    confidence: float = 0.0
    target_value: float | None = None
    evidence_value: float | None = None
    unit: str | None = None


_NUMBER = re.compile(
    r"(?P<sign>-?)\s*(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>万亿美元|万亿元|亿美元|亿元|万美元|万元|万台|万家|万|亿美元|美元|人民币|元|台|家|%|％|年)?"
)


def _scale(unit: str | None) -> float:
    return {
        "万亿美元": 1_000_000_000_000,
        "万亿元": 1_000_000_000_000,
        "亿美元": 100_000_000,
        "亿元": 100_000_000,
        "万美元": 10_000,
        "万元": 10_000,
        "万台": 10_000,
        "万家": 10_000,
        "万": 10_000,
    }.get(unit or "", 1.0)


def _kind(unit: str | None) -> str:
    value = unit or ""
    if "美元" in value:
        return "USD"
    if value in {"万亿元", "亿元", "万元", "人民币", "元"}:
        return "CNY"
    if value in {"万台", "台"}:
        return "COUNT_UNIT"
    if value in {"万家", "家"}:
        return "COUNT_ENTITY"
    if value in {"%", "％"}:
        return "PERCENT"
    return "NUMBER"


def _measured_number(text: str) -> tuple[float, str | None] | None:
    for match in _NUMBER.finditer(text or ""):
        raw = float(match.group("sign") + match.group("num"))
        unit = match.group("unit")
        # A four-digit year is context, never the measured value.
        suffix = (text or "")[match.end():match.end() + 1]
        if unit == "年" or (1900 <= raw <= 2100 and suffix == "年"):
            continue
        return raw * _scale(unit), unit
    return None


def _operator(text: str) -> str:
    if any(token in text for token in ("不会超过", "不超过", "不高于", "至多", "最多", "没有超过")):
        return "LE"
    if any(token in text for token in ("不足", "低于", "少于", "未达到")):
        return "LT"
    if any(token in text for token in ("不低于", "至少", "不少于")):
        return "GE"
    if any(token in text for token in ("超过", "高于", "大于")):
        return "GT"
    return "EQ"


class NumericRelationVerifier:
    """Compare one claim with one explicit evidence span, without conversion."""

    def verify(self, target_text: str, evidence_text: str, *, llm_relation: str | None = None) -> NumericRelationResult:
        target = _measured_number(target_text)
        evidence = _measured_number(evidence_text)
        if not target or not evidence:
            return NumericRelationResult(None, final_relation=llm_relation, override_reason="numeric_value_unavailable")
        target_value, target_unit = target
        evidence_value, evidence_unit = evidence
        target_kind, evidence_kind = _kind(target_unit), _kind(evidence_unit)
        if target_kind != evidence_kind and "NUMBER" not in {target_kind, evidence_kind}:
            return NumericRelationResult(
                None, final_relation=llm_relation, override_reason="unit_mismatch",
                target_value=target_value, evidence_value=evidence_value,
            )
        operator = _operator(target_text)
        tolerance = max(abs(target_value) * 0.05, 1e-9)
        supported = {
            "LT": evidence_value < target_value,
            "LE": evidence_value <= target_value,
            "GT": evidence_value > target_value,
            "GE": evidence_value >= target_value,
            "EQ": abs(evidence_value - target_value) <= tolerance,
        }[operator]
        relation = "SUPPORT" if supported else "REFUTE"
        normalized_llm = str(llm_relation or "").upper()
        conflict = normalized_llm in {"SUPPORT", "REFUTE"} and normalized_llm != relation
        return NumericRelationResult(
            relation,
            conflict,
            relation,
            "deterministic_numeric_relation",
            0.95,
            target_value,
            evidence_value,
            evidence_unit or target_unit,
        )

    compare = verify


__all__ = ["NumericRelationResult", "NumericRelationVerifier"]
