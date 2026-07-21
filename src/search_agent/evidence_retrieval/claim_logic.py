"""Deterministic claim decomposition and reverse-hypothesis normalization.

The parser is intentionally conservative: it never invents a value and always
keeps the original text as the fallback claim.  It is used for retrieval
queries and verification hints; the LLM remains the source of semantic
judgement when a sentence cannot be parsed deterministically.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .query_normalization import extract_numeric_expressions, normalize_query_preserving_numbers


class ClaimLogicOperator(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    SINGLE = "SINGLE"


class AtomicClaim(BaseModel):
    claim_id: str
    subject: str = ""
    time_scope: str | None = None
    metric: str = ""
    operator: str = "ASSERT"
    value: float | str | None = None
    unit: str | None = None
    qualifier: str | None = None
    polarity: str = "POSITIVE"
    source_text_span: str = ""


class AtomicClaimGroup(BaseModel):
    atomic_claims: list[AtomicClaim] = Field(default_factory=list)
    logic_operator: ClaimLogicOperator = ClaimLogicOperator.SINGLE
    original_target_text: str = ""


_NUM = re.compile(r"(?P<sign>-?)\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>万亿美元|万亿元|万亿|亿美元|亿元|万元|万台|万美元|万|亿|千|%|％|美元|台|年)?")
_YEAR = re.compile(r"(?:19|20)\d{2}(?:\s*(?:年|年至|到|-|—)\s*(?:19|20)?\d{2}\s*年?)?")
_METRICS = ("出货量", "市场规模", "增长率", "同比增长", "同比下降", "份额", "占比", "价格", "数量", "企业数", "产品数", "成功率")


def _numeric_value(raw: str, unit: str | None) -> float:
    value = float(raw)
    if unit in {"万", "万台", "万元", "万美元"}:
        return value * 10_000
    if unit in {"亿", "亿元", "亿美元"}:
        return value * 100_000_000
    if unit in {"万亿", "万亿元", "万亿美元"}:
        return value * 1_000_000_000_000
    if unit == "千":
        return value * 1_000
    return value


def _operator(text: str, *, reverse: bool) -> str:
    if any(x in text for x in ("不超过", "不高于", "至多", "没有超过")):
        return "LE"
    if any(x in text for x in ("不足", "低于", "少于", "未达到")):
        return "LT"
    if any(x in text for x in ("不低于", "至少", "不少于")):
        return "GE"
    if any(x in text for x in ("超过", "高于", "大于")):
        return "GT"
    if any(x in text for x in ("约", "左右", "大约")):
        return "APPROX_EQ"
    return "ASSERT"


def atomize_claim(text: str, *, line_type: str | None = None) -> AtomicClaimGroup:
    original = str(text or "").strip()
    reverse = str(line_type or "").lower() == "reverse"
    separator = re.search(r"且|并且|以及|同时|和|或", original)
    logic = ClaimLogicOperator.OR if "或" in original else ClaimLogicOperator.AND if separator or re.search(r"[，,；;]", original) else ClaimLogicOperator.SINGLE
    # Split only on logical conjunctions; commas inside a numeric range stay in
    # one segment.  If there are no conjunctions, a numeric sentence may still
    # contain multiple independent clauses separated by Chinese punctuation.
    parts = [p.strip(" ，,；;。！？?!") for p in re.split(r"(?:且|并且|以及|同时|或)", original) if p.strip(" ，,；;。！？?!")]
    if len(parts) == 1 and re.search(r"[，,；;]", original):
        parts = [p.strip() for p in re.split(r"[，,；;]", original) if p.strip()]
    time_only = re.compile(
        r"^(?:预计)?(?:截至|截止|到)?\s*(?:19|20)\d{2}(?:年(?:底|末)?)?$"
        r"|^(?:19|20)\d{2}\s*(?:[-—–~～至到])\s*(?:19|20)\d{2}年?$"
        r"|^未来[一二三四五六七八九十\d]+年$"
    )
    merged_parts: list[str] = []
    pending_time = ""
    for part in parts:
        if time_only.fullmatch(part.strip()):
            pending_time = " ".join(filter(None, [pending_time, part.strip()]))
            continue
        merged_parts.append("，".join(filter(None, [pending_time, part])))
        pending_time = ""
    if pending_time and merged_parts:
        merged_parts[-1] = "，".join([pending_time, merged_parts[-1]])
    parts = merged_parts or [original]
    logic = ClaimLogicOperator.OR if "或" in original else ClaimLogicOperator.AND if len(parts) > 1 else ClaimLogicOperator.SINGLE
    claims: list[AtomicClaim] = []
    for index, part in enumerate(parts or [original], 1):
        matches = list(_NUM.finditer(part))
        numbers = [m for m in matches if m.group("unit") != "年"][:1] or matches[:1]
        value: float | str | None = None
        unit: str | None = None
        if numbers:
            m = numbers[0]
            unit = m.group("unit")
            value = _numeric_value(m.group("sign") + m.group("num"), unit)
        metric = next((item for item in _METRICS if item in part), "")
        year = _YEAR.search(part)
        subject = part
        if numbers:
            subject = part[: numbers[0].start()].strip(" ：:，,") or part
        claims.append(AtomicClaim(
            claim_id=f"a{index}", subject=subject[:120], time_scope=year.group(0) if year else None,
            metric=metric, operator=_operator(part, reverse=reverse), value=value, unit=unit,
            qualifier=part, polarity="NEGATIVE_HYPOTHESIS" if reverse else "POSITIVE",
            source_text_span=part,
        ))
    if not claims:
        claims = [AtomicClaim(claim_id="a1", subject=original, qualifier=original, polarity="NEGATIVE_HYPOTHESIS" if reverse else "POSITIVE", source_text_span=original)]
    return AtomicClaimGroup(atomic_claims=claims, logic_operator=logic, original_target_text=original)


def normalize_reverse_hypothesis(text: str) -> dict[str, Any]:
    """Turn a yes/no reverse question into a neutral searchable hypothesis."""
    original = str(text or "").strip()
    normalized = re.sub(r"[？?]", "", original)
    normalized = re.sub(r"是否|是不是|有无", "", normalized)
    normalized = normalized.replace("没有超过", "不超过").replace("没有低于", "不低于")
    normalized = re.sub(r"^(请问|请判断)\s*", "", normalized)
    numeric = {value for value in extract_numeric_expressions(original) if not value.endswith("年")}
    neutral = original
    for value in sorted(numeric, key=len, reverse=True):
        neutral = neutral.replace(value, " ")
    neutral = re.sub(r"是否|是不是|有没有|有无|不足|低于|少于|超过|高于|不超过|不低于|未达到|没有", " ", neutral)
    neutral = normalize_query_preserving_numbers(neutral)
    if not neutral:
        neutral = normalize_query_preserving_numbers(re.sub(r"[？?]", "", original))
    return {
        "original_target_text": original,
        "normalized_hypothesis": normalized,
        "neutral_retrieval_query": neutral,
        "polarity": "NEGATIVE_HYPOTHESIS",
        "atomic_claims": atomize_claim(normalized, line_type="reverse").model_dump(mode="json"),
    }


def apply_claim_logic(relations: list[str], operator: ClaimLogicOperator | str, *, polarity: str = "POSITIVE") -> str:
    """Aggregate atomic SUPPORT/REFUTE/NEUTRAL relations conservatively."""
    values = [
        {"SUPPORTED": "SUPPORT", "REFUTED": "REFUTE"}.get(str(item).upper(), str(item).upper())
        for item in relations
    ]
    op = operator if isinstance(operator, ClaimLogicOperator) else ClaimLogicOperator(str(operator).upper())
    if op == ClaimLogicOperator.AND:
        if "REFUTE" in values:
            result = "REFUTED"
        elif values and all(item == "SUPPORT" for item in values):
            result = "SUPPORTED"
        elif "CONFLICT" in values:
            result = "CONFLICT"
        else:
            result = "INCONCLUSIVE"
    elif op == ClaimLogicOperator.OR:
        if "SUPPORT" in values:
            result = "SUPPORTED"
        elif values and all(item == "REFUTE" for item in values):
            result = "REFUTED"
        elif "CONFLICT" in values:
            result = "CONFLICT"
        else:
            result = "INCONCLUSIVE"
    elif op == ClaimLogicOperator.NOT:
        if "SUPPORT" in values:
            result = "REFUTED"
        elif "REFUTE" in values:
            result = "SUPPORTED"
        else:
            result = "INCONCLUSIVE"
    else:
        result = values[0] + ("D" if values[0] in {"SUPPORT", "REFUTE"} else "") if values else "INCONCLUSIVE"
        result = {"SUPPORTD": "SUPPORTED", "REFUTED": "REFUTED", "NEUTRAL": "INCONCLUSIVE"}.get(result, result)
    return result


__all__ = ["AtomicClaim", "AtomicClaimGroup", "ClaimLogicOperator", "atomize_claim", "normalize_reverse_hypothesis", "apply_claim_logic"]
