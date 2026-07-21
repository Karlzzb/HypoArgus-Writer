"""One-shot deterministic, benefit-oriented gap query planning."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .query_normalization import extract_numeric_expressions, normalize_query_preserving_numbers
from .slot_aggregation import SlotType, infer_slot_type

ALLOWED_GAP_REASONS = {
    "MISSING_REQUIRED_SLOTS",
}


@dataclass
class GapRetrievalPlan:
    triggered: bool
    queries: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    round: int = 0
    gap_reason: str | None = None


def _claim_parts(task) -> tuple[str, str, str, str, str]:
    target = str(task.target_text)
    claims = list(getattr(task, "atomic_claims", None) or [])
    subject_parts = []
    for claim in claims:
        value = str(getattr(claim, "subject", "")).strip()
        value = re.sub(r"(?:预计)?(?:截至|截止|到)?\s*(?:19|20)\d{2}(?:年(?:底|末)?)?", " ", value)
        for expression in extract_numeric_expressions(value):
            value = value.replace(expression, " ")
        value = re.sub(r"是否|不足|低于|少于|超过|高于|不超过|不低于|未达到", " ", value)
        value = normalize_query_preserving_numbers(value)
        if value:
            subject_parts.append(value)
    subject = " ".join(dict.fromkeys(subject_parts))
    if not subject:
        # Retain a bounded factual prefix, never the surrounding paragraph.
        subject = re.split(r"[，。；;！？!?]", target, maxsplit=1)[0].strip()
    years = " ".join(dict.fromkeys(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", target)))
    metrics = " ".join(dict.fromkeys(
        str(getattr(claim, "metric", "")).strip() for claim in claims
        if str(getattr(claim, "metric", "")).strip()
    ))
    numbers = " ".join(extract_numeric_expressions(target))
    regions = " ".join(dict.fromkeys(re.findall(r"全球|中国|美国|欧洲|亚太|印度|日本|韩国", target)))
    return subject, years, metrics, numbers, regions


def _slot_query(subject: str, years: str, metrics: str, numbers: str, regions: str, slot: str) -> str:
    kind = infer_slot_type(slot)
    if kind == SlotType.CURRENCY:
        parts = (years, regions, subject, metrics or "市场规模", numbers, "币种 货币单位")
    elif kind == SlotType.PERCENTAGE:
        qualifier = "同比增长率" if any(key in slot for key in ("同比", "增长", "增速")) else "市场份额 占比"
        parts = (years, regions, subject, metrics, numbers, slot, qualifier)
    elif kind == SlotType.COUNT:
        parts = (years, regions, subject, metrics or slot, numbers, "数量 单位")
    elif kind in {SlotType.YEAR, SlotType.FORECAST_YEAR, SlotType.TIME_RANGE}:
        parts = (years, regions, subject, metrics, numbers, slot, "预测" if kind == SlotType.FORECAST_YEAR else "报告")
    else:
        # Keep the missing slot explicit and make the one-shot gap query distinct
        # from the initial broad query even when the slot token already appears in
        # the subject (for example: ``target metric``).
        parts = (years, regions, subject, metrics, numbers, slot, "官方 报告")
    tokens = []
    seen = set()
    for part in parts:
        raw_part = str(part or "")
        expressions = extract_numeric_expressions(raw_part)
        remainder = raw_part
        for expression in expressions:
            remainder = remainder.replace(expression, " ")
        if expressions and not remainder.strip():
            cleaned_tokens = expressions
        else:
            cleaned = normalize_query_preserving_numbers(raw_part)
            cleaned = re.sub(r"(?<=\d)(?=[A-Za-z\u3400-\u9fff])", " ", cleaned)
            cleaned = re.sub(r"(?<=[A-Za-z\u3400-\u9fff])(?=\d)", " ", cleaned)
            cleaned_tokens = cleaned.split()
        for token in cleaned_tokens:
            if token.casefold() not in seen:
                seen.add(token.casefold())
                tokens.append(token)
    return " ".join(tokens)


def validate_gap_query(query: str, target_text: str) -> bool:
    value = normalize_query_preserving_numbers(query)
    if not value or re.search(r"(?:19|20)\d{2}(?:19|20)\d{2}", value):
        return False
    if re.search(r"\d+(?:\.\d+)?[%％]\d+(?:\.\d+)?[%％]", value):
        return False
    target_numbers = extract_numeric_expressions(target_text)
    compact = value.replace(" ", "")
    if any(number.replace(" ", "") not in compact for number in target_numbers):
        return False
    target_units = re.findall(r"万亿美元|万亿元|亿美元|亿元|万元|万台|美元|人民币|%|％|台|元", target_text)
    if any(unit not in value for unit in target_units):
        return False
    return len(re.findall(r"[。！？!?；;]", value)) < 3


def plan_gap_retrieval(task, *, gap_reason: str | None, missing_slots: list[str], round_number: int = 0) -> GapRetrievalPlan:
    if round_number >= 1 or gap_reason not in ALLOWED_GAP_REASONS or not missing_slots:
        return GapRetrievalPlan(False, [], list(missing_slots), round_number, gap_reason)
    subject, years, metrics, numbers, regions = _claim_parts(task)
    queries = [_slot_query(subject, years, metrics, numbers, regions, slot) for slot in missing_slots if slot]
    queries = [query for query in dict.fromkeys(queries) if query and validate_gap_query(query, task.target_text)]
    return GapRetrievalPlan(bool(queries), queries, list(missing_slots), 1 if queries else round_number, gap_reason)


__all__ = ["ALLOWED_GAP_REASONS", "GapRetrievalPlan", "plan_gap_retrieval", "validate_gap_query"]
