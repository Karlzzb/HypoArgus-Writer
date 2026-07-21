"""Typed deterministic slot extraction and cross-evidence aggregation."""
from __future__ import annotations

from enum import Enum
from datetime import datetime, timezone
from typing import Any, Iterable, Literal
import re
from pydantic import BaseModel


class SlotType(str, Enum):
    YEAR = "YEAR"
    FORECAST_YEAR = "FORECAST_YEAR"
    TIME_RANGE = "TIME_RANGE"
    COUNT = "COUNT"
    PERCENTAGE = "PERCENTAGE"
    CURRENCY = "CURRENCY"
    MARKET_SIZE = "MARKET_SIZE"
    REGION = "REGION"
    ENTITY = "ENTITY"
    METRIC = "METRIC"
    RANGE = "RANGE"
    TEXT_VALUE = "TEXT_VALUE"
    NUMERIC_VALUE = "NUMERIC_VALUE"
    UNIT = "UNIT"


class TimeSlotEvidence(BaseModel):
    slot_name: str
    slot_type: Literal["YEAR", "FORECAST_YEAR", "TIME_RANGE"]
    value: str
    normalized_value: int | list[int]
    source_span: str


_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_FORECAST_YEAR_RE = re.compile(
    r"(?:(?:预计|预测|预期)\s*(?:到|至)?\s*(?:19|20)\d{2}(?:年)?(?:底|末)?"
    r"|(?:到|至)\s*(?:19|20)\d{2}(?:年)?(?:底|末)?"
    r"|(?:19|20)\d{2}[EF])",
    re.I,
)
_FUTURE_YEARS_RE = re.compile(r"未来\s*([一二三四五六七八九十\d]+)\s*年")
_TIME_RANGE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*(?:至|到|[-~～—–])\s*((?:19|20)\d{2})(?:年)?")
_PERCENT_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(%|％|个百分点)|百分之\s*(\d+(?:\.\d+)?)")
_COUNT_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(万|亿|千|百)?\s*(台|个|家|座|套|件|人|辆|架|艘|只|学分|学时|小时|周|门|项|平方米|分|名|位|期|次|届|科|班|组|团|类|种|篇|本|册|卷|部|集|场|处|站|栋|层|间|套|批|轮|道|条|块|张|件|份)(?![元美币])")
_MARKET_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(万亿|千亿|百亿|亿|万|千)?\s*(美元|美金|人民币|元|欧元|日元|港元)")
_RANGE_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:至|到|[-~～—])\s*(\d+(?:\.\d+)?)(?:\s*([%％]|万亿元|亿元|万元|美元|元|万台|台|个|家))?")
_CURRENCY_RE = re.compile(r"美元|美金|人民币|欧元|日元|港元|元")
_REGION_RE = re.compile(r"全球|中国大陆|中国|美国|欧洲|欧盟|亚太|亚洲|北美|南美|非洲|日本|韩国|印度")
_ENTITY_RE = re.compile(r"[A-Za-z\u3400-\u9fff][A-Za-z0-9\u3400-\u9fff·&（）()\-]{1,30}(?:公司|集团|研究院|研究所|协会|机构|大学|实验室)")
# Numeric value: any standalone number (not part of a year or version number).
# Excludes years (19xx/20xx) and version numbers (letter+digit like OAuth2).
_NUMERIC_VALUE_RE = re.compile(r"(?<!\d)(?<![A-Za-z])(\d+(?:\.\d+)?)(?!\d)(?![A-Za-z年级])")
# Unit: any measurement unit — education domain + general.
_UNIT_RE = re.compile(
    r"(学分|学时|小时|周|门|项|平方米|分|月|年|人|家|所|个|元|万元|亿元|美元|倍|个百分点|％|%|台|座|套|件|辆|架|艘|只|名|位|期|次|届|科|班|组|团|类|种|篇|本|册|卷|部|集|场|处|站|栋|层|间|批|轮|道|条|块|张|份)"
)


def infer_slot_type(slot_name: str) -> SlotType:
    value = str(slot_name or "").casefold()
    # HypoArgus generates type-label slots like "数值" and "单位" — these are not
    # semantic slot names but type requests: "数值" = "there should be a number",
    # "单位" = "there should be a unit". Map them to typed slots.
    if value in ("数值", "numeric_value", "number", "数字"):
        return SlotType.NUMERIC_VALUE
    if value in ("单位", "unit", "计量单位"):
        return SlotType.UNIT
    if any(key in value for key in ("预测年份", "预计年份", "forecast year", "forecast_year")):
        return SlotType.FORECAST_YEAR
    if any(key in value for key in ("时间区间", "年份区间", "时间范围", "预测区间", "time range", "time_range")):
        return SlotType.TIME_RANGE
    if any(key in value for key in ("年份", "年度", "year", "日期", "时间")):
        return SlotType.YEAR
    if any(key in value for key in ("同比", "环比", "增长率", "增速", "份额", "占比", "比例", "percentage", "percent")):
        return SlotType.PERCENTAGE
    if any(key in value for key in ("币种", "货币单位", "currency")):
        return SlotType.CURRENCY
    if any(key in value for key in ("市场规模", "融资规模", "金额", "市值", "营收", "销售额", "market size")):
        return SlotType.MARKET_SIZE
    if any(key in value for key in ("区间", "范围", "range")):
        return SlotType.RANGE
    if any(key in value for key in ("地区", "地域", "区域", "国家", "region")):
        return SlotType.REGION
    if any(key in value for key in ("企业", "公司", "机构", "主体", "entity")) and not any(
        key in value for key in ("数量", "数目", "家数", "份额", "占比")
    ):
        return SlotType.ENTITY
    if any(key in value for key in ("出货", "销量", "数量", "数目", "企业数", "家数", "count")):
        return SlotType.COUNT
    if any(key in value for key in ("指标", "口径", "metric")):
        return SlotType.METRIC
    return SlotType.TEXT_VALUE


def _semantic_markers(slot: str, slot_type: SlotType) -> list[str]:
    groups = [
        (("同比增长率", "增长率", "同比增速", "增速", "同比增长"),
         ("同比增长", "同比增速", "较上年增长", "较上年", "CAGR", "增长率", "增速")),
        (("环比",), ("环比",)),
        (("企业份额", "中国企业份额", "出货量份额", "市场份额", "占比", "份额"),
         ("占全球", "市场份额", "合计占据", "出货量份额", "企业份额", "占比", "份额", "前六", "前十")),
        (("出货量",), ("出货量", "出货", "销量", "交付量", "安装量")),
        (("企业数量", "企业数", "家数", "整机企业"),
         ("企业数量", "企业数", "公司数量", "家", "整机企业")),
        (("市场规模",), ("市场规模", "产业规模", "销售额", "总规模", "市场总规模")),
       (("融资规模", "融资金额"), ("融资", "融资金额", "融资规模")),
       (("价格", "平均价格", "产品价格", "售价"), ("价格", "售价", "单价", "平均价格")),
       (("成功率",), ("成功率", "成功", "完成率")),
       (("数据中心", "机器人数据中心"), ("数据中心", "机器人数据中心")),
    ]
    markers: list[str] = []
    for triggers, values in groups:
        if any(trigger in slot for trigger in triggers):
            markers.extend(values)
    if slot_type == SlotType.REGION:
        markers.extend(["全球", "中国", "美国", "欧洲", "亚太"])
    return list(dict.fromkeys(markers))


def _subject_markers(slot: str) -> list[str]:
    """Extract subject keywords from the slot name for semantic anchoring."""
    subjects: list[str] = []
    if "中国" in slot or "中国企业" in slot:
        subjects.extend(["中国", "中国企业", "国内"])
    if "全球" in slot:
        subjects.extend(["全球", "世界", "国际"])
    if "人形机器人" in slot or "人形" in slot:
        subjects.extend(["人形机器人", "人形"])
    if "具身智能" in slot:
        subjects.extend(["具身智能", "具身"])
    if "机器人" in slot:
        subjects.extend(["机器人"])
    return list(dict.fromkeys(subjects))


def _choose_match(slot: str, slot_type: SlotType, text: str, matches: list[re.Match[str]]) -> re.Match[str] | None:
    if not matches:
        return None
    if len(matches) == 1:
        markers = _semantic_markers(slot, slot_type)
        subjects = _subject_markers(slot)
        if markers or subjects:
            start, end = matches[0].span()
            window = text[max(0, start - 30):min(len(text), end + 20)]
            has_metric = any(marker in window for marker in markers) if markers else True
            has_subject = any(sub in window for sub in subjects) if subjects else True
            if not has_metric or not has_subject:
                return None
        return matches[0]
    markers = _semantic_markers(slot, slot_type)
    subjects = _subject_markers(slot)
    if not markers and not subjects:
        # Generic temporal slots such as ``年份`` may legitimately have more
        # than one occurrence. Use the earliest grounded span deterministically
        # instead of dropping the slot merely because no subject marker exists.
        return min(matches, key=lambda match: match.start())
    ranked: list[tuple[int, int, int, re.Match[str]]] = []
    for match in matches:
        start, end = match.span()
        window = text[max(0, start - 30):min(len(text), end + 20)]
        metric_hits = sum(marker in window for marker in markers) if markers else 0
        subject_hits = sum(sub in window for sub in subjects) if subjects else 0
        distances = [abs(text.find(marker) - start) for marker in markers if text.find(marker) >= 0]
        distance = min(distances) if distances else 10_000
        # Require at least one metric marker AND one subject marker when both exist
        if markers and subject_hits == 0 and subjects:
            continue
        if markers and metric_hits == 0:
            continue
        ranked.append((-metric_hits - subject_hits, distance, match.start(), match))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    if not ranked:
        return None
    if not markers or -ranked[0][0] == 0:
        return None
    return ranked[0][3]


def _normalized_number(raw: str) -> float | int | None:
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if not match:
        return None
    value = float(match.group())
    return int(value) if value.is_integer() else value


def _detail(slot: str, slot_type: SlotType, value: str, quote: str, candidate_id: str) -> dict[str, Any]:
    normalized: Any = value
    unit: str | None = None
    if slot_type in {SlotType.YEAR, SlotType.COUNT, SlotType.PERCENTAGE, SlotType.MARKET_SIZE, SlotType.RANGE}:
        normalized = _normalized_number(value)
    elif slot_type == SlotType.FORECAST_YEAR:
        year = _YEAR_RE.search(value)
        if year:
            normalized = int(year.group(1))
        else:
            future = _FUTURE_YEARS_RE.search(value)
            cn = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            count = int(future.group(1)) if future and future.group(1).isdigit() else cn.get(future.group(1), 0) if future else 0
            current = datetime.now(timezone.utc).year
            normalized = list(range(current + 1, current + count + 1))
    elif slot_type == SlotType.TIME_RANGE:
        years_in_range = [int(year) for year in _YEAR_RE.findall(value)]
        normalized = years_in_range[:2]
    if slot_type == SlotType.PERCENTAGE:
        unit = "%"
    elif slot_type == SlotType.COUNT:
        match = re.search(r"台|个|家|座|套|件|人|辆|架|艘|只", value)
        unit = match.group() if match else None
    elif slot_type in {SlotType.CURRENCY, SlotType.MARKET_SIZE}:
        match = re.search(r"美元|美金|人民币|欧元|日元|港元|元", value)
        unit = match.group() if match else value if slot_type == SlotType.CURRENCY else None
    # Extract semantic binding context from the quote
    years = _YEAR_RE.findall(quote)
    regions = _REGION_RE.findall(quote)
    return {
        "slot_name": slot,
        "slot_type": slot_type.value,
        "covered": True,
        "value": value,
        "normalized_value": normalized,
        "unit": unit,
        "quote": quote,
        "source_span": quote,
        "candidate_id": candidate_id,
        "time_scope": years[0] if years else None,
        "region": regions[0] if regions else None,
        "subject": _subject_markers(slot),
        "metric": _semantic_markers(slot, slot_type),
    }


def _extract_typed(slot: str, slot_type: SlotType, text: str, candidate_id: str) -> dict[str, Any] | None:
    patterns: dict[SlotType, re.Pattern[str]] = {
        SlotType.YEAR: _YEAR_RE,
        SlotType.FORECAST_YEAR: _FORECAST_YEAR_RE,
        SlotType.TIME_RANGE: _TIME_RANGE_RE,
        SlotType.PERCENTAGE: _PERCENT_RE,
        SlotType.COUNT: _COUNT_RE,
        SlotType.MARKET_SIZE: _MARKET_RE,
        SlotType.RANGE: _RANGE_RE,
        SlotType.CURRENCY: _CURRENCY_RE,
        SlotType.REGION: _REGION_RE,
        SlotType.ENTITY: _ENTITY_RE,
        SlotType.NUMERIC_VALUE: _NUMERIC_VALUE_RE,
        SlotType.UNIT: _UNIT_RE,
    }
    if slot_type == SlotType.METRIC:
        if slot and slot in text:
            return _detail(slot, slot_type, slot, slot, candidate_id)
        return None
    if slot_type == SlotType.TEXT_VALUE:
        if slot and slot in text:
            return _detail(slot, slot_type, slot, slot, candidate_id)
        return None
    matches = list(patterns[slot_type].finditer(text))
    if slot_type == SlotType.FORECAST_YEAR:
        matches.extend(_FUTURE_YEARS_RE.finditer(text))
        matches.sort(key=lambda match: match.start())
    chosen = _choose_match(slot, slot_type, text, matches)
    if chosen is None:
        return None
    value = chosen.group(0).strip()
    if slot_type == SlotType.PERCENTAGE and value.startswith("百分之"):
        value = f"{chosen.group(3)}%"
    quote = text[max(0, chosen.start() - 24):min(len(text), chosen.end() + 24)].strip()
    return _detail(slot, slot_type, value, quote, candidate_id)


def validate_slot_value(slot_type: SlotType | str, value: Any, *, quote: str = "") -> bool:
    try:
        kind = slot_type if isinstance(slot_type, SlotType) else SlotType(str(slot_type))
    except ValueError:
        return False
    raw = str(value or "").strip()
    context = f"{raw} {quote}".strip()
    if not raw:
        return False
    if kind == SlotType.YEAR:
        match = _YEAR_RE.fullmatch(raw)
        return bool(match and 1900 <= int(match.group(1)) <= 2100)
    if kind == SlotType.FORECAST_YEAR:
        return bool(_FORECAST_YEAR_RE.search(context) or _FUTURE_YEARS_RE.search(context))
    if kind == SlotType.TIME_RANGE:
        return bool(_TIME_RANGE_RE.search(context))
    if kind == SlotType.PERCENTAGE:
        return bool(_PERCENT_RE.search(context))
    if kind == SlotType.COUNT:
        return bool(_COUNT_RE.search(context)) and not bool(_YEAR_RE.fullmatch(raw))
    if kind == SlotType.CURRENCY:
        return bool(re.search(r"美元|美金|人民币|欧元|日元|港元|元", context))
    if kind == SlotType.MARKET_SIZE:
        return bool(_MARKET_RE.search(context))
    if kind == SlotType.RANGE:
        return bool(_RANGE_RE.search(context))
    if kind == SlotType.REGION:
        return bool(_REGION_RE.search(context))
    if kind == SlotType.ENTITY:
        return bool(_ENTITY_RE.search(context))
    if kind == SlotType.NUMERIC_VALUE:
        return bool(_NUMERIC_VALUE_RE.search(context))
    if kind == SlotType.UNIT:
        return bool(_UNIT_RE.search(context))
    return True


def infer_slot_evidence(required_slots: Iterable[str], content: str, *, candidate_id: str = "") -> dict[str, dict[str, Any]]:
    """Extract only values compatible with each required slot's type."""
    text = str(content or "")
    output: dict[str, dict[str, Any]] = {}
    for raw_slot in required_slots:
        slot = str(raw_slot)
        slot_type = infer_slot_type(slot)
        detail = _extract_typed(slot, slot_type, text, candidate_id)
        if detail is not None and validate_slot_value(slot_type, detail["value"], quote=detail["quote"]):
            output[slot] = detail
    return output


def normalize_slot_evidence(
    required_slots: Iterable[str],
    content: str,
    provided: dict[str, dict[str, Any]] | None,
    *,
    candidate_id: str = "",
) -> dict[str, dict[str, Any]]:
    """Merge deterministic extraction with only type-valid, grounded LLM values."""
    required = [str(slot) for slot in required_slots]
    output = infer_slot_evidence(required, content, candidate_id=candidate_id)
    for slot, raw_detail in (provided or {}).items():
        slot = str(slot)
        if slot not in required or not isinstance(raw_detail, dict):
            continue
        slot_type = infer_slot_type(slot)
        value = raw_detail.get("value")
        quote = str(raw_detail.get("quote") or "")
        if quote and quote not in content:
            continue
        if not validate_slot_value(slot_type, value, quote=quote):
            continue
        output[slot] = _detail(
            slot,
            slot_type,
            str(value),
            quote or str(value),
            str(raw_detail.get("candidate_id") or candidate_id),
        )
    return output


def aggregate_slot_evidence(items: Iterable[dict[str, Any]]) -> tuple[dict[str, list[str]], list[str]]:
    coverage: dict[str, list[str]] = {}
    conflicts: list[str] = []
    scopes: dict[str, set[tuple[str, str, str, str]]] = {}
    values: dict[str, set[str]] = {}
    for item in items:
        evidence_id = str(item.get("evidence_id") or item.get("candidate_id") or "")
        slots = item.get("slot_evidence") or {}
        if not slots and item.get("covered_slots"):
            slots = {str(slot): {} for slot in item["covered_slots"]}
        for slot, detail in slots.items():
            detail = detail or {}
            if detail.get("covered", True) is False:
                continue
            scope = (
                str(detail.get("year") or detail.get("time_scope") or ""),
                str(detail.get("region") or detail.get("region_scope") or ""),
                str(detail.get("subject") or ""),
                str(detail.get("statistical_scope") or ""),
            )
            previous_scopes = scopes.setdefault(str(slot), set())
            if previous_scopes and scope != ("", "", "", "") and all(
                value != ("", "", "", "") for value in previous_scopes
            ) and scope not in previous_scopes:
                # Different scopes are not a factual conflict and must not be
                # mixed into the same aggregate value set.
                continue
            previous_scopes.add(scope)
            current_value = str(detail.get("normalized_value", detail.get("value", "")))
            previous_values = values.setdefault(str(slot), set())
            if current_value and previous_values and current_value not in previous_values:
                conflicts.append(f"{slot}:value_conflict")
            if current_value:
                previous_values.add(current_value)
            ids = coverage.setdefault(str(slot), [])
            if evidence_id and evidence_id not in ids:
                ids.append(evidence_id)
    return coverage, list(dict.fromkeys(conflicts))


__all__ = [
    "SlotType", "TimeSlotEvidence", "aggregate_slot_evidence", "infer_slot_evidence",
    "infer_slot_type", "normalize_slot_evidence", "validate_slot_value",
]
