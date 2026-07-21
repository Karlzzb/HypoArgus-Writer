"""Deterministic matching and parameter extraction for the 12 registered scenarios."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .retrievers.bm25_retriever import BM25Retriever, tokenize
from .schemas import RetrievalTask


@dataclass(slots=True)
class StructuredMatch:
    scenario_key: str | None = None
    scenario_name: str | None = None
    params: dict[str, Any] | None = None
    status: str = "not_matched"
    score: float = 0.0
    hit_terms: list[str] | None = None
    missing_params: list[str] | None = None


def _explicit_hints(task: RetrievalTask) -> tuple[str | None, dict[str, Any]]:
    scenario = None
    params: dict[str, Any] = {}
    for ref in task.source_refs:
        if not isinstance(ref, dict):
            continue
        scenario = scenario or ref.get("scenario_key") or ref.get("scenario_name")
        raw = ref.get("params") or ref.get("structured_params")
        if isinstance(raw, dict):
            params.update(raw)
    return str(scenario) if scenario else None, params


def _first(patterns: list[str], text: str, flags: int = 0) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1).strip(" ，。；：、!?！？\"'“”")
    return None


def _extract_params(task: RetrievalTask, schema: dict[str, Any], explicit: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(filter(None, [task.target_text, task.boundary or "", task.paragraph_text]))
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    output = {key: value for key, value in explicit.items() if key in properties}
    dates = re.findall(r"(?:19|20)\d{2}[-/.年](?:0?[1-9]|1[0-2])[-/.月](?:0?[1-9]|[12]\d|3[01])日?", text)
    normalized_dates = [re.sub(r"[/.年月]", "-", value).rstrip("日-") for value in dates]
    years = re.findall(r"((?:19|20)\d{2})(?:年|级)?", text)

    inferred: dict[str, Any] = {
        "enterprise_name": _first([
            r"(?:企业|公司)(?:名称)?[为是：:]?\s*[“\"']?([\u3400-\u9fffA-Za-z0-9（）()·&]{2,40}(?:公司|集团|企业)?)",
            r"([\u3400-\u9fffA-Za-z0-9（）()·&]{2,40}(?:有限公司|股份公司|集团))",
        ], text),
        "major_name": _first([r"([\u3400-\u9fffA-Za-z0-9+.#-]{2,24})专业"], text),
        "position_name": _first([r"([\u3400-\u9fffA-Za-z0-9+.#-]{2,24})(?:岗位|职位)"], text),
        "school_id": _first([r"(?:school[_ ]?id|学校ID|院校ID)[=：:\s]+([A-Za-z0-9_-]+)"], text, re.I),
        "my_school_id": _first([r"(?:my_school_id|本校(?:学校)?ID)[=：:\s]+([A-Za-z0-9_-]+)"], text, re.I),
        "label_type": _first([r"(产教融合型企业|上市企业|高新技术企业|专精特新企业)"], text),
        "peer_level": _first([r"对标院校层次[为是：:\s]+([\u3400-\u9fffA-Za-z0-9_-]{2,20})"], text),
        "peer_double": _first([r"(?:双高校|peer_double)[为是=：:\s]+(0|1|是|否)"], text, re.I),
        "province": _first([r"([\u3400-\u9fff]{2,12}(?:省|自治区))"], text),
        "city": _first([r"([\u3400-\u9fff]{2,12}市)"], text),
        "grade": _first([r"((?:19|20)\d{2}级)"], text),
    }
    numeric_patterns = {
        "salary_upper_bound": r"(?:薪资上限|salary_upper_bound)[为是=：:\s]*(\d+(?:\.\d+)?)",
        "min_sample": r"(?:最小样本(?:数)?|min_sample)[为是=：:\s]*(\d+(?:\.\d+)?)",
    }
    for name, pattern in numeric_patterns.items():
        value = _first([pattern], text, re.I)
        if value is not None:
            inferred[name] = float(value)
    if normalized_dates:
        inferred["start_date"] = normalized_dates[0]
        inferred["end_date"] = normalized_dates[-1]
    if years:
        inferred["start_year"] = years[0]
        inferred["end_year"] = years[-1]

    for name, rule in properties.items():
        if name in output:
            continue
        value = inferred.get(name)
        if value not in (None, ""):
            if name in {"major_name", "enterprise_name", "position_name"}:
                value = re.sub(r"^(?:统计|查询|验证|分析|比较|评估)", "", str(value))
            if name == "peer_double":
                value = {"是": "1", "否": "0"}.get(str(value), value)
            expected = (rule or {}).get("type") if isinstance(rule, dict) else None
            if expected == "integer" and str(value).replace(".0", "").isdigit():
                value = int(float(value))
            output[name] = value
        elif isinstance(rule, dict) and "default" in rule:
            output[name] = rule["default"]
    return output


def _scenario_document(key: str, scenario: Any) -> str:
    schema = getattr(scenario, "params_schema", {}) or {}
    properties = schema.get("properties", {})
    descriptions = " ".join(str(rule.get("description") or "") for rule in properties.values() if isinstance(rule, dict))
    return " ".join([
        key,
        getattr(scenario, "scenario_name", ""),
        getattr(scenario, "description", ""),
        " ".join(getattr(scenario, "keywords", ()) or ()),
        " ".join(properties), descriptions,
        " ".join(getattr(scenario, "return_columns", ()) or ()),
    ])


def match_structured_scenario(task: RetrievalTask, scenarios: dict[str, Any], threshold: float = 0.32, margin: float = 0.03) -> StructuredMatch:
    if not scenarios:
        return StructuredMatch(status="registry_unavailable", hit_terms=[], missing_params=[])
    explicit_name, explicit_params = _explicit_hints(task)
    target = " ".join([task.target_text, *task.required_slots, task.boundary or ""])
    documents = [(key, scenario, _scenario_document(key, scenario)) for key, scenario in scenarios.items()]
    bm25 = BM25Retriever(text_getter=lambda row: row[2]).retrieve(target, documents, len(documents))
    max_score = max((score for _, score in bm25), default=0) or 1.0
    task_tokens = set(tokenize(target))
    meaningful_task_tokens = {token for token in task_tokens if len(token) >= 2}
    ranked: list[tuple[float, str, Any, list[str]]] = []
    for (key, scenario, document), raw_score in bm25:
        display = getattr(scenario, "scenario_name", "")
        words = set(tokenize(document))
        hits = sorted(task_tokens & words)
        if explicit_name and explicit_name in {key, display}:
            score = 10.0
        elif display and display in task.target_text:
            score = 5.0
        else:
            keyword_hits = sum(keyword in task.target_text for keyword in (getattr(scenario, "keywords", ()) or ()))
            meaningful_words = {token for token in words if len(token) >= 2}
            coverage = len(meaningful_task_tokens & meaningful_words) / max(1, len(meaningful_task_tokens))
            score = 0.25 * (raw_score / max_score) + 0.50 * coverage + 0.25 * min(1.0, keyword_hits / 2)
        ranked.append((score, key, scenario, hits))
    ranked.sort(reverse=True, key=lambda row: row[0])
    best_score, key, scenario, hit_terms = ranked[0]
    if best_score < threshold:
        return StructuredMatch(status="not_matched", score=best_score, hit_terms=hit_terms, missing_params=[])
    if len(ranked) > 1 and best_score < 5 and best_score - ranked[1][0] < margin:
        return StructuredMatch(status="ambiguous", score=best_score, hit_terms=hit_terms, missing_params=[])
    schema = getattr(scenario, "params_schema", {}) or {}
    params = _extract_params(task, schema, explicit_params)
    required = set(schema.get("required", []))
    missing = sorted(name for name in required if params.get(name) in (None, ""))
    display = getattr(scenario, "scenario_name", key)
    if missing:
        return StructuredMatch(key, display, params, "missing_params", best_score, hit_terms, missing)
    return StructuredMatch(key, display, params, "matched", best_score, hit_terms, [])
