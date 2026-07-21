"""Deterministic multi-query planning for Public KB retrieval."""
from __future__ import annotations

import re
from typing import Iterable

from .claim_logic import normalize_reverse_hypothesis
from .query_normalization import extract_numeric_expressions, normalize_query_preserving_numbers


def _compact(value: str) -> str:
    return " ".join(normalize_query_preserving_numbers(value).split())[:500]


def build_kb_query_variants(task, *, max_variants: int = 4) -> list[dict[str, str]]:
    target = task.normalized_hypothesis or task.target_text
    line_type = getattr(getattr(task, "line_type", None), "value", getattr(task, "line_type", None))
    neutral = normalize_reverse_hypothesis(task.target_text)["neutral_retrieval_query"] if line_type == "reverse" else target
    numbers = " ".join(extract_numeric_expressions(target))
    subject = " ".join(str(getattr(x, "subject", "")) for x in getattr(task, "atomic_claims", [])[:3] if getattr(x, "subject", ""))
    if not subject:
        subject = target
    variants = [
        ("full", target),
        ("neutral", neutral),
        ("subject_metric_year", " ".join([subject, *getattr(task, "required_slots", [])[:3]])),
        ("numeric_gap", " ".join([subject, numbers, *getattr(task, "required_slots", [])[:2]])),
    ]
    # A single atomic claim whose subject, year and numeric value are already
    # explicit in the full target does not benefit from redundant paraphrases.
    # The V10 contract explicitly permits omitting redundant variants when the
    # complete query covers the claim; this also avoids needless KB fan-out.
    atomics = list(getattr(task, "atomic_claims", []) or [])
    if len(atomics) <= 1:
        claim = atomics[0] if atomics else None
        subject_text = str(getattr(claim, "subject", "") or "")
        numbers_in_target = extract_numeric_expressions(target)
        year_present = bool(re.search(r"(?:19|20)\d{2}", target))
        subject_present = not subject_text or all(token in target for token in re.findall(r"[\u3400-\u9fffA-Za-z]{2,}", subject_text))
        if subject_present and (numbers_in_target or claim is None) and (year_present or not getattr(task, "required_slots", [])):
            variants = variants[:1]
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for kind, query in variants:
        query = _compact(query)
        if query and query.casefold() not in seen:
            seen.add(query.casefold())
            output.append({"query_id": f"{task.task_id}:kb:{len(output)+1}", "variant": kind, "query": query})
        if len(output) >= max_variants:
            break
    return output


def dedupe_query_variants(variants: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result = []
    for item in variants:
        query = _compact(item.get("query", ""))
        if query.casefold() in seen or not query:
            continue
        seen.add(query.casefold())
        result.append({**item, "query": query})
    return result


__all__ = ["build_kb_query_variants", "dedupe_query_variants"]
