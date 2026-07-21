"""Query normalization and exact numeric-expression extraction.

The same parser is shared by Web query construction, BM25 tokenization and
exact-value ranking so decimals/ranges/models cannot be normalized differently
at different stages of the pipeline.
"""

from __future__ import annotations

import re


_UNIT = r"(?:万亿美元|万亿元|亿美元|亿元人民币|亿元|万元|万台|万人|万件|万套|美元|人民币|万|亿|年|元|台|人|个|项|件|套)?"
_NUMBER = r"-?\d+(?:\.\d+)?(?:%|％)?"
_RANGE_SEP = r"(?:至|到|—|–|-|~|～)"
_EXPRESSION_RE = re.compile(
    rf"(?<![A-Za-z0-9]){_NUMBER}(?:\s*{_UNIT})?(?:\s*{_RANGE_SEP}\s*{_NUMBER}(?:\s*{_UNIT})?)?"
    r"|\b[A-Za-z]{1,12}-\d+(?:\.\d+)?\b"
    r"|\bFigure\s+\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)


def extract_numeric_expressions(text: str) -> list[str]:
    """Return stable, de-duplicated numeric/model expressions."""
    output: list[str] = []
    seen: set[str] = set()
    for match in _EXPRESSION_RE.finditer(text or ""):
        raw = match.group(0).strip()
        value = re.sub(r"\s+", " ", raw) if raw.casefold().startswith("figure") else re.sub(r"\s+", "", raw)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def normalize_query_preserving_numbers(text: str) -> str:
    """Normalize punctuation/spacing without damaging factual expressions."""
    raw = str(text or "")
    protected: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        marker = f"NUMEXPRTOKEN{len(protected)}X"
        raw_match = match.group(0).strip()
        protected[marker] = (
            re.sub(r"\s+", " ", raw_match)
            if raw_match.casefold().startswith("figure")
            else re.sub(r"\s+", "", raw_match)
        )
        return marker

    guarded = _EXPRESSION_RE.sub(protect, raw)
    # ASCII full stop and hyphen may be semantic inside numbers/models, but all
    # such cases are protected above. They are safe to treat as punctuation now.
    cleaned = re.sub(r"[，。；：、！？,.!?;:()（）\[\]【】{}<>《》\"'“”‘’]+", " ", guarded)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for marker, value in protected.items():
        cleaned = cleaned.replace(marker, value)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_reverse_hypothesis(text: str):
    """Compatibility export for callers that keep query helpers together."""
    from .claim_logic import normalize_reverse_hypothesis as _normalize
    return _normalize(text)


__all__ = ["extract_numeric_expressions", "normalize_query_preserving_numbers", "normalize_reverse_hypothesis"]
