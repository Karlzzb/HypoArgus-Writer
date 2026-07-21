"""Bounded same-document adjacent chunk context assembly."""
from __future__ import annotations

from typing import Iterable


def build_adjacent_context(center: dict, chunks: Iterable[dict], *, max_chars: int = 4000, window: int = 1) -> dict:
    rows = list(chunks)
    knowledge = center.get("knowledge_id")
    file_id = center.get("file_id")
    index = center.get("chunk_index")
    eligible = [row for row in rows if row.get("knowledge_id") == knowledge and row.get("file_id") == file_id]
    if index is None:
        eligible = [center]
    else:
        eligible = [row for row in eligible if isinstance(row.get("chunk_index"), int) and abs(row["chunk_index"] - index) <= window]
    eligible = sorted({str(row.get("chunk_id")): row for row in [*eligible, center]}.values(), key=lambda row: row.get("chunk_index", index or 0))
    included = [row for row in eligible if row.get("chunk_id")]
    text_parts: list[str] = []
    total = 0
    for row in included:
        content = str(row.get("text") or "")
        if total >= max_chars:
            break
        piece = content[:max_chars-total]
        text_parts.append(piece)
        total += len(piece)
    return {
        "center_chunk_id": center.get("chunk_id"),
        "included_chunk_ids": [row.get("chunk_id") for row in included],
        "text": "\n".join(text_parts),
        "text_length": total,
    }


__all__ = ["build_adjacent_context"]
