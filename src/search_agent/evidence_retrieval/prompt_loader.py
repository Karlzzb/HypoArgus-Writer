"""Versioned prompt resources used by production LLM adapters."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files


_PROMPTS = {
    "evidence_judge": "evidence_judge_zh.txt",
}


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    try:
        filename = _PROMPTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown prompt: {name}") from exc
    text = files("search_agent.evidence_retrieval").joinpath("prompts", filename).read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"prompt resource is empty: {filename}")
    return text
