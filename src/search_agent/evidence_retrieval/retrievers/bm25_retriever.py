"""Small dependency-free BM25 implementation with CJK-aware tokenization."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence, TypeVar

from ..query_normalization import extract_numeric_expressions


T = TypeVar("T")


def tokenize(text: str) -> list[str]:
    raw = text or ""
    text = raw.lower()
    latin = re.findall(r"[a-z0-9]+", text)
    # Preserve decimals, ranges and model identifiers as exact BM25 terms in
    # addition to their loose alphanumeric components.
    exact = [value.casefold().replace(" ", "") for value in extract_numeric_expressions(text)]
    acronyms = [value.casefold() for value in re.findall(r"\b[A-Z]{2,10}\b", raw)]
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", text)
    cjk: list[str] = []
    for run in cjk_runs:
        cjk.extend(list(run))
        cjk.extend(run[i:i + 2] for i in range(len(run) - 1))
    return list(dict.fromkeys([*exact, *acronyms, *latin, *cjk]))


class BM25Retriever:
    def __init__(self, text_getter=lambda x: x.text, *, k1: float = 1.5, b: float = 0.75):
        self.text_getter = text_getter
        self.k1 = k1
        self.b = b

    def retrieve(self, query: str, documents: Sequence[T], top_k: int) -> list[tuple[T, float]]:
        if not documents:
            return []
        tokenized = [tokenize(self.text_getter(doc)) for doc in documents]
        q = set(tokenize(query))
        average = sum(map(len, tokenized)) / len(tokenized) or 1
        dfs = Counter(token for tokens in tokenized for token in set(tokens))
        scored: list[tuple[T, float]] = []
        for doc, tokens in zip(documents, tokenized):
            counts = Counter(tokens)
            score = 0.0
            for token in q:
                df = dfs[token]
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                tf = counts[token]
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * len(tokens) / average)) if tf else 0
            scored.append((doc, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
