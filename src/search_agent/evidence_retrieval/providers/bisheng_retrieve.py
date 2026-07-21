"""Async client for Bisheng's server-side vector retrieval endpoint."""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

import httpx

from ..config import EvidenceRetrievalConfig
from ..errors import ErrorCode, RetrievalError


AccessChecker = Callable[[str, str], bool | Awaitable[bool]]
AuthHeadersProvider = Callable[[list[str], str | None, str], Mapping[str, str] | Awaitable[Mapping[str, str]]]

_PARAGRAPH = re.compile(r"<paragraph_content>(.*?)</paragraph_content>", re.I | re.S)
_TITLE = re.compile(r"<file_title>(.*?)</file_title>", re.I | re.S)
_TAGS = re.compile(r"</?(?:file_title|file_abstract|paragraph_content)>\s*", re.I)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_bisheng_text(raw: str, *, max_chars: int) -> tuple[str, str | None]:
    """Return deterministic Judge text and an optional embedded title."""
    raw = str(raw or "")
    title_match = _TITLE.search(raw)
    paragraph_match = _PARAGRAPH.search(raw)
    selected = paragraph_match.group(1) if paragraph_match else raw
    selected = _MARKDOWN_IMAGE.sub("", selected)
    selected = _TAGS.sub("", selected)
    selected = _CONTROL.sub("", selected)
    selected = re.sub(r"[ \t]+", " ", selected)
    selected = re.sub(r"\n\s*\n+", "\n", selected).strip()
    if not selected:
        selected = _CONTROL.sub("", _MARKDOWN_IMAGE.sub("", raw)).strip()
    return selected[:max_chars], (title_match.group(1).strip() if title_match else None)


@dataclass(slots=True)
class BishengRetrieveChunk:
    knowledge_id: str
    file_id: str
    file_name: str
    chunk_id: str
    chunk_index: int | None
    page: int | None
    text: str
    score: float
    rank: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BishengRetrieveResult:
    chunks: list[BishengRetrieveChunk] = field(default_factory=list)
    resolved_knowledge_ids: list[str] = field(default_factory=list)
    missing_knowledge_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BishengRetrieveClient:
    """Reuse one HTTP client and delegate embedding/vector search to Bisheng."""

    def __init__(
        self,
        config: EvidenceRetrievalConfig,
        client: httpx.AsyncClient | None = None,
        *,
        access_checker: AccessChecker | None = None,
        auth_headers_provider: AuthHeadersProvider | None = None,
    ):
        self.config = config
        self._owns_client = client is None
        timeout = httpx.Timeout(
            config.bisheng_retrieve_timeout_ms / 1000,
            connect=config.bisheng_retrieve_connect_timeout_ms / 1000,
            read=config.bisheng_retrieve_read_timeout_ms / 1000,
        )
        self.client = client or httpx.AsyncClient(
            base_url=config.bisheng_retrieve_base_url or config.bisheng_base_url or "http://invalid.local",
            timeout=timeout,
            http2=True,
            limits=httpx.Limits(
                max_connections=config.bisheng_max_connections,
                max_keepalive_connections=config.bisheng_keepalive_connections,
                keepalive_expiry=30,
            ),
            trust_env=False,
        )
        self.access_checker = access_checker
        self.auth_headers_provider = auth_headers_provider

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _headers(self, ids: list[str], user_id: str | None, origin: str) -> dict[str, str]:
        if self.auth_headers_provider is not None:
            value = self.auth_headers_provider(ids, user_id, origin)
            if inspect.isawaitable(value):
                value = await value
            return dict(value)
        headers: dict[str, str] = {}
        if self.config.bisheng_token:
            headers["Authorization"] = f"Bearer {self.config.bisheng_token.get_secret_value()}"
        return headers

    async def _check_selected_access(self, ids: list[str], user_id: str | None) -> None:
        if not user_id:
            raise RetrievalError(ErrorCode.KB_UNAUTHORIZED, "selected knowledge requires a user id", "selected_kb_retrieval", "bisheng_vector_retrieve")
        if self.access_checker is not None:
            for knowledge_id in ids:
                allowed = self.access_checker(knowledge_id, user_id)
                if inspect.isawaitable(allowed):
                    allowed = await allowed
                if not allowed:
                    raise RetrievalError(ErrorCode.KB_UNAUTHORIZED, "knowledge access denied", "selected_kb_retrieval", "bisheng_vector_retrieve")
            return
        if not self.config.bisheng_access_path:
            raise RetrievalError(ErrorCode.KB_UNAUTHORIZED, "selected knowledge requires an access checker or access endpoint", "selected_kb_retrieval", "bisheng_vector_retrieve")
        for knowledge_id in ids:
            path = self.config.bisheng_access_path.format(knowledge_id=knowledge_id)
            try:
                response = await self.client.get(path, params={"user_id": user_id}, headers=await self._headers([knowledge_id], user_id, "upstream_selected"))
                if response.status_code in {401, 403}:
                    raise RetrievalError(ErrorCode.KB_UNAUTHORIZED, "knowledge access denied", "selected_kb_retrieval", "bisheng_vector_retrieve")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("allowed") is False:
                    raise RetrievalError(ErrorCode.KB_UNAUTHORIZED, "knowledge access denied", "selected_kb_retrieval", "bisheng_vector_retrieve")
            except RetrievalError:
                raise
            except httpx.TimeoutException as exc:
                raise RetrievalError(
                    ErrorCode.KB_TIMEOUT, "knowledge access check timed out",
                    "selected_kb_retrieval", "bisheng_vector_retrieve", True,
                    "http_read_timeout",
                ) from exc
            except (httpx.HTTPError, ValueError) as exc:
                raise RetrievalError(ErrorCode.KB_PROVIDER_ERROR, "knowledge access check failed", "selected_kb_retrieval", "bisheng_vector_retrieve", True) from exc

    async def retrieve(
        self,
        *,
        knowledge_ids: list[str],
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        user_id: str | None,
        origin: str,
    ) -> BishengRetrieveResult:
        ids = list(dict.fromkeys(str(value) for value in knowledge_ids if str(value).strip()))
        if not ids:
            return BishengRetrieveResult()
        if not (self.config.bisheng_retrieve_base_url or self.config.bisheng_base_url):
            raise RetrievalError(ErrorCode.KB_PROVIDER_ERROR, "knowledge retrieve service is not configured", "kb_retrieve", "bisheng_vector_retrieve")
        if origin == "upstream_selected":
            await self._check_selected_access(ids, user_id)

        payload_ids: list[int | str] = [int(value) if value.isdecimal() else value for value in ids]
        payload = {
            "knowledge_ids": payload_ids,
            "query": query,
            "top_k": top_k or self.config.bisheng_retrieve_top_k,
            "score_threshold": score_threshold if score_threshold is not None else self.config.bisheng_retrieve_score_threshold,
        }
        last: Exception | None = None
        response_payload: Any = None
        for attempt in range(self.config.bisheng_retrieve_retry_count + 1):
            try:
                response = await self.client.post(
                    self.config.bisheng_retrieve_path,
                    json=payload,
                    headers=await self._headers(ids, user_id, origin),
                )
                response.raise_for_status()
                response_payload = response.json()
                break
            except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
                last = exc
                # A read/connect timeout already consumes almost the whole
                # provider budget (10s inside a 12s SearchAgent wait layer).
                # Retrying it would only be cancelled by the outer wait_for
                # and misclassify the timeout layer.  Fast HTTP/JSON failures
                # may still use the configured retry.
                if isinstance(exc, httpx.TimeoutException):
                    break
                if attempt < self.config.bisheng_retrieve_retry_count:
                    await asyncio.sleep(0.05 * (2**attempt))
        if response_payload is None:
            if isinstance(last, httpx.TimeoutException):
                layer = "http_read_timeout" if isinstance(last, httpx.ReadTimeout) else "provider_timeout"
                raise RetrievalError(
                    ErrorCode.KB_TIMEOUT, "knowledge retrieve request timed out",
                    "kb_retrieve", "bisheng_vector_retrieve", True, layer,
                ) from last
            if (
                isinstance(last, httpx.HTTPStatusError)
                and last.response.status_code == 504
            ):
                raise RetrievalError(
                    ErrorCode.KB_TIMEOUT, "knowledge retrieve gateway timed out",
                    "kb_retrieve", "bisheng_vector_retrieve", True, "gateway_timeout",
                ) from last
            raise RetrievalError(ErrorCode.KB_PROVIDER_ERROR, "knowledge retrieve request failed", "kb_retrieve", "bisheng_vector_retrieve", True) from last

        if not isinstance(response_payload, dict):
            raise RetrievalError(ErrorCode.KB_PROVIDER_ERROR, "knowledge retrieve response is not an object", "kb_retrieve", "bisheng_vector_retrieve")
        if response_payload.get("success") is False:
            raise RetrievalError(ErrorCode.KB_PROVIDER_ERROR, "knowledge retrieve service reported failure", "kb_retrieve", "bisheng_vector_retrieve", True)

        errors = [str(value)[:200] for value in response_payload.get("errors", [])]
        chunks: list[BishengRetrieveChunk] = []
        for index, row in enumerate(response_payload.get("chunks", []) or []):
            if not isinstance(row, dict):
                errors.append(f"invalid_chunk:{index}")
                continue
            required = (row.get("knowledge_id"), row.get("file_id"), row.get("chunk_id"))
            if any(value is None for value in required):
                errors.append(f"invalid_chunk:{index}")
                continue
            try:
                text, embedded_title = clean_bisheng_text(str(row.get("text") or ""), max_chars=self.config.bisheng_retrieve_max_text_chars)
                if not text:
                    errors.append(f"empty_chunk:{index}")
                    continue
                chunks.append(BishengRetrieveChunk(
                    knowledge_id=str(row["knowledge_id"]),
                    file_id=str(row["file_id"]),
                    file_name=str(row.get("file_name") or embedded_title or ""),
                    chunk_id=str(row["chunk_id"]),
                    chunk_index=int(row["chunk_index"]) if row.get("chunk_index") is not None else None,
                    page=int(row["page"]) if row.get("page") is not None else None,
                    text=text,
                    score=float(row.get("score") or 0),
                    rank=int(row["rank"]) if row.get("rank") is not None else None,
                    metadata=dict(row.get("metadata") or {}),
                ))
            except (TypeError, ValueError):
                errors.append(f"invalid_chunk:{index}")
        return BishengRetrieveResult(
            chunks=chunks,
            resolved_knowledge_ids=[str(value) for value in response_payload.get("resolved_knowledge_ids", [])],
            missing_knowledge_ids=[str(value) for value in response_payload.get("missing_knowledge_ids", [])],
            errors=errors,
        )
