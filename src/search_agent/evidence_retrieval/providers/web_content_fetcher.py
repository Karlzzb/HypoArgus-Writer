"""SSRF-safe web body fetch, extraction and chunking."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import os
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import urljoin, urlsplit

import httpx

from ..config import EvidenceRetrievalConfig
from ..errors import ErrorCode, RetrievalError
from ..schemas import ErrorDetail, EvidenceCandidate, SourceRef, SourceType, stable_json_hash


_TEXT_TYPES = ("text/html", "text/plain", "application/xhtml+xml", "application/pdf")
_DROP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"}


@dataclass(slots=True)
class FetchResult:
    candidates: list[EvidenceCandidate] = field(default_factory=list)
    errors: list[ErrorDetail] = field(default_factory=list)
    degraded_to_snippet: bool = False


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in _DROP_TAGS:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in _DROP_TAGS and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth and data.strip():
            self.parts.append(data.strip())


def _unsafe_ip(ip: str) -> bool:
    address = ipaddress.ip_address(ip)
    return not address.is_global or address.is_loopback or address.is_link_local or address.is_private or address.is_reserved


class WebContentFetcher:
    def __init__(self, config: EvidenceRetrievalConfig, client: httpx.AsyncClient | None = None, resolver=None, peer_ip_getter=None):
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=config.web_fetch_timeout_ms / 1000,
            follow_redirects=False,
            trust_env=False,
        )
        self.resolver = resolver or self._resolve
        self.peer_ip_getter = peer_ip_getter

    async def close(self):
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    async def _resolve(host: str) -> list[str]:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None, type=socket.SOCK_STREAM)
        return list({item[4][0] for item in infos})

    async def validate_url(self, url: str) -> frozenset[str]:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise RetrievalError(ErrorCode.WEB_WHITELIST_BLOCKED, "URL scheme or authority is not allowed", "fetch_web_content", "web_content_fetcher")
        host = parts.hostname.rstrip(".").lower()
        if host == "localhost" or host.endswith(".localhost") or host == "169.254.169.254":
            raise RetrievalError(ErrorCode.WEB_WHITELIST_BLOCKED, "local or metadata endpoint is blocked", "fetch_web_content", "web_content_fetcher")
        allowed = [x.lower().lstrip(".") for x in self.config.web_allowed_domains]
        blocked = [x.lower().lstrip(".") for x in self.config.web_blocked_domains]
        if any(host == d or host.endswith("." + d) for d in blocked):
            raise RetrievalError(ErrorCode.WEB_WHITELIST_BLOCKED, "domain is blocked", "fetch_web_content", "web_content_fetcher")
        if self.config.web_whitelist_enabled and not any(host == d or host.endswith("." + d) for d in allowed):
            raise RetrievalError(ErrorCode.WEB_WHITELIST_BLOCKED, "domain is outside the configured allowlist", "fetch_web_content", "web_content_fetcher")
        try:
            if _unsafe_ip(host):
                raise RetrievalError(ErrorCode.WEB_WHITELIST_BLOCKED, "non-public address is blocked", "fetch_web_content", "web_content_fetcher")
            return frozenset({str(ipaddress.ip_address(host))})
        except ValueError:
            try:
                addresses = await self.resolver(host)
            except Exception as exc:
                raise RetrievalError(ErrorCode.WEB_FETCH_ERROR, "host resolution failed", "fetch_web_content", "web_content_fetcher", True) from exc
            if not addresses or any(_unsafe_ip(ip) for ip in addresses):
                raise RetrievalError(ErrorCode.WEB_WHITELIST_BLOCKED, "host resolves to a non-public address", "fetch_web_content", "web_content_fetcher")
            return frozenset(str(ipaddress.ip_address(ip)) for ip in addresses)

    def _connected_peer_ip(self, response: httpx.Response) -> str | None:
        if self.peer_ip_getter is not None:
            value = self.peer_ip_getter(response)
            return str(value) if value else None
        stream = response.extensions.get("network_stream")
        if stream is None:
            return None
        try:
            server = stream.get_extra_info("server_addr")
            if server:
                return str(server[0] if isinstance(server, tuple) else server)
            sock = stream.get_extra_info("socket")
            if sock is not None:
                return str(sock.getpeername()[0])
        except (AttributeError, OSError, TypeError):
            return None
        return None

    async def _trusted_proxy_ips(self) -> frozenset[str]:
        if not self.config.web_trusted_proxy_enabled:
            return frozenset()
        ips: set[str] = set()
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
            value = os.environ.get(name) or os.environ.get(name.lower())
            host = urlsplit(value).hostname if value else None
            if not host:
                continue
            try:
                ips.update(str(ipaddress.ip_address(address)) for address in await self.resolver(host))
            except Exception:
                continue
        return frozenset(ips)

    def _validate_connected_peer(self, response: httpx.Response, resolved: frozenset[str], trusted_proxy_ips: frozenset[str] = frozenset()) -> None:
        peer = self._connected_peer_ip(response)
        # Injected transports (notably MockTransport) have no real socket. A
        # production-owned client fails closed when the peer cannot be proven.
        if peer is None and not self._owns_client:
            return
        try:
            normalized = str(ipaddress.ip_address(peer)) if peer else None
        except ValueError:
            normalized = None
        via_trusted_proxy = normalized is not None and normalized in trusted_proxy_ips
        if normalized is None or (not via_trusted_proxy and (_unsafe_ip(normalized) or normalized not in resolved)):
            raise RetrievalError(
                ErrorCode.WEB_WHITELIST_BLOCKED,
                "connected peer does not match the validated public DNS result",
                "fetch_web_content", "web_content_fetcher",
            )

    async def fetch(self, candidate: EvidenceCandidate) -> FetchResult:
        url = candidate.source_ref.url
        if not url:
            error = ErrorDetail(code=ErrorCode.WEB_FETCH_ERROR.value, node="fetch_web_content", tool="web_content_fetcher", reason="candidate URL is missing")
            return FetchResult([candidate], [error], True)
        current = url
        try:
            for _ in range(5):
                resolved = await self.validate_url(current)
                trusted_proxy_ips = await self._trusted_proxy_ips()
                async with self.client.stream("GET", current, headers={"User-Agent": "SearchAgentEvidenceFetcher/1.0"}) as response:
                    self._validate_connected_peer(response, resolved, trusted_proxy_ips)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise RetrievalError(ErrorCode.WEB_FETCH_ERROR, "redirect without location", "fetch_web_content", "web_content_fetcher")
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type and content_type not in _TEXT_TYPES:
                        raise RetrievalError(ErrorCode.WEB_CONTENT_UNSUPPORTED, "content type is unsupported", "fetch_web_content", "web_content_fetcher")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > self.config.web_max_response_bytes:
                            raise RetrievalError(ErrorCode.WEB_FETCH_ERROR, "response exceeds size limit", "fetch_web_content", "web_content_fetcher")
                    # PDF parsing and large HTML extraction are CPU-bound and
                    # must not block the shared async event loop/deadline.
                    text = await asyncio.to_thread(self._extract, bytes(body), content_type)
                    if not text.strip():
                        raise RetrievalError(ErrorCode.WEB_CONTENT_UNSUPPORTED, "no extractable text", "fetch_web_content", "web_content_fetcher")
                    return FetchResult(self._chunks(candidate, current, text))
            raise RetrievalError(ErrorCode.WEB_FETCH_ERROR, "too many redirects", "fetch_web_content", "web_content_fetcher")
        except RetrievalError as exc:
            error = ErrorDetail(code=exc.code.value, node=exc.node, tool=exc.tool, retryable=exc.retryable, reason=exc.message)
            if exc.code == ErrorCode.WEB_WHITELIST_BLOCKED:
                return FetchResult([], [error], False)
            return FetchResult([candidate], [error], True)
        except httpx.TimeoutException:
            error = ErrorDetail(code=ErrorCode.WEB_FETCH_ERROR.value, node="fetch_web_content", tool="web_content_fetcher", retryable=True, reason="web content request timed out")
            return FetchResult([candidate], [error], True)
        except (httpx.HTTPError, UnicodeError) as exc:
            # Search snippets remain unmodified and explicitly low-trust.
            error = ErrorDetail(code=ErrorCode.WEB_FETCH_ERROR.value, node="fetch_web_content", tool="web_content_fetcher", retryable=True, reason=f"{type(exc).__name__}: web content fetch failed")
            return FetchResult([candidate], [error], True)

    def _extract(self, body: bytes, content_type: str) -> str:
        if content_type == "application/pdf":
            if len(body) < self.config.web_pdf_min_bytes or not body.startswith(b"%PDF-"):
                raise RetrievalError(ErrorCode.WEB_CONTENT_TYPE_MISMATCH, "PDF content type did not contain a valid PDF payload", "fetch_web_content", "web_content_fetcher")
            try:
                from pypdf import PdfReader
                return "\n".join((page.extract_text() or "") for page in PdfReader(BytesIO(body)).pages)
            except Exception as exc:
                raise RetrievalError(ErrorCode.WEB_CONTENT_UNSUPPORTED, "PDF has no directly extractable text", "fetch_web_content", "web_content_fetcher") from exc
        text = body.decode("utf-8", errors="replace")
        if "html" not in content_type and "<html" not in text[:500].lower():
            return text
        parser = _TextExtractor()
        parser.feed(text)
        return re.sub(r"\s+", " ", "\n".join(parser.parts)).strip()

    def _chunks(self, source: EvidenceCandidate, url: str, text: str) -> list[EvidenceCandidate]:
        size = self.config.web_chunk_chars
        chunks: list[str] = []
        cursor = 0
        while cursor < len(text):
            end = min(len(text), cursor + size)
            if end < len(text):
                split = max(text.rfind("。", cursor, end), text.rfind("\n", cursor, end), text.rfind(" ", cursor, end))
                if split > cursor + size // 2:
                    end = split + 1
            chunks.append(text[cursor:end].strip())
            cursor = end
        return [EvidenceCandidate(
            candidate_id=f"webbody-{stable_json_hash([source.task_id, url, i, chunk])[:20]}",
            task_id=source.task_id, source_type=SourceType.WEB, source_name="web_content_fetcher",
            source_ref=SourceRef(url=url, query_id=source.source_ref.query_id), title=source.title,
            content=chunk, metadata={**source.metadata, "chunk_index": i},
            initial_score=source.initial_score, snippet_only=False,
        ) for i, chunk in enumerate(chunks) if chunk]
