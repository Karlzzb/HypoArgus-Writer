"""Volcano web search client with bounded retries and normalized candidates."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..config import EvidenceRetrievalConfig
from ..errors import ErrorCode, RetrievalError
from ..schemas import EvidenceCandidate, QueryItem, SourceRef, SourceType, stable_json_hash


class VolcanoWebSearchClient:
    def __init__(self, config: EvidenceRetrievalConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url=config.volcano_base_url,
            timeout=config.web_search_timeout_ms / 1000,
            follow_redirects=False,
            trust_env=False,
        )
        self.diagnostics_by_query: dict[tuple[str, str], dict[str, Any]] = {}

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def search(self, task_id: str, query: QueryItem) -> list[EvidenceCandidate]:
        key = self.config.volcano_api_key
        if key is None:
            raise RetrievalError(ErrorCode.WEB_PROVIDER_ERROR, "web search credential is not configured", "volcano_web_search", "volcano_global_search")
        headers = {"Authorization": f"Bearer {key.get_secret_value()}"}
        payload = {
            "query": query.query,
            "doc_count": self.config.web_doc_count,
            "max_snippet_length": self.config.web_max_snippet_length,
            "max_image_count_per_doc": self.config.web_max_image_count_per_doc,
        }
        last: Exception | None = None
        for attempt in range(self.config.web_retry_count + 1):
            try:
                response = await self.client.post(self.config.volcano_search_path, json=payload, headers=headers)
                response.raise_for_status()
                return self._normalize(task_id, query.query_id, response.json(), http_status=response.status_code)
            except httpx.TimeoutException as exc:
                last = exc
                if attempt < self.config.web_retry_count:
                    await asyncio.sleep(0.05 * (2**attempt))
            except RetrievalError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt < self.config.web_retry_count:
                    await asyncio.sleep(0.05 * (2**attempt))
        code = ErrorCode.WEB_TIMEOUT if isinstance(last, httpx.TimeoutException) else ErrorCode.WEB_PROVIDER_ERROR
        raise RetrievalError(code, "web search request failed", "volcano_web_search", "volcano_global_search", code == ErrorCode.WEB_TIMEOUT) from last

    def _normalize(self, task_id: str, query_id: str, payload: Any, *, http_status: int | None = None) -> list[EvidenceCandidate]:
        recognized_path = "root_list"
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("Result"), dict):
                result = payload["Result"]
                rows = result.get("Documents", [])
                recognized_path = "Result.Documents" if "Documents" in result else ""
            elif isinstance(payload.get("result"), dict):
                result = payload["result"]
                rows = result.get("Documents", []) or result.get("documents", []) or result.get("results", [])
                result_key = next((name for name in ("Documents", "documents", "results") if name in result), None)
                recognized_path = f"result.{result_key}" if result_key else ""
            elif isinstance(payload.get("data"), dict):
                data = payload["data"]
                key = next((name for name in ("results", "items", "web_pages", "web_results", "webPages") if name in data), None)
                rows = data.get(key, []) if key else []
                if key == "webPages" and isinstance(rows, dict):
                    rows = rows.get("value", [])
                    recognized_path = "data.webPages.value"
                else:
                    recognized_path = f"data.{key}" if key else ""
            elif isinstance(payload.get("data"), list):
                rows = payload["data"]
                recognized_path = "data"
            elif isinstance(payload.get("results"), list):
                rows = payload["results"]
                recognized_path = "results"
            elif isinstance(payload.get("items"), list):
                rows = payload["items"]
                recognized_path = "items"
            elif isinstance(payload.get("web_results"), list):
                rows = payload["web_results"]
                recognized_path = "web_results"
            elif isinstance(payload.get("webPages"), dict) and isinstance(payload["webPages"].get("value"), list):
                rows = payload["webPages"]["value"]
                recognized_path = "webPages.value"
            elif isinstance(payload.get("choices"), list):
                rows = []
                for choice in payload["choices"]:
                    message = choice.get("message") if isinstance(choice, dict) else None
                    content = message.get("content") if isinstance(message, dict) else None
                    if not content:
                        continue
                    try:
                        nested = json.loads(content) if isinstance(content, str) else content
                        if isinstance(nested, dict):
                            nested_rows = nested.get("results") or nested.get("items") or nested.get("web_results") or []
                            if isinstance(nested_rows, list):
                                rows.extend(nested_rows)
                    except (TypeError, ValueError):
                        continue
                recognized_path = "choices[].message.content" if rows else ""
            else:
                rows = []
                recognized_path = ""
        else:
            rows = []
            recognized_path = "empty" if payload in (None, "") else ""
        if not isinstance(rows, list):
            recognized_path = ""
            rows = []
        raw_count = len(rows)
        if not recognized_path and payload not in (None, {}, [], ""):
            raw_count = 1
        out: list[EvidenceCandidate] = []
        seen: set[str] = set()
        invalid_count = 0
        for index, row in enumerate(rows[: self.config.web_doc_count]):
            if not isinstance(row, dict):
                invalid_count += 1
                continue
            url = str(row.get("url") or row.get("link") or row.get("Url") or "").strip()
            if not url or url in seen:
                invalid_count += 1
                continue
            parts = urlsplit(url)
            host = (parts.hostname or "").lower().rstrip(".")
            if parts.scheme not in {"http", "https"} or not host or parts.username or parts.password:
                invalid_count += 1
                continue
            allowed = [x.lower().lstrip(".") for x in self.config.web_allowed_domains]
            blocked = [x.lower().lstrip(".") for x in self.config.web_blocked_domains]
            if any(host == d or host.endswith("." + d) for d in blocked):
                invalid_count += 1
                continue
            if self.config.web_whitelist_enabled and not any(host == d or host.endswith("." + d) for d in allowed):
                invalid_count += 1
                continue
            seen.add(url)
            raw_snippet = row.get("snippet") or row.get("summary") or row.get("description") or row.get("Snippet") or ""
            if isinstance(raw_snippet, list):
                snippet = " ".join(
                    str(part.get("Text") or part.get("text") or "") if isinstance(part, dict) else str(part)
                    for part in raw_snippet
                ).strip()
            else:
                snippet = str(raw_snippet).strip()
            document_info = row.get("DocumentInfo") if isinstance(row.get("DocumentInfo"), dict) else {}
            out.append(EvidenceCandidate(
                candidate_id=f"web-{stable_json_hash([task_id, query_id, url])[:20]}",
                task_id=task_id,
                source_type=SourceType.WEB,
                source_name="volcano_global_search",
                source_ref=SourceRef(url=url, query_id=query_id),
                title=str(row.get("title") or row.get("Title") or ""),
                content=snippet[: self.config.web_max_snippet_length],
                metadata={
                    "rank": row.get("Rank") or index + 1,
                    "published_at": row.get("published_at") or row.get("publish_time") or row.get("date") or document_info.get("PublishTime"),
                    "provider_raw_result_count": raw_count,
                    "provider_parsed_result_count": 0,  # filled below
                    "provider_parse_error_count": invalid_count,
                },
                initial_score=float(row.get("score") or 0),
                snippet_only=True,
            ))
        diagnostics = {
            "http_status": http_status,
            "payload_type": type(payload).__name__,
            "top_level_keys": sorted(str(key) for key in payload)[:30] if isinstance(payload, dict) else [],
            "data_keys": sorted(str(key) for key in payload.get("data", {}))[:30] if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else [],
            "result_keys": sorted(str(key) for key in payload.get("Result", {}))[:30] if isinstance(payload, dict) and isinstance(payload.get("Result"), dict) else [],
            "recognized_result_path": recognized_path or None,
            "raw_result_count": raw_count,
            "parsed_result_count": len(out),
            "parse_error_count": invalid_count + int(not bool(recognized_path)),
            "provider_raw_result_count": raw_count,
            "provider_parsed_result_count": len(out),
            "provider_parse_error_count": invalid_count + int(not recognized_path),
            "invalid_url_count": invalid_count,
        }
        self.diagnostics_by_query[(task_id, query_id)] = diagnostics
        for candidate in out:
            candidate.metadata.update(diagnostics)
        metadata = payload.get("ResponseMetadata", {}) if isinstance(payload, dict) else {}
        provider_error = metadata.get("Error") if isinstance(metadata, dict) else None
        result_error_code = payload.get("Result", {}).get("ErrorCode") if isinstance(payload, dict) and isinstance(payload.get("Result"), dict) else None
        if provider_error or result_error_code not in (None, 0, "0"):
            raise RetrievalError(
                ErrorCode.WEB_PROVIDER_ERROR,
                "web provider returned an error envelope",
                "volcano_web_search",
                "volcano_global_search",
                retryable=True,
                details=diagnostics,
            )
        if not recognized_path and payload not in (None, {}, [], ""):
            raise RetrievalError(
                ErrorCode.WEB_RESPONSE_PARSE_ERROR,
                "web provider returned a non-empty payload with an unrecognized result path",
                "volcano_web_search",
                "volcano_global_search",
                details=diagnostics,
            )
        return out
