
"""Scenario-only structured query client. There is intentionally no SQL API."""

from __future__ import annotations

import re
import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

import httpx

from ..config import EvidenceRetrievalConfig
from ..structured.scenario_registry import SCENARIO_REGISTRY
from ..errors import ErrorCode, RetrievalError
from ..structured.contracts import StructuredQueryResponse


_SQL_PATTERN = re.compile(r"\b(select|insert|update|delete|drop|alter|create|truncate|union)\b", re.I)
_monotonic = time.monotonic


class StructuredHealthCheckMeta(TypedDict):
    cache_hit: bool
    check_executed: bool
    ttl_seconds: float
    status: str


@dataclass(slots=True)
class StructuredScenario:
    name: str
    scenario_name: str = ""
    description: str = ""
    params_schema: dict[str, Any] = field(default_factory=dict)
    keywords: tuple[str, ...] = ()
    return_columns: tuple[str, ...] = ()
    healthy: bool = True


class StructuredQueryClient:
    def __init__(self, config: EvidenceRetrievalConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._owns_client = client is None
        # Scenario matching is local and many requests never execute a
        # structured query.  Defer SSL/client construction until the first
        # real HTTP call instead of charging every full-chain CLI run.
        self.client = client
        self._scenario_cache: tuple[float, dict[str, StructuredScenario]] | None = None
        self._health_cache: tuple[float, bool, float] | None = None
        # Loop-agnostic guard: protects the TTL health cache snapshot only, and is
        # *never* held across the HTTP await in ``_actual_health_check``. An
        # ``asyncio.Lock`` held across that await would bind to whichever event loop
        # first contended for it; HypoArgus drives retrieval via a dedicated daemon
        # worker loop, but if the runtime is ever touched from a second loop (worker
        # loop closed+recreated, or a non-bridge ``await``), a loop-bound lock raises
        # ``... is bound to a different event loop``. ``threading.Lock`` is not
        # loop-affine and (held only microseconds around sync cache I/O) cannot
        # deadlock the single-loop async scheduler. Cost: on a cold cache, concurrent
        # callers may each probe once (thundering herd) — health checks are idempotent
        # and TTL-cached, so duplicates are harmless.
        self._health_lock = threading.Lock()

    def _http(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.config.structured_base_url or "http://invalid.local",
                timeout=self.config.structured_timeout_ms / 1000,
                trust_env=False,
            )
        return self.client

    def _headers(self):
        token = self.config.structured_token
        return {"Authorization": f"Bearer {token.get_secret_value()}"} if token else {}

    async def close(self):
        if self._owns_client and self.client is not None:
            client = self.client
            self.client = None
            await client.aclose()

    async def _request_json(self, method: str, path: str, **kwargs) -> Any:
        last: Exception | None = None
        for attempt in range(self.config.structured_retry_count + 1):
            try:
                response = await self._http().request(method, path, headers=self._headers(), **kwargs)
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError:
                    if response.text.strip():
                        return {"status": response.text.strip()}
                    raise
            except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {401, 403}:
                    raise RetrievalError(
                        ErrorCode.STRUCTURED_PERMISSION_DENIED,
                        "structured query permission denied",
                        "structured_query",
                        "structured_query",
                    ) from exc
                last = exc
                if attempt < self.config.structured_retry_count:
                    await asyncio.sleep(0.05 * (2**attempt))
        if isinstance(last, httpx.TimeoutException):
            raise RetrievalError(ErrorCode.STRUCTURED_TIMEOUT, "structured request timed out", "structured_query", "structured_query", True) from last
        raise RetrievalError(ErrorCode.STRUCTURED_UNAVAILABLE, "structured request failed", "structured_query", "structured_query", True) from last

    @staticmethod
    def _health_meta(
        result: bool,
        *,
        cache_hit: bool,
        check_executed: bool,
        ttl_seconds: float,
        status: str | None = None,
    ) -> StructuredHealthCheckMeta:
        return {
            "cache_hit": cache_hit,
            "check_executed": check_executed,
            "ttl_seconds": ttl_seconds,
            "status": status or ("healthy" if result else "unhealthy"),
        }

    def _cached_health(
        self,
        now: float,
    ) -> tuple[bool, StructuredHealthCheckMeta] | None:
        cache = self._health_cache
        if cache is None or cache[0] <= now:
            return None
        _, result, ttl_seconds = cache
        return result, self._health_meta(
            result,
            cache_hit=True,
            check_executed=False,
            ttl_seconds=ttl_seconds,
        )

    async def _actual_health_check(self) -> bool:
        try:
            payload = await self._request_json("GET", "/health")
            if isinstance(payload, dict):
                status = str(payload.get("status", payload.get("health", "healthy"))).lower()
                return status in {"ok", "healthy", "up", "true", "1"}
            return True
        except RetrievalError:
            # Keep the historical public health-probe contract: provider
            # failures are represented as False while query() still raises.
            return False

    async def healthy_with_meta(
        self,
        force: bool = False,
    ) -> tuple[bool, StructuredHealthCheckMeta]:
        """Return health plus per-call cache metadata without changing ``healthy()``."""
        if not self.config.structured_base_url:
            return False, self._health_meta(
                False,
                cache_hit=False,
                check_executed=False,
                ttl_seconds=0.0,
                status="not_configured",
            )

        if not force:
            cached = self._cached_health(_monotonic())
            if cached is not None:
                return cached

        # Double-checked cache under the (loop-agnostic) thread lock. The lock is
        # released before the HTTP await below on purpose — see ``_health_lock`` notes.
        with self._health_lock:
            if not force:
                cached = self._cached_health(_monotonic())
                if cached is not None:
                    return cached

        result = await self._actual_health_check()
        ttl_seconds = float(
            self.config.structured_health_cache_ttl_seconds
            if result
            else self.config.structured_health_failure_cache_ttl_seconds
        )
        with self._health_lock:
            self._health_cache = (_monotonic() + ttl_seconds, result, ttl_seconds)
        return result, self._health_meta(
            result,
            cache_hit=False,
            check_executed=True,
            ttl_seconds=ttl_seconds,
        )

    async def healthy(self, force: bool = False) -> bool:
        result, _ = await self.healthy_with_meta(force=force)
        return result

    async def scenarios(self, force: bool = False) -> dict[str, StructuredScenario]:
        if not force and self._scenario_cache and self._scenario_cache[0] > _monotonic():
            return self._scenario_cache[1]
        scenarios = {
            key: StructuredScenario(
                name=key,
                scenario_name=str(row.get("scenario_name") or key),
                description=str(row.get("description") or ""),
                params_schema=self._normalize_params_schema(row.get("params") or {}),
                keywords=tuple(str(value) for value in row.get("keywords", [])),
                return_columns=tuple(str(value) for value in row.get("return_columns", [])),
            )
            for key, row in SCENARIO_REGISTRY.items()
        }
        self._scenario_cache = (_monotonic() + self.config.structured_scenarios_cache_ttl_seconds, scenarios)
        return scenarios

    async def scenario(self, name: str) -> StructuredScenario:
        scenarios = await self.scenarios()
        key = name if name in scenarios else next(
            (scenario_key for scenario_key, value in scenarios.items() if value.scenario_name == name),
            None,
        )
        if key is None:
            raise RetrievalError(ErrorCode.STRUCTURED_SCENARIO_NOT_FOUND, "scenario is not registered", "structured_query", "structured_query")
        return scenarios[key]

    async def detailed_scenarios(self, force: bool = False) -> dict[str, StructuredScenario]:
        """Return router-ready scenarios with authoritative detail schemas."""
        summaries = await self.scenarios(force=force)
        if not summaries:
            return {}
        return summaries

    async def tables(self) -> list[dict[str, Any]]:
        """Read registration metadata only; returned data is never an SQL surface."""
        try:
            payload = await self._request_json("GET", "/tables")
            rows = payload.get("data", payload.get("tables", [])) if isinstance(payload, dict) else payload
            return [dict(row) for row in rows or [] if isinstance(row, dict)]
        except RetrievalError as exc:
            raise RetrievalError(ErrorCode.STRUCTURED_UNAVAILABLE, "table registry unavailable", "structured_query", "structured_query", True) from exc

    @staticmethod
    def _normalize_params_schema(raw: Any) -> dict[str, Any]:
        """Convert the deployed scenario `params` contract to JSON Schema."""
        if isinstance(raw, list):
            names = [str(name) for name in raw if str(name).strip()]
            return {
                "type": "object", "required": names,
                "properties": {name: {} for name in names},
                "additionalProperties": False,
            }
        if not isinstance(raw, dict):
            return {}
        if "properties" in raw or "required" in raw:
            return dict(raw)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, rule in raw.items():
            if isinstance(rule, dict):
                clean = {key: value for key, value in rule.items() if key != "required"}
                properties[str(name)] = clean
                if bool(rule.get("required")):
                    required.append(str(name))
            else:
                properties[str(name)] = {}
        return {
            "type": "object", "required": required,
            "properties": properties, "additionalProperties": False,
        }

    @staticmethod
    def _validate_params(schema: dict[str, Any], params: dict[str, Any]) -> None:
        if any(_SQL_PATTERN.search(str(k)) or _SQL_PATTERN.search(str(v)) for k, v in params.items()):
            raise RetrievalError(ErrorCode.STRUCTURED_PARAM_INVALID, "SQL-like input is forbidden", "structured_query", "structured_query")
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        if required - set(params) or (properties and set(params) - set(properties)):
            raise RetrievalError(ErrorCode.STRUCTURED_PARAM_INVALID, "parameters do not match scenario schema", "structured_query", "structured_query")
        if schema.get("additionalProperties") is False and set(params) - set(properties):
            raise RetrievalError(ErrorCode.STRUCTURED_PARAM_INVALID, "additional parameters are forbidden", "structured_query", "structured_query")
        type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        for name, value in params.items():
            rule = properties.get(name) or {}
            expected = type_map.get(rule.get("type"))
            if expected and (isinstance(value, bool) and rule.get("type") in {"integer", "number"} or not isinstance(value, expected)):
                raise RetrievalError(ErrorCode.STRUCTURED_PARAM_INVALID, f"invalid type for parameter {name}", "structured_query", "structured_query")
            if "enum" in rule and value not in rule["enum"]:
                raise RetrievalError(ErrorCode.STRUCTURED_PARAM_INVALID, f"parameter {name} is outside enum", "structured_query", "structured_query")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in rule and value < rule["minimum"] or "maximum" in rule and value > rule["maximum"]:
                    raise RetrievalError(ErrorCode.STRUCTURED_PARAM_INVALID, f"parameter {name} is outside numeric bounds", "structured_query", "structured_query")
            if isinstance(value, str) and rule.get("pattern") and not re.fullmatch(rule["pattern"], value):
                raise RetrievalError(ErrorCode.STRUCTURED_PARAM_INVALID, f"parameter {name} does not match pattern", "structured_query", "structured_query")
            if isinstance(value, list) and isinstance(rule.get("items"), dict):
                item_type = type_map.get(rule["items"].get("type"))
                if item_type and any(not isinstance(item, item_type) for item in value):
                    raise RetrievalError(ErrorCode.STRUCTURED_PARAM_INVALID, f"parameter {name} has invalid array items", "structured_query", "structured_query")

    async def query(self, scenario_name: str, params: dict[str, Any]) -> StructuredQueryResponse:
        scenario = await self.scenario(scenario_name)
        if not scenario.healthy:
            raise RetrievalError(ErrorCode.STRUCTURED_SCENARIO_NOT_FOUND, "scenario is not registered and healthy", "structured_query", "structured_query")
        self._validate_params(scenario.params_schema, params)
        try:
            payload = await self._request_json(
                "POST", "/query",
                json={"scenario_name": scenario.scenario_name, "params": params},
            )
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            rows = data.get("rows", data.get("records", [])) if isinstance(data, dict) else data
            normalized = [dict(row) for row in rows or [] if isinstance(row, dict)]
            meta = data if isinstance(data, dict) else {}
            root = payload if isinstance(payload, dict) else {}
            columns = meta.get("columns") or root.get("columns") or list(scenario.return_columns)
            return StructuredQueryResponse(
                rows=normalized,
                columns=[str(value) for value in columns or []],
                row_count=int(meta.get("row_count", root.get("row_count", len(normalized))) or len(normalized)),
                dataset_id=meta.get("dataset_id") or root.get("dataset_id"),
                query_execution_id=meta.get("query_execution_id") or root.get("query_execution_id"),
                server_elapsed_ms=meta.get("server_elapsed_ms") or root.get("server_elapsed_ms"),
            )
        except RetrievalError as exc:
            if exc.code in {ErrorCode.STRUCTURED_TIMEOUT, ErrorCode.STRUCTURED_PARAM_INVALID, ErrorCode.STRUCTURED_SCENARIO_NOT_FOUND}:
                raise
            raise RetrievalError(ErrorCode.STRUCTURED_UNAVAILABLE, "structured query failed", "structured_query", "structured_query", True) from exc