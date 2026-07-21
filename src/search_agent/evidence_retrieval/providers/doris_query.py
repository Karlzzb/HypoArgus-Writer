"""Direct, pooled, read-only Doris client for audited Function Tool scenarios."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError

from ..config import EvidenceRetrievalConfig
from ..errors import ErrorCode, RetrievalError
from ..structured.contracts import StructuredQueryResponse
from ..structured.scenario_loader import load_scenario_templates
from ..structured.scenario_registry import SCENARIO_REGISTRY

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


class DorisQueryClient:
    """One process-wide SQLAlchemy QueuePool; every exposed query is pre-audited."""

    def __init__(self, config: EvidenceRetrievalConfig, engine: Engine | None = None):
        self.config = config
        self._engine_instance = engine
        self._owns_engine = engine is None
        self._engine_lock = asyncio.Lock()
        self._health_lock = asyncio.Lock()
        self._health_cache: tuple[float, bool, float] | None = None
        self._query_cache: dict[
            tuple[str, str], tuple[float, StructuredQueryResponse]
        ] = {}
        self._query_inflight: dict[
            tuple[str, str], asyncio.Task[StructuredQueryResponse]
        ] = {}
        self.query_cache_hits = 0
        self.query_singleflight_hits = 0
        self._closed = False

    @property
    def configured(self) -> bool:
        return bool(
            self.config.doris_host
            and self.config.doris_username
            and self.config.doris_password
        )

    def _create_engine(self) -> Engine:
        password = self.config.doris_password
        if not self.configured or password is None:
            raise RetrievalError(
                ErrorCode.STRUCTURED_UNAVAILABLE,
                "Doris is not configured",
                "structured_query",
                "doris",
            )
        url = URL.create(
            "mysql+pymysql",
            username=self.config.doris_username,
            password=password.get_secret_value(),
            host=self.config.doris_host,
            port=self.config.doris_port,
            database=self.config.doris_database or None,
        )
        return create_engine(
            url,
            pool_size=self.config.doris_pool_size,
            max_overflow=self.config.doris_max_overflow,
            pool_timeout=self.config.doris_pool_timeout_seconds,
            pool_recycle=self.config.doris_pool_recycle_seconds,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": self.config.doris_connect_timeout_seconds,
                "read_timeout": self.config.doris_read_timeout_seconds,
                "write_timeout": self.config.doris_read_timeout_seconds,
                "charset": self.config.doris_charset,
            },
        )

    async def _engine(self) -> Engine:
        if self._closed:
            raise RetrievalError(
                ErrorCode.STRUCTURED_UNAVAILABLE,
                "Doris client is closed",
                "structured_query",
                "doris",
            )
        if self._engine_instance is not None:
            return self._engine_instance
        async with self._engine_lock:
            if self._engine_instance is None:
                self._engine_instance = self._create_engine()
            return self._engine_instance

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

    def _cached_health(self) -> tuple[bool, StructuredHealthCheckMeta] | None:
        cached = self._health_cache
        if cached is None or cached[0] <= _monotonic():
            return None
        _, result, ttl = cached
        return result, self._health_meta(
            result, cache_hit=True, check_executed=False, ttl_seconds=ttl
        )

    @staticmethod
    def _health_sync(engine: Engine) -> None:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    async def healthy_with_meta(
        self, force: bool = False
    ) -> tuple[bool, StructuredHealthCheckMeta]:
        if not self.configured:
            return False, self._health_meta(
                False,
                cache_hit=False,
                check_executed=False,
                ttl_seconds=0.0,
                status="not_configured",
            )
        if not force:
            cached = self._cached_health()
            if cached is not None:
                return cached
        async with self._health_lock:
            if not force:
                cached = self._cached_health()
                if cached is not None:
                    return cached
            try:
                engine = await self._engine()
                await asyncio.wait_for(
                    asyncio.to_thread(self._health_sync, engine),
                    timeout=self.config.doris_connect_timeout_seconds + 1,
                )
                result = True
            except (TimeoutError, SQLAlchemyError, RetrievalError):
                result = False
            ttl = float(
                self.config.structured_health_cache_ttl_seconds
                if result
                else self.config.structured_health_failure_cache_ttl_seconds
            )
            self._health_cache = (_monotonic() + ttl, result, ttl)
            return result, self._health_meta(
                result, cache_hit=False, check_executed=True, ttl_seconds=ttl
            )

    async def healthy(self, force: bool = False) -> bool:
        result, _ = await self.healthy_with_meta(force=force)
        return result

    @staticmethod
    def _params_schema(raw: dict[str, Any]) -> dict[str, Any]:
        required: list[str] = []
        properties: dict[str, Any] = {}
        for name, rule_value in raw.items():
            rule = dict(rule_value) if isinstance(rule_value, dict) else {}
            if rule.get("required") and "default" not in rule:
                required.append(str(name))
            properties[str(name)] = {
                key: value for key, value in rule.items() if key not in {"required", "wildcard"}
            }
        return {
            "type": "object",
            "required": required,
            "properties": properties,
            "additionalProperties": False,
        }

    async def scenarios(self, force: bool = False) -> dict[str, StructuredScenario]:
        del force
        templates = load_scenario_templates()
        return {
            key: StructuredScenario(
                name=key,
                scenario_name=template.name,
                description=template.description,
                params_schema=self._params_schema(template.params),
                keywords=tuple(str(value) for value in SCENARIO_REGISTRY[key]["keywords"]),
                return_columns=template.return_columns,
            )
            for key, template in templates.items()
        }

    async def detailed_scenarios(self, force: bool = False) -> dict[str, StructuredScenario]:
        return await self.scenarios(force=force)

    async def scenario(self, name: str) -> StructuredScenario:
        scenarios = await self.scenarios()
        if name in scenarios:
            return scenarios[name]
        for value in scenarios.values():
            if value.scenario_name == name:
                return value
        raise RetrievalError(
            ErrorCode.STRUCTURED_SCENARIO_NOT_FOUND,
            "scenario is not registered",
            "structured_query",
            "doris",
        )

    async def tables(self) -> list[dict[str, Any]]:
        return []

    def _query_sync(
        self, engine: Engine, scenario_key: str, params: dict[str, Any]
    ) -> StructuredQueryResponse:
        template = load_scenario_templates()[scenario_key]
        try:
            bound = template.bind_params(params)
        except (TypeError, ValueError) as exc:
            raise RetrievalError(
                ErrorCode.STRUCTURED_PARAM_INVALID,
                str(exc),
                "structured_query",
                scenario_key,
            ) from exc
        row_limit = min(template.limit, self.config.structured_max_rows)
        sql = f"{template.sql}\nLIMIT {row_limit}"
        started = time.monotonic()
        try:
            with engine.connect() as connection:
                connection.exec_driver_sql(
                    f"SET query_timeout = {self.config.doris_query_timeout_seconds}"
                )
                result = connection.execute(text(sql), bound)
                rows = [dict(row) for row in result.mappings().all()]
                columns = [str(column) for column in result.keys()]
        except OperationalError as exc:
            code = getattr(getattr(exc, "orig", None), "args", [None])[0]
            error_code = (
                ErrorCode.STRUCTURED_PERMISSION_DENIED
                if code in {1044, 1045, 1142, 1227}
                else ErrorCode.STRUCTURED_UNAVAILABLE
            )
            raise RetrievalError(
                error_code,
                "Doris query failed",
                "structured_query",
                scenario_key,
                retryable=error_code == ErrorCode.STRUCTURED_UNAVAILABLE,
            ) from exc
        except DBAPIError as exc:
            raise RetrievalError(
                ErrorCode.STRUCTURED_UNAVAILABLE,
                "Doris query failed",
                "structured_query",
                scenario_key,
                retryable=bool(exc.connection_invalidated),
            ) from exc
        elapsed = int((time.monotonic() - started) * 1000)
        return StructuredQueryResponse(
            rows=rows,
            columns=columns or list(template.return_columns),
            row_count=len(rows),
            dataset_id=self.config.doris_dataset_id,
            query_execution_id=f"doris-{uuid.uuid4().hex}",
            server_elapsed_ms=elapsed,
        )

    async def _query_uncached(
        self, scenario_name: str, params: dict[str, Any]
    ) -> StructuredQueryResponse:
        scenario = await self.scenario(scenario_name)
        engine = await self._engine()
        last: RetrievalError | None = None
        for attempt in range(self.config.structured_retry_count + 1):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._query_sync, engine, scenario.name, params),
                    timeout=self.config.structured_timeout_ms / 1000,
                )
            except TimeoutError as exc:
                last = RetrievalError(
                    ErrorCode.STRUCTURED_TIMEOUT,
                    "Doris query timed out",
                    "structured_query",
                    scenario.name,
                    retryable=True,
                )
                if attempt >= self.config.structured_retry_count:
                    raise last from exc
            except RetrievalError as exc:
                last = exc
                if not exc.retryable or attempt >= self.config.structured_retry_count:
                    raise
            await asyncio.sleep(0.05 * (2**attempt))
        assert last is not None
        raise last

    async def query(
        self, scenario_name: str, params: dict[str, Any]
    ) -> StructuredQueryResponse:
        """Execute one stable scenario+parameter query with short singleflight.

        A report often maps several Search Tasks to the same Doris scenario.
        Reusing that immutable read result keeps the DB call request-scoped in
        practice without creating a new connection or repeating identical SQL.
        """

        key = (
            scenario_name,
            json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
        )
        now = _monotonic()
        cached = self._query_cache.get(key)
        if cached is not None and cached[0] > now:
            self.query_cache_hits += 1
            return cached[1]
        future = self._query_inflight.get(key)
        if future is None:
            future = asyncio.create_task(self._query_uncached(scenario_name, params))
            self._query_inflight[key] = future
        else:
            self.query_singleflight_hits += 1
        try:
            response = await asyncio.shield(future)
            ttl = float(self.config.structured_query_cache_ttl_seconds)
            if ttl > 0:
                self._query_cache[key] = (_monotonic() + ttl, response)
            return response
        finally:
            if future.done():
                self._query_inflight.pop(key, None)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._query_cache.clear()
        engine = self._engine_instance
        self._engine_instance = None
        if self._owns_engine and engine is not None:
            await asyncio.to_thread(engine.dispose)

    close = aclose


__all__ = ["DorisQueryClient", "StructuredHealthCheckMeta", "StructuredScenario"]
