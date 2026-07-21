"""Real Function Tool registry for all audited Structured scenarios."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from .contracts import NoStructuredQueryArgs, StructuredQueryResponse, StructuredToolResult
from .scenario_registry import SCENARIO_REGISTRY
from ..schemas import StrictModel


_TOOL_NAMES = {
    "scenario_1": "query_enterprise_qualification",
    "scenario_2": "query_enterprise_intern_count",
    "scenario_3": "compare_hightech_intern_salary",
    "scenario_4": "query_major_intern_salary_quality",
    "scenario_5": "query_major_job_recruitment_trend",
    "scenario_6": "query_peer_major_gap",
    "scenario_7": "query_position_skill_graph",
    "scenario_8": "query_regional_talent_demand_heatmap",
    "scenario_9": "query_regional_salary_heat_index",
    "scenario_10": "query_major_internship_distribution",
    "scenario_11": "query_major_student_growth",
    "scenario_12": "query_student_origin_internship_city",
}


@dataclass(slots=True)
class StructuredToolDefinition:
    scenario_key: str
    domain: str
    tool: BaseTool
    args_schema: type[BaseModel]
    return_columns: tuple[str, ...]
    scenario_name: str = ""
    description: str = ""
    keywords: tuple[str, ...] = ()
    fallback_safe: bool = False
    permission: str = "READ_ONLY"
    result_mapper: str = "map_structured_tool_result"


def _python_type(rule: dict[str, Any]):
    return {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list[Any],
        "object": dict[str, Any],
    }.get(str(rule.get("type") or "string"), Any)


def _args_model(scenario_key: str, row: dict[str, Any]) -> type[BaseModel]:
    fields: dict[str, tuple[Any, Any]] = {}
    for name, raw_rule in (row.get("params") or {}).items():
        rule = raw_rule if isinstance(raw_rule, dict) else {}
        annotation = _python_type(rule)
        required = bool(rule.get("required")) and "default" not in rule
        if not required:
            annotation = annotation | None
        default = ... if required else rule.get("default")
        fields[str(name)] = (
            annotation,
            Field(default=default, description=str(rule.get("description") or name)),
        )
    fields["target_task_ids"] = (
        list[str],
        Field(description="该结构化调用服务的 SearchAgent task_id；不会传给数据库接口"),
    )
    fields["tool_call_id"] = (
        str | None,
        Field(default=None, description="Internal ToolNode call identifier; callers must omit it."),
    )
    return create_model(
        f"{scenario_key.title().replace('_', '')}Args",
        __base__=StrictModel,
        **fields,
    )


def _domain(row: dict[str, Any]) -> str:
    text = f"{row.get('scenario_name', '')} {row.get('description', '')}"
    if "薪资" in text:
        return "salary"
    if "实习" in text:
        return "internship"
    if "企业" in text:
        return "enterprise"
    return "employment"


def _normalize_response(value: Any, default_columns: tuple[str, ...]) -> StructuredQueryResponse:
    if isinstance(value, StructuredQueryResponse):
        return value
    rows = [dict(row) for row in (value or []) if isinstance(row, dict)]
    return StructuredQueryResponse(rows=rows, columns=list(default_columns), row_count=len(rows))


def _error_result(tool_call_id: str, tool_name: str, scenario_key: str, arguments: dict[str, Any], targets: list[str], exc: BaseException) -> StructuredToolResult:
    from ..errors import ErrorCode, RetrievalError
    if isinstance(exc, RetrievalError) and exc.code == ErrorCode.STRUCTURED_PARAM_INVALID:
        status, code = "INVALID_ARGUMENT", exc.code.value
    elif isinstance(exc, RetrievalError) and exc.code == ErrorCode.STRUCTURED_PERMISSION_DENIED:
        status, code = "PERMISSION_DENIED", exc.code.value
    else:
        status, code = "TOOL_ERROR", getattr(getattr(exc, "code", None), "value", type(exc).__name__)
    return StructuredToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        scenario_key=scenario_key,
        status=status,
        query_summary=f"Read-only {scenario_key} query failed",
        arguments=arguments,
        target_task_ids=targets,
        error_code=str(code),
        error_message=f"{type(exc).__name__}: {exc}",
    )


def build_structured_tool_registry(
    client: Any,
    organization_context: dict[str, Any] | None = None,
    available_scenario_keys: set[str] | None = None,
) -> dict[str, StructuredToolDefinition]:
    organization_context = organization_context or {}
    registry: dict[str, StructuredToolDefinition] = {}
    for scenario_key, row in SCENARIO_REGISTRY.items():
        if available_scenario_keys is not None and scenario_key not in available_scenario_keys:
            continue
        args_schema = _args_model(scenario_key, row)
        tool_name = _TOOL_NAMES[scenario_key]
        return_columns = tuple(str(value) for value in row.get("return_columns", []))

        async def execute_scenario(
            tool_call_id: str | None = None,
            _scenario_key=scenario_key,
            _tool_name=tool_name,
            _return_columns=return_columns,
            **kwargs,
        ):
            tool_call_id = str(tool_call_id or kwargs.pop("tool_call_id", None) or f"direct-{uuid.uuid4().hex}")
            target_task_ids = list(kwargs.pop("target_task_ids", []))
            if "school_id" in (SCENARIO_REGISTRY[_scenario_key].get("params") or {}) and not kwargs.get("school_id"):
                kwargs["school_id"] = organization_context.get("school_id")
            if "my_school_id" in (SCENARIO_REGISTRY[_scenario_key].get("params") or {}) and not kwargs.get("my_school_id"):
                kwargs["my_school_id"] = organization_context.get("school_id")
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            try:
                response = _normalize_response(await client.query(_scenario_key, kwargs), _return_columns)
                result = StructuredToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=_tool_name,
                    scenario_key=_scenario_key,
                    status="SUCCESS" if response.rows else "NO_DATA",
                    query_summary=f"{_scenario_key} read-only query",
                    arguments=kwargs,
                    columns=response.columns or list(_return_columns),
                    rows=response.rows,
                    row_count=response.row_count,
                    dataset_id=response.dataset_id,
                    query_execution_id=response.query_execution_id,
                    server_elapsed_ms=response.server_elapsed_ms,
                    target_task_ids=target_task_ids,
                )
            except BaseException as exc:
                result = _error_result(tool_call_id, _tool_name, _scenario_key, kwargs, target_task_ids, exc)
            return result.model_dump(mode="json")

        description = (
            f"场景：{row.get('scenario_name')}. {row.get('description')}。"
            f"仅当段落明确需要该业务数据时使用；不适用于公开市场、白皮书或通用 Web 事实。"
            f"只读权限；返回字段：{', '.join(return_columns)}。"
        )
        tool = StructuredTool.from_function(
            coroutine=execute_scenario,
            name=tool_name,
            description=description,
            args_schema=args_schema,
        )
        registry[scenario_key] = StructuredToolDefinition(
            scenario_key=scenario_key,
            domain=_domain(row),
            tool=tool,
            args_schema=args_schema,
            return_columns=return_columns,
            scenario_name=str(row.get("scenario_name") or scenario_key),
            description=str(row.get("description") or ""),
            keywords=tuple(str(value) for value in row.get("keywords", [])),
            fallback_safe=not any(
                bool(rule.get("required")) and "default" not in rule
                for rule in (row.get("params") or {}).values()
                if isinstance(rule, dict)
            ),
        )

    async def no_structured_query(
        reason: str,
        evaluated_task_ids: list[str],
        tool_call_id: str | None = None,
    ):
        return StructuredToolResult(
            tool_call_id=str(tool_call_id or f"direct-{uuid.uuid4().hex}"),
            tool_name="no_structured_query",
            scenario_key="no_structured_query",
            status="SUCCESS",
            query_summary=reason,
            arguments={"reason": reason},
            columns=[], rows=[], row_count=0,
            target_task_ids=evaluated_task_ids,
        ).model_dump(mode="json")

    no_tool = StructuredTool.from_function(
        coroutine=no_structured_query,
        name="no_structured_query",
        description=(
            "当且仅当段落不适用于任何已注册结构化业务场景时调用。"
            "必须说明原因并列出所有已评估 task_id；该工具不会访问数据库。"
        ),
        args_schema=NoStructuredQueryArgs,
    )
    registry["no_structured_query"] = StructuredToolDefinition(
        scenario_key="no_structured_query",
        domain="none",
        tool=no_tool,
        args_schema=NoStructuredQueryArgs,
        return_columns=(),
    )
    return registry


def tools_from_registry(registry: dict[str, StructuredToolDefinition]) -> list[BaseTool]:
    return [definition.tool for definition in registry.values()]


__all__ = ["StructuredToolDefinition", "build_structured_tool_registry", "tools_from_registry"]
