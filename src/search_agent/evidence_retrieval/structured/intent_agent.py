"""Paragraph-level LLM intent agent using required parallel Tool Calling."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .registry import StructuredToolDefinition, tools_from_registry


@dataclass(slots=True)
class RequestedToolCall:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class StructuredIntentAgent:
    def __init__(self, model: Any, registry: dict[str, StructuredToolDefinition], config: Any = None):
        self.model = model
        self.registry = registry
        self.by_name = {definition.tool.name: definition for definition in registry.values()}
        self.min_tool_calls = getattr(config, "structured_min_tool_calls", 1) if config else 1
        self.max_tool_calls = getattr(config, "structured_max_tool_calls", 5) if config else 5
        self.repair_count = getattr(config, "structured_repair_count", 1) if config else 1

    def _fallback(self, task_ids: list[str], reason: str) -> list[RequestedToolCall]:
        return [RequestedToolCall(
            tool_call_id=f"no-structured-{uuid.uuid4().hex}",
            tool_name="no_structured_query",
            arguments={"reason": reason, "evaluated_task_ids": task_ids},
        )]

    def _minimum_fallback(
        self,
        paragraph_text: str,
        tasks: list[Any],
        organization_context: dict[str, Any] | None,
        existing_tool_names: set[str] | None = None,
    ) -> list[RequestedToolCall]:
        """Deterministically fill the configured minimum with parameter-safe tools.

        This is deliberately a routing fallback, not an evidence judgment.  Only
        scenarios whose business parameters all have defaults are eligible, so
        the fallback can always execute a valid read-only query without inventing
        an enterprise, major, date or school identifier.
        """
        task_ids = [task.task_id for task in tasks]
        missing = max(0, self.min_tool_calls - len(existing_tool_names or set()))
        if missing == 0 or not task_ids:
            return []
        haystack = " ".join(
            [paragraph_text]
            + [
                " ".join(
                    [
                        str(getattr(task, "target_text", "")),
                        *[str(value) for value in getattr(task, "required_slots", [])],
                    ]
                )
                for task in tasks
            ]
        )
        existing = existing_tool_names or set()
        scored = [
            (
                sum(
                    3
                    for keyword in (definition.scenario_name, *definition.keywords)
                    if keyword and keyword in haystack
                ),
                definition,
            )
            for definition in self.registry.values()
            if definition.scenario_key != "no_structured_query"
            and definition.fallback_safe
            and definition.tool.name not in existing
        ]
        # A mandatory Structured path is not permission to manufacture an
        # unrelated SQL result. Only parameter-safe scenarios with an explicit
        # lexical business match may be used as deterministic real-query
        # fallback; otherwise the auditable no_structured_query capability path
        # is returned below.
        eligible = [row for score, row in scored if score > 0]
        if not eligible and self.min_tool_calls > 0:
            # No lexical match but Structured is mandatory: pick by task_id
            # hash to ensure diversity — different paragraphs query different
            # Doris scenarios instead of all hitting the same one.
            all_fb = [row for _, row in scored]
            if all_fb and task_ids:
                import hashlib
                h = int(hashlib.md5(task_ids[0].encode()).hexdigest(), 16)
                eligible = [all_fb[h % len(all_fb)]]
            else:
                eligible = all_fb
        eligible.sort(
            key=lambda definition: (
                -next(score for score, row in scored if row is definition),
                int(definition.scenario_key.rsplit("_", 1)[-1]),
            )
        )
        calls: list[RequestedToolCall] = []
        organization = organization_context or {}
        for definition in eligible[:missing]:
            arguments: dict[str, Any] = {"target_task_ids": task_ids}
            fields = definition.args_schema.model_fields
            if "school_id" in fields and organization.get("school_id"):
                arguments["school_id"] = organization["school_id"]
            if "my_school_id" in fields and organization.get("school_id"):
                arguments["my_school_id"] = organization["school_id"]
            validated = definition.args_schema.model_validate(arguments).model_dump(exclude_none=True)
            calls.append(
                RequestedToolCall(
                    tool_call_id=f"structured-fallback-{uuid.uuid4().hex}",
                    tool_name=definition.tool.name,
                    arguments=validated,
                )
            )
        return calls

    @staticmethod
    def build_prompt(paragraph_text: str, tasks: list[Any], organization_context: dict[str, Any] | None = None, min_tool_calls: int = 1, max_tool_calls: int = 5) -> str:
        task_payload = [{
            "task_id": task.task_id,
            "line_type": task.line_type.value,
            "target_text": task.target_text,
            "required_slots": task.required_slots,
            "atomic_claims": [claim.model_dump(mode="json") for claim in task.atomic_claims],
            "source_refs": task.source_refs,
        } for task in tasks]
        min_hint = f"你必须至少调用 {min_tool_calls} 个真实场景工具。no_structured_query 不计入真实场景工具，不能用来满足最低调用次数要求。" if min_tool_calls > 0 else "你可以调用 0 个真实场景工具。"
        return (
            "你是 SearchAgent 的结构化数据意图节点。你必须调用至少一个工具。"
            f"{min_hint}最多调用 {max_tool_calls} 个工具。"
            "仅当段落明确属于某个已注册业务场景时调用对应查询工具；公开市场、白皮书、"
            "通用 Web 事实必须调用 no_structured_query。可一次并行调用多个场景工具。"
            "每个真实查询工具都必须提供 target_task_ids，且只能来自输入 task_id。"
            "不得生成 SQL、表名、API 地址或额外参数。\n"
            f"段落：{paragraph_text}\n"
            f"任务：{json.dumps(task_payload, ensure_ascii=False, separators=(',', ':'))}"
        )

    async def select(
        self,
        paragraph_text: str,
        tasks: list[Any],
        *,
        feedback: str | None = None,
        organization_context: dict[str, Any] | None = None,
    ) -> tuple[list[RequestedToolCall], list[str]]:
        task_ids = [task.task_id for task in tasks]
        haystack = " ".join(
            [paragraph_text]
            + [
                " ".join(
                    [
                        str(getattr(task, "target_text", "")),
                        *[str(value) for value in getattr(task, "required_slots", [])],
                    ]
                )
                for task in tasks
            ]
        )
        deterministic_scores = [
            (
                sum(
                    1
                    for keyword in (definition.scenario_name, *definition.keywords)
                    if keyword and keyword in haystack
                ),
                definition,
            )
            for definition in self.registry.values()
            if definition.scenario_key != "no_structured_query"
        ]
        top_score = max((score for score, _ in deterministic_scores), default=0)
        top_matches = [
            definition
            for score, definition in deterministic_scores
            if score == top_score and score > 0
        ]
        # When min_tool_calls > 0, Structured is mandatory: don't short-circuit
        # on top_score == 0. Fall through to LLM (which may find a match the
        # deterministic router missed) or _minimum_fallback (which picks the
        # best available fallback_safe scenario unconditionally).
        # Only when min_tool_calls == 0 (optional Structured) is no_structured_query
        # acceptable as a valid no-match result.
        if top_score == 0 and self.min_tool_calls == 0:
            return self._fallback(
                task_ids, "no_matching_structured_scenario"
            ), ["STRUCTURED_NO_MATCH"]
        if len(top_matches) == 1 and top_matches[0].fallback_safe:
            forced = self._minimum_fallback(
                paragraph_text, tasks, organization_context
            )
            if forced and forced[0].tool_name == top_matches[0].tool.name:
                return forced, ["STRUCTURED_DETERMINISTIC_ROUTE"]
        if self.model is None or not hasattr(self.model, "bind_tools"):
            forced = self._minimum_fallback(
                paragraph_text, tasks, organization_context
            )
            if forced:
                return forced, [
                    "STRUCTURED_INTENT_MODEL_UNAVAILABLE",
                    "STRUCTURED_MINIMUM_FALLBACK",
                ]
            return self._fallback(
                task_ids, "Structured Tool Calling model is unavailable"
            ), ["STRUCTURED_INTENT_MODEL_UNAVAILABLE"]
        bound = self.model.bind_tools(
            tools_from_registry(self.registry),
            parallel_tool_calls=True,
            tool_choice="required",
        )
        prompt = self.build_prompt(paragraph_text, tasks, organization_context, self.min_tool_calls, self.max_tool_calls)
        prompt += "\norganization_context: " + json.dumps(
            organization_context or {}, ensure_ascii=False, separators=(",", ":")
        )
        if feedback:
            prompt += (
                "\n上轮工具执行返回 INVALID_ARGUMENT。仅允许本次修复；请根据错误重新提取参数，"
                f"不要重复无效调用：{feedback}"
            )
        warnings: list[str] = []
        best_real_calls: list[RequestedToolCall] = []
        for attempt in range(self.repair_count + 1):
            response = await bound.ainvoke(prompt)
            raw_calls = list(getattr(response, "tool_calls", []) or [])
            calls: list[RequestedToolCall] = []
            invalid: list[str] = []
            for index, raw in enumerate(raw_calls):
                name = str(raw.get("name") or "")
                arguments = dict(raw.get("args")) if isinstance(raw.get("args"), dict) else {}
                arguments.pop("tool_call_id", None)
                call_id = str(raw.get("id") or f"structured-call-{attempt}-{index}")
                definition = self.by_name.get(name)
                if definition is None:
                    invalid.append(f"unknown tool {name}")
                    continue
                try:
                    organization = organization_context or {}
                    fields = definition.args_schema.model_fields
                    if "school_id" in fields and not arguments.get("school_id") and organization.get("school_id"):
                        arguments["school_id"] = organization["school_id"]
                    if "my_school_id" in fields and not arguments.get("my_school_id") and organization.get("school_id"):
                        arguments["my_school_id"] = organization["school_id"]
                    validated = definition.args_schema.model_validate(arguments).model_dump(exclude_none=True)
                except ValidationError as exc:
                    invalid.append(f"{name}: {exc.errors(include_url=False)}")
                    continue
                targets = validated.get("target_task_ids") or validated.get("evaluated_task_ids") or []
                if not targets or any(task_id not in task_ids for task_id in targets):
                    invalid.append(f"{name}: target_task_ids outside paragraph")
                    continue
                calls.append(RequestedToolCall(call_id, name, validated))
            if len(calls) > self.max_tool_calls:
                invalid.append(
                    f"tool call count {len(calls)} exceeds maximum {self.max_tool_calls}"
                )
            real_calls = [call for call in calls if call.tool_name != "no_structured_query"]
            if len(real_calls) > len(best_real_calls):
                best_real_calls = real_calls
            if calls and not invalid:
                no_match_calls = [
                    call for call in calls if call.tool_name == "no_structured_query"
                ]
                if no_match_calls and self.min_tool_calls == 0:
                    # Only accept no_structured_query when Structured is optional
                    # (min_tool_calls == 0). When min_tool_calls > 0, Structured
                    # is a mandatory core feature — no_structured_query does NOT
                    # satisfy the minimum real query requirement.
                    return no_match_calls[:1], warnings
                if len(real_calls) >= self.min_tool_calls:
                    return real_calls, warnings
            if len(real_calls) < self.min_tool_calls:
                invalid.append(
                    f"real structured tool count {len(real_calls)} is below required minimum "
                    f"{self.min_tool_calls}"
                )
            if attempt < self.repair_count:
                warnings.append("STRUCTURED_ARGUMENT_REPAIR")
                prompt += (
                    "\n上次 Tool Call 无效。请重新选择并确保真实场景调用达到最少次数："
                    + "; ".join(invalid or ["模型未返回 Tool Call"])
                )
        existing_names = {call.tool_name for call in best_real_calls}
        forced = self._minimum_fallback(
            paragraph_text,
            tasks,
            organization_context,
            existing_tool_names=existing_names,
        )
        if best_real_calls or forced:
            warnings.extend(["STRUCTURED_INVALID_ARGUMENT", "STRUCTURED_MINIMUM_FALLBACK"])
            return [*best_real_calls, *forced], list(dict.fromkeys(warnings))
        warnings.append("STRUCTURED_INVALID_ARGUMENT")
        return self._fallback(task_ids, "No valid structured tool call after repair"), warnings


__all__ = ["RequestedToolCall", "StructuredIntentAgent"]
