"""检索引擎运行时边界：真实一次性调用封装与可注入的假实现（唯一新增测试接缝）。

引擎保持无状态一次性调用（issue #31 决策）：每次检索构建新的引擎运行时，
调用完成即关闭，失败向上抛由调用方整体重调；不移植源项目的守护线程事件循环
桥接——本项目 Subagent 协议本身是异步的，且检索并行分支各自运行在独立事件
循环，一次性运行时天然规避 httpx 客户端跨事件循环复用的亲和性问题。

调用检索图前把 LangChain 环境配置过滤到只保留回调：父图 checkpointer 配置
经 contextvar 泄漏进检索子图会引发 loop 亲和性问题（源项目适配层的已知坑）。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol


EngineEventHook = Callable[[str, dict[str, Any]], None]
"""引擎事件回调：(引擎事件名, 已消毒载荷)；SafeTraceEmitter 按可调用分派。

进度事件带 ``progress.`` 前缀，其余为引擎诊断事件（pair.consistency、
parallel.finalize 等）；翻译与过滤是适配层桥的职责，运行时只负责透传。
"""


class SearchAgentRuntimeSeam(Protocol):
    """引擎运行时边界协议：引擎公开入参 dict 进，（公开出参, 诊断出参）出。

    诊断出参与引擎批处理诊断同形（含 ``flow_metrics`` 键），只在运行时
    边界内传出供可观测接入，不进入检索结果契约。
    """

    async def retrieve(
        self,
        payload: dict[str, Any],
        *,
        on_engine_event: EngineEventHook | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


def ambient_callbacks() -> list[Any]:
    """从当前 LangChain 运行配置提取回调清单：适配层只向检索图透传回调。"""
    from langchain_core.runnables.config import var_child_runnable_config

    raw = (var_child_runnable_config.get() or {}).get("callbacks")
    if isinstance(raw, (list, tuple)):
        return list(raw)
    handlers = getattr(raw, "handlers", None)
    return list(handlers) if handlers else []


async def _invoke_engine_once(
    payload: dict[str, Any], callbacks: list[Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """构建一次性引擎运行时并调用：编图 → 调用 → 关闭，任何失败向上抛。

    显式传入回调列表（含空列表）即宿主管理模式：引擎不再自动挂载
    自己的 Langfuse handler，避免与本项目可观测接入层双上报。
    """
    from search_agent.api import SearchAgentRuntime

    runtime = SearchAgentRuntime.from_env(callbacks=callbacks)
    try:
        return await runtime.ainvoke_with_diagnostics(payload)
    finally:
        await runtime.aclose()


class EngineRuntime:
    """真实引擎运行时封装：配置过滤到只剩回调 + 无状态一次性调用。

    invoke 参数是配置过滤逻辑的测试注入点（缺省真实引擎一次性调用）。
    """

    def __init__(
        self,
        invoke: Callable[
            [dict[str, Any], list[Any]],
            Awaitable[tuple[dict[str, Any], dict[str, Any]]],
        ] = _invoke_engine_once,
    ) -> None:
        self._invoke = invoke

    async def retrieve(
        self,
        payload: dict[str, Any],
        *,
        on_engine_event: EngineEventHook | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from langchain_core.runnables.config import var_child_runnable_config

        callbacks = ambient_callbacks()
        # 引擎事件回调不是 LangChain handler：追加进引擎回调清单（引擎的
        # SafeTraceEmitter 按可调用分派、载荷先消毒），但不进 contextvar
        # 环境配置，避免 LangChain 运行把它当 handler 调用。
        engine_callbacks = (
            [*callbacks, on_engine_event] if on_engine_event is not None else callbacks
        )
        # 收窄环境配置：检索图经 contextvar 只能看到回调，父图的
        # configurable（thread_id / checkpoint_ns / checkpointer）一律不透传。
        token = var_child_runnable_config.set({"callbacks": callbacks})
        try:
            return await self._invoke(payload, engine_callbacks)
        finally:
            var_child_runnable_config.reset(token)


class FakeSearchAgentRuntime:
    """引擎运行时边界的假实现：模拟时延与副作用，记录全部调用载荷。

    与真实封装同协议注入 make_search_agent，供离线契约测试与中断续跑
    E2E（issue #37）使用：latency_seconds 模拟外部检索耗时，side_effect
    （同步或异步可调用，入参为载荷）模拟计数、崩溃注入等副作用，
    output_builder 缺省用 fake_engine_output 产出确定性合法出参。
    注入 on_engine_event 时按真实引擎的事件形状回放逐项进度事件
    （fake_engine_progress_events），并回带 fake_engine_diagnostics 假诊断。
    """

    def __init__(
        self,
        *,
        output_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        latency_seconds: float = 0.0,
        side_effect: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._output_builder = output_builder or fake_engine_output
        self._latency_seconds = latency_seconds
        self._side_effect = side_effect

    async def retrieve(
        self,
        payload: dict[str, Any],
        *,
        on_engine_event: EngineEventHook | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.payloads.append(payload)
        if self._latency_seconds > 0:
            await asyncio.sleep(self._latency_seconds)
        if self._side_effect is not None:
            result = self._side_effect(payload)
            if inspect.isawaitable(result):
                await result
        if on_engine_event is not None:
            for event, event_payload in fake_engine_progress_events(payload):
                on_engine_event(event, event_payload)
        return self._output_builder(payload), fake_engine_diagnostics(payload)


_FAKE_SOURCE_TYPES = ("WEB", "KNOWLEDGE_BASE", "STRUCTURED_DATA")


def _lined_items(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """引擎入参的（线别, 检索项）平铺清单：正向在前、反向在后。"""
    paragraph = payload["paragraph"]
    return [("forward", item) for item in paragraph.get("forward_items", [])] + [
        ("reverse", item) for item in paragraph.get("reverse_items", [])
    ]


def fake_engine_progress_events(
    payload: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """确定性回放引擎进度事件：与真实引擎同名同形，供离线事件断言。

    每个检索项回放任务开始 / 检索完成 / 裁决完成三条 ``progress.`` 事件，
    整批一条裁决批次事件；末尾附一条非进度诊断事件（parallel.finalize），
    供适配层桥的前缀过滤离线验证。载荷只含元数据，不含任何正文字段。
    """
    request_id = payload["request_id"]
    events: list[tuple[str, dict[str, Any]]] = []
    for line, item in _lined_items(payload):
        base = {
            "request_id": request_id,
            "task_id": f"task-{item['item_id']}",
            "item_id": item["item_id"],
            "line_type": line,
        }
        events.append(("progress.task.start", dict(base)))
        events.append(("progress.task.retrieved", {**base, "candidate_count": 1}))
        events.append(
            (
                "progress.verdict.done",
                {**base, "verdict": "SUPPORTED" if line == "forward" else "REFUTED"},
            )
        )
    events.append(
        (
            "progress.judge.batches_done",
            {"request_id": request_id, "batch_count": 1},
        )
    )
    events.append(
        (
            "parallel.finalize",
            {"request_id": request_id, "flow_mode": "parallel_sources"},
        )
    )
    return events


def fake_engine_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    """确定性构造引擎诊断出参：与真实批处理诊断同形（含 flow_metrics）。

    只放摘要提取会消费的计数与耗时键，供离线测试与中断续跑复用。
    """
    item_count = len(_lined_items(payload))
    return {
        "request_id": payload["request_id"],
        "flow_metrics": {
            "total_elapsed_ms": 12,
            "deadline_reached": False,
            "call_counts": {"web_search": item_count, "web_fetch": item_count},
            "gap_retrieval": {
                "triggered_count": 0,
                "resolved_task_count": 0,
                "unresolved_task_count": 0,
            },
            "judge_integrity": {
                "judge_input_candidate_count": item_count,
                "judge_returned_candidate_count": item_count,
                "judge_missing_candidate_count": 0,
            },
        },
    }


def fake_engine_output(payload: dict[str, Any]) -> dict[str, Any]:
    """确定性构造引擎公开出参：每个检索项一条裁决与一条专属引文。

    正向项判 SUPPORTED 且引文列为支撑，反向项判 REFUTED 且引文列为反驳；
    来源通道按检索项 id 字节和分派三值（与打桩同法，跨调用稳定），
    仅联网来源带确定性链接。结构与 search-agent-output/v1 同形，
    保证适配层消费的字段齐全。
    """
    paragraph = payload["paragraph"]
    lined_items = _lined_items(payload)
    results: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    for line, item in lined_items:
        item_id = item["item_id"]
        citation_id = f"cit-{item_id}"
        source_type = _FAKE_SOURCE_TYPES[
            sum(item_id.encode()) % len(_FAKE_SOURCE_TYPES)
        ]
        supporting = line == "forward"
        citations.append(
            {
                "citation_id": citation_id,
                "task_ids": [f"task-{item_id}"],
                "content": f"假引擎检索到的证据正文（{item['target_text']}）",
                "summary": f"假引擎摘录：针对「{item['target_text']}」的证据。",
                "title": f"假引擎来源标题（{item_id}）",
                "source_type": source_type,
                "source_name": f"假引擎来源（{source_type}）",
                "url": (
                    f"https://fake-engine.example/{item_id}"
                    if source_type == "WEB"
                    else None
                ),
                "relation": "SUPPORT" if supporting else "REFUTE",
                "status": "ACCEPTED",
                "judgment": {
                    "confidence": 0.9,
                    "directness": 0.8,
                    "reason": "假引擎确定性裁决",
                    "scope_compatible": True,
                    "quote_match_mode": "SNIPPET",
                },
                "provenance": {
                    "retrieved_at": "2026-01-01T00:00:00+00:00",
                    "content_fingerprint": f"fp-{item_id}",
                    "source_evidence_fingerprint": f"sfp-{item_id}",
                },
            }
        )
        results.append(
            {
                "task_id": f"task-{item_id}",
                "item_id": item_id,
                "node_id": f"node-{item_id}",
                "line_type": line,
                "target_text": item["target_text"],
                "run_status": "SUCCESS",
                "verdict": "SUPPORTED" if supporting else "REFUTED",
                "confidence": 0.9,
                "conclusion_summary": "假引擎结论摘要",
                "citation_ids": [citation_id],
                "supporting_citation_ids": [citation_id] if supporting else [],
                "refuting_citation_ids": [] if supporting else [citation_id],
            }
        )
    return {
        "schema_version": "search-agent-output/v1",
        "request_id": payload["request_id"],
        "document_id": payload["document_id"],
        "paragraph_id": paragraph["paragraph_id"],
        "run_status": {
            "status": "SUCCESS",
            "completed_task_count": len(results),
        },
        "results": results,
        "citations": citations,
        "warnings": [],
        "trace": {},
    }
