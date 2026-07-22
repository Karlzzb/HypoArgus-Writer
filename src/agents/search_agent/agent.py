"""search_agent 真实现适配器：黑盒 dict 进/出的异步调用 + 信号量限流 + 工厂。

调用形态与打桩完全一致（ADR-0001 约束 3：任务包 dict 进/出，不做子图化），
工厂签名沿用打桩的事件钩子注入位；契约映射见 mapping，引擎运行时边界见
runtime。并行章节分支的外部调用总并发经线程信号量限流：检索分支各自运行
在独立事件循环（reference_orchestrator 每分支 asyncio.run）或主循环
（修订增量检索），asyncio 信号量绑定单循环不可跨分支共享，故用线程信号量
并经 to_thread 获取，等待时不阻塞事件循环。
"""

import os
import time
from typing import Any

from agents.concurrency import make_thread_permit
from agents.contracts import DIAGNOSTICS_SUMMARY_KEY, SubagentAdapter
from agents.search_agent.mapping import (
    engine_payload_from_task,
    search_result_from_engine_output,
)
from agents.search_agent.runtime import EngineRuntime, SearchAgentRuntimeSeam
from agents.search_agent.stub import UNIT
from domain.env_config import read_positive_int
from domain.events import SUBAGENT_PROGRESS, EventHook, noop_hook
from llm.observability import update_current_span_metadata

MAX_CONCURRENT_CALLS_ENV = "SEARCH_AGENT_MAX_CONCURRENT_CALLS"
"""适配层外部调用总并发阈值的环境变量名。"""

DEFAULT_MAX_CONCURRENT_CALLS = 2
"""并发阈值缺省值：引擎单次调用内部已有多路通道并发，外层从紧避免击穿限流。"""

MIN_PASS_PER_CHAPTER_ENV = "SEARCH_AGENT_MIN_PASS_PER_CHAPTER"
"""每章 pass 落库下限的环境变量名：低于此值发薄弱章警告（杠杆①）。"""

DEFAULT_MIN_PASS_PER_CHAPTER = 3
"""每章 pass 落库下限缺省值：低于 3 条强支撑素材即判本章检索薄弱、显式暴露。"""

ENGINE_PROGRESS_PREFIX = "progress."
"""引擎进度事件的稳定前缀：桥只翻译此前缀事件，其余引擎诊断事件不进宿主进度。"""

_PROGRESS_METADATA_KEYS = frozenset(
    {
        "task_id",
        "item_id",
        "line_type",
        "verdict",
        "channel",
        "candidate_count",
        "batch_count",
        "round",
        "query_count",
        "elapsed_ms",
    }
)
"""进度事件载荷的白名单键：只透传 id / 计数 / 步骤要点等元数据。

引擎侧 redact 已把正文类字段兜底消毒，桥再收一层白名单，保证宿主进度
事件绝不携带正文全文（issue #36 验收 4）。
"""

_SUMMARY_TOP_KEYS = ("total_elapsed_ms", "deadline_reached", "call_counts")
_SUMMARY_GAP_KEYS = ("triggered_count", "resolved_task_count", "unresolved_task_count")
_SUMMARY_JUDGE_KEYS = (
    "judge_input_candidate_count",
    "judge_returned_candidate_count",
    "judge_missing_candidate_count",
)


def diagnostics_summary(flow_metrics: dict[str, Any]) -> dict[str, Any]:
    """从引擎 flow_metrics 提取结束事件的诊断摘要子集：只留关键计数与耗时。

    全量诊断走 Langfuse span 元数据；结束事件只带排障最常用的子集
    （总耗时、外部调用计数、补漏轮次、裁决完整性、截止命中）。
    """
    summary: dict[str, Any] = {
        key: flow_metrics[key] for key in _SUMMARY_TOP_KEYS if key in flow_metrics
    }
    gap = flow_metrics.get("gap_retrieval")
    if isinstance(gap, dict):
        summary["gap_retrieval"] = {
            key: gap[key] for key in _SUMMARY_GAP_KEYS if key in gap
        }
    judge = flow_metrics.get("judge_integrity")
    if isinstance(judge, dict):
        summary["judge_integrity"] = {
            key: judge[key] for key in _SUMMARY_JUDGE_KEYS if key in judge
        }
    return summary


def make_search_agent(
    event_hook: EventHook = noop_hook,
    *,
    runtime: SearchAgentRuntimeSeam | None = None,
    max_concurrent_calls: int | None = None,
    min_pass_per_chapter: int | None = None,
) -> SubagentAdapter:
    """构造 search_agent 真实现适配器：工厂签名与打桩一致（事件钩子注入）。

    runtime 是检索引擎运行时边界的测试接缝（缺省真实引擎一次性调用，
    构造零环境依赖、首次调用才触碰引擎配置）；max_concurrent_calls 未注入时
    按环境变量 SEARCH_AGENT_MAX_CONCURRENT_CALLS（缺省 2）；min_pass_per_chapter
    未注入时按环境变量 SEARCH_AGENT_MIN_PASS_PER_CHAPTER（缺省 3），每章 pass
    落库低于此值即发薄弱章警告并计入诊断摘要（杠杆①，不阻断不补检）。
    引擎调用失败即整体失败向上抛（无状态一次性调用，不做局部重试）。
    """
    effective_runtime = runtime or EngineRuntime()
    limit = (
        max_concurrent_calls
        if max_concurrent_calls is not None
        else read_positive_int(
            os.environ, MAX_CONCURRENT_CALLS_ENV, DEFAULT_MAX_CONCURRENT_CALLS
        )
    )
    permit = make_thread_permit(limit)
    min_pass = (
        min_pass_per_chapter
        if min_pass_per_chapter is not None
        else read_positive_int(
            os.environ, MIN_PASS_PER_CHAPTER_ENV, DEFAULT_MIN_PASS_PER_CHAPTER
        )
    )

    async def run(task: dict[str, Any]) -> dict[str, Any]:
        payload = engine_payload_from_task(task)
        chapter_id = task["chapter_id"]

        def progress(step: str, **extra: Any) -> None:
            """发进度事件：载荷统一带 unit / chapter_id / mode / step，只放元数据。"""
            event_hook(
                SUBAGENT_PROGRESS,
                {
                    "unit": UNIT,
                    "chapter_id": chapter_id,
                    "mode": None,
                    "step": step,
                    **extra,
                },
            )

        paragraph = payload["paragraph"]
        item_total = len(paragraph["forward_items"]) + len(paragraph["reverse_items"])
        verdict_done_count = 0

        def on_engine_event(event: str, event_payload: dict[str, Any]) -> None:
            """进度桥：引擎 progress. 前缀事件翻译为宿主进度，白名单透传元数据。

            裁决完成事件附带 done_count/item_total 累计计数，供订阅方渲染
            「检索项 x/y 裁决完成」级别的进度；引擎事件在其自身事件循环内
            顺序分派，计数无并发竞争。
            """
            if not event.startswith(ENGINE_PROGRESS_PREFIX):
                return
            step = event[len(ENGINE_PROGRESS_PREFIX) :]
            extra = {
                key: value
                for key, value in event_payload.items()
                if key in _PROGRESS_METADATA_KEYS
            }
            if step == "verdict.done":
                nonlocal verdict_done_count
                verdict_done_count += 1
                extra["done_count"] = verdict_done_count
                extra["item_total"] = item_total
            progress(step, **extra)

        await permit.acquire()
        # 适配层自身的最低进度保证：即使引擎内部事件全丢，一次调用也有
        # engine_call_start / engine_call_end 首尾两步。
        progress("engine_call_start", hypothesis_count=len(task["hypotheses"]))
        started = time.monotonic()
        try:
            output, diagnostics = await effective_runtime.retrieve(
                payload, on_engine_event=on_engine_event
            )
        finally:
            permit.release()
        result: dict[str, Any] = dict(search_result_from_engine_output(output, task))
        materials = result["materials"]
        pass_count = sum(1 for m in materials if m["verdict"] == "pass")
        weak_count = sum(1 for m in materials if m["verdict"] == "inconclusive")
        summary: dict[str, Any] = {}
        flow_metrics = diagnostics.get("flow_metrics")
        if isinstance(flow_metrics, dict) and flow_metrics:
            # 诊断三去向之二（issue #31）：全量进当前 subagent:search_agent
            # 的 Langfuse span 元数据，摘要子集经保留键随结束事件上报。
            update_current_span_metadata({"search_agent_flow_metrics": flow_metrics})
            summary.update(diagnostics_summary(flow_metrics))
        if weak_count:
            # 弱佐证单独计数进诊断摘要（杠杆②）：本章仅弱佐证 N 条，供排障与暴露。
            summary["weak_evidence_count"] = weak_count
        if pass_count < min_pass:
            # 薄弱章显式暴露（杠杆①）：pass 落库低于下限即发警告事件并计入摘要，
            # 计数只数 pass、不数 inconclusive；不阻断不补检、检索保持单轮。
            summary["pass_below_threshold"] = {
                "pass_count": pass_count,
                "threshold": min_pass,
            }
            progress(
                "weak_chapter_warning",
                pass_count=pass_count,
                threshold=min_pass,
                weak_evidence_count=weak_count,
            )
        if summary:
            result[DIAGNOSTICS_SUMMARY_KEY] = summary
        progress(
            "engine_call_end",
            materials=len(materials),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        return result

    return SubagentAdapter(UNIT, run, event_hook)
