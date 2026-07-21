"""search_agent 真实现适配器：黑盒 dict 进/出的异步调用 + 信号量限流 + 工厂。

调用形态与打桩完全一致（ADR-0001 约束 3：任务包 dict 进/出，不做子图化），
工厂签名沿用打桩的事件钩子注入位；契约映射见 mapping，引擎运行时边界见
runtime。并行章节分支的外部调用总并发经线程信号量限流：检索分支各自运行
在独立事件循环（reference_orchestrator 每分支 asyncio.run）或主循环
（修订增量检索），asyncio 信号量绑定单循环不可跨分支共享，故用线程信号量
并经 to_thread 获取，等待时不阻塞事件循环。
"""

import asyncio
import os
import threading
from typing import Any

from agents.contracts import SubagentAdapter
from agents.search_agent.mapping import (
    engine_payload_from_task,
    search_result_from_engine_output,
)
from agents.search_agent.runtime import EngineRuntime, SearchAgentRuntimeSeam
from agents.search_agent.stub import UNIT
from domain.env_config import read_positive_int
from domain.events import EventHook, noop_hook

MAX_CONCURRENT_CALLS_ENV = "SEARCH_AGENT_MAX_CONCURRENT_CALLS"
"""适配层外部调用总并发阈值的环境变量名。"""

DEFAULT_MAX_CONCURRENT_CALLS = 2
"""并发阈值缺省值：引擎单次调用内部已有多路通道并发，外层从紧避免击穿限流。"""


def make_search_agent(
    event_hook: EventHook = noop_hook,
    *,
    runtime: SearchAgentRuntimeSeam | None = None,
    max_concurrent_calls: int | None = None,
) -> SubagentAdapter:
    """构造 search_agent 真实现适配器：工厂签名与打桩一致（事件钩子注入）。

    runtime 是检索引擎运行时边界的测试接缝（缺省真实引擎一次性调用，
    构造零环境依赖、首次调用才触碰引擎配置）；max_concurrent_calls 未注入时
    按环境变量 SEARCH_AGENT_MAX_CONCURRENT_CALLS（缺省 2）。
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
    semaphore = threading.Semaphore(limit)

    async def acquire_permit() -> None:
        """经工作线程获取信号量，等待时不阻塞事件循环。

        排队等待中任务被取消时，工作线程仍会完成获取——挂回调把届时
        拿到的许可立即归还，保证取消不泄漏许可位。
        """
        acquire_future = asyncio.ensure_future(asyncio.to_thread(semaphore.acquire))
        try:
            await asyncio.shield(acquire_future)
        except asyncio.CancelledError:
            acquire_future.add_done_callback(lambda _: semaphore.release())
            raise

    async def run(task: dict[str, Any]) -> dict[str, Any]:
        payload = engine_payload_from_task(task)
        await acquire_permit()
        try:
            output = await effective_runtime.retrieve(payload)
        finally:
            semaphore.release()
        return dict(search_result_from_engine_output(output, task))

    return SubagentAdapter(UNIT, run, event_hook)
