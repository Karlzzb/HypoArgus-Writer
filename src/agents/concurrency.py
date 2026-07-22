"""子智能体外部调用的线程信号量限流（取消安全），供各子智能体共用。

并行章节分支各自运行在独立事件循环（reference_orchestrator 每分支 asyncio.run），
或在主循环（修订增量检索、章级评审），asyncio 信号量绑定单循环不可跨分支共享，
故用线程信号量并经 ``to_thread`` 获取——等待时不阻塞事件循环。抽为共享工具供
search_agent 与 chapter_reviewer 复用（ADR-0006），保证限流语义单一事实源。
"""

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ThreadPermit:
    """跨事件循环共享的线程信号量许可：``acquire``/``release`` 或 ``hold`` 上下文。

    ``acquire`` 经工作线程获取，等待时不阻塞事件循环；排队等待中任务被取消时，
    工作线程仍会完成获取——挂回调把届时拿到的许可立即归还，保证取消不泄漏许可位。
    """

    def __init__(self, limit: int) -> None:
        self._semaphore = threading.Semaphore(limit)

    async def acquire(self) -> None:
        acquire_future = asyncio.ensure_future(asyncio.to_thread(self._semaphore.acquire))
        try:
            await asyncio.shield(acquire_future)
        except asyncio.CancelledError:
            acquire_future.add_done_callback(lambda _: self._semaphore.release())
            raise

    def release(self) -> None:
        self._semaphore.release()

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        """在许可内执行受限区段：进入即获取、退出（含异常）即归还。"""
        await self.acquire()
        try:
            yield
        finally:
            self.release()


def make_thread_permit(limit: int) -> ThreadPermit:
    """构造并发上限为 ``limit`` 的线程信号量许可。"""
    return ThreadPermit(limit)
