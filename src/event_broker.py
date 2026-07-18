"""SSE 事件发布订阅枢纽：跨线程发布、异步订阅、历史回放。

LangGraph 图在工作线程中同步执行并从那里发布事件，
SSE 订阅者在服务的 asyncio 事件循环中消费。
因此 publish/close 走 call_soon_threadsafe 调度到 loop 线程，
所有内部状态变更都只在 loop 线程内发生，无需加锁。

订阅可能晚于任务开跑，subscribe 先回放订阅时刻的全部历史快照，
再进入实时推送；注册队列与拍快照在同一个同步步骤内完成，不漏不重。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

# 结束哨兵：投入订阅队列表示枢纽已关闭，订阅迭代到此正常结束。
_CLOSE_SENTINEL: Any = object()


class EventHub:
    """单通道事件枢纽：载荷类型不限，过滤由调用方负责。"""

    def __init__(self, loop: asyncio.AbstractEventLoop, max_history: int = 10000) -> None:
        self._loop = loop
        self._max_history = max_history
        self._history: list[Any] = []
        self._queues: set[asyncio.Queue[Any]] = set()
        self._closed = False
        self._dropped = 0

    @property
    def closed(self) -> bool:
        """枢纽是否已关闭。"""
        return self._closed

    @property
    def dropped(self) -> int:
        """历史超上限被丢弃的最旧事件条数，供上层观测。"""
        return self._dropped

    @property
    def subscriber_count(self) -> int:
        """当前在线订阅者数量（只读，供测试与观测）。"""
        return len(self._queues)

    def _call_in_loop(self, callback: Any) -> None:
        """把状态变更调度到 loop 线程：已在 loop 线程则直接执行。"""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            callback()
        else:
            self._loop.call_soon_threadsafe(callback)

    def publish(self, item: Any) -> None:
        """线程安全发布：追加历史并推送给所有在线订阅者。

        历史超过 max_history 时丢弃最旧一条并累加 dropped 计数。
        枢纽关闭后发布被静默忽略。
        """

        def _do_publish() -> None:
            if self._closed:
                return
            self._history.append(item)
            if len(self._history) > self._max_history:
                del self._history[0]
                self._dropped += 1
            for queue in self._queues:
                queue.put_nowait(item)

        self._call_in_loop(_do_publish)

    def close(self) -> None:
        """线程安全、幂等关闭：标记关闭并向所有订阅者投递结束哨兵。"""

        def _do_close() -> None:
            if self._closed:
                return
            self._closed = True
            for queue in self._queues:
                queue.put_nowait(_CLOSE_SENTINEL)

        self._call_in_loop(_do_close)

    async def subscribe(self) -> AsyncIterator[Any]:
        """异步生成器：先回放历史快照，再持续产出实时事件直到枢纽关闭。

        注册队列与拍历史快照在同一个同步步骤内完成（中间无 await），
        因此快照之后发布的事件必然进入队列，不漏不重。
        已关闭的枢纽上仍可订阅：回放完历史即结束。
        订阅者被取消时在 finally 中注销自己的队列，不泄漏。
        """
        # 本方法只会在 loop 线程内被调用，直接同步完成注册与快照。
        queue: asyncio.Queue[Any] = asyncio.Queue()
        snapshot = list(self._history)
        if self._closed:
            queue.put_nowait(_CLOSE_SENTINEL)
        else:
            self._queues.add(queue)
        try:
            for item in snapshot:
                yield item
            while True:
                item = await queue.get()
                if item is _CLOSE_SENTINEL:
                    return
                yield item
        finally:
            self._queues.discard(queue)
