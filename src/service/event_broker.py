"""SSE 事件发布订阅枢纽：世代 id、Last-Event-ID 续传、reconcile_required。

LangGraph 图在工作线程中同步执行并从那里发布事件，SSE 订阅者在服务的
asyncio 事件循环中消费。因此 publish/close 走 call_soon_threadsafe 调度到
loop 线程，所有内部状态变更都只在 loop 线程内发生，无需加锁。

传输语义（issue #54 传输底座）：
- 事件 id 形如 ``{epoch}-{seq}``：epoch 为进程（应用实例）启动标识，seq 在
  单流内单调递增，由 publish 在入站时分配并盖戳到 dict 载荷的 ``event_id``。
- 不带 ``Last-Event-ID`` 的新订阅只收实时事件，不再回放历史。
- 带 ``Last-Event-ID`` 的续传只补该 id 之后仍保留在缓冲内的事件；世代失配
  或所求位置已被淘汰时，立即向该订阅者下发 ``reconcile_required`` 控制事件
  后转实时推送，绝不静默错位续推。
- 有界环形缓冲保留近期事件供续传；超限淘汰最旧并累加 ``dropped`` 计数。
- ``reconcile_required`` 是每订阅者的控制事件，不入缓冲、不广播给其他订阅者，
  但占用一个单调 seq（不与任何真实事件 id 冲突）。
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator

# 结束哨兵：投入订阅队列表示枢纽已关闭，订阅迭代到此正常结束。
_CLOSE_SENTINEL: Any = object()

# reconcile_required 载荷里指明的对账 REST 口子（issue #54 时点已存在的只读通道）。
_RECONCILE_VIA_BUSINESS: tuple[str, ...] = (
    "GET /tasks/{id}",
    "GET /tasks/{id}/checkpoints",
    "GET /tasks/{id}/bibliography",
)


def new_epoch() -> str:
    """进程（应用实例）启动标识：uuid hex，无连字符，便于 ``{epoch}-{seq}`` 解析。"""
    return uuid.uuid4().hex


def parse_event_id(event_id: str) -> tuple[str, int]:
    """``{epoch}-{seq}`` → (epoch, seq)；epoch 为空或 seq 非整数时抛 ValueError。

    epoch 本身不含连字符（见 new_epoch），故按最后一个连字符切分；容忍测试
    用的带连字符 epoch 串（按最后连字符切分仍正确）。
    """
    epoch, sep, seq_str = event_id.rpartition("-")
    if not sep or not epoch or not seq_str:
        raise ValueError(f"非法 event id：{event_id!r}")
    return epoch, int(seq_str)


class EventHub:
    """单通道事件枢纽：载荷类型不限，事件 id 与续传语义由本类负责。

    publish 接收任意 ``item``：若 ``item`` 是可变映射（业务事件 dict），盖戳
    ``event_id = {epoch}-{seq}``；信封等不可变载荷自带拓扑 event_id，传输 id
    仍取 ``{epoch}-{seq}``（由订阅端按 seq 组装）。

    thread_id 非空时（业务通道）reconcile_required 载荷带 thread_id 与对账
    REST 口子；为 None 时（全局可视化通道）载荷带"无可重取 REST"说明。
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        epoch: str,
        thread_id: str | None = None,
        max_history: int = 10000,
    ) -> None:
        self._loop = loop
        self._epoch = epoch
        self._thread_id = thread_id
        self._max_history = max_history
        self._buffer: deque[tuple[int, Any]] = deque()
        self._queues: set[asyncio.Queue[Any]] = set()
        self._closed = False
        self._dropped = 0
        self._seq = 0

    @property
    def epoch(self) -> str:
        """本枢纽所属世代（进程启动标识）。"""
        return self._epoch

    @property
    def closed(self) -> bool:
        """枢纽是否已关闭。"""
        return self._closed

    @property
    def dropped(self) -> int:
        """缓冲超上限被淘汰的最旧事件条数，供上层观测。"""
        return self._dropped

    @property
    def subscriber_count(self) -> int:
        """当前在线订阅者数量（只读，供测试与观测）。"""
        return len(self._queues)

    @property
    def latest_seq(self) -> int:
        """已分配的最高 seq（无事件时为 0）。"""
        return self._seq

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
        """线程安全发布：分配 seq、盖戳传输 id、入缓冲并推给所有在线订阅者。

        缓冲超过 max_history 时淘汰最旧一条并累加 dropped 计数。
        枢纽关闭后发布被静默忽略。
        """

        def _do_publish() -> None:
            if self._closed:
                return
            self._seq += 1
            seq = self._seq
            if isinstance(item, dict):
                item["event_id"] = f"{self._epoch}-{seq}"
            self._buffer.append((seq, item))
            if len(self._buffer) > self._max_history:
                self._buffer.popleft()
                self._dropped += 1
            for queue in self._queues:
                queue.put_nowait((seq, item))

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

    async def subscribe(
        self, last_event_id: str | None = None
    ) -> AsyncIterator[tuple[int, Any]]:
        """异步生成器：按 Last-Event-ID 续传或纯实时，产出 (seq, item) 直到关闭。

        续传裁决在同一同步步骤内完成（无 await）：
        - 无 Last-Event-ID：纯实时，不回放。
        - Last-Event-ID 非法：下发 reconcile_required(malformed) 后转实时。
        - 世代失配：下发 reconcile_required(epoch_mismatch) 后转实时。
        - 同世代但所求位置已被淘汰：下发 reconcile_required(position_dropped) 后转实时。
        - 同世代且位置仍在缓冲内：回放该 id 之后仍保留的事件，再续实时。
        reconcile_required 仅投给本订阅者，不入缓冲、不广播。
        订阅者被取消时在 finally 中注销自己的队列，不泄漏。
        """
        queue: asyncio.Queue[Any] = asyncio.Queue()
        replay: list[tuple[int, Any]] = []
        reconcile: tuple[int, dict[str, Any]] | None = None

        # 本方法只在 loop 线程内被调用；以下裁决与注册在同一同步步骤内完成。
        if last_event_id is None:
            self._queues.add(queue)
        else:
            try:
                epoch, seq = parse_event_id(last_event_id)
            except ValueError:
                self._queues.add(queue)
                reconcile = self._build_reconcile("malformed", last_event_id)
            else:
                if epoch != self._epoch:
                    self._queues.add(queue)
                    reconcile = self._build_reconcile(
                        "epoch_mismatch", last_event_id
                    )
                else:
                    oldest = (
                        self._buffer[0][0] if self._buffer else self._seq + 1
                    )
                    if seq + 1 < oldest:
                        self._queues.add(queue)
                        reconcile = self._build_reconcile(
                            "position_dropped", last_event_id
                        )
                    else:
                        replay = [
                            (s, it) for s, it in self._buffer if s > seq
                        ]
                        self._queues.add(queue)
        if self._closed:
            queue.put_nowait(_CLOSE_SENTINEL)
        try:
            if reconcile is not None:
                yield reconcile
            for item in replay:
                yield item
            while True:
                item = await queue.get()
                if item is _CLOSE_SENTINEL:
                    return
                yield item
        finally:
            self._queues.discard(queue)

    def _build_reconcile(
        self, reason: str, last_event_id: str | None
    ) -> tuple[int, dict[str, Any]]:
        """构造 reconcile_required 控制事件并盖戳传输 id，仅投给当前订阅者。

        占用一个单调 seq（不入缓冲、不广播）：与任何真实事件 id 都不冲突，
        reconcile 之后实时事件必然 seq 更大，客户端据此续传不会重收 reconcile。
        """
        self._seq += 1
        seq = self._seq
        item: dict[str, Any] = {
            "type": "reconcile_required",
            "reason": reason,
            "last_event_id": last_event_id,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if self._thread_id is not None:
            item["thread_id"] = self._thread_id
            item["reconcile_via"] = list(_RECONCILE_VIA_BUSINESS)
        else:
            # 全局可视化通道无可重取的 REST：仅元数据观测，重新订阅实时即可。
            item["reconcile_via"] = []
            item["note"] = (
                "graph_events 为元数据观测通道，无可重取的 REST；请重新订阅实时事件。"
            )
        item["event_id"] = f"{self._epoch}-{seq}"
        return (seq, item)
