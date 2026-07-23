"""SSE 事件发布订阅枢纽：世代 id、Last-Event-ID 续传、reconcile_required、
慢消费者两级丢弃背压。

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

背压语义（issue #55 两级丢弃——信号必达，产物可丢可取）：
- 每订阅者一条有界队列（``max_queue``）。慢消费者灌满时按两级丢弃挤位：
  可丢级（``content_delta`` / ``product`` 整块，REST 可对账）淘汰最旧一条并
  累加 ``dropped``；不可丢控制信号（``review_required`` / ``finalized`` /
  ``error`` / ``reconcile_required``）满时挤掉可丢级帧为其让位，全队列无可丢
  级可挤时强制超容入队——信号体积极小、罕见，保信号必达，无内存风险。
- ``dropped`` 累计历史缓冲淘汰与订阅者队列丢弃两部分，供运维观测背压健康。
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, AsyncIterator

# 结束哨兵：投入订阅队列表示枢纽已关闭，订阅迭代到此正常结束。
_CLOSE_SENTINEL: Any = object()

# 可丢级事件类型：业务通道的逐字帧与产物整块，丢了靠 REST 对账重取。
# 其余类型（status / review_required / finalized / error / reconcile_required
# 等控制信号、可视化信封）一律不可丢——契约一句话：信号必达，产物可丢可取。
_DROPPABLE_TYPES: frozenset[str] = frozenset({"content_delta", "product"})

# 每订阅者队列的缺省容量：正常速率消费者不丢不重，慢消费者灌满后丢最旧可丢级。
_DEFAULT_MAX_QUEUE = 256

# 容量可经同名环境变量覆盖（由 app 装配时读取），具体阈值据线上负载调定。
_DEFAULT_MAX_QUEUE_ENV = "SSE_MAX_QUEUE"

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


def _is_droppable(item: Any) -> bool:
    """是否可丢级：业务通道的 ``content_delta`` 与 ``product`` 整块（REST 可对账）。

    dict 业务事件按 ``type`` 判定；EventEnvelope（可视化信封）、哨兵等非 dict
    载荷一律不可丢——可视化通道元数据专用、必达。
    """
    return isinstance(item, dict) and item.get("type") in _DROPPABLE_TYPES


class _BackpressureQueue:
    """单订阅者有界队列 + 两级丢弃背压（单消费者：一条 SSE 流）。

    契约：信号必达，产物可丢可取。
    - 满时挤位只淘汰最旧的可丢级帧（``content_delta`` / ``product``），绝不淘汰
      信号；经 ``on_drop`` 回调上交枢纽累计 ``dropped``。
    - 无可丢级可挤时：可丢级 incoming 直接丢弃（产物可丢）；不可丢信号强制
      超容入队（保必达，信号体积极小、罕见，无内存风险）。
    - 消费者挂起等待（缓冲空）时直接交付，不经缓冲、不占位、不触发背压。
    - 入队永不阻塞（``put_nowait``）；取空时 ``get`` 挂起 Future 等 put 唤醒。
    """

    def __init__(
        self, maxsize: int, on_drop: Callable[[int], None]
    ) -> None:
        self._buf: deque = deque()
        self._maxsize = maxsize
        self._on_drop = on_drop
        self._getter: asyncio.Future[Any] | None = None

    def put_nowait(self, value: Any) -> None:
        """入队一条 (seq, item) 或结束哨兵；永不阻塞。"""
        # 有消费者挂起等待：直接交付，不经缓冲（不占位、不触发背压）。
        getter = self._getter
        if getter is not None and not getter.done():
            self._getter = None
            getter.set_result(value)
            return
        if value is _CLOSE_SENTINEL:
            self._buf.append(value)
            return
        if len(self._buf) >= self._maxsize:
            if not self._evict_oldest_droppable():
                # 无可丢级可挤
                if _is_droppable(value):
                    self._on_drop(1)  # 丢弃 incoming 可丢级
                    return
                # 不可丢信号：强制超容入队，保必达
        self._buf.append(value)

    def _evict_oldest_droppable(self) -> bool:
        """淘汰缓冲内最旧的可丢级帧，返回是否淘汰成功。"""
        for i, entry in enumerate(self._buf):
            # 仅 (seq, item) 元组可判级；结束哨兵是裸对象，跳过（且哨兵入队后
            # 枢纽必已关闭，不会再有 publish 触发淘汰，此处仅作防御）。
            if isinstance(entry, tuple) and _is_droppable(entry[1]):
                del self._buf[i]
                self._on_drop(1)
                return True
        return False

    async def get(self) -> Any:
        """取下一条（(seq, item) 或哨兵）；空时挂起等待 put 直接交付。"""
        if self._buf:
            return self._buf.popleft()
        # 缓冲空：登记 Future 等 put 直接交付。先存局部变量再 await——
        # put_nowait 在 set_result 前即清空 self._getter，取消路径须凭局部
        # future 判断是否已交付、归还缓冲，否则交付而未取的帧会丢（违信号必达）。
        fut = asyncio.get_running_loop().create_future()
        self._getter = fut
        try:
            return await fut
        except asyncio.CancelledError:
            # 消费者被取消：若 future 已被交付但未取走，归还缓冲以免丢帧。
            if fut.done() and not fut.cancelled():
                value = fut.result()
                if value is not _CLOSE_SENTINEL:
                    self._buf.appendleft(value)
            raise
        finally:
            self._getter = None


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
        max_queue: int = _DEFAULT_MAX_QUEUE,
    ) -> None:
        self._loop = loop
        self._epoch = epoch
        self._thread_id = thread_id
        self._max_history = max_history
        self._max_queue = max_queue
        self._buffer: deque[tuple[int, Any]] = deque()
        self._queues: set[_BackpressureQueue] = set()
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
        """被丢弃的事件总数：历史缓冲淘汰 + 慢消费者队列丢弃，供运维观测。"""
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
        推给订阅者时走其有界队列的两级丢弃背压（可丢级丢最旧、信号必达）。
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
        实时事件经本订阅者的有界队列（两级丢弃背压）；订阅者被取消时在 finally
        中注销自己的队列，不泄漏。
        """
        queue = _BackpressureQueue(self._max_queue, self._account_drop)
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

    def _account_drop(self, count: int) -> None:
        """订阅者队列两级丢弃时回调：累加 dropped（loop 线程内，无需加锁）。"""
        self._dropped += count

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
