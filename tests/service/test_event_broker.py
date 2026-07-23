"""event_broker 事件枢纽的单元测试（传输底座：世代 id + Last-Event-ID 续传）。

覆盖：世代 id `{epoch}-{seq}` 单流单调递增、不带 Last-Event-ID 的新订阅只收
实时事件（不回放）、带 Last-Event-ID 续传只补该 id 之后的事件、世代失配与
位置丢弃立即下发 reconcile_required、跨线程发布、关闭语义、订阅者注销、
有界缓冲淘汰与 dropped 计数。

另覆盖两级丢弃背压（issue #55）：慢消费者灌满可丢级丢最旧且 dropped 准确、
控制信号满时仍全部送达、subscriber_count 可观测、正常速率不丢不重。
"""

import asyncio
import threading
from typing import Any

from service.event_broker import (
    EventHub,
    _BackpressureQueue,
    _DROPPABLE_TYPES,
    _is_droppable,
    new_epoch,
    parse_event_id,
)


async def _collect(hub: EventHub, last_event_id: str | None = None) -> list[Any]:
    """消费订阅直到枢纽关闭，返回收到的 (seq, item) 列表。"""
    return [item async for item in hub.subscribe(last_event_id)]


def _ids(events: list[Any]) -> list[str]:
    """从 (seq, item) 列表取 event_id（item["event_id"]）。"""
    return [item["event_id"] for _, item in events]


def test_世代id单流单调递增且形如epoch_seq():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        consumer = asyncio.create_task(_collect(hub))
        await asyncio.sleep(0)  # 完成注册（实时订阅）
        hub.publish({"type": "status", "data": {"n": 1}})
        hub.publish({"type": "status", "data": {"n": 2}})
        hub.close()
        events = await consumer
        ids = _ids(events)
        assert ids == ["epA-1", "epA-2"]
        assert [seq for seq, _ in events] == [1, 2]

    asyncio.run(main())


def test_不带LastEventID的新订阅不回放只收实时():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        hub.publish({"type": "status", "data": {"n": 1}})
        consumer = asyncio.create_task(_collect(hub))  # 新订阅（无 Last-Event-ID）
        await asyncio.sleep(0)  # 完成注册
        hub.publish({"type": "status", "data": {"n": 2}})
        hub.close()
        events = await consumer
        # 历史的 n=1 不回放，只收到订阅之后的 n=2。
        assert [item["data"]["n"] for _, item in events] == [2]

    asyncio.run(main())


def test_带LastEventID续传只补该id之后的事件():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        hub.publish({"type": "status", "data": {"n": 1}})
        hub.publish({"type": "status", "data": {"n": 2}})
        hub.publish({"type": "status", "data": {"n": 3}})
        consumer = asyncio.create_task(_collect(hub, last_event_id="epA-1"))
        await asyncio.sleep(0)  # 完成注册与续传回放
        hub.publish({"type": "status", "data": {"n": 4}})
        hub.close()
        events = await consumer
        # 只补 id 之后的事件：2、3（缓冲内续传）+ 4（实时），不重复 1。
        assert [item["data"]["n"] for _, item in events] == [2, 3, 4]

    asyncio.run(main())


def test_续传id已是最新的只收实时不重复():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        hub.publish({"type": "status", "data": {"n": 1}})
        consumer = asyncio.create_task(_collect(hub, last_event_id="epA-1"))
        await asyncio.sleep(0)
        hub.publish({"type": "status", "data": {"n": 2}})
        hub.close()
        events = await consumer
        assert [item["data"]["n"] for _, item in events] == [2]

    asyncio.run(main())


def test_世代失配立即下发reconcile_required后转实时():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epB")
        hub.publish({"type": "status", "data": {"n": 1}})
        consumer = asyncio.create_task(_collect(hub, last_event_id="epA-2"))
        await asyncio.sleep(0)
        hub.publish({"type": "status", "data": {"n": 2}})
        hub.close()
        events = await consumer
        # 首帧即 reconcile_required，载荷指明 reason 与 last_event_id。
        first = events[0][1]
        assert first["type"] == "reconcile_required"
        assert first["reason"] == "epoch_mismatch"
        assert first["last_event_id"] == "epA-2"
        # 随后转实时推送。
        assert [item["type"] for _, item in events] == [
            "reconcile_required",
            "status",
        ]

    asyncio.run(main())


def test_位置落入丢弃区间下发reconcile_required():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA", max_history=3)
        for n in range(5):
            hub.publish({"type": "status", "data": {"n": n}})
        assert hub.dropped == 2  # seq 1、2 已淘汰，缓冲只留 3、4、5
        consumer = asyncio.create_task(_collect(hub, last_event_id="epA-1"))
        await asyncio.sleep(0)
        hub.close()
        events = await consumer
        first = events[0][1]
        assert first["type"] == "reconcile_required"
        assert first["reason"] == "position_dropped"

    asyncio.run(main())


def test_工作线程发布事件到达loop内订阅者且顺序保持():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        consumer = asyncio.create_task(_collect(hub))
        await asyncio.sleep(0)

        def worker() -> None:
            for i in range(50):
                hub.publish({"type": "status", "data": {"i": i}})
            hub.close()

        thread = threading.Thread(target=worker)
        thread.start()
        events = await consumer
        thread.join()
        assert [item["data"]["i"] for _, item in events] == list(range(50))

    asyncio.run(main())


def test_close后订阅者正常结束且close幂等():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        consumer = asyncio.create_task(_collect(hub))
        await asyncio.sleep(0)
        hub.publish({"type": "status", "data": {"n": 1}})
        hub.close()
        hub.close()  # 幂等
        events = await consumer
        assert [item["data"]["n"] for _, item in events] == [1]
        assert hub.closed

    asyncio.run(main())


def test_关闭后publish被忽略():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        hub.publish({"type": "status", "data": {"n": 1}})
        hub.close()
        hub.publish({"type": "status", "data": {"n": 2}})
        await asyncio.sleep(0)
        assert await _collect(hub) == []

    asyncio.run(main())


def test_已关闭枢纽带LastEventID仍可续传缓冲内事件():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        hub.publish({"type": "status", "data": {"n": 1}})
        hub.publish({"type": "status", "data": {"n": 2}})
        hub.close()
        # 任务终态后客户端带 Last-Event-ID 续传，仍能取到缓冲内剩余事件。
        events = await _collect(hub, last_event_id="epA-1")
        assert [item["data"]["n"] for _, item in events] == [2]

    asyncio.run(main())


def test_订阅者中途取消_内部队列被注销():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        consumer = asyncio.create_task(_collect(hub))
        await asyncio.sleep(0)
        assert hub.subscriber_count == 1
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        assert hub.subscriber_count == 0
        hub.close()

    asyncio.run(main())


def test_多订阅者各自独立收实时事件():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        first = asyncio.create_task(_collect(hub))
        second = asyncio.create_task(_collect(hub))
        await asyncio.sleep(0)
        hub.publish({"type": "status", "data": {"n": 1}})
        hub.close()
        assert [item["data"]["n"] for _, item in await first] == [1]
        assert [item["data"]["n"] for _, item in await second] == [1]

    asyncio.run(main())


def test_parse_event_id合法与非法():
    assert parse_event_id("epA-3") == ("epA", 3)
    assert parse_event_id("ep-with-dash-9") == ("ep-with-dash", 9)
    for bad in ("", "noseq", "-5", "epA-", "epA-x"):
        try:
            parse_event_id(bad)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝非法 event id：{bad!r}")


def test_new_epoch无连字符():
    e1 = new_epoch()
    e2 = new_epoch()
    assert "-" not in e1
    assert e1 != e2


# ---- 两级丢弃背压（issue #55：信号必达，产物可丢可取）----


def test_可丢级类型分类():
    """可丢级 = content_delta / product；控制信号与信封一律不可丢。"""
    assert _is_droppable({"type": "content_delta", "delta": "x"})
    assert _is_droppable({"type": "product", "kind": "chapter_ready"})
    assert not _is_droppable({"type": "status"})
    assert not _is_droppable({"type": "review_required"})
    assert not _is_droppable({"type": "finalized"})
    assert not _is_droppable({"type": "error"})
    assert not _is_droppable({"type": "reconcile_required"})
    assert not _is_droppable(object())  # 信封/哨兵等非 dict
    assert _DROPPABLE_TYPES == frozenset({"content_delta", "product"})


def test_慢消费者可丢级满时丢最旧且dropped准确():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA", max_queue=3)
        gate = asyncio.Event()
        received: list[Any] = []

        async def slow() -> None:
            # 首帧后阻塞消费，撑满队列触发背压。
            async for seq, item in hub.subscribe():
                received.append((seq, item))
                await gate.wait()

        consumer = asyncio.create_task(slow())
        await asyncio.sleep(0)  # 完成注册并挂起在首帧 get
        # 发布 10 条可丢级逐字帧：第 1 条直交付给消费者，其余灌满容量 3 后丢最旧。
        for i in range(10):
            hub.publish({"type": "content_delta", "delta": str(i)})
        await asyncio.sleep(0)  # 让消费者取走首帧后回到 gate 阻塞
        gate.set()
        hub.close()
        await consumer
        # 第 1 条（直交付）+ 队列留存的最新 3 条；中间 6 条淘汰。
        deltas = [
            item["delta"] for _, item in received if isinstance(item, dict)
        ]
        assert deltas == ["0", "7", "8", "9"]
        assert hub.dropped == 6

    asyncio.run(main())


def test_慢消费者控制信号满时仍全部送达():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA", max_queue=3)
        gate = asyncio.Event()
        received: list[Any] = []

        async def slow() -> None:
            async for seq, item in hub.subscribe():
                received.append((seq, item))
                await gate.wait()

        consumer = asyncio.create_task(slow())
        await asyncio.sleep(0)
        # 4 条可丢级：首条直交付，3 条灌满队列。
        for i in range(4):
            hub.publish({"type": "content_delta", "delta": f"d{i}"})
        # 4 条控制信号：前 3 条各挤掉 1 条可丢级；第 4 条全信号队列强制超容必达。
        # 注：生产中 reconcile_required 由 subscribe() 直接 yield（逐订阅者、
        # 不过队列），本处经 publish 合成投递以驱动 force-deliver 分支；
        # review_required / finalized / error 才是经 publish 过队列的真路径。
        signals = [
            {"type": "review_required", "round": 1},
            {"type": "finalized", "chapters": []},
            {"type": "error", "message": "boom"},
            {"type": "reconcile_required", "reason": "position_dropped"},
        ]
        for sig in signals:
            hub.publish(sig)
        gate.set()
        hub.close()
        await consumer
        sig_types = [
            item["type"]
            for _, item in received
            if isinstance(item, dict)
            and item["type"]
            in {"review_required", "finalized", "error", "reconcile_required"}
        ]
        # 4 条控制信号全部送达、顺序保持。
        assert sig_types == [
            "review_required",
            "finalized",
            "error",
            "reconcile_required",
        ]
        # 恰 3 条可丢级被挤掉（第 4 条信号无可丢级可挤、强制超容，不新增丢弃）。
        assert hub.dropped == 3
        # 首条可丢级仍送达（直交付，未被淘汰）。
        droppable = [
            item["delta"]
            for _, item in received
            if isinstance(item, dict) and item["type"] == "content_delta"
        ]
        assert droppable == ["d0"]

    asyncio.run(main())


def test_subscriber_count可观测():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        assert hub.subscriber_count == 0
        c1 = asyncio.create_task(_collect(hub))
        c2 = asyncio.create_task(_collect(hub))
        await asyncio.sleep(0)  # 两个订阅者完成注册
        assert hub.subscriber_count == 2
        c1.cancel()
        try:
            await c1
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)  # 取消者注销
        assert hub.subscriber_count == 1
        hub.close()
        await c2
        assert hub.subscriber_count == 0

    asyncio.run(main())


def test_正常速率消费者不丢不重():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA", max_queue=8)

        async def drain() -> list[Any]:
            return [item async for item in hub.subscribe()]

        consumer = asyncio.create_task(drain())
        await asyncio.sleep(0)
        # 发布量超过队列容量，但每条发布间让消费者排空，故不丢不重。
        for i in range(20):
            hub.publish({"type": "content_delta", "delta": str(i)})
            await asyncio.sleep(0)
        hub.close()
        events = await consumer
        deltas = [
            item["delta"] for _, item in events if isinstance(item, dict)
        ]
        assert deltas == [str(i) for i in range(20)]
        assert hub.dropped == 0

    asyncio.run(main())


def test_取消消费者时已交付未取帧归还缓冲():
    """直接交付给 future 的帧若消费者在取走前被取消，须归还缓冲以免丢帧。

    覆盖 _BackpressureQueue.get 的取消竞态：put_nowait 在 set_result 前即
    清空 self._getter，取消路径须凭局部 future 判断已交付并归还——否则
    交付而未取的控制信号会丢，违「信号必达」。
    """

    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), epoch="epA")
        q = _BackpressureQueue(4, hub._account_drop)

        async def consume() -> Any:
            return await q.get()

        t = asyncio.create_task(consume())
        await asyncio.sleep(0)  # 挂起在首帧 get（缓冲空，future 已登记）
        # 直接交付一帧给 future（不入缓冲），消费者尚未取走即取消。
        q.put_nowait((1, {"type": "review_required", "round": 1}))
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        # 已交付未取的帧应已归还缓冲；后续仍可取到，不丢信号。
        item = await q.get()
        assert item == (1, {"type": "review_required", "round": 1})

    asyncio.run(main())
