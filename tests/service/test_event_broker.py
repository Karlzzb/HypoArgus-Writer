"""event_broker 事件枢纽的单元测试（传输底座：世代 id + Last-Event-ID 续传）。

覆盖：世代 id `{epoch}-{seq}` 单流单调递增、不带 Last-Event-ID 的新订阅只收
实时事件（不回放）、带 Last-Event-ID 续传只补该 id 之后的事件、世代失配与
位置丢弃立即下发 reconcile_required、跨线程发布、关闭语义、订阅者注销、
有界缓冲淘汰与 dropped 计数。
"""

import asyncio
import threading
from typing import Any

from service.event_broker import EventHub, new_epoch, parse_event_id


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
