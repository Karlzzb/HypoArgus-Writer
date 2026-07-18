"""event_broker 事件枢纽的单元测试。

覆盖：先发布后订阅的历史回放、多订阅者独立接收、
工作线程跨线程发布、关闭语义（幂等/忽略发布/关闭后订阅）、
订阅者取消后队列注销、历史超上限淘汰与 dropped 计数。
"""

import asyncio
import threading
from typing import Any

from event_broker import EventHub


async def _collect_all(hub: EventHub) -> list[Any]:
    """消费订阅直到枢纽关闭，返回收到的全部事件。"""
    received: list[Any] = []
    async for item in hub.subscribe():
        received.append(item)
    return received


def test_先发布后订阅_历史回放不漏事件():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop())
        hub.publish("历史1")
        hub.publish("历史2")
        consumer = asyncio.create_task(_collect_all(hub))
        await asyncio.sleep(0)  # 让订阅者完成注册与历史回放。
        hub.publish("实时3")
        hub.close()
        assert await consumer == ["历史1", "历史2", "实时3"]

    asyncio.run(main())


def test_多订阅者各自独立收到全部事件():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop())
        hub.publish("a")
        first = asyncio.create_task(_collect_all(hub))
        second = asyncio.create_task(_collect_all(hub))
        await asyncio.sleep(0)
        hub.publish("b")
        hub.publish("c")
        hub.close()
        assert await first == ["a", "b", "c"]
        assert await second == ["a", "b", "c"]

    asyncio.run(main())


def test_工作线程发布_事件到达loop内订阅者且顺序保持():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop())
        consumer = asyncio.create_task(_collect_all(hub))
        await asyncio.sleep(0)

        def worker() -> None:
            for i in range(50):
                hub.publish(i)
            hub.close()

        thread = threading.Thread(target=worker)
        thread.start()
        received = await consumer
        thread.join()
        assert received == list(range(50))

    asyncio.run(main())


def test_close后订阅者正常结束且close幂等():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop())
        consumer = asyncio.create_task(_collect_all(hub))
        await asyncio.sleep(0)
        hub.publish("唯一事件")
        hub.close()
        hub.close()  # 幂等：重复关闭不抛错、不重复投递哨兵。
        assert await consumer == ["唯一事件"]
        assert hub.closed

    asyncio.run(main())


def test_关闭后publish被忽略():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop())
        hub.publish("关闭前")
        hub.close()
        hub.publish("关闭后")
        await asyncio.sleep(0)
        assert await _collect_all(hub) == ["关闭前"]

    asyncio.run(main())


def test_关闭后新订阅仅回放历史即结束():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop())
        hub.publish("h1")
        hub.publish("h2")
        hub.close()
        assert await _collect_all(hub) == ["h1", "h2"]

    asyncio.run(main())


def test_订阅者中途取消_内部队列被注销():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop())
        consumer = asyncio.create_task(_collect_all(hub))
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


def test_历史超上限淘汰最旧且dropped计数正确():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop(), max_history=3)
        for i in range(5):
            hub.publish(i)
        assert hub.dropped == 2
        hub.close()
        # 新订阅者只回放留存的最新 3 条历史。
        assert await _collect_all(hub) == [2, 3, 4]

    asyncio.run(main())


def test_订阅期间发布的实时事件不与回放重复():
    async def main() -> None:
        hub = EventHub(asyncio.get_running_loop())
        hub.publish("历史")
        received: list[Any] = []

        async def consume() -> None:
            async for item in hub.subscribe():
                received.append(item)
                # 回放第一条时立即再发布，验证不重复不丢失。
                if item == "历史":
                    hub.publish("回放期间发布")

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        hub.close()
        await consumer
        assert received == ["历史", "回放期间发布"]

    asyncio.run(main())
