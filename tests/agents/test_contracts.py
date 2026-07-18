"""子智能体适配层的契约测试：启动 / 结束事件挂钩。

直接以最小异步实现测 SubagentAdapter 本身，不依赖任何具体子智能体。
"""

import asyncio
from typing import Any

from agents.contracts import SubagentAdapter
from domain.events import SUBAGENT_END, SUBAGENT_START


def test_适配层_一次运行依次发出启动与结束事件() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def record_hook(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    async def run_impl(task: dict[str, Any]) -> dict[str, Any]:
        return {"echo": task}

    adapter = SubagentAdapter("search_agent", run_impl, record_hook)
    result = asyncio.run(adapter.run({"k": "v"}))

    assert result == {"echo": {"k": "v"}}
    assert [event_type for event_type, _ in events] == [SUBAGENT_START, SUBAGENT_END]
    for _, payload in events:
        assert payload["unit"] == "search_agent"
