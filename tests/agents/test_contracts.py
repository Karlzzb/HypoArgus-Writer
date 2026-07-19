"""子智能体适配层的契约测试：启动 / 结束事件挂钩。

直接以最小异步实现测 SubagentAdapter 本身，不依赖任何具体子智能体。
"""

import asyncio
from typing import Any

from agents.contracts import SubagentAdapter
from domain.events import SUBAGENT_END, SUBAGENT_START


def _make_recorder() -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    """构造把 (事件类型, 载荷) 收进列表的挂钩。"""
    events: list[tuple[str, dict[str, Any]]] = []

    def record_hook(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    return events, record_hook


async def _echo_impl(task: dict[str, Any]) -> dict[str, Any]:
    return {"echo": task}


def test_适配层_一次运行依次发出启动与结束事件() -> None:
    events, record_hook = _make_recorder()

    adapter = SubagentAdapter("search_agent", _echo_impl, record_hook)
    result = asyncio.run(adapter.run({"k": "v"}))

    assert result == {"echo": {"k": "v"}}
    assert [event_type for event_type, _ in events] == [SUBAGENT_START, SUBAGENT_END]
    # 任务包无 chapter_id / chapter_spec / mode 时，两键取 None 兜底。
    for _, payload in events:
        assert payload == {"unit": "search_agent", "chapter_id": None, "mode": None}


def test_适配层_检索任务包_载荷带顶层章节id且mode为None() -> None:
    events, record_hook = _make_recorder()

    adapter = SubagentAdapter("search_agent", _echo_impl, record_hook)
    task = {
        "chapter_id": "ch-1",
        "hypotheses": [],
        "genre": "行业白皮书",
        "existing_materials_digest": "",
    }
    asyncio.run(adapter.run(task))

    assert [event_type for event_type, _ in events] == [SUBAGENT_START, SUBAGENT_END]
    for _, payload in events:
        assert payload == {"unit": "search_agent", "chapter_id": "ch-1", "mode": None}


def test_适配层_改写任务包_载荷带章节骨架id与mode() -> None:
    events, record_hook = _make_recorder()

    adapter = SubagentAdapter("rewriter_loop", _echo_impl, record_hook)
    task = {
        "mode": "draft",
        "chapter_spec": {"id": "ch-1", "title": "示例章节", "points": [], "hypotheses": []},
        "materials": [],
        "prev_chapter_summary": "",
    }
    asyncio.run(adapter.run(task))

    assert [event_type for event_type, _ in events] == [SUBAGENT_START, SUBAGENT_END]
    for _, payload in events:
        assert payload == {"unit": "rewriter_loop", "chapter_id": "ch-1", "mode": "draft"}
