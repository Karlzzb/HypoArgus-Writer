"""search_agent 子智能体：按章节批量检索假说证据。

本期为打桩实现；真实现落地时按 contracts.SearchTask/SearchResult
同一接口规范替换，工厂签名不变。
"""

from typing import Any

from agents.contracts import MaterialPayload, SubagentAdapter
from domain.events import EventHook, noop_hook

UNIT = "search_agent"


async def stub_search_agent_run(task: dict[str, Any]) -> dict[str, Any]:
    """search_agent 打桩：每条假说生成一条 pass 素材，确定性回链假说 ID。"""
    materials: list[MaterialPayload] = [
        MaterialPayload(
            id=f"m-{hypothesis['id']}",
            hypothesis_id=hypothesis["id"],
            source=f"打桩来源（{task['genre'] or '未识别品类'}）",
            excerpt=f"打桩摘录：支撑假说「{hypothesis['text']}」的模拟证据。",
            relevance_score=0.9,
            verdict="pass",
        )
        for hypothesis in task["hypotheses"]
    ]
    return {"materials": materials}


def make_stub_search_agent(event_hook: EventHook = noop_hook) -> SubagentAdapter:
    """构造 search_agent 打桩适配器。"""
    return SubagentAdapter(UNIT, stub_search_agent_run, event_hook)
