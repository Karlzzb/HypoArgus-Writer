"""search_agent 子智能体打桩：按章节批量生成确定性模拟素材。

真实现（agents.search_agent.agent.make_search_agent）已按
contracts.SearchTask/SearchResult 同一接口规范落地并成为缺省装配；
打桩同包共存，供空转链路与测试显式注入（工厂签名与真实现一致）。
"""

from typing import Any

from agents.contracts import MaterialPayload, SourceKind, SubagentAdapter
from agents.search_agent.mapping import material_id_from_source_ref
from domain.events import EventHook, noop_hook

UNIT = "search_agent"

_STUB_SOURCE_KINDS: tuple[SourceKind, ...] = ("web", "knowledge_base", "structured_data")


def _stub_source_kind(hypothesis_id: str) -> SourceKind:
    """按假说 ID 字节和确定性分派来源通道。

    与假说在任务包中的位置无关：同一假说跨调用稳定，不同假说（含跨章）
    分散到三条通道，使一次完整任务的书目输出覆盖多种类型标识。
    """
    return _STUB_SOURCE_KINDS[sum(hypothesis_id.encode()) % len(_STUB_SOURCE_KINDS)]


async def stub_search_agent_run(task: dict[str, Any]) -> dict[str, Any]:
    """search_agent 打桩：每条假说生成一条 pass 素材，确定性回链假说 ID。

    来源通道按假说 ID 确定性分派三值；联网来源带确定性的打桩链接，
    知识库与结构化来源无链接（与真实通道的链接有无语义一致）。
    """
    materials: list[MaterialPayload] = []
    for hypothesis in task["hypotheses"]:
        source_kind = _stub_source_kind(hypothesis["id"])
        source_ref = {
            "stub_source": "search_agent",
            "chapter_id": task["chapter_id"],
            "hypothesis_id": hypothesis["id"],
        }
        if source_kind == "web":
            source_ref["url"] = f"https://stub.example/{hypothesis['id']}"
        materials.append(
            MaterialPayload(
                id=material_id_from_source_ref(source_kind, source_ref),
                hypothesis_id=hypothesis["id"],
                source=f"打桩来源（{task['genre'] or '未识别品类'}）",
                url=source_ref.get("url"),
                source_kind=source_kind,
                source_ref=source_ref,
                excerpt=f"打桩摘录：支撑假说「{hypothesis['text']}」的模拟证据。",
                relevance_score=0.9,
                verdict="pass",
            )
        )
    return {"materials": materials}


def make_stub_search_agent(event_hook: EventHook = noop_hook) -> SubagentAdapter:
    """构造 search_agent 打桩适配器。"""
    return SubagentAdapter(UNIT, stub_search_agent_run, event_hook)
