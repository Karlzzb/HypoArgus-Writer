"""子智能体黑盒适配层与本期打桩实现。

调用形态为黑盒异步可调用，不强制改造为 LangGraph 子图；
子智能体启动/结束事件由适配层发出（本期挂钩点就位，缺省空实现）。
任务包与结果的字段规范来自设计定稿（字段即决策），见 PRD「子智能体接入」；
打桩返回结构合规的确定性模拟数据，真实现迁移时按同一接口规范替换。
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal, NotRequired, Protocol, TypedDict

SUBAGENT_START = "subagent_start"
SUBAGENT_END = "subagent_end"

EventHook = Callable[[str, dict[str, Any]], None]
"""事件挂钩：(事件类型, 载荷)；本期由调用方注入，缺省空实现。"""


def _noop_hook(event_type: str, payload: dict[str, Any]) -> None:
    """缺省事件挂钩：不做任何事，真实事件通道在后续 issue 接入。"""


class HypothesisPayload(TypedDict):
    """任务包中的假说条目。"""

    id: str
    text: str
    refute_condition: str


class MaterialPayload(TypedDict):
    """检索结果中的素材条目：逐条回链假说 ID。"""

    id: str
    hypothesis_id: str
    source: str
    excerpt: str
    relevance_score: float
    verdict: Literal["pass", "fail"]


class SearchTask(TypedDict):
    """search_agent 任务包：一次给整章假说列表（按章节批量调用）。"""

    chapter_id: str
    hypotheses: list[HypothesisPayload]
    genre: str
    existing_materials_digest: str


class SearchResult(TypedDict):
    """search_agent 检索结果。"""

    materials: list[MaterialPayload]


class PointPayload(TypedDict):
    """章节骨架中的论点条目。"""

    id: str
    text: str


class ChapterSpecPayload(TypedDict):
    """rewriter_loop 任务包中的章节骨架。"""

    id: str
    title: str
    points: list[PointPayload]
    hypotheses: list[HypothesisPayload]


class RevisionDirectivePayload(TypedDict):
    """rewriter_loop 任务包中的修订指令条目。"""

    type: Literal["rewrite_only", "evidence_augmented"]
    instruction: str


class SelfCheckPayload(TypedDict):
    """rewriter_loop 单章自检结果。"""

    citations_ok: bool
    issues: list[str]


class RewriteTask(TypedDict):
    """rewriter_loop 任务包：统包首写（draft）与纯改写（revise）两种模式。"""

    mode: Literal["draft", "revise"]
    chapter_spec: ChapterSpecPayload
    materials: list[MaterialPayload]
    prev_chapter_summary: str
    revision_directives: NotRequired[list[RevisionDirectivePayload]]
    current_text: NotRequired[str]


class RewriteResult(TypedDict):
    """rewriter_loop 改写结果。"""

    chapter_text: str
    """章节正文，含原位角标（形如 [素材id]）。"""
    chapter_summary: str
    self_check: SelfCheckPayload


class Subagent(Protocol):
    """子智能体黑盒调用协议：编排节点只依赖此协议，不关心真实现或打桩。"""

    @property
    def unit(self) -> str:
        """运行单元名，用于事件上报。"""
        ...

    async def run(self, task: dict[str, Any]) -> dict[str, Any]: ...


class SubagentAdapter:
    """黑盒适配层：包装异步可调用，调用前后发出子智能体启动/结束事件。"""

    def __init__(
        self,
        unit: str,
        run_impl: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        event_hook: EventHook = _noop_hook,
    ) -> None:
        self.unit = unit
        self._run_impl = run_impl
        self._event_hook = event_hook

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        self._event_hook(SUBAGENT_START, {"unit": self.unit})
        result = await self._run_impl(task)
        self._event_hook(SUBAGENT_END, {"unit": self.unit})
        return result


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


async def stub_rewriter_loop_run(task: dict[str, Any]) -> dict[str, Any]:
    """rewriter_loop 打桩：产出含原位角标的确定性正文、章节摘要与自检结果。

    draft 模式承接上一章摘要生成正文；revise 模式在 current_text 基础上
    逐条附注修订指令，保证两种模式的接口都可空转。
    """
    spec = task["chapter_spec"]
    pass_materials = [
        material for material in task["materials"] if material["verdict"] == "pass"
    ]

    if task["mode"] == "revise":
        directives = task.get("revision_directives", [])
        notes = "".join(
            f"（修订落实：{directive['instruction']}）" for directive in directives
        )
        chapter_text = f"{task.get('current_text', '')}{notes}"
    else:
        paragraphs: list[str] = []
        prev_summary = task["prev_chapter_summary"]
        if prev_summary:
            paragraphs.append(f"承接上一章：{prev_summary}")
        paragraphs.append(f"本章《{spec['title']}》围绕以下论点展开（打桩正文）。")
        for point in spec["points"]:
            paragraphs.append(f"论点：{point['text']}（打桩论证）")
        for material in pass_materials:
            paragraphs.append(
                f"素材佐证假说 {material['hypothesis_id']}（打桩）[{material['id']}]"
            )
        chapter_text = "\n\n".join(paragraphs)

    point_digest = "；".join(point["text"] for point in spec["points"])
    chapter_summary = f"《{spec['title']}》要点：{point_digest or '（无论点）'}（打桩摘要）"
    return {
        "chapter_text": chapter_text,
        "chapter_summary": chapter_summary,
        "self_check": SelfCheckPayload(citations_ok=True, issues=[]),
    }


def make_stub_search_agent(event_hook: EventHook = _noop_hook) -> SubagentAdapter:
    """构造 search_agent 打桩适配器。"""
    return SubagentAdapter("search_agent", stub_search_agent_run, event_hook)


def make_stub_rewriter_loop(event_hook: EventHook = _noop_hook) -> SubagentAdapter:
    """构造 rewriter_loop 打桩适配器。"""
    return SubagentAdapter("rewriter_loop", stub_rewriter_loop_run, event_hook)
