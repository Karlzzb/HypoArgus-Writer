"""子智能体契约：任务包/结果的字段规范、黑盒调用协议与事件适配层。

调用形态为黑盒异步可调用，不强制改造为 LangGraph 子图；
编排节点只依赖本模块的 Subagent 协议，不关心真实现或打桩。
任务包与结果的字段规范来自设计定稿（字段即决策），见 PRD「子智能体接入」。
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal, NotRequired, Protocol, TypedDict

from domain.events import SUBAGENT_END, SUBAGENT_START, EventHook, noop_hook


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
        event_hook: EventHook = noop_hook,
    ) -> None:
        self.unit = unit
        self._run_impl = run_impl
        self._event_hook = event_hook

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        self._event_hook(SUBAGENT_START, {"unit": self.unit})
        result = await self._run_impl(task)
        self._event_hook(SUBAGENT_END, {"unit": self.unit})
        return result
