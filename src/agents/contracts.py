"""子智能体契约：任务包/结果的字段规范、黑盒调用协议与事件适配层。

调用形态为黑盒异步可调用，不强制改造为 LangGraph 子图；
编排节点只依赖本模块的 Subagent 协议，不关心真实现或打桩。
任务包与结果的字段规范以本文件为唯一事实源（字段即决策）。
"""

from collections.abc import Awaitable, Callable
from typing import (
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

# TypedDict 取 typing_extensions 版本：Python 3.12 前 Pydantic 不接受
# typing.TypedDict（独立检索接口的响应模型直接复用 MaterialPayload）。
from typing_extensions import NotRequired, TypedDict

from domain.events import SUBAGENT_END, SUBAGENT_START, EventHook, noop_hook
from domain.state import Material, SourceKind


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
    url: str | None
    """来源链接：联网来源必带真实链接，知识库与结构化来源可为 None。"""
    source_kind: SourceKind
    excerpt: str
    relevance_score: float
    verdict: Literal["pass", "fail"]


def material_from_payload(payload: MaterialPayload, chapter_id: str) -> Material:
    """检索结果素材条目转结构化引文库条目：契约到 State 的唯一映射点。

    chapter_id 由编排方按当前检索章节补齐（契约条目只回链假说，不带章节）。
    """
    return Material(
        id=payload["id"],
        hypothesis_id=payload["hypothesis_id"],
        chapter_id=chapter_id,
        source=payload["source"],
        url=payload["url"],
        source_kind=payload["source_kind"],
        excerpt=payload["excerpt"],
        relevance_score=payload["relevance_score"],
        verdict=payload["verdict"],
    )


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
    chapter_type: str | None
    """章型：State 大纲随章携带的骨架事实（ADR-0005），编排层原样透传，
    lint 直接消费、不从位置或标题反推；自由结构模式为 None。"""
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


class RuleViolationEntry(TypedDict):
    """分区式修订说明·规则违规区单条：位置摘录 + 修改指导 + error/warn 定级。

    确定性 lint 违规与四维 LLM 自审违规折成同形一条：``location_excerpt`` 对
    确定性违规可为空串（lint 不总能定位到片段），自审违规给正文片段；
    ``severity`` 是引用类门禁之外的软/硬定级依据（error 阻断、warn 提示）。
    """

    rule: str
    location_excerpt: str
    guidance: str
    severity: Literal["error", "warn"]


class ConflictHintEntry(TypedDict):
    """分区式修订说明·冲突提示区单条：用户指令与规则冲突处，用户指令优先。"""

    description: str


class RevisionNotePayload(TypedDict):
    """分区式修订说明（chapter_reviewer 产物，部分取代 ADR-0004 的 self_check 折叠）。

    四区语义：用户指令区（原文逐字保留、零改写）、规则违规区（逐条含位置摘录、
    修改指导与定级）、冲突提示区（用户指令优先）、passed 结论（error 级违规为空即过）。
    """

    user_directives: str
    """用户指令区：revise 时取用户意见原文、逐字保留；draft 无用户意见为空串。"""
    rule_violations: list[RuleViolationEntry]
    conflict_hints: list[ConflictHintEntry]
    passed: bool
    """error 级违规为空即过（warn 级不阻断）。"""


class RewriteTask(TypedDict):
    """rewriter_loop 任务包：统包首写（draft）与纯改写（revise）两种模式。"""

    mode: Literal["draft", "revise"]
    doc_type: str
    """文种：State 中经文种注册表锚定的确定性事实（ADR-0005），编排层原样携带。"""
    doc_variant: str | None
    """文种内变体（目前仅人才培养方案声明本科/高职）；无变体为 None。"""
    chapter_spec: ChapterSpecPayload
    materials: list[MaterialPayload]
    prev_chapter_summary: str
    revision_directives: NotRequired[list[RevisionDirectivePayload]]
    """旧修订指令字段（ADR-0004）：与 ``revision_note`` 并存（expand），删除留 T3b。"""
    revision_note: NotRequired[RevisionNotePayload]
    """新分区式修订说明字段（ADR-0006）：本期仅落契约，rewriter 消费留 T3。"""
    current_text: NotRequired[str]


class RewriteResult(TypedDict):
    """rewriter_loop 改写结果。"""

    chapter_text: str
    """章节正文，含原位角标（形如 [素材id]）。"""
    chapter_summary: str
    self_check: SelfCheckPayload
    doc_type: str
    """回带任务包携带的文种：产物按哪套文种规则产出，随结果自证、供排障回放。"""
    doc_variant: str | None
    """回带任务包携带的变体，语义同 doc_type。"""


class ReviewTask(TypedDict):
    """chapter_reviewer 任务包：对一章成稿做章级评审（确定性 lint + 四维自审）。

    输入齐备一章评审所需的全部上下文：章骨架（含 id/论点/假说）、章文本、素材、
    摘要链（前章摘要）；revise 模式另携用户意见原文供修订说明的用户指令区逐字保留。
    """

    mode: Literal["review", "revise"]
    doc_type: str
    """文种：与写作任务包同源，经契约携带（ADR-0005），评审按文种加载校验与裁决项。"""
    doc_variant: str | None
    chapter_spec: ChapterSpecPayload
    chapter_text: str
    materials: list[MaterialPayload]
    prev_chapter_summary: str
    user_feedback: NotRequired[str]
    """revise 时的用户意见原文：逐字进入修订说明的用户指令区，评审不改写。"""


class ReviewResult(TypedDict):
    """chapter_reviewer 评审结果：分区式修订说明 + 按引用类规则折叠的自检。"""

    revision_note: RevisionNotePayload
    self_check: SelfCheckPayload
    """按引用类规则折叠：终态正文仍存引用类违规则 citations_ok=False，交全局终审裁决。"""


@runtime_checkable
class Subagent(Protocol):
    """子智能体黑盒调用协议：编排节点只依赖此协议，不关心真实现或打桩。

    runtime_checkable 供装配层区分实例与工厂形态的注入（结构性判定：
    有 run 与 unit 属性即视为实例）。
    """

    @property
    def unit(self) -> str:
        """运行单元名，用于事件上报。"""
        ...

    async def run(self, task: dict[str, Any]) -> dict[str, Any]: ...


DIAGNOSTICS_SUMMARY_KEY = "_diagnostics_summary"
"""结果 dict 中的诊断摘要保留键：适配层弹出并入结束事件载荷，不进对外契约。

真实现 run 把本次调用的诊断摘要子集（计数、耗时等元数据）放在此键下，
SubagentAdapter 在发结束事件前弹出，作为 ``diagnostics`` 字段随
SUBAGENT_END 上报；返回给编排方的结果不含此键，各结果契约保持不变。
"""


class SubagentAdapter:
    """黑盒适配层：包装异步可调用，调用前后发出子智能体启动/结束事件。

    结果 dict 若携带 ``DIAGNOSTICS_SUMMARY_KEY`` 保留键，弹出后并入
    结束事件载荷（详见该常量 docstring）。
    """

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
        context = self._event_context(task)
        self._event_hook(SUBAGENT_START, {"unit": self.unit, **context})
        result = await self._run_impl(task)
        summary = result.pop(DIAGNOSTICS_SUMMARY_KEY, None)
        end_payload = {"unit": self.unit, **context}
        if isinstance(summary, dict) and summary:
            end_payload["diagnostics"] = summary
        self._event_hook(SUBAGENT_END, end_payload)
        return result

    @staticmethod
    def _event_context(task: dict[str, Any]) -> dict[str, Any]:
        """从任务包提取事件业务上下文：章节 id 与调用模式，取不到则为 None。

        检索任务包的章节 id 在顶层，改写任务包的在章节骨架内；
        调用模式仅改写任务包携带（draft/revise），检索任务包为 None。
        """
        chapter_id = task.get("chapter_id")
        if chapter_id is None:
            chapter_spec = task.get("chapter_spec")
            if isinstance(chapter_spec, dict):
                chapter_id = chapter_spec.get("id")
        return {"chapter_id": chapter_id, "mode": task.get("mode")}
