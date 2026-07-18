"""writing_orchestrator 主节点：逐章串行写作的纯调度逻辑。

写作由 rewriter_loop 子智能体承担，本节点不调 LLM，只负责：
按大纲顺序逐章组装任务包（draft 模式）、串行驱动异步适配器
（后一章必须等前一章结果，用其 chapter_summary 承接摘要链）、
把改写结果与单章自检转为 ChapterDraft 写回 State。
"""

import asyncio
from typing import Protocol

from state import (
    ChapterDraft,
    ChapterSpec,
    Material,
    SelfCheck,
    WorkflowStatus,
    WritingAgentState,
)
from subagents import (
    ChapterSpecPayload,
    HypothesisPayload,
    MaterialPayload,
    PointPayload,
    RewriteTask,
    Subagent,
)


class WritingOrchestratorNode(Protocol):
    """节点函数类型：入参与返回均为图状态（state 具名，满足 LangGraph 节点协议）。"""

    def __call__(self, state: WritingAgentState) -> WritingAgentState: ...


def _chapter_spec_payload(chapter: ChapterSpec) -> ChapterSpecPayload:
    """章节骨架转任务包字典：论点列表 + 该章全部假说扁平列表。"""
    return ChapterSpecPayload(
        id=chapter.id,
        title=chapter.title,
        points=[PointPayload(id=point.id, text=point.text) for point in chapter.points],
        hypotheses=[
            HypothesisPayload(
                id=hypothesis.id,
                text=hypothesis.text,
                refute_condition=hypothesis.refute_condition,
            )
            for point in chapter.points
            for hypothesis in point.hypotheses
        ],
    )


def _chapter_materials(
    citation_library: list[Material], chapter_id: str
) -> list[MaterialPayload]:
    """筛选该章且 verdict=pass 的素材，转为任务包字典。"""
    return [
        MaterialPayload(
            id=material.id,
            hypothesis_id=material.hypothesis_id,
            source=material.source,
            excerpt=material.excerpt,
            relevance_score=material.relevance_score,
            verdict=material.verdict,
        )
        for material in citation_library
        if material.chapter_id == chapter_id and material.verdict == "pass"
    ]


def make_writing_orchestrator_node(
    rewriter_loop: Subagent,
) -> WritingOrchestratorNode:
    """构造 writing_orchestrator 节点函数：注入 rewriter_loop 黑盒适配器。"""

    async def _write_all_chapters(state: WritingAgentState) -> list[ChapterDraft]:
        """章节严格串行写作：await 前一章结果后才组装并下发后一章任务包。"""
        citation_library = state.get("citation_library", [])
        drafts: list[ChapterDraft] = []
        prev_chapter_summary = ""
        for chapter in state.get("outline", []):
            task = RewriteTask(
                mode="draft",
                chapter_spec=_chapter_spec_payload(chapter),
                materials=_chapter_materials(citation_library, chapter.id),
                prev_chapter_summary=prev_chapter_summary,
            )
            result = await rewriter_loop.run(dict(task))
            self_check = result["self_check"]
            drafts.append(
                ChapterDraft(
                    chapter_id=chapter.id,
                    text=result["chapter_text"],
                    summary=result["chapter_summary"],
                    self_check=SelfCheck(
                        citations_ok=self_check["citations_ok"],
                        issues=self_check["issues"],
                    ),
                )
            )
            prev_chapter_summary = result["chapter_summary"]
        return drafts

    def node(state: WritingAgentState) -> WritingAgentState:
        chapter_drafts = asyncio.run(_write_all_chapters(state))
        return WritingAgentState(
            chapter_drafts=chapter_drafts,
            status=WorkflowStatus.ARTICLE_WRITING,
            current_node_llm_config={"unit": "writing_orchestrator"},
        )

    return node
