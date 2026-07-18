"""reference_orchestrator 主节点：逐章批量调度 search_agent 的检索编排逻辑。

按 outline 章节顺序逐章一次调用适配器，任务包携带该章全部假说的扁平列表；
没有任何假说的章节直接跳过，不产生适配器调用。
适配器是异步可调用，节点保持同步函数形态：一次 asyncio.run 驱动全部章节串行调用。
返回素材逐条转为结构化引文库条目，pass 与 fail 都入库（verdict 供后续环节筛选）。
本节点不直接调 LLM，检索调度为纯程序逻辑。
"""

import asyncio
from typing import Protocol

from state import ChapterSpec, Material, WorkflowStatus, WritingAgentState
from subagents import HypothesisPayload, SearchTask, Subagent


class ReferenceOrchestratorNode(Protocol):
    """节点函数类型：入参与返回均为图状态（state 具名，满足 LangGraph 节点协议）。"""

    def __call__(self, state: WritingAgentState) -> WritingAgentState: ...


def _flatten_hypotheses(chapter: ChapterSpec) -> list[HypothesisPayload]:
    """把章节内全部论点下的假说按顺序拉平为任务包条目。"""
    return [
        HypothesisPayload(
            id=hypothesis.id,
            text=hypothesis.text,
            refute_condition=hypothesis.refute_condition,
        )
        for point in chapter.points
        for hypothesis in point.hypotheses
    ]


def make_reference_orchestrator_node(
    search_agent: Subagent,
) -> ReferenceOrchestratorNode:
    """构造 reference_orchestrator 节点函数。"""

    async def _run_all_chapters(state: WritingAgentState) -> list[Material]:
        """串行逐章调用 search_agent，边调用边累积引文库。"""
        genre = state.get("genre", "")
        library: list[Material] = []
        for chapter in state.get("outline", []):
            hypotheses = _flatten_hypotheses(chapter)
            if not hypotheses:
                continue
            task = SearchTask(
                chapter_id=chapter.id,
                hypotheses=hypotheses,
                genre=genre,
                existing_materials_digest=f"引文库已有素材 {len(library)} 条",
            )
            result = await search_agent.run(dict(task))
            for material in result["materials"]:
                library.append(
                    Material(
                        id=material["id"],
                        hypothesis_id=material["hypothesis_id"],
                        chapter_id=chapter.id,
                        source=material["source"],
                        url=None,
                        excerpt=material["excerpt"],
                        relevance_score=material["relevance_score"],
                        verdict=material["verdict"],
                    )
                )
        return library

    def node(state: WritingAgentState) -> WritingAgentState:
        citation_library = asyncio.run(_run_all_chapters(state))
        return WritingAgentState(
            citation_library=citation_library,
            status=WorkflowStatus.REFERENCE_FETCHING,
            current_node_llm_config={"unit": "reference_orchestrator"},
        )

    return node
