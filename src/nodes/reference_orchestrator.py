"""reference_orchestrator 主节点：检索阶段经 Send 并行扇出的单章检索调度。

framework_orchestrator 后的条件边为每个待检索章节各发一个 Send（见 graph.py），
本节点每个并行分支只检索一章：Send 载荷携带目标章 id 与装配所需的状态切片，
素材经 citation_library 的合并 reducer 汇入主状态并按超步落 checkpoint
（ADR-0001 约束 1：崩溃重跑只损失进行中的分支），
每个检索分支完成后经条件 Send 进入管线首写节点，全部分支完成后汇合进入终审。

任务包的 existing_materials_digest 只反映既有引文库（扇出前的快照）：
并行后轮内跨章摘要链取消，跨章去重收敛到合并 reducer（按 URL）。
没有任何假说的章节不产生 Send；已有素材入库的章节（恢复续跑等场景）不重发。
返回素材逐条转为结构化引文库条目，pass 与 fail 都入库（verdict 供后续环节筛选）。
本节点不直接调 LLM，检索调度为纯程序逻辑（保持非子图边界，ADR-0001 约束 3）。
"""

from typing import Any, Final, Protocol, cast

from assembly.assembler_config import AssemblerConfig
from assembly.context_assembler import assemble
from domain.env_config import read_positive_int
from domain.state import WorkflowStatus, WritingAgentState
from agents.contracts import MaterialPayload, SearchTask, Subagent, material_from_payload
from nodes.chapter_drafter import DRAFT_CHAPTER_ID_KEY
from nodes.writing_orchestrator import (
    chapter_by_id,
    chapter_points,
    flatten_hypotheses,
)

REFERENCE_CHAPTER_ID_KEY: Final = "reference_chapter_id"
"""Send 载荷中目标章 id 的键名：任务态专用，不是主状态字段。"""


class ReferenceSendPayload(WritingAgentState):
    """Send 载荷类型：主状态切片 + 目标章 id（任务态专用键）。"""

    reference_chapter_id: str


class ReferenceOrchestratorNode(Protocol):
    """节点函数类型：入参是 Send 载荷（任务态），返回主状态的部分更新。"""

    def __call__(self, state: ReferenceSendPayload) -> WritingAgentState: ...


def reference_send_payloads(state: WritingAgentState) -> list[ReferenceSendPayload]:
    """为全部待检索章节构造 Send 载荷：目标章 id + 装配所需的状态切片。

    载荷只携带 search_agent 配方与任务包实际消费的字段（目标章骨架、
    既有引文库、文体），不复制整个主状态，控制 checkpoint 中 pending Send
    的体积；引文库整体携带，供 citation_digest 段给出既有引文库摘要。
    没有假说的章节跳过；已有素材入库的章节（恢复续跑等场景）不重发。
    """
    retrieved = {
        material.chapter_id for material in state.get("citation_library", [])
    }
    return [
        ReferenceSendPayload(
            reference_chapter_id=chapter.id,
            # 首写需要前章规划摘要链；该只读结构随同章节管线携带。
            outline=state.get("outline", []),
            citation_library=state.get("citation_library", []),
            genre=state.get("genre", ""),
        )
        for chapter in state.get("outline", [])
        if chapter.id not in retrieved and flatten_hypotheses(chapter)
    ]


def make_reference_orchestrator_node(
    search_agent: Subagent,
    assembler_config: AssemblerConfig | None = None,
) -> ReferenceOrchestratorNode:
    """构造 reference_orchestrator 节点函数：注入 search_agent 适配器。

    assembler_config 为 None 时在节点执行时读取环境变量装配配置。
    """

    def node(state: ReferenceSendPayload) -> WritingAgentState:
        chapter_id = state[REFERENCE_CHAPTER_ID_KEY]
        chapter = chapter_by_id(state, chapter_id)
        context = assemble(state, "search_agent", config=assembler_config)
        task = SearchTask(
            chapter_id=chapter_id,
            points=chapter_points(chapter),
            hypotheses=flatten_hypotheses(chapter),
            genre=state.get("genre", ""),
            existing_materials_digest=context.text("citation_digest"),
        )
        result = _run_search_agent(search_agent, dict(task))
        materials = [
            material_from_payload(material, chapter_id)
            for material in cast(list[MaterialPayload], result["materials"])
        ]
        return WritingAgentState(
            citation_library=materials,
            status=WorkflowStatus.REFERENCE_FETCHING,
            # 传给同一 Send 分支的管线首写节点；主状态合并后的残留值无业务意义。
            **{DRAFT_CHAPTER_ID_KEY: chapter_id},
            current_node_llm_config={"unit": "reference_orchestrator"},
        )

    return node


SEARCH_TIMEOUT_ENV: Final = "REFERENCE_SEARCH_TIMEOUT_SECONDS"
"""单章检索硬超时（秒）的环境变量名：缺省 600。"""

_DEFAULT_SEARCH_TIMEOUT_SECONDS: Final = 600


def _run_search_agent(search_agent: Subagent, task: dict[str, Any]) -> dict[str, Any]:
    """在当前节点线程内执行异步检索适配器，避免嵌套占用 Pregel 执行器。

    外层包 asyncio.wait_for 硬超时（issue #80 放大器）：检索源间歇停顿
    （如 Volcano 返回 http-200 空 Result 后挂起）时该分支不得永不返回——
    超时抛 TimeoutError 由图的错误路径兜底，任务失败可观测而非静默挂死。
    """
    import asyncio
    import os

    timeout_seconds = read_positive_int(
        os.environ, SEARCH_TIMEOUT_ENV, _DEFAULT_SEARCH_TIMEOUT_SECONDS
    )

    async def _bounded() -> dict[str, Any]:
        return await asyncio.wait_for(search_agent.run(task), timeout=timeout_seconds)

    return asyncio.run(_bounded())
