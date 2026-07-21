"""chapter_drafter 主节点：首写阶段经 Send 并行扇出的单章首写。

reference_orchestrator 后的条件边为每个未写章节各发一个 Send（见 graph.py），
本节点每个并行分支只写一章：Send 载荷携带目标章 id 与装配所需的状态切片，
产物经 chapter_drafts 的合并 reducer 汇入主状态并按超步落 checkpoint
（ADR-0001 约束 1：崩溃重跑只损失进行中的分支），
全部分支完成后汇合进入 citation_validator。

前文承接用框架生成的规划摘要链（extract_planned_summary_chain 段，
各章用其之前各章的规划摘要衔接），不依赖前章实际写成的摘要，
各章因此可以并行；修订与终审回退仍由 writing_orchestrator 串行处理。

写作由 rewriter_loop 子智能体承担（保持非子图边界，ADR-0001 约束 3），
本节点自身不调 LLM。状态回写只含带合并 / keep_last reducer 的字段，
避免并行分支在同一超步触发 LastValue 写入冲突。
"""

import asyncio
from typing import Final, Protocol

from assembly.assembler_config import AssemblerConfig, load_assembler_config
from assembly.context_assembler import assemble
from domain.doc_types import carried_doc_facts
from domain.state import (
    ChapterDraft,
    SelfCheck,
    WorkflowStatus,
    WritingAgentState,
)
from agents.contracts import RewriteTask, Subagent
from nodes.writing_orchestrator import (
    chapter_by_id,
    chapter_spec_payload,
    materials_from_segment,
)

DRAFT_CHAPTER_ID_KEY: Final = "draft_chapter_id"
"""Send 载荷中目标章 id 的键名：任务态专用，不是主状态字段。"""


class DraftSendPayload(WritingAgentState):
    """Send 载荷类型：主状态切片 + 目标章 id（任务态专用键）。"""

    draft_chapter_id: str


class ChapterDrafterNode(Protocol):
    """节点函数类型：入参是 Send 载荷（任务态），返回主状态的部分更新。"""

    def __call__(self, state: DraftSendPayload) -> WritingAgentState: ...


def draft_send_payloads(state: WritingAgentState) -> list[DraftSendPayload]:
    """为全部未写章节构造 Send 载荷：目标章 id + 装配所需的状态切片。

    载荷只携带 chapter_drafter 配方实际消费的字段（大纲、该章素材、
    文种事实），不复制整个主状态，控制 checkpoint 中 pending Send 的体积；
    引文库按目标章过滤，与装配提取器的按章过滤语义一致。
    """
    drafted = {draft.chapter_id for draft in state.get("chapter_drafts", [])}
    outline = state.get("outline", [])
    return [
        DraftSendPayload(
            draft_chapter_id=chapter.id,
            outline=outline,
            citation_library=[
                material
                for material in state.get("citation_library", [])
                if material.chapter_id == chapter.id
            ],
            doc_type=state.get("doc_type", ""),
            doc_variant=state.get("doc_variant"),
        )
        for chapter in outline
        if chapter.id not in drafted
    ]


def make_chapter_drafter_node(
    rewriter_loop: Subagent,
    assembler_config: AssemblerConfig | None = None,
) -> ChapterDrafterNode:
    """构造 chapter_drafter 节点函数：注入 rewriter_loop 适配器。

    assembler_config 为 None 时在节点执行时读取环境变量装配配置。
    """

    def node(state: DraftSendPayload) -> WritingAgentState:
        config = assembler_config
        if config is None:
            config = load_assembler_config()
        chapter_id = state[DRAFT_CHAPTER_ID_KEY]
        chapter = chapter_by_id(state, chapter_id)
        context = assemble(
            state, "chapter_drafter", config=config, chapter_id=chapter_id
        )
        doc_type, doc_variant = carried_doc_facts(state)
        task = RewriteTask(
            mode="draft",
            doc_type=doc_type,
            doc_variant=doc_variant,
            chapter_spec=chapter_spec_payload(chapter),
            materials=materials_from_segment(context.text("chapter_materials")),
            prev_chapter_summary=context.text("summary_chain"),
        )
        result = asyncio.run(rewriter_loop.run(dict(task)))
        self_check = result["self_check"]
        draft = ChapterDraft(
            chapter_id=chapter_id,
            text=result["chapter_text"],
            summary=result["chapter_summary"],
            self_check=SelfCheck(
                citations_ok=self_check["citations_ok"],
                issues=self_check["issues"],
            ),
        )
        return WritingAgentState(
            chapter_drafts=[draft],
            status=WorkflowStatus.ARTICLE_WRITING,
            current_node_llm_config={"unit": "chapter_drafter"},
        )

    return node
