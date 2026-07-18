"""writing_orchestrator 主节点：首写 / 修订 / 终审回退三种模式的纯调度逻辑。

写作由 rewriter_loop 子智能体承担，补充佐证的增量检索由 search_agent 承担，
本节点不调 LLM，只负责按 State 分派模式并组装任务包：

- 修订模式（pending_directives 非空）：按目标章节分组指令，
  补充佐证章节先经 search_agent 增量检索入库（既有 id 去重），
  再逐章调 rewriter_loop（mode=revise）定向改写，其他章节草稿原样保留。
- 终审回退模式（citation_report 未通过且有不合格章节）：
  只对不合格章节按报告问题拼出纯改写指令做定向重写。
- 首写模式（其余情形）：按大纲顺序逐章串行 draft，
  后一章必须等前一章结果，用其 chapter_summary 承接摘要链。

修订与回退模式各章独立，prev_chapter_summary 取大纲中前一章的现有摘要即可，
无需重建串行摘要链。
"""

import asyncio
from typing import Protocol

from state import (
    ChapterDraft,
    ChapterSpec,
    CitationReport,
    Material,
    RevisionDirective,
    SelfCheck,
    WorkflowStatus,
    WritingAgentState,
)
from subagents import (
    ChapterSpecPayload,
    HypothesisPayload,
    MaterialPayload,
    PointPayload,
    RevisionDirectivePayload,
    RewriteTask,
    SearchTask,
    Subagent,
)


class WritingOrchestratorNode(Protocol):
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


def _chapter_spec_payload(chapter: ChapterSpec) -> ChapterSpecPayload:
    """章节骨架转任务包字典：论点列表 + 该章全部假说扁平列表。"""
    return ChapterSpecPayload(
        id=chapter.id,
        title=chapter.title,
        points=[PointPayload(id=point.id, text=point.text) for point in chapter.points],
        hypotheses=_flatten_hypotheses(chapter),
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


def _grouped_directives(
    directives: list[RevisionDirective], outline: list[ChapterSpec]
) -> dict[str, list[RevisionDirective]]:
    """按目标章节分组指令（同章多条指令合并一次改写），并做防御性校验。

    目标章节不在大纲中时抛 ValueError（上游 human_review_gate 已过滤，这里是防御）。
    """
    outline_ids = {chapter.id for chapter in outline}
    grouped: dict[str, list[RevisionDirective]] = {}
    for directive in directives:
        if directive.target_chapter_id not in outline_ids:
            raise ValueError(
                f"修订指令的目标章节 {directive.target_chapter_id} 不在大纲中"
            )
        grouped.setdefault(directive.target_chapter_id, []).append(directive)
    return grouped


def _report_repair_payload(
    report: CitationReport, chapter_id: str
) -> RevisionDirectivePayload:
    """由终审报告中该章各 issue 的 detail 拼出一句话中文修复指令。"""
    details = "；".join(
        issue.detail for issue in report.issues if issue.chapter_id == chapter_id
    )
    return RevisionDirectivePayload(
        type="rewrite_only",
        instruction=f"根据引文终审发现的问题修复本章：{details or '（无具体问题描述）'}",
    )


def make_writing_orchestrator_node(
    rewriter_loop: Subagent, search_agent: Subagent
) -> WritingOrchestratorNode:
    """构造 writing_orchestrator 节点函数：注入 rewriter_loop 与 search_agent 适配器。"""

    async def _write_all_chapters(state: WritingAgentState) -> list[ChapterDraft]:
        """首写模式章节严格串行：await 前一章结果后才组装并下发后一章任务包。"""
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

    async def _augment_evidence(
        state: WritingAgentState,
        grouped: dict[str, list[RevisionDirective]],
        library: list[Material],
    ) -> None:
        """对含补充佐证指令的章节做增量检索：新素材入库，既有 id 的条目跳过。"""
        genre = state.get("genre", "")
        known_ids = {material.id for material in library}
        for chapter in state.get("outline", []):
            chapter_directives = grouped.get(chapter.id, [])
            if not any(
                directive.type == "evidence_augmented"
                for directive in chapter_directives
            ):
                continue
            task = SearchTask(
                chapter_id=chapter.id,
                hypotheses=_flatten_hypotheses(chapter),
                genre=genre,
                existing_materials_digest=f"引文库已有素材 {len(library)} 条",
            )
            result = await search_agent.run(dict(task))
            # 素材必须逐条回链本章假说；回链不上的脏数据不入库。
            chapter_hypothesis_ids = {
                hypothesis["id"] for hypothesis in task["hypotheses"]
            }
            for material in result["materials"]:
                if material["id"] in known_ids:
                    continue
                if material["hypothesis_id"] not in chapter_hypothesis_ids:
                    continue
                known_ids.add(material["id"])
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

    async def _revise_targets(
        state: WritingAgentState,
        payloads_by_chapter: dict[str, list[RevisionDirectivePayload]],
        library: list[Material],
    ) -> tuple[list[ChapterDraft], list[str]]:
        """按大纲顺序对目标章节逐章调 rewriter_loop（mode=revise），其余章节原样保留。

        prev_chapter_summary 取大纲中前一章的现有摘要（首章为空串）；
        目标章节没有现存草稿时抛 ValueError（防御性校验）。
        """
        drafts_by_id = {
            draft.chapter_id: draft for draft in state.get("chapter_drafts", [])
        }
        new_drafts: list[ChapterDraft] = []
        revised_ids: list[str] = []
        prev_chapter_summary = ""
        for chapter in state.get("outline", []):
            draft = drafts_by_id.get(chapter.id)
            if chapter.id in payloads_by_chapter:
                if draft is None:
                    raise ValueError(f"目标章节 {chapter.id} 没有现存草稿可供修订")
                task = RewriteTask(
                    mode="revise",
                    chapter_spec=_chapter_spec_payload(chapter),
                    materials=_chapter_materials(library, chapter.id),
                    prev_chapter_summary=prev_chapter_summary,
                    revision_directives=payloads_by_chapter[chapter.id],
                    current_text=draft.text,
                )
                result = await rewriter_loop.run(dict(task))
                self_check = result["self_check"]
                new_drafts.append(
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
                revised_ids.append(chapter.id)
            elif draft is not None:
                new_drafts.append(draft)
            # prev_chapter_summary 用现有摘要，不受本轮改写影响。
            prev_chapter_summary = draft.summary if draft is not None else ""
        return new_drafts, revised_ids

    async def _run_directive_revision(
        state: WritingAgentState, directives: list[RevisionDirective]
    ) -> tuple[list[ChapterDraft], list[Material], list[str]]:
        """修订模式：增量检索后按指令定向改写。"""
        grouped = _grouped_directives(directives, state.get("outline", []))
        library = list(state.get("citation_library", []))
        await _augment_evidence(state, grouped, library)
        payloads_by_chapter = {
            chapter_id: [
                RevisionDirectivePayload(
                    type=directive.type, instruction=directive.instruction
                )
                for directive in chapter_directives
            ]
            for chapter_id, chapter_directives in grouped.items()
        }
        new_drafts, revised_ids = await _revise_targets(
            state, payloads_by_chapter, library
        )
        return new_drafts, library, revised_ids

    async def _run_report_fallback(
        state: WritingAgentState, report: CitationReport
    ) -> tuple[list[ChapterDraft], list[Material], list[str]]:
        """终审回退模式：只重写不合格章节，指令由报告问题拼出。"""
        library = list(state.get("citation_library", []))
        payloads_by_chapter = {
            chapter_id: [_report_repair_payload(report, chapter_id)]
            for chapter_id in report.failed_chapter_ids
        }
        new_drafts, revised_ids = await _revise_targets(
            state, payloads_by_chapter, library
        )
        return new_drafts, library, revised_ids

    def node(state: WritingAgentState) -> WritingAgentState:
        pending_directives = state.get("pending_directives", [])
        report = state.get("citation_report")
        if pending_directives:
            chapter_drafts, library, revised_ids = asyncio.run(
                _run_directive_revision(state, pending_directives)
            )
        elif report is not None and not report.passed and report.failed_chapter_ids:
            chapter_drafts, library, revised_ids = asyncio.run(
                _run_report_fallback(state, report)
            )
        else:
            return WritingAgentState(
                chapter_drafts=asyncio.run(_write_all_chapters(state)),
                revised_chapter_ids=[],
                status=WorkflowStatus.ARTICLE_WRITING,
                current_node_llm_config={"unit": "writing_orchestrator"},
            )
        return WritingAgentState(
            chapter_drafts=chapter_drafts,
            citation_library=library,
            revised_chapter_ids=revised_ids,
            pending_directives=[],
            status=WorkflowStatus.ARTICLE_WRITING,
            current_node_llm_config={"unit": "writing_orchestrator"},
        )

    return node
