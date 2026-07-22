"""writing_orchestrator 主节点：图内自环、每超步只处理一章的纯调度逻辑。

写作由 rewriter_loop 子智能体承担，补充佐证的增量检索由 search_agent 承担，
本节点不调 LLM。每次节点调用（一个超步）只处理一章：由 next_writing_step
从 State 纯数据推导下一步该做什么与目标章，处理完该章即把产物落 State 返回，
条件边（见 graph.py 的 route_after_writing_orchestrator）据同一判别函数决定
回到本节点写下一章还是前进终审。章级产物按超步自然落 checkpoint，
崩溃重跑只损失进行中的一章（ADR-0001 约束 1）。

三种模式的单章选取（游标全部从既有字段推导，不新增游标字段）：

- 修订模式（pending_directives 非空）：目标章 = 按大纲顺序第一个有待执行
  指令的章。该章若含补充佐证指令，先经 search_agent 增量检索入库
  （既有 id 去重、素材必须回链本章假说），再调 rewriter_loop（mode=revise）
  定向改写；执行完剔除该章全部指令。
- 终审回退模式（citation_report 未通过且 failed_chapter_ids 中还有章
  未在本轮修复）：目标章 = 按大纲顺序第一个「不合格且未修复」的章，
  按报告问题拼出纯改写指令做定向重写。
- 首写模式（其余情形）：目标章 = 大纲中第一个没有草稿的章，
  前章草稿均已在 State 中，用其摘要承接摘要链。
  正常路径下首写已由 chapter_drafter 并行扇出承担（见 graph.py），
  此分支仅作防御保留，不在主路径可达。

素材与前文摘要一律经 context_assembler 现场装配：任务包的
prev_chapter_summary 注入装配后的 summary_chain 段（该章之前的全部前章摘要链，
超阈值即压缩、未超时原样拼接，首章为空串），素材取装配后的 chapter_materials 段。
"""

import asyncio
import json
from typing import Literal, Protocol

from assembly.assembler_config import AssemblerConfig, load_assembler_config
from assembly.context_assembler import assemble, assemble_with
from domain.doc_types import carried_doc_facts
from domain.state import (
    ChapterDraft,
    ChapterSpec,
    CitationReport,
    Material,
    RevisionDirective,
    SelfCheck,
    WorkflowStatus,
    WritingAgentState,
)
from agents.chapter_reviewer import make_stub_chapter_reviewer
from agents.contracts import (
    ChapterSpecPayload,
    HypothesisPayload,
    MaterialPayload,
    PointPayload,
    RevisionDirectivePayload,
    RewriteTask,
    SearchTask,
    Subagent,
    material_from_payload,
)
from nodes.chapter_write_loop import resolve_max_rewrites, run_chapter_write_loop

# 单超步的判别结果：模式（修订 / 终审回退 / 首写）与目标章 id。
WritingStep = tuple[Literal["revise", "fallback", "draft"], str]


class WritingOrchestratorNode(Protocol):
    """节点函数类型：入参与返回均为图状态（state 具名，满足 LangGraph 节点协议）。"""

    def __call__(self, state: WritingAgentState) -> WritingAgentState: ...


def next_writing_step(state: WritingAgentState) -> WritingStep | None:
    """从 State 纯数据推导下一个超步的模式与目标章；全部完成时返回 None。

    节点选章与图路由共用此单一事实源，保证两处判定严格一致、不死循环不漏章。
    修订指令的目标章不在大纲中时抛 ValueError（上游已过滤，这里是防御）。
    """
    outline = state.get("outline", [])
    pending_directives = state.get("pending_directives", [])
    if pending_directives:
        grouped = _grouped_directives(pending_directives, outline)
        for chapter in outline:
            if chapter.id in grouped:
                return ("revise", chapter.id)
    report = state.get("citation_report")
    # 终审回退只在重试预算内的失败报告上触发：citation_validator 写失败报告时
    # 必然把 citation_retry_count 递增到至少 1，而 human_review_gate 开新一轮
    # 修订时会把它重置为 0——由此保证超限后残留的旧失败报告不会在修订轮
    # 结束后触发计划外的回退重写（绕过重试上限判定）。
    if (
        report is not None
        and not report.passed
        and report.failed_chapter_ids
        and state.get("citation_retry_count", 0) >= 1
    ):
        failed = set(report.failed_chapter_ids)
        revised = set(state.get("revised_chapter_ids", []))
        for chapter in outline:
            if chapter.id in failed and chapter.id not in revised:
                return ("fallback", chapter.id)
    drafted = {draft.chapter_id for draft in state.get("chapter_drafts", [])}
    for chapter in outline:
        if chapter.id not in drafted:
            return ("draft", chapter.id)
    return None


def flatten_hypotheses(chapter: ChapterSpec) -> list[HypothesisPayload]:
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


def chapter_points(chapter: ChapterSpec) -> list[PointPayload]:
    """把章节论点按顺序转为任务包条目：检索任务包与写作任务包共用。"""
    return [PointPayload(id=point.id, text=point.text) for point in chapter.points]


def chapter_spec_payload(chapter: ChapterSpec) -> ChapterSpecPayload:
    """章节骨架转任务包字典：论点列表 + 该章全部假说扁平列表。"""
    return ChapterSpecPayload(
        id=chapter.id,
        title=chapter.title,
        chapter_type=chapter.chapter_type,
        points=chapter_points(chapter),
        hypotheses=flatten_hypotheses(chapter),
    )


def materials_from_segment(chapter_materials_json: str) -> list[MaterialPayload]:
    """把 chapter_materials 段（该章 verdict=pass 素材的 JSON）转为任务包条目。

    段文本由 context_assembler.extract_chapter_materials 装配（已按章过滤并只留
    通过校验的素材），此处只取任务包所需字段，丢弃 chapter_id 等无关字段。
    段缺失（空串）时视为该章无素材。
    """
    if not chapter_materials_json:
        return []
    return [
        MaterialPayload(
            id=material["id"],
            hypothesis_id=material["hypothesis_id"],
            source=material["source"],
            url=material["url"],
            source_kind=material["source_kind"],
            excerpt=material["excerpt"],
            relevance_score=material["relevance_score"],
            verdict=material["verdict"],
        )
        for material in json.loads(chapter_materials_json)
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


def chapter_by_id(state: WritingAgentState, chapter_id: str) -> ChapterSpec:
    """按 id 取大纲章节；判别函数已保证目标章在大纲中。"""
    for chapter in state.get("outline", []):
        if chapter.id == chapter_id:
            return chapter
    raise ValueError(f"目标章节 {chapter_id} 不在大纲中")


def make_writing_orchestrator_node(
    rewriter_loop: Subagent,
    search_agent: Subagent,
    assembler_config: AssemblerConfig | None = None,
    chapter_reviewer: Subagent | None = None,
    *,
    max_rewrites: int | None = None,
) -> WritingOrchestratorNode:
    """构造 writing_orchestrator 节点函数：注入 rewriter_loop 与 search_agent 适配器。

    assembler_config 为 None 时在节点执行时读取环境变量装配配置；
    max_rewrites 为 None 时读环境变量 CHAPTER_MAX_REWRITES（缺省 1）。
    chapter_reviewer 为章级评审子智能体（ADR-0006）：防御性首写分支（_draft_chapter）
    经其跑写→评→重写循环；未注入时回落打桩评审（正常路径首写由 chapter_drafter
    并行扇出承担，本分支不在主路径可达）。修订/终审回退分支的评审消费留 T3b。
    """
    effective_chapter_reviewer = chapter_reviewer or make_stub_chapter_reviewer()
    resolved_max_rewrites = resolve_max_rewrites(max_rewrites)

    async def _draft_chapter(
        state: WritingAgentState, chapter: ChapterSpec, config: AssemblerConfig
    ) -> ChapterDraft:
        """首写单章：经写→评→重写循环产出成稿（ADR-0006 T3，见 chapter_write_loop）。

        前章草稿已逐超步落在 State 的 chapter_drafts 中（本章尚无草稿），
        summary_chain 段由此给出该章之前的全部前章摘要链（超阈值即压缩，
        未超时为原样拼接；首章为空串），循环由此得到完整前文链。
        """
        context = assemble(
            state,
            "writing_orchestrator",
            config=config,
            chapter_id=chapter.id,
        )
        doc_type, doc_variant = carried_doc_facts(state)
        return await run_chapter_write_loop(
            rewriter_loop=rewriter_loop,
            chapter_reviewer=effective_chapter_reviewer,
            max_rewrites=resolved_max_rewrites,
            doc_type=doc_type,
            doc_variant=doc_variant,
            chapter_spec=chapter_spec_payload(chapter),
            materials=materials_from_segment(context.text("chapter_materials")),
            prev_chapter_summary=context.text("summary_chain"),
        )

    async def _augment_evidence(
        state: WritingAgentState,
        chapter: ChapterSpec,
        library: list[Material],
        config: AssemblerConfig,
    ) -> None:
        """对目标章做增量检索：新素材入库，既有 id 的条目跳过。

        任务包的 existing_materials_digest 经 citation_digest 段装配得到：
        以当前引文库现场覆盖 state 再装配，反映本轮增量前的引文库状态。
        素材必须逐条回链本章假说；回链不上的脏数据不入库。
        """
        known_ids = {material.id for material in library}
        context = assemble_with(
            state,
            {"citation_library": list(library)},
            "search_agent",
            config=config,
        )
        task = SearchTask(
            chapter_id=chapter.id,
            points=chapter_points(chapter),
            hypotheses=flatten_hypotheses(chapter),
            genre=state.get("genre", ""),
            existing_materials_digest=context.text("citation_digest"),
        )
        result = await search_agent.run(dict(task))
        chapter_hypothesis_ids = {hypothesis["id"] for hypothesis in task["hypotheses"]}
        for material in result["materials"]:
            if material["id"] in known_ids:
                continue
            if material["hypothesis_id"] not in chapter_hypothesis_ids:
                continue
            known_ids.add(material["id"])
            library.append(material_from_payload(material, chapter.id))

    async def _revise_chapter(
        state: WritingAgentState,
        chapter: ChapterSpec,
        payloads: list[RevisionDirectivePayload],
        library: list[Material],
        config: AssemblerConfig,
    ) -> ChapterDraft:
        """对目标章调 rewriter_loop（mode=revise）定向改写。

        素材与前文摘要链一律经装配入口取得：素材以增量检索后的引文库现场覆盖
        state 再按 chapter_id 装配；prev_chapter_summary 注入 summary_chain 段
        （该章之前的前章摘要链，首章为空串），不受本轮改写影响。
        目标章节没有现存草稿时抛 ValueError（防御性校验）。
        """
        draft = next(
            (
                draft
                for draft in state.get("chapter_drafts", [])
                if draft.chapter_id == chapter.id
            ),
            None,
        )
        if draft is None:
            raise ValueError(f"目标章节 {chapter.id} 没有现存草稿可供修订")
        context = assemble_with(
            state,
            {"citation_library": list(library)},
            "writing_orchestrator",
            config=config,
            chapter_id=chapter.id,
        )
        doc_type, doc_variant = carried_doc_facts(state)
        task = RewriteTask(
            mode="revise",
            doc_type=doc_type,
            doc_variant=doc_variant,
            chapter_spec=chapter_spec_payload(chapter),
            materials=materials_from_segment(context.text("chapter_materials")),
            prev_chapter_summary=context.text("summary_chain"),
            revision_directives=payloads,
            current_text=draft.text,
        )
        result = await rewriter_loop.run(dict(task))
        self_check = result["self_check"]
        return ChapterDraft(
            chapter_id=chapter.id,
            text=result["chapter_text"],
            summary=result["chapter_summary"],
            self_check=SelfCheck(
                citations_ok=self_check["citations_ok"],
                issues=self_check["issues"],
            ),
        )

    async def _run_directive_step(
        state: WritingAgentState, chapter_id: str, config: AssemblerConfig
    ) -> tuple[ChapterDraft, list[Material]]:
        """修订模式单超步：目标章若含补充佐证指令先增量检索，再按指令定向改写。"""
        grouped = _grouped_directives(
            state.get("pending_directives", []), state.get("outline", [])
        )
        chapter = chapter_by_id(state, chapter_id)
        chapter_directives = grouped[chapter_id]
        library = list(state.get("citation_library", []))
        if any(
            directive.type == "evidence_augmented" for directive in chapter_directives
        ):
            await _augment_evidence(state, chapter, library, config)
        payloads = [
            RevisionDirectivePayload(
                type=directive.type, instruction=directive.instruction
            )
            for directive in chapter_directives
        ]
        new_draft = await _revise_chapter(state, chapter, payloads, library, config)
        return new_draft, library

    def _replace_draft(
        state: WritingAgentState, new_draft: ChapterDraft
    ) -> list[ChapterDraft]:
        """读旧草稿列表，替换目标章草稿，其余章草稿对象原样保留。

        chapter_drafts 带按 chapter_id 合并的 reducer：回写完整列表时逐项
        同 id 替换，与旧的整值覆盖语义等价。
        """
        return [
            new_draft if draft.chapter_id == new_draft.chapter_id else draft
            for draft in state.get("chapter_drafts", [])
        ]

    def _revised_step_update(
        state: WritingAgentState, new_draft: ChapterDraft, chapter_id: str
    ) -> WritingAgentState:
        """修订/回退单超步的公共收尾：替换该章草稿、追加本轮已修改章节。"""
        return WritingAgentState(
            chapter_drafts=_replace_draft(state, new_draft),
            revised_chapter_ids=[*state.get("revised_chapter_ids", []), chapter_id],
            status=WorkflowStatus.ARTICLE_WRITING,
            current_node_llm_config={"unit": "writing_orchestrator"},
        )

    def node(state: WritingAgentState) -> WritingAgentState:
        config = assembler_config
        if config is None:
            config = load_assembler_config()
        llm_config = {"unit": "writing_orchestrator"}
        step = next_writing_step(state)
        if step is None:
            # 防御兜底：路由不会把「无事可做」的 state 送进来；万一发生，
            # 不调子智能体，只推进状态机，路由随后前进终审。
            return WritingAgentState(
                status=WorkflowStatus.ARTICLE_WRITING,
                current_node_llm_config=llm_config,
            )
        mode, chapter_id = step
        if mode == "revise":
            new_draft, library = asyncio.run(
                _run_directive_step(state, chapter_id, config)
            )
            update = _revised_step_update(state, new_draft, chapter_id)
            update["citation_library"] = library
            update["pending_directives"] = [
                directive
                for directive in state.get("pending_directives", [])
                if directive.target_chapter_id != chapter_id
            ]
            return update
        if mode == "fallback":
            report = state.get("citation_report")
            assert report is not None  # 判别函数已保证。
            chapter = chapter_by_id(state, chapter_id)
            library = list(state.get("citation_library", []))
            new_draft = asyncio.run(
                _revise_chapter(
                    state,
                    chapter,
                    [_report_repair_payload(report, chapter_id)],
                    library,
                    config,
                )
            )
            return _revised_step_update(state, new_draft, chapter_id)
        chapter = chapter_by_id(state, chapter_id)
        draft = asyncio.run(_draft_chapter(state, chapter, config))
        return WritingAgentState(
            chapter_drafts=[*state.get("chapter_drafts", []), draft],
            revised_chapter_ids=[],
            status=WorkflowStatus.ARTICLE_WRITING,
            current_node_llm_config=llm_config,
        )

    return node
