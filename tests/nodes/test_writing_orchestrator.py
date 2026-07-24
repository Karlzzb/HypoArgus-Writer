"""writing_orchestrator 节点单元测试：用假子智能体适配器验证自环形态下的三种模式。

节点每个超步只处理一章：测试辅助函数模拟图循环——调 node(state) → 把返回
更新 merge 进 state（整值覆盖语义）→ 用共享判别函数 next_writing_step 判断
是否继续，直到前进为止。覆盖点：首写模式逐超步各一次调用且顺序与大纲一致、
章级增量落 state；摘要链承接（prev_chapter_summary）；素材过滤（只有该章
pass 与 inconclusive 素材进任务包）；改写结果与单章自检入 State；状态机推进到
ARTICLE_WRITING；修订模式按指令定向改写与增量检索入库去重、逐章消费指令
队列；终审回退模式只重写不合格章节、revised_chapter_ids 逐超步累积；
判别函数各模式与全部完成情形的返回值；非法目标章节防御性抛错。
"""

from typing import Any, cast

import pytest

from domain.state import (
    ArgumentPoint,
    ChapterDraft,
    ChapterSpec,
    CitationIssue,
    CitationReport,
    Hypothesis,
    Material,
    RevisionDirective,
    SourceKind,
    WorkflowStatus,
    WritingAgentState,
)
from agents.contracts import MaterialPayload, SubagentAdapter
from nodes.writing_orchestrator import (
    REVISION_CHAPTER_ID_KEY,
    WritingOrchestratorNode,
    make_writing_orchestrator_node,
    next_writing_step,
    revision_send_payloads,
)


class 记录式假改写适配器(SubagentAdapter):
    """记录任务包顺序并返回确定性结果的假 rewriter_loop。

    draft 模式各章摘要不同以便验证摘要链，ch2 返回 citations_ok=False 带问题清单；
    revise 模式返回带「修订后」标记的正文与摘要，便于断言草稿确实被改写。
    """

    def __init__(self) -> None:
        super().__init__("rewriter_loop", self._run)
        self.tasks: list[dict[str, Any]] = []

    async def _run(self, task: dict[str, Any]) -> dict[str, Any]:
        self.tasks.append(task)
        chapter_id = task["chapter_spec"]["id"]
        if task["mode"] == "revise":
            return {
                "chapter_text": f"{chapter_id} 修订后正文",
                "chapter_summary": f"{chapter_id} 修订后摘要",
                "self_check": {"citations_ok": True, "issues": []},
            }
        if chapter_id == "ch2":
            self_check = {"citations_ok": False, "issues": ["角标 m-x 不在素材列表中"]}
        else:
            self_check = {"citations_ok": True, "issues": []}
        return {
            "chapter_text": f"{chapter_id} 的正文 [m-{chapter_id}]",
            "chapter_summary": f"{chapter_id} 的摘要",
            "self_check": self_check,
        }


class 回带自检假评审适配器(SubagentAdapter):
    """按章节回带与 rewriter 一致自检、且判 passed=True 的假 chapter_reviewer。

    首写循环在评审通过时以评审的 self_check 作为该章成稿自检。为验证 rewriter
    的逐章自检（ch2 失败带问题清单）确实入 State，评审须原样回带 rewriter 对
    该章产出的自检，并判通过以走干净退出路径（单次 rewriter 草稿调用）。
    """

    def __init__(self) -> None:
        super().__init__("chapter_reviewer", self._run)
        self.tasks: list[dict[str, Any]] = []

    async def _run(self, task: dict[str, Any]) -> dict[str, Any]:
        self.tasks.append(task)
        chapter_id = task["chapter_spec"]["id"]
        if chapter_id == "ch2":
            self_check = {"citations_ok": False, "issues": ["角标 m-x 不在素材列表中"]}
        else:
            self_check = {"citations_ok": True, "issues": []}
        return {
            "revision_note": {
                "user_directives": "",
                "rule_violations": [],
                "conflict_hints": [],
                "passed": True,
            },
            "self_check": self_check,
        }


class 记录式假检索适配器(SubagentAdapter):
    """记录任务包并按章节返回预设素材的假 search_agent。"""

    def __init__(
        self, materials_by_chapter: dict[str, list[MaterialPayload]] | None = None
    ) -> None:
        super().__init__("search_agent", self._run)
        self.tasks: list[dict[str, Any]] = []
        self._materials_by_chapter = materials_by_chapter or {}

    async def _run(self, task: dict[str, Any]) -> dict[str, Any]:
        self.tasks.append(task)
        return {"materials": self._materials_by_chapter.get(task["chapter_id"], [])}


def _hypothesis(hyp_id: str) -> Hypothesis:
    return Hypothesis(
        id=hyp_id,
        text=f"假说 {hyp_id}",
        refute_condition=f"出现反例即证伪 {hyp_id}",
        angle="假设",
    )


def _material(
    mat_id: str,
    hypothesis_id: str,
    chapter_id: str,
    verdict: str,
    *,
    source_kind: SourceKind = "web",
) -> Material:
    source_ref = (
        {"url": f"https://example.com/{mat_id}"}
        if source_kind == "web"
        else {"knowledge_id": "kb", "file_id": mat_id, "chunk_id": "c1"}
    )
    return Material(
        id=mat_id,
        hypothesis_id=hypothesis_id,
        chapter_id=chapter_id,
        source=f"来源 {mat_id}",
        url=source_ref.get("url"),
        source_kind=source_kind,
        source_ref=source_ref,
        excerpt=f"摘录 {mat_id}",
        relevance_score=0.8,
        verdict=verdict,  # type: ignore[arg-type]
    )


def _make_state() -> WritingAgentState:
    """三章大纲：ch1 两论点各一假说，ch2 一论点两假说，ch3 无论点。

    引文库同时含 pass、inconclusive 与 fail 素材，用于验证素材过滤。
    """
    outline = [
        ChapterSpec(
            id="ch1",
            title="第一章",
            chapter_type="维度章",
            points=[
                ArgumentPoint(
                    id="ch1-p1", text="论点一", hypotheses=[_hypothesis("ch1-p1-h1")]
                ),
                ArgumentPoint(
                    id="ch1-p2", text="论点二", hypotheses=[_hypothesis("ch1-p2-h1")]
                ),
            ],
        ),
        ChapterSpec(
            id="ch2",
            title="第二章",
            points=[
                ArgumentPoint(
                    id="ch2-p1",
                    text="论点三",
                    hypotheses=[_hypothesis("ch2-p1-h1"), _hypothesis("ch2-p1-h2")],
                ),
            ],
        ),
        ChapterSpec(id="ch3", title="第三章", points=[]),
    ]
    citation_library = [
        _material("m-1", "ch1-p1-h1", "ch1", "pass"),
        _material("m-2", "ch1-p2-h1", "ch1", "fail", source_kind="knowledge_base"),
        _material("m-3", "ch2-p1-h1", "ch2", "pass"),
        _material("m-4", "ch2-p1-h2", "ch2", "inconclusive"),
    ]
    return WritingAgentState(
        outline=outline,
        citation_library=citation_library,
        status=WorkflowStatus.REFERENCE_FETCHING,
    )


def _drive_to_completion(
    node: WritingOrchestratorNode, state: WritingAgentState
) -> tuple[WritingAgentState, list[WritingAgentState], int]:
    """模拟图自环：逐超步调 node 并 merge 更新，直到判别函数判定前进为止。

    merge 用 dict.update（整值覆盖语义，与图 state 的缺省 reducer 一致）。
    返回（最终 state, 各超步更新列表, 超步数）；防御性设上限防死循环。
    """
    current = cast(WritingAgentState, dict(state))
    updates: list[WritingAgentState] = []
    steps = 0
    while next_writing_step(current) is not None:
        assert steps < 20, "自环超步数异常，疑似死循环"
        update = node(current)
        updates.append(update)
        cast(dict, current).update(update)
        steps += 1
    return current, updates, steps


def _run_node() -> tuple[记录式假改写适配器, WritingAgentState]:
    adapter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(
        adapter, 记录式假检索适配器(), chapter_reviewer=回带自检假评审适配器()
    )
    final, _, _ = _drive_to_completion(node, _make_state())
    return adapter, final


def test_逐超步各一次调用_顺序与大纲一致():
    adapter, _ = _run_node()
    assert [task["chapter_spec"]["id"] for task in adapter.tasks] == [
        "ch1",
        "ch2",
        "ch3",
    ]
    assert all(task["mode"] == "draft" for task in adapter.tasks)


def test_首写三章恰好三超步_每超步草稿只多一章():
    """自环形态验收：每超步恰好一次 rewriter 调用、返回更新里草稿只比之前多一章。

    这是章级增量落 state 的直接证据：checkpointer 按超步落盘时，
    崩溃重跑只损失进行中的一章。
    """
    adapter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(adapter, 记录式假检索适配器())
    _, updates, steps = _drive_to_completion(node, _make_state())
    assert steps == 3
    assert len(adapter.tasks) == 3
    # 每超步返回的更新里 chapter_drafts 恰好比上一超步多一章。
    assert [
        [draft.chapter_id for draft in update["chapter_drafts"]] for update in updates
    ] == [["ch1"], ["ch1", "ch2"], ["ch1", "ch2", "ch3"]]


def test_任务包章节骨架含论点与全章扁平假说():
    adapter, _ = _run_node()
    spec_ch1 = adapter.tasks[0]["chapter_spec"]
    assert spec_ch1["title"] == "第一章"
    # 章型骨架事实原样透传进任务包（ADR-0005）；大纲未携带时为 None。
    assert spec_ch1["chapter_type"] == "维度章"
    assert adapter.tasks[2]["chapter_spec"]["chapter_type"] is None
    assert [point["id"] for point in spec_ch1["points"]] == ["ch1-p1", "ch1-p2"]
    assert [point["text"] for point in spec_ch1["points"]] == ["论点一", "论点二"]
    assert [hyp["id"] for hyp in spec_ch1["hypotheses"]] == ["ch1-p1-h1", "ch1-p2-h1"]
    spec_ch2 = adapter.tasks[1]["chapter_spec"]
    assert [hyp["id"] for hyp in spec_ch2["hypotheses"]] == ["ch2-p1-h1", "ch2-p1-h2"]
    assert all(
        hyp["refute_condition"] for hyp in spec_ch1["hypotheses"] + spec_ch2["hypotheses"]
    )


def test_摘要链承接_首章为空_后章收到完整前章摘要链():
    adapter, _ = _run_node()
    # prev_chapter_summary 注入 summary_chain 段：首章为空，
    # 后章收到该章之前的全部前章摘要链（带章节标题前缀），而非仅紧邻一章。
    assert adapter.tasks[0]["prev_chapter_summary"] == ""
    assert adapter.tasks[1]["prev_chapter_summary"] == "【第一章】ch1 的摘要"
    assert (
        adapter.tasks[2]["prev_chapter_summary"]
        == "【第一章】ch1 的摘要\n【第二章】ch2 的摘要"
    )


def test_素材过滤_只有本章pass与弱佐证素材进任务包():
    adapter, _ = _run_node()
    # ch1：m-2 verdict=fail 被过滤，他章素材不进。
    assert [material["id"] for material in adapter.tasks[0]["materials"]] == ["m-1"]
    material = adapter.tasks[0]["materials"][0]
    assert material["hypothesis_id"] == "ch1-p1-h1"
    assert material["source"] == "来源 m-1"
    assert material["source_ref"] == {"url": "https://example.com/m-1"}
    assert material["excerpt"] == "摘录 m-1"
    assert material["relevance_score"] == 0.8
    assert material["verdict"] == "pass"
    # ch2：pass 与 inconclusive 都进，verdict 保留；ch3：无素材。
    assert [material["id"] for material in adapter.tasks[1]["materials"]] == [
        "m-3",
        "m-4",
    ]
    assert [material["verdict"] for material in adapter.tasks[1]["materials"]] == [
        "pass",
        "inconclusive",
    ]
    assert adapter.tasks[2]["materials"] == []


def test_改写结果与自检入State():
    _, final = _run_node()
    drafts = final["chapter_drafts"]
    assert [draft.chapter_id for draft in drafts] == ["ch1", "ch2", "ch3"]
    assert drafts[0].text == "ch1 的正文 [m-ch1]"
    assert drafts[0].summary == "ch1 的摘要"
    assert drafts[0].self_check.citations_ok is True
    assert drafts[0].self_check.issues == []
    # ch2 的自检失败结果（含问题清单）必须原样入 State。
    assert drafts[1].self_check.citations_ok is False
    assert drafts[1].self_check.issues == ["角标 m-x 不在素材列表中"]


def test_状态机推进到ARTICLE_WRITING_且记录运行单元():
    _, final = _run_node()
    assert final["status"] == WorkflowStatus.ARTICLE_WRITING
    assert final["current_node_llm_config"] == {"unit": "writing_orchestrator"}


def test_首写模式回归_revised_chapter_ids为空():
    """无待执行指令、无失败终审报告时走首写模式，revised_chapter_ids 为空表示全量核查。"""
    _, final = _run_node()
    assert final["revised_chapter_ids"] == []


def _existing_drafts() -> list[ChapterDraft]:
    """三章现有草稿：文本与摘要带「旧」标记，便于断言是否被改写。"""
    return [
        ChapterDraft(chapter_id=cid, text=f"{cid} 旧正文", summary=f"{cid} 旧摘要")
        for cid in ("ch1", "ch2", "ch3")
    ]


def _make_revision_state(directives: list[RevisionDirective]) -> WritingAgentState:
    """在基础三章状态上叠加现有草稿与待执行修订指令。"""
    state = _make_state()
    state["chapter_drafts"] = _existing_drafts()
    state["pending_directives"] = directives
    state["revised_chapter_ids"] = []
    state["status"] = WorkflowStatus.AWAIT_USER_REVIEW
    return state


def _new_material(mat_id: str, hypothesis_id: str) -> MaterialPayload:
    return MaterialPayload(
        id=mat_id,
        hypothesis_id=hypothesis_id,
        source=f"增量来源 {mat_id}",
        url=f"https://example.com/{mat_id}",
        source_kind="web",
        excerpt=f"增量摘录 {mat_id}",
        relevance_score=0.9,
        verdict="pass",
    )


def test_修订模式_混合分支逐超步执行():
    """ch1 纯改写 + ch2 补充佐证：两目标章各占一个超步，其他章节草稿原样保留。"""
    rewriter = 记录式假改写适配器()
    search = 记录式假检索适配器(
        {"ch2": [_new_material("m-new", "ch2-p1-h1")]}
    )
    node = make_writing_orchestrator_node(rewriter, search)
    state = _make_revision_state(
        [
            RevisionDirective(
                target_chapter_id="ch1", type="rewrite_only", instruction="收紧第一章语气"
            ),
            RevisionDirective(
                target_chapter_id="ch2",
                type="evidence_augmented",
                instruction="补充第二章数据佐证",
            ),
        ]
    )
    original_drafts = state["chapter_drafts"]
    final, updates, steps = _drive_to_completion(node, state)
    assert steps == 2

    # 逐超步消费指令队列：第一超步后只剩另一章（ch2）的指令。
    assert [
        directive.target_chapter_id for directive in updates[0]["pending_directives"]
    ] == ["ch2"]
    assert updates[0]["revised_chapter_ids"] == ["ch1"]

    # search_agent 只为 ch2 被调用一次，任务包同 reference_orchestrator 的 SearchTask。
    assert len(search.tasks) == 1
    search_task = search.tasks[0]
    assert search_task["chapter_id"] == "ch2"
    assert [hyp["id"] for hyp in search_task["hypotheses"]] == ["ch2-p1-h1", "ch2-p1-h2"]
    # digest 由 citation_digest 段装配：4 条素材，ch1 通过 1 未通过 1、ch2 通过 1 弱佐证 1。
    assert search_task["existing_materials_digest"] == (
        "引文库共 4 条素材。\n"
        "章节 ch1：通过 1 条，弱佐证 0 条，未通过 1 条\n"
        "章节 ch2：通过 1 条，弱佐证 1 条，未通过 0 条"
    )

    # rewriter_loop 两次均 mode=revise，任务包带评审装配的分区式修订说明与 current_text：
    # 未显式注入评审时回落打桩评审，用户意见原文逐字进用户指令区。
    assert [task["chapter_spec"]["id"] for task in rewriter.tasks] == ["ch1", "ch2"]
    assert all(task["mode"] == "revise" for task in rewriter.tasks)
    assert rewriter.tasks[0]["revision_note"]["user_directives"] == "收紧第一章语气"
    assert rewriter.tasks[0]["revision_note"]["rule_violations"] == []
    assert rewriter.tasks[0]["current_text"] == "ch1 旧正文"
    assert rewriter.tasks[0]["prev_chapter_summary"] == ""
    assert rewriter.tasks[1]["revision_note"]["user_directives"] == "补充第二章数据佐证"
    assert rewriter.tasks[1]["current_text"] == "ch2 旧正文"
    # 章级落 state 的自然结果：ch1 的改写在前一超步已入 state，
    # ch2 超步装配的摘要链承接的是 ch1 修订后的最新摘要。
    assert rewriter.tasks[1]["prev_chapter_summary"] == "【第一章】ch1 修订后摘要"
    # ch2 任务包素材含新增素材 m-new。
    assert [material["id"] for material in rewriter.tasks[1]["materials"]] == [
        "m-3",
        "m-4",
        "m-new",
    ]

    # 仅 ch1、ch2 草稿变化，ch3 草稿对象原样保留。
    drafts = final["chapter_drafts"]
    assert [draft.chapter_id for draft in drafts] == ["ch1", "ch2", "ch3"]
    assert drafts[0].text == "ch1 修订后正文"
    assert drafts[1].text == "ch2 修订后正文"
    assert drafts[2] is original_drafts[2]
    assert final["revised_chapter_ids"] == ["ch1", "ch2"]
    assert final["pending_directives"] == []
    assert final["status"] == WorkflowStatus.ARTICLE_WRITING
    assert final["current_node_llm_config"] == {"unit": "writing_orchestrator"}


class 记录式装配假评审适配器(SubagentAdapter):
    """记录评审任务包并装配「用户指令 + 一条 warn 违规」修订说明的假 chapter_reviewer。

    用于验证 ADR-0007 评审前置：修订模式先经章级评审装配分区式修订说明，
    评审产物必须原封不动地成为 rewriter revise 任务包的 revision_note。
    """

    def __init__(self) -> None:
        super().__init__("chapter_reviewer", self._run)
        self.tasks: list[dict[str, Any]] = []

    async def _run(self, task: dict[str, Any]) -> dict[str, Any]:
        self.tasks.append(task)
        return {
            "revision_note": {
                "user_directives": task.get("user_feedback", ""),
                "rule_violations": [
                    {
                        "rule": "oral_blacklist",
                        "location_excerpt": "我们认为",
                        "guidance": "改为书面语表述",
                        "severity": "warn",
                    }
                ],
                "conflict_hints": [],
                "passed": True,
            },
            "self_check": {"citations_ok": True, "issues": []},
        }


def test_修订模式_评审前置装配修订说明_且恰一次改写无二次重写():
    """issue #47 验收：revise 路径先调 chapter_reviewer（mode=revise）装配修订说明。

    同章多条指令的 instruction 合并为用户意见原文逐字交给评审；评审产物原封
    不动驱动 rewriter 恰一次改写——rewriter revise 调用计数 == 目标章数，
    改写后只跑纯函数 re-lint，绝无二次评审、二次重写叠加（ADR-0007）。
    """
    rewriter = 记录式假改写适配器()
    reviewer = 记录式装配假评审适配器()
    node = make_writing_orchestrator_node(
        rewriter, 记录式假检索适配器(), chapter_reviewer=reviewer
    )
    state = _make_revision_state(
        [
            RevisionDirective(
                target_chapter_id="ch1", type="rewrite_only", instruction="收紧语气"
            ),
            RevisionDirective(
                target_chapter_id="ch1", type="rewrite_only", instruction="压缩篇幅"
            ),
            RevisionDirective(
                target_chapter_id="ch2", type="rewrite_only", instruction="精简第二章"
            ),
        ]
    )
    final, _, steps = _drive_to_completion(node, state)
    assert steps == 2

    # 评审前置：每个目标章恰一次 mode=revise 评审，现存草稿与合并指令入任务包。
    assert [task["mode"] for task in reviewer.tasks] == ["revise", "revise"]
    assert [task["chapter_spec"]["id"] for task in reviewer.tasks] == ["ch1", "ch2"]
    assert reviewer.tasks[0]["chapter_text"] == "ch1 旧正文"
    # 同章多条指令按顺序合并为用户意见原文（逐字，不改写）。
    assert reviewer.tasks[0]["user_feedback"] == "收紧语气\n压缩篇幅"
    assert reviewer.tasks[1]["user_feedback"] == "精简第二章"

    # 恰一次改写、无二次重写叠加：rewriter revise 调用计数 == 目标章数（2）。
    assert len(rewriter.tasks) == 2
    assert all(task["mode"] == "revise" for task in rewriter.tasks)
    # 评审装配的分区式修订说明原封不动进 rewriter 任务包。
    assert rewriter.tasks[0]["revision_note"]["user_directives"] == "收紧语气\n压缩篇幅"
    assert rewriter.tasks[0]["revision_note"]["rule_violations"] == [
        {
            "rule": "oral_blacklist",
            "location_excerpt": "我们认为",
            "guidance": "改为书面语表述",
            "severity": "warn",
        }
    ]
    assert final["chapter_drafts"][0].text == "ch1 修订后正文"
    assert final["chapter_drafts"][1].text == "ch2 修订后正文"


def test_修订模式_只作用于指定章节():
    """三章大纲只改 ch2，一个超步完成，ch1 与 ch3 草稿对象原样保留。"""
    rewriter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(rewriter, 记录式假检索适配器())
    state = _make_revision_state(
        [
            RevisionDirective(
                target_chapter_id="ch2", type="rewrite_only", instruction="精简第二章"
            )
        ]
    )
    original_drafts = state["chapter_drafts"]
    final, _, steps = _drive_to_completion(node, state)
    assert steps == 1
    assert [task["chapter_spec"]["id"] for task in rewriter.tasks] == ["ch2"]
    drafts = final["chapter_drafts"]
    assert drafts[0] is original_drafts[0]
    assert drafts[1].text == "ch2 修订后正文"
    assert drafts[2] is original_drafts[2]
    assert final["revised_chapter_ids"] == ["ch2"]


def test_增量素材入库与去重():
    """新素材入引文库；返回与既有 id 重复的素材跳过不重复入库。"""
    search = 记录式假检索适配器(
        {
            "ch2": [
                _new_material("m-3", "ch2-p1-h1"),
                _new_material("m-new", "ch2-p1-h2"),
            ]
        }
    )
    node = make_writing_orchestrator_node(记录式假改写适配器(), search)
    state = _make_revision_state(
        [
            RevisionDirective(
                target_chapter_id="ch2",
                type="evidence_augmented",
                instruction="补充第二章佐证",
            )
        ]
    )
    final, _, _ = _drive_to_completion(node, state)
    library = final["citation_library"]
    assert [material.id for material in library] == ["m-1", "m-2", "m-3", "m-4", "m-new"]
    added = library[-1]
    assert added.chapter_id == "ch2"
    assert added.hypothesis_id == "ch2-p1-h2"
    assert added.source == "增量来源 m-new"
    # url 与 source_kind 从检索结果透传入库，不再硬编码。
    assert added.url == "https://example.com/m-new"
    assert added.source_kind == "web"


def _make_fallback_state(failed_chapter_ids: list[str]) -> WritingAgentState:
    """在基础三章状态上叠加现有草稿与失败终审报告（终审回退模式入口）。

    citation_retry_count 置 1 模拟 document_reviewer 写失败报告时的递增：
    回退只在重试预算内触发，预算由该字段承载。
    """
    state = _make_state()
    state["chapter_drafts"] = _existing_drafts()
    state["pending_directives"] = []
    state["revised_chapter_ids"] = []
    state["citation_retry_count"] = 1
    state["citation_report"] = CitationReport(
        passed=False,
        issues=[
            CitationIssue(
                kind="orphan_marker",
                chapter_id=chapter_id,
                material_id="m-x",
                detail=f"章节 {chapter_id} 角标 m-x 无对应素材",
            )
            for chapter_id in failed_chapter_ids
        ],
        failed_chapter_ids=failed_chapter_ids,
    )
    return state


def test_终审回退模式_只重写不合格章节():
    """终审失败定向回退：只有 failed_chapter_ids 中的章节被 revise。"""
    rewriter = 记录式假改写适配器()
    search = 记录式假检索适配器()
    node = make_writing_orchestrator_node(rewriter, search)
    state = _make_fallback_state(["ch2"])
    original_drafts = state["chapter_drafts"]
    final, _, steps = _drive_to_completion(node, state)
    assert steps == 1
    assert search.tasks == []
    assert [task["chapter_spec"]["id"] for task in rewriter.tasks] == ["ch2"]
    task = rewriter.tasks[0]
    assert task["mode"] == "revise"
    assert task["current_text"] == "ch2 旧正文"
    # prev_chapter_summary 注入 summary_chain 段（带章节标题前缀）。
    assert task["prev_chapter_summary"] == "【第一章】ch1 旧摘要"
    # 终审报告直接组装分区式修订说明：每条 issue 折成 error 级规则违规，
    # rule 带 document_review. 前缀、guidance 为 issue 明细；无用户指令。
    note = task["revision_note"]
    assert note["user_directives"] == ""
    assert len(note["rule_violations"]) == 1
    violation = note["rule_violations"][0]
    assert violation["rule"] == "document_review.orphan_marker"
    assert violation["severity"] == "error"
    assert violation["guidance"] == "章节 ch2 角标 m-x 无对应素材"
    assert note["passed"] is False
    drafts = final["chapter_drafts"]
    assert drafts[0] is original_drafts[0]
    assert drafts[1].text == "ch2 修订后正文"
    assert drafts[2] is original_drafts[2]
    assert final["revised_chapter_ids"] == ["ch2"]
    assert final["pending_directives"] == []


def test_终审回退模式_两不合格章需两超步_逐超步累积revised_chapter_ids():
    """两不合格章各占一个超步；第一超步后 revised_chapter_ids 已含第一章。"""
    rewriter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(rewriter, 记录式假检索适配器())
    state = _make_fallback_state(["ch1", "ch3"])
    final, updates, steps = _drive_to_completion(node, state)
    assert steps == 2
    assert updates[0]["revised_chapter_ids"] == ["ch1"]
    assert [task["chapter_spec"]["id"] for task in rewriter.tasks] == ["ch1", "ch3"]
    assert final["revised_chapter_ids"] == ["ch1", "ch3"]
    drafts = final["chapter_drafts"]
    assert drafts[0].text == "ch1 修订后正文"
    assert drafts[1].text == "ch2 旧正文"
    assert drafts[2].text == "ch3 修订后正文"


def test_判别函数_各模式与全部完成的返回值():
    """next_writing_step 是节点选章与图路由共用的单一事实源，逐情形验证。"""
    # 首写：第一个无草稿的章。
    state = _make_state()
    assert next_writing_step(state) == ("draft", "ch1")
    state["chapter_drafts"] = _existing_drafts()[:1]
    assert next_writing_step(state) == ("draft", "ch2")
    # 修订：按大纲顺序第一个有待执行指令的章（优先级最高）。
    state = _make_revision_state(
        [
            RevisionDirective(
                target_chapter_id="ch3", type="rewrite_only", instruction="改三"
            ),
            RevisionDirective(
                target_chapter_id="ch2", type="rewrite_only", instruction="改二"
            ),
        ]
    )
    assert next_writing_step(state) == ("revise", "ch2")
    # 终审回退：第一个「不合格且未修复」的章；已修复章不再选中。
    state = _make_fallback_state(["ch1", "ch3"])
    assert next_writing_step(state) == ("fallback", "ch1")
    state["revised_chapter_ids"] = ["ch1"]
    assert next_writing_step(state) == ("fallback", "ch3")
    # 全部完成（草稿齐、无指令、失败章均已修复）→ None，路由前进终审。
    state["revised_chapter_ids"] = ["ch1", "ch3"]
    assert next_writing_step(state) is None
    # 终审通过的报告不触发回退。
    state = _make_state()
    state["chapter_drafts"] = _existing_drafts()
    state["citation_report"] = CitationReport(passed=True)
    assert next_writing_step(state) is None
    # 超限后残留的旧失败报告（human_review_gate 开新一轮已把重试计数重置为 0）
    # 不触发回退：避免修订轮结束后绕过重试上限的计划外重写。
    state = _make_fallback_state(["ch1"])
    state["citation_retry_count"] = 0
    assert next_writing_step(state) is None


def test_路由与判别函数一致():
    """直调路径按判别函数自环，人工修订按章节构造 Send。"""
    from graph import route_after_writing_orchestrator

    state = _make_state()
    assert route_after_writing_orchestrator(state) == "writing_orchestrator"
    state["chapter_drafts"] = _existing_drafts()
    assert route_after_writing_orchestrator(state) == "document_reviewer"
    state["pending_directives"] = [
        RevisionDirective(
            target_chapter_id="ch1", type="rewrite_only", instruction="改一"
        )
    ]
    routed = route_after_writing_orchestrator(state)
    assert isinstance(routed, list)
    assert [send.node for send in routed] == ["writing_orchestrator"]
    assert routed[0].arg["directive_chapter_id"] == "ch1"


def test_防御兜底_无事可做时只推进状态机不调子智能体():
    """理论上路由不会把「无事可做」的 state 送进节点；万一发生不调子智能体。"""
    rewriter = 记录式假改写适配器()
    search = 记录式假检索适配器()
    node = make_writing_orchestrator_node(rewriter, search)
    state = _make_state()
    state["chapter_drafts"] = _existing_drafts()
    update = node(state)
    assert rewriter.tasks == []
    assert search.tasks == []
    assert update == {
        "status": WorkflowStatus.ARTICLE_WRITING,
        "current_node_llm_config": {"unit": "writing_orchestrator"},
    }


def test_非法目标章节抛ValueError():
    """指令目标章节不在大纲/草稿中：防御性抛错（上游 human_review_gate 已过滤）。"""
    node = make_writing_orchestrator_node(记录式假改写适配器(), 记录式假检索适配器())
    state = _make_revision_state(
        [
            RevisionDirective(
                target_chapter_id="ch9", type="rewrite_only", instruction="改写不存在的章"
            )
        ]
    )
    with pytest.raises(ValueError):
        node(state)


def test_修订目标章无现存草稿抛ValueError():
    """目标章在大纲中但没有草稿：防御性抛错。"""
    node = make_writing_orchestrator_node(记录式假改写适配器(), 记录式假检索适配器())
    state = _make_revision_state(
        [
            RevisionDirective(
                target_chapter_id="ch2", type="rewrite_only", instruction="精简第二章"
            )
        ]
    )
    state["chapter_drafts"] = _existing_drafts()[:1]
    with pytest.raises(ValueError):
        node(state)


def test_任务包携带State锚定的文种与变体_首写与修订两路径():
    """State 的 doc_type/doc_variant 经任务包契约原样携带（ADR-0005），只读透传。"""
    rewriter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(rewriter, 记录式假检索适配器())
    state = _make_state()
    state["doc_type"] = "人才培养方案"
    state["doc_variant"] = "高职"
    _drive_to_completion(node, state)
    assert all(task["doc_type"] == "人才培养方案" for task in rewriter.tasks)
    assert all(task["doc_variant"] == "高职" for task in rewriter.tasks)

    # 修订路径（第二处任务包构造点）同样携带。
    rewriter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(rewriter, 记录式假检索适配器())
    state = _make_revision_state(
        [
            RevisionDirective(
                target_chapter_id="ch1", type="rewrite_only", instruction="收紧语气"
            )
        ]
    )
    state["doc_type"] = "汇报材料"
    state["doc_variant"] = None
    _drive_to_completion(node, state)
    assert [task["mode"] for task in rewriter.tasks] == ["revise"]
    assert rewriter.tasks[0]["doc_type"] == "汇报材料"
    assert rewriter.tasks[0]["doc_variant"] is None


def test_State缺文种字段_任务包回落通用公文兑底():
    """过渡兼容：旧存档 State 无 doc_type/doc_variant 时任务包按兑底文种携带。"""
    rewriter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(rewriter, 记录式假检索适配器())
    _drive_to_completion(node, _make_state())
    assert all(task["doc_type"] == "通用公文" for task in rewriter.tasks)
    assert all(task["doc_variant"] is None for task in rewriter.tasks)


def test_回退并行扇出载荷_每个未修订失败章各一载荷且素材按章过滤():
    """route_after_document_reviewer 据此发 Send：每个本轮未修订的失败章一个载荷，
    引文库按章过滤；本轮已修订的章不再发。"""
    state = _make_fallback_state(["ch1", "ch3"])
    state["citation_library"] = [
        _material("m1", "h1", "ch1", "pass"),
        _material("m3", "h3", "ch3", "pass"),
        _material("m2", "h2", "ch2", "pass"),
    ]
    payloads = revision_send_payloads(state)
    assert [p[REVISION_CHAPTER_ID_KEY] for p in payloads] == ["ch1", "ch3"]
    ch1 = next(p for p in payloads if p[REVISION_CHAPTER_ID_KEY] == "ch1")
    assert [m.id for m in ch1["citation_library"]] == ["m1"]
    ch3 = next(p for p in payloads if p[REVISION_CHAPTER_ID_KEY] == "ch3")
    assert [m.id for m in ch3["citation_library"]] == ["m3"]
    # 载荷携带共享终审报告与全部草稿（summary_chain / _existing_draft 所需）
    assert ch1["citation_report"] is state["citation_report"]
    assert [d.chapter_id for d in ch1["chapter_drafts"]] == ["ch1", "ch2", "ch3"]
    # 本轮已修订的章不再发
    state["revised_chapter_ids"] = ["ch1"]
    assert [p[REVISION_CHAPTER_ID_KEY] for p in revision_send_payloads(state)] == ["ch3"]
    # 通过或无失败章时无载荷
    passed = _make_fallback_state(["ch1"])
    passed["citation_report"] = CitationReport(passed=True)
    assert revision_send_payloads(passed) == []


def test_回退并行扇出分支_单章改写回写单元素列表():
    """Send 载荷指定单章时走并行回退分支：只改写该章、回写单元素列表
    （chapter_drafts=[new_draft]、revised_chapter_ids=[chapter_id]），
    交合并 reducer 汇入，避免并行分支各回写整表互相覆盖。"""
    rewriter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(rewriter, 记录式假检索适配器())
    state = _make_fallback_state(["ch2"])
    cast(dict[str, Any], state)[REVISION_CHAPTER_ID_KEY] = "ch2"
    result = node(state)
    assert [task["chapter_spec"]["id"] for task in rewriter.tasks] == ["ch2"]
    assert [task["mode"] for task in rewriter.tasks] == ["revise"]
    # 回写单元素列表，交合并 reducer 汇入
    assert [d.chapter_id for d in result["chapter_drafts"]] == ["ch2"]
    assert result["chapter_drafts"][0].text == "ch2 修订后正文"
    assert result["revised_chapter_ids"] == ["ch2"]
    assert result["status"] == WorkflowStatus.ARTICLE_WRITING
    assert result["current_node_llm_config"]["unit"] == "writing_orchestrator"
