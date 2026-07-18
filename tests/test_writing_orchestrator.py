"""writing_orchestrator 节点单元测试：用假子智能体适配器验证三种模式的调度。

覆盖点：首写模式逐章各一次调用且顺序与大纲一致；摘要链承接（prev_chapter_summary）；
素材过滤（只有该章 verdict=pass 的素材进任务包）；改写结果与单章自检入 State；
状态机推进到 ARTICLE_WRITING；修订模式按指令定向改写与增量检索入库去重；
终审回退模式只重写不合格章节；非法目标章节防御性抛错。
"""

from typing import Any

import pytest

from state import (
    ArgumentPoint,
    ChapterDraft,
    ChapterSpec,
    CitationIssue,
    CitationReport,
    Hypothesis,
    Material,
    RevisionDirective,
    WorkflowStatus,
    WritingAgentState,
)
from subagents import MaterialPayload, SubagentAdapter
from writing_orchestrator import make_writing_orchestrator_node


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
    mat_id: str, hypothesis_id: str, chapter_id: str, verdict: str
) -> Material:
    return Material(
        id=mat_id,
        hypothesis_id=hypothesis_id,
        chapter_id=chapter_id,
        source=f"来源 {mat_id}",
        excerpt=f"摘录 {mat_id}",
        relevance_score=0.8,
        verdict=verdict,  # type: ignore[arg-type]
    )


def _make_state() -> WritingAgentState:
    """三章大纲：ch1 两论点各一假说，ch2 一论点两假说，ch3 无论点。

    引文库同时含 pass 与 fail 素材，用于验证素材过滤。
    """
    outline = [
        ChapterSpec(
            id="ch1",
            title="第一章",
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
        _material("m-2", "ch1-p2-h1", "ch1", "fail"),
        _material("m-3", "ch2-p1-h1", "ch2", "pass"),
        _material("m-4", "ch2-p1-h2", "ch2", "pass"),
    ]
    return WritingAgentState(
        outline=outline,
        citation_library=citation_library,
        status=WorkflowStatus.REFERENCE_FETCHING,
    )


def _run_node() -> tuple[记录式假改写适配器, WritingAgentState]:
    adapter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(adapter, 记录式假检索适配器())
    update = node(_make_state())
    return adapter, update


def test_逐章各一次调用_顺序与大纲一致():
    adapter, _ = _run_node()
    assert [task["chapter_spec"]["id"] for task in adapter.tasks] == [
        "ch1",
        "ch2",
        "ch3",
    ]
    assert all(task["mode"] == "draft" for task in adapter.tasks)


def test_任务包章节骨架含论点与全章扁平假说():
    adapter, _ = _run_node()
    spec_ch1 = adapter.tasks[0]["chapter_spec"]
    assert spec_ch1["title"] == "第一章"
    assert [point["id"] for point in spec_ch1["points"]] == ["ch1-p1", "ch1-p2"]
    assert [point["text"] for point in spec_ch1["points"]] == ["论点一", "论点二"]
    assert [hyp["id"] for hyp in spec_ch1["hypotheses"]] == ["ch1-p1-h1", "ch1-p2-h1"]
    spec_ch2 = adapter.tasks[1]["chapter_spec"]
    assert [hyp["id"] for hyp in spec_ch2["hypotheses"]] == ["ch2-p1-h1", "ch2-p1-h2"]
    assert all(
        hyp["refute_condition"] for hyp in spec_ch1["hypotheses"] + spec_ch2["hypotheses"]
    )


def test_摘要链承接_首章为空_后章取前章摘要():
    adapter, _ = _run_node()
    assert adapter.tasks[0]["prev_chapter_summary"] == ""
    assert adapter.tasks[1]["prev_chapter_summary"] == "ch1 的摘要"
    assert adapter.tasks[2]["prev_chapter_summary"] == "ch2 的摘要"


def test_素材过滤_只有本章pass素材进任务包():
    adapter, _ = _run_node()
    # ch1：m-2 verdict=fail 被过滤，他章素材不进。
    assert [material["id"] for material in adapter.tasks[0]["materials"]] == ["m-1"]
    material = adapter.tasks[0]["materials"][0]
    assert material["hypothesis_id"] == "ch1-p1-h1"
    assert material["source"] == "来源 m-1"
    assert material["excerpt"] == "摘录 m-1"
    assert material["relevance_score"] == 0.8
    assert material["verdict"] == "pass"
    # ch2：两条 pass 素材都进；ch3：无素材。
    assert [material["id"] for material in adapter.tasks[1]["materials"]] == [
        "m-3",
        "m-4",
    ]
    assert adapter.tasks[2]["materials"] == []


def test_改写结果与自检入State():
    _, update = _run_node()
    drafts = update["chapter_drafts"]
    assert [draft.chapter_id for draft in drafts] == ["ch1", "ch2", "ch3"]
    assert drafts[0].text == "ch1 的正文 [m-ch1]"
    assert drafts[0].summary == "ch1 的摘要"
    assert drafts[0].self_check.citations_ok is True
    assert drafts[0].self_check.issues == []
    # ch2 的自检失败结果（含问题清单）必须原样入 State。
    assert drafts[1].self_check.citations_ok is False
    assert drafts[1].self_check.issues == ["角标 m-x 不在素材列表中"]


def test_状态机推进到ARTICLE_WRITING_且记录运行单元():
    _, update = _run_node()
    assert update["status"] == WorkflowStatus.ARTICLE_WRITING
    assert update["current_node_llm_config"] == {"unit": "writing_orchestrator"}


def test_首写模式回归_revised_chapter_ids为空():
    """无待执行指令、无失败终审报告时走首写模式，revised_chapter_ids 为空表示全量核查。"""
    _, update = _run_node()
    assert update["revised_chapter_ids"] == []


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
    state["status"] = WorkflowStatus.AWAIT_USER_REVIEW
    return state


def _new_material(mat_id: str, hypothesis_id: str) -> MaterialPayload:
    return MaterialPayload(
        id=mat_id,
        hypothesis_id=hypothesis_id,
        source=f"增量来源 {mat_id}",
        excerpt=f"增量摘录 {mat_id}",
        relevance_score=0.9,
        verdict="pass",
    )


def test_修订模式_混合分支同轮执行():
    """ch1 纯改写 + ch2 补充佐证：同轮各自执行，其他章节草稿原样保留。"""
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
    update = node(state)

    # search_agent 只为 ch2 被调用一次，任务包同 reference_orchestrator 的 SearchTask。
    assert len(search.tasks) == 1
    search_task = search.tasks[0]
    assert search_task["chapter_id"] == "ch2"
    assert [hyp["id"] for hyp in search_task["hypotheses"]] == ["ch2-p1-h1", "ch2-p1-h2"]
    assert search_task["existing_materials_digest"] == "引文库已有素材 4 条"

    # rewriter_loop 两次均 mode=revise，任务包带正确 directives 与 current_text。
    assert [task["chapter_spec"]["id"] for task in rewriter.tasks] == ["ch1", "ch2"]
    assert all(task["mode"] == "revise" for task in rewriter.tasks)
    assert rewriter.tasks[0]["revision_directives"] == [
        {"type": "rewrite_only", "instruction": "收紧第一章语气"}
    ]
    assert rewriter.tasks[0]["current_text"] == "ch1 旧正文"
    assert rewriter.tasks[0]["prev_chapter_summary"] == ""
    assert rewriter.tasks[1]["revision_directives"] == [
        {"type": "evidence_augmented", "instruction": "补充第二章数据佐证"}
    ]
    assert rewriter.tasks[1]["current_text"] == "ch2 旧正文"
    assert rewriter.tasks[1]["prev_chapter_summary"] == "ch1 旧摘要"
    # ch2 任务包素材含新增素材 m-new。
    assert [material["id"] for material in rewriter.tasks[1]["materials"]] == [
        "m-3",
        "m-4",
        "m-new",
    ]

    # 仅 ch1、ch2 草稿变化，ch3 草稿对象原样保留。
    drafts = update["chapter_drafts"]
    assert [draft.chapter_id for draft in drafts] == ["ch1", "ch2", "ch3"]
    assert drafts[0].text == "ch1 修订后正文"
    assert drafts[1].text == "ch2 修订后正文"
    assert drafts[2] is state["chapter_drafts"][2]
    assert update["revised_chapter_ids"] == ["ch1", "ch2"]
    assert update["pending_directives"] == []
    assert update["status"] == WorkflowStatus.ARTICLE_WRITING
    assert update["current_node_llm_config"] == {"unit": "writing_orchestrator"}


def test_修订模式_只作用于指定章节():
    """三章大纲只改 ch2，ch1 与 ch3 草稿对象原样保留。"""
    rewriter = 记录式假改写适配器()
    node = make_writing_orchestrator_node(rewriter, 记录式假检索适配器())
    state = _make_revision_state(
        [
            RevisionDirective(
                target_chapter_id="ch2", type="rewrite_only", instruction="精简第二章"
            )
        ]
    )
    update = node(state)
    assert [task["chapter_spec"]["id"] for task in rewriter.tasks] == ["ch2"]
    drafts = update["chapter_drafts"]
    assert drafts[0] is state["chapter_drafts"][0]
    assert drafts[1].text == "ch2 修订后正文"
    assert drafts[2] is state["chapter_drafts"][2]
    assert update["revised_chapter_ids"] == ["ch2"]


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
    update = node(state)
    library = update["citation_library"]
    assert [material.id for material in library] == ["m-1", "m-2", "m-3", "m-4", "m-new"]
    added = library[-1]
    assert added.chapter_id == "ch2"
    assert added.hypothesis_id == "ch2-p1-h2"
    assert added.source == "增量来源 m-new"


def test_终审回退模式_只重写不合格章节():
    """终审失败定向回退：只有 failed_chapter_ids 中的章节被 revise。"""
    rewriter = 记录式假改写适配器()
    search = 记录式假检索适配器()
    node = make_writing_orchestrator_node(rewriter, search)
    state = _make_state()
    state["chapter_drafts"] = _existing_drafts()
    state["pending_directives"] = []
    state["citation_report"] = CitationReport(
        passed=False,
        issues=[
            CitationIssue(
                kind="orphan_marker",
                chapter_id="ch2",
                material_id="m-x",
                detail="角标 m-x 无对应素材",
            )
        ],
        failed_chapter_ids=["ch2"],
    )
    update = node(state)
    assert search.tasks == []
    assert [task["chapter_spec"]["id"] for task in rewriter.tasks] == ["ch2"]
    task = rewriter.tasks[0]
    assert task["mode"] == "revise"
    assert task["current_text"] == "ch2 旧正文"
    assert task["prev_chapter_summary"] == "ch1 旧摘要"
    directives = task["revision_directives"]
    assert len(directives) == 1
    assert directives[0]["type"] == "rewrite_only"
    assert "角标 m-x 无对应素材" in directives[0]["instruction"]
    drafts = update["chapter_drafts"]
    assert drafts[0] is state["chapter_drafts"][0]
    assert drafts[1].text == "ch2 修订后正文"
    assert drafts[2] is state["chapter_drafts"][2]
    assert update["revised_chapter_ids"] == ["ch2"]
    assert update["pending_directives"] == []


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
