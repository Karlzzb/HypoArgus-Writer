"""writing_orchestrator 节点单元测试：用假 rewriter_loop 适配器验证串行调度。

覆盖点：逐章各一次调用且顺序与大纲一致；摘要链承接（prev_chapter_summary）；
素材过滤（只有该章 verdict=pass 的素材进任务包）；改写结果与单章自检入 State；
状态机推进到 ARTICLE_WRITING。
"""

from typing import Any

from state import (
    ArgumentPoint,
    ChapterSpec,
    Hypothesis,
    Material,
    WorkflowStatus,
    WritingAgentState,
)
from subagents import SubagentAdapter
from writing_orchestrator import make_writing_orchestrator_node


class 记录式假改写适配器(SubagentAdapter):
    """记录任务包顺序并返回确定性结果的假 rewriter_loop。

    各章摘要不同以便验证摘要链；ch2 返回 citations_ok=False 带问题清单。
    """

    def __init__(self) -> None:
        super().__init__("rewriter_loop", self._run)
        self.tasks: list[dict[str, Any]] = []

    async def _run(self, task: dict[str, Any]) -> dict[str, Any]:
        self.tasks.append(task)
        chapter_id = task["chapter_spec"]["id"]
        if chapter_id == "ch2":
            self_check = {"citations_ok": False, "issues": ["角标 m-x 不在素材列表中"]}
        else:
            self_check = {"citations_ok": True, "issues": []}
        return {
            "chapter_text": f"{chapter_id} 的正文 [m-{chapter_id}]",
            "chapter_summary": f"{chapter_id} 的摘要",
            "self_check": self_check,
        }


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
    node = make_writing_orchestrator_node(adapter)
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
