"""reference_orchestrator 节点测试：用假适配器验证逐章调度与结构化引文库入库。

假适配器记录收到的任务包并返回预置素材（含 verdict=fail 条目），
覆盖调用次数与顺序、无假说章节跳过、素材回链、摘要递增与状态机推进。
"""

from typing import Any

from reference_orchestrator import make_reference_orchestrator_node
from state import (
    ArgumentPoint,
    ChapterSpec,
    Hypothesis,
    WorkflowStatus,
    WritingAgentState,
)


class 假检索适配器:
    """记录收到的任务包，按章节 id 返回预置素材列表。"""

    def __init__(self, materials_by_chapter: dict[str, list[dict[str, Any]]]) -> None:
        self.unit = "search_agent"
        self.tasks: list[dict[str, Any]] = []
        self._materials_by_chapter = materials_by_chapter

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        self.tasks.append(task)
        return {"materials": self._materials_by_chapter.get(task["chapter_id"], [])}


def _hypothesis(hyp_id: str) -> Hypothesis:
    return Hypothesis(
        id=hyp_id,
        text=f"假说 {hyp_id}",
        refute_condition=f"证伪条件 {hyp_id}",
        angle="假设",
    )


def _material(mat_id: str, hyp_id: str, verdict: str = "pass") -> dict[str, Any]:
    return {
        "id": mat_id,
        "hypothesis_id": hyp_id,
        "source": f"来源 {mat_id}",
        "excerpt": f"摘录 {mat_id}",
        "relevance_score": 0.8,
        "verdict": verdict,
    }


def _build_state() -> WritingAgentState:
    """三章骨架：ch1 两论点共三假说、ch2 无假说、ch3 一假说。"""
    outline = [
        ChapterSpec(
            id="ch1",
            title="第一章",
            points=[
                ArgumentPoint(
                    id="ch1-p1",
                    text="论点一",
                    hypotheses=[_hypothesis("ch1-p1-h1"), _hypothesis("ch1-p1-h2")],
                ),
                ArgumentPoint(
                    id="ch1-p2",
                    text="论点二",
                    hypotheses=[_hypothesis("ch1-p2-h1")],
                ),
            ],
        ),
        ChapterSpec(
            id="ch2",
            title="第二章",
            points=[ArgumentPoint(id="ch2-p1", text="论点三", hypotheses=[])],
        ),
        ChapterSpec(
            id="ch3",
            title="第三章",
            points=[
                ArgumentPoint(
                    id="ch3-p1", text="论点四", hypotheses=[_hypothesis("ch3-p1-h1")]
                )
            ],
        ),
    ]
    return WritingAgentState(
        genre="行业评论",
        outline=outline,
        citation_library=[],
        status=WorkflowStatus.FRAMEWORK_BUILDING,
    )


def _make_adapter() -> 假检索适配器:
    return 假检索适配器(
        {
            "ch1": [
                _material("m-ch1-p1-h1", "ch1-p1-h1"),
                _material("m-ch1-p1-h2", "ch1-p1-h2", verdict="fail"),
                _material("m-ch1-p2-h1", "ch1-p2-h1"),
            ],
            "ch3": [_material("m-ch3-p1-h1", "ch3-p1-h1")],
        }
    )


def test_每章恰好一次调用且顺序与任务包字段正确():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)

    node(_build_state())

    assert [task["chapter_id"] for task in adapter.tasks] == ["ch1", "ch3"]
    for task in adapter.tasks:
        assert task["genre"] == "行业评论"
    # 假说扁平列表与骨架一致：跨论点按顺序拉平，字段完整。
    assert adapter.tasks[0]["hypotheses"] == [
        {
            "id": "ch1-p1-h1",
            "text": "假说 ch1-p1-h1",
            "refute_condition": "证伪条件 ch1-p1-h1",
        },
        {
            "id": "ch1-p1-h2",
            "text": "假说 ch1-p1-h2",
            "refute_condition": "证伪条件 ch1-p1-h2",
        },
        {
            "id": "ch1-p2-h1",
            "text": "假说 ch1-p2-h1",
            "refute_condition": "证伪条件 ch1-p2-h1",
        },
    ]
    assert adapter.tasks[1]["hypotheses"] == [
        {
            "id": "ch3-p1-h1",
            "text": "假说 ch3-p1-h1",
            "refute_condition": "证伪条件 ch3-p1-h1",
        }
    ]


def test_无假说章节被跳过不调用适配器():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)

    node(_build_state())

    assert "ch2" not in {task["chapter_id"] for task in adapter.tasks}


def test_素材入库回链假说与章节且fail素材保留():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)

    update = node(_build_state())

    library = update["citation_library"]
    assert [material.id for material in library] == [
        "m-ch1-p1-h1",
        "m-ch1-p1-h2",
        "m-ch1-p2-h1",
        "m-ch3-p1-h1",
    ]
    by_id = {material.id for material in library}
    assert "m-ch1-p1-h2" in by_id, "fail 素材也必须入库供后续环节筛选"
    for material in library:
        assert material.url is None
        assert material.id == f"m-{material.hypothesis_id}"
        assert material.chapter_id == material.hypothesis_id.split("-")[0]
    verdicts = {material.id: material.verdict for material in library}
    assert verdicts["m-ch1-p1-h2"] == "fail"
    assert verdicts["m-ch1-p1-h1"] == "pass"


def test_existing_materials_digest经引文库摘要段逐章反映增长():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)

    node(_build_state())

    digests = [task["existing_materials_digest"] for task in adapter.tasks]
    # digest 改由 citation_digest 段装配：ch1 调用前引文库为空；
    # ch3 调用前已累积 ch1 返回的 3 条素材（2 通过 1 未通过，均属 ch1）。
    assert digests[0] == "引文库共 0 条素材。"
    assert digests[1] == "引文库共 3 条素材。\n章节 ch1：通过 2 条，未通过 1 条"


def test_状态机推进到REFERENCE_FETCHING且记录节点配置():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)

    update = node(_build_state())

    assert update["status"] == WorkflowStatus.REFERENCE_FETCHING
    assert update["current_node_llm_config"] == {"unit": "reference_orchestrator"}
