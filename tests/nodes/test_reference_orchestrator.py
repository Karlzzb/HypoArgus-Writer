"""reference_orchestrator 检索并行扇出测试：载荷切分与单章分支节点。

假适配器记录收到的任务包并返回预置素材（含 verdict=fail 条目），
覆盖载荷切分（无假说跳过、已检索不重发、状态切片字段）、
单章分支的任务包字段、素材回链、既有引文库摘要与状态机推进。
"""

from typing import Any

from nodes.reference_orchestrator import (
    REFERENCE_CHAPTER_ID_KEY,
    make_reference_orchestrator_node,
    reference_send_payloads,
)
from domain.state import (
    ArgumentPoint,
    ChapterSpec,
    Hypothesis,
    Material,
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


def _material(
    mat_id: str,
    hyp_id: str,
    verdict: str = "pass",
    url: str | None = None,
    source_kind: str = "knowledge_base",
) -> dict[str, Any]:
    return {
        "id": mat_id,
        "hypothesis_id": hyp_id,
        "source": f"来源 {mat_id}",
        "url": url,
        "source_kind": source_kind,
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
            "ch3": [
                _material(
                    "m-ch3-p1-h1",
                    "ch3-p1-h1",
                    url="https://example.com/ch3",
                    source_kind="web",
                )
            ],
        }
    )


def _ch1_library_material() -> Material:
    return Material(
        id="m-ch1-p1-h1",
        hypothesis_id="ch1-p1-h1",
        chapter_id="ch1",
        source="来源 m-ch1-p1-h1",
        excerpt="摘录 m-ch1-p1-h1",
        relevance_score=0.8,
        verdict="pass",
    )


def test_载荷切分_有假说章节各一个且无假说章节跳过():
    payloads = reference_send_payloads(_build_state())

    assert [payload[REFERENCE_CHAPTER_ID_KEY] for payload in payloads] == [
        "ch1",
        "ch3",
    ]


def test_载荷切分_已有素材入库章节不重发():
    state = _build_state()
    state["citation_library"] = [_ch1_library_material()]

    payloads = reference_send_payloads(state)

    assert [payload[REFERENCE_CHAPTER_ID_KEY] for payload in payloads] == ["ch3"]


def test_载荷只携带状态切片_大纲按目标章过滤且引文库整体携带():
    state = _build_state()
    state["citation_library"] = [_ch1_library_material()]

    payloads = reference_send_payloads(state)

    payload = payloads[0]
    assert [chapter.id for chapter in payload["outline"]] == ["ch3"]
    assert payload["citation_library"] == state["citation_library"]
    assert payload["genre"] == "行业评论"
    assert "chapter_drafts" not in payload


def test_单章分支任务包字段与假说扁平列表正确():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)
    payloads = reference_send_payloads(_build_state())

    for payload in payloads:
        node(payload)

    assert [task["chapter_id"] for task in adapter.tasks] == ["ch1", "ch3"]
    for task in adapter.tasks:
        assert task["genre"] == "行业评论"
    # 论点列表随任务包携带，供查询构造聚合论点+假说（杠杆①）。
    assert adapter.tasks[0]["points"] == [
        {"id": "ch1-p1", "text": "论点一"},
        {"id": "ch1-p2", "text": "论点二"},
    ]
    assert adapter.tasks[1]["points"] == [{"id": "ch3-p1", "text": "论点四"}]
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


def test_单章分支素材入库回链假说与章节且fail素材保留():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)
    payloads = reference_send_payloads(_build_state())

    updates = [node(payload) for payload in payloads]

    # 每个分支只回写目标章素材，跨分支合并交给 citation_library reducer。
    assert [material.id for material in updates[0]["citation_library"]] == [
        "m-ch1-p1-h1",
        "m-ch1-p1-h2",
        "m-ch1-p2-h1",
    ]
    assert [material.id for material in updates[1]["citation_library"]] == [
        "m-ch3-p1-h1"
    ]
    library = [
        material for update in updates for material in update["citation_library"]
    ]
    by_material_id = {material.id: material for material in library}
    assert "m-ch1-p1-h2" in by_material_id, "fail 素材也必须入库供后续环节筛选"
    for material in library:
        assert material.id == f"m-{material.hypothesis_id}"
        assert material.chapter_id == material.hypothesis_id.split("-")[0]
    # url 与 source_kind 从检索结果透传入库，不再硬编码。
    assert by_material_id["m-ch3-p1-h1"].url == "https://example.com/ch3"
    assert by_material_id["m-ch3-p1-h1"].source_kind == "web"
    assert by_material_id["m-ch1-p1-h1"].url is None
    assert by_material_id["m-ch1-p1-h1"].source_kind == "knowledge_base"
    verdicts = {material.id: material.verdict for material in library}
    assert verdicts["m-ch1-p1-h2"] == "fail"
    assert verdicts["m-ch1-p1-h1"] == "pass"


def test_existing_materials_digest只反映既有引文库():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)
    state = _build_state()
    state["citation_library"] = [_ch1_library_material()]

    for payload in reference_send_payloads(state):
        node(payload)

    # 语义降档：digest 是扇出前既有引文库的快照，不再逐章反映轮内增长。
    assert [task["existing_materials_digest"] for task in adapter.tasks] == [
        "引文库共 1 条素材。\n章节 ch1：通过 1 条，弱佐证 0 条，未通过 0 条"
    ]


def test_首轮空引文库时digest为零条素材():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)

    for payload in reference_send_payloads(_build_state()):
        node(payload)

    assert {task["existing_materials_digest"] for task in adapter.tasks} == {
        "引文库共 0 条素材。"
    }


def test_状态机推进到REFERENCE_FETCHING且记录节点配置():
    adapter = _make_adapter()
    node = make_reference_orchestrator_node(adapter)
    payload = reference_send_payloads(_build_state())[0]

    update = node(payload)

    assert update["status"] == WorkflowStatus.REFERENCE_FETCHING
    assert update["current_node_llm_config"] == {"unit": "reference_orchestrator"}
