"""chapter_drafter 节点与首写扇出路由的单元测试。

覆盖：Send 载荷构造（选章与状态切片）、reference_orchestrator 后的条件路由、
节点单分支执行（任务包字段、规划摘要链承接、只回写带 reducer 的字段）。
"""

from typing import Any

from langgraph.types import Send

from agents.chapter_reviewer import make_stub_chapter_reviewer
from agents.contracts import SubagentAdapter
from assembly.assembler_config import AssemblerConfig
from domain.state import (
    ArgumentPoint,
    ChapterDraft,
    ChapterSpec,
    Hypothesis,
    Material,
    WorkflowStatus,
    WritingAgentState,
)
from graph import route_after_reference_join
from nodes.chapter_drafter import (
    DRAFT_CHAPTER_ID_KEY,
    draft_send_payloads,
    make_chapter_drafter_node,
)

_CONFIG = AssemblerConfig(
    summary_chain_max_chars=800,
    summary_digest_max_chars=60,
    ledger_keep_rounds=2,
    ledger_digest_max_chars=60,
    document_text_max_chars=30000,
)


def _chapter(chapter_id: str, title: str, planned: str) -> ChapterSpec:
    return ChapterSpec(
        id=chapter_id,
        title=title,
        planned_summary=planned,
        points=[
            ArgumentPoint(
                id=f"{chapter_id}-p1",
                text=f"{title}论点",
                hypotheses=[
                    Hypothesis(
                        id=f"{chapter_id}-p1-h1",
                        text=f"{title}假说",
                        refute_condition="出现公开反例即证伪",
                        angle="假设",
                    )
                ],
            )
        ],
    )


def _material(chapter_id: str) -> Material:
    return Material(
        id=f"m-{chapter_id}",
        hypothesis_id=f"{chapter_id}-p1-h1",
        chapter_id=chapter_id,
        source="来源",
        url=None,
        excerpt="摘录",
        relevance_score=0.9,
        verdict="pass",
    )


def _state() -> WritingAgentState:
    return WritingAgentState(
        outline=[
            _chapter("ch1", "第一章", "规划一。"),
            _chapter("ch2", "第二章", "规划二。"),
        ],
        citation_library=[_material("ch1"), _material("ch2")],
        chapter_drafts=[],
        doc_type="通用公文",
        doc_variant=None,
    )


def test_载荷构造_全部未写章节各一份且引文库按章过滤():
    payloads = draft_send_payloads(_state())
    assert [payload[DRAFT_CHAPTER_ID_KEY] for payload in payloads] == ["ch1", "ch2"]
    for payload in payloads:
        chapter_id = payload[DRAFT_CHAPTER_ID_KEY]
        assert [m.chapter_id for m in payload["citation_library"]] == [chapter_id]
        assert payload["doc_type"] == "通用公文"
        assert len(payload["outline"]) == 2


def test_载荷构造_已写章节不再扇出():
    state = _state()
    state["chapter_drafts"] = [
        ChapterDraft(chapter_id="ch1", text="正文", summary="摘要")
    ]
    payloads = draft_send_payloads(state)
    assert [payload[DRAFT_CHAPTER_ID_KEY] for payload in payloads] == ["ch2"]


def test_路由_未写章节扇出Send_全部已写直进终审():
    routed = route_after_reference_join(_state())
    assert isinstance(routed, list)
    assert all(isinstance(send, Send) for send in routed)
    assert [send.node for send in routed] == ["chapter_drafter", "chapter_drafter"]

    state = _state()
    state["chapter_drafts"] = [
        ChapterDraft(chapter_id="ch1", text="a", summary="s"),
        ChapterDraft(chapter_id="ch2", text="b", summary="s"),
    ]
    assert route_after_reference_join(state) == "document_reviewer"


def test_节点单分支_任务包承接规划摘要链且只回写reducer字段():
    tasks: list[dict[str, Any]] = []

    async def _recording_run(task: dict[str, Any]) -> dict[str, Any]:
        tasks.append(task)
        return {
            "chapter_text": "第二章正文 [m-ch2]",
            "chapter_summary": "第二章摘要",
            "self_check": {"citations_ok": True, "issues": []},
        }

    node = make_chapter_drafter_node(
        SubagentAdapter("rewriter_loop", _recording_run),
        make_stub_chapter_reviewer(),
        _CONFIG,
    )
    payloads = draft_send_payloads(_state())
    ch2_payload = payloads[1]
    update = node(ch2_payload)

    # 任务包：draft 模式、目标章骨架、按章素材、规划摘要链承接前章。
    (task,) = tasks
    assert task["mode"] == "draft"
    assert task["chapter_spec"]["id"] == "ch2"
    assert [m["id"] for m in task["materials"]] == ["m-ch2"]
    assert task["prev_chapter_summary"] == "【第一章】规划一。"

    # 回写只含带合并 / keep_last reducer 的字段，避免并行分支写入冲突。
    assert set(update) == {"chapter_drafts", "status", "current_node_llm_config"}
    (draft,) = update["chapter_drafts"]
    assert draft.chapter_id == "ch2"
    assert draft.text == "第二章正文 [m-ch2]"
    assert update["status"] == WorkflowStatus.ARTICLE_WRITING
    assert update["current_node_llm_config"] == {"unit": "chapter_drafter"}


def test_节点单分支_首章规划摘要链为空串():
    tasks: list[dict[str, Any]] = []

    async def _recording_run(task: dict[str, Any]) -> dict[str, Any]:
        tasks.append(task)
        return {
            "chapter_text": "第一章正文 [m-ch1]",
            "chapter_summary": "第一章摘要",
            "self_check": {"citations_ok": True, "issues": []},
        }

    node = make_chapter_drafter_node(
        SubagentAdapter("rewriter_loop", _recording_run),
        make_stub_chapter_reviewer(),
        _CONFIG,
    )
    node(draft_send_payloads(_state())[0])
    assert tasks[0]["prev_chapter_summary"] == ""
