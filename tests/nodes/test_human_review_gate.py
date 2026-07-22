"""human_review_gate 节点测试：经真实 LangGraph 中断机制验证。

构建只含此节点的最小 StateGraph，用内存存档器编译；首次 invoke 触发中断，
再以 Command(resume=决定) 恢复，覆盖 finalize 与 revise 两条恢复路径、
意见解析的程序侧过滤，以及契约不符/解析失败时的安全汇点重新中断行为。
"""

import json
import uuid
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from graph import checkpoint_serializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from assembly.assembler_config import AssemblerConfig
from assembly.context_assembler import assemble
from nodes.human_review_gate import make_human_review_gate_node
from llm.llm_client import FakeLLM
from domain.state import (
    ChapterSpec,
    RevisionDirective,
    RevisionRound,
    WorkflowStatus,
    WritingAgentState,
)

OUTLINE = [
    ChapterSpec(id="ch1", title="引言"),
    ChapterSpec(id="ch2", title="课程体系"),
]


def _build_graph(fake: FakeLLM, assembler_config: AssemblerConfig | None = None) -> Any:
    """构建只含 human_review_gate 节点的最小图，用内存存档器编译。"""
    node = make_human_review_gate_node(lambda unit: fake, assembler_config)
    builder = StateGraph(WritingAgentState)
    builder.add_node("human_review_gate", node)
    builder.add_edge(START, "human_review_gate")
    builder.add_edge("human_review_gate", END)
    return builder.compile(checkpointer=InMemorySaver(serde=checkpoint_serializer()))


def _state(**overrides: Any) -> WritingAgentState:
    """构造带大纲与既有台账的进入中断点前的状态。"""
    state = WritingAgentState(
        outline=OUTLINE,
        chapter_drafts=[],
        revision_ledger=[
            RevisionRound(round_no=1, raw_feedback="旧一轮意见", directives=[])
        ],
        pending_directives=[],
        citation_warnings=["素材 m1 未决"],
        status=WorkflowStatus.AWAIT_USER_REVIEW,
        iteration_round=1,
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _interrupt_then_resume(
    fake: FakeLLM, decision: Any, **state_overrides: Any
) -> dict[str, Any]:
    """首次 invoke 触发中断后立即以给定决定恢复，返回最终状态。"""
    graph = _build_graph(fake)
    config = _config()
    first = graph.invoke(_state(**state_overrides), config)
    assert "__interrupt__" in first
    return dict(graph.invoke(Command(resume=decision), config))


def test_首次运行停在中断点且载荷只含元数据() -> None:
    graph = _build_graph(FakeLLM())
    result = graph.invoke(_state(), _config())

    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert payload["iteration_round"] == 1
    assert payload["chapter_ids"] == ["ch1", "ch2"]
    assert payload["citation_warnings"] == ["素材 m1 未决"]
    # 载荷只含元数据，不得携带任何章节正文。
    assert "chapter_drafts" not in payload
    assert "outline" not in payload


def test_finalize恢复_定稿且不调LLM() -> None:
    fake = FakeLLM()
    result = _interrupt_then_resume(fake, {"action": "finalize"})

    assert result["status"] == WorkflowStatus.FINISHED
    assert result["pending_directives"] == []
    assert result["citation_warnings"] == []
    assert result["current_node_llm_config"] == {"unit": "human_review_gate"}
    assert fake.calls == []


def test_revise恢复_混合两类诉求解析为修订指令() -> None:
    fake = FakeLLM(
        [
            json.dumps(
                [
                    {
                        "target_chapter_id": "ch1",
                        "type": "rewrite_only",
                        "instruction": "把引言写得更简洁",
                    },
                    {
                        "target_chapter_id": "ch2",
                        "type": "evidence_augmented",
                        "instruction": "为课程体系补充行业数据佐证",
                    },
                ],
                ensure_ascii=False,
            )
        ]
    )
    result = _interrupt_then_resume(
        fake, {"action": "revise", "feedback": "引言太啰嗦；课程体系缺数据支撑"}
    )

    assert result["pending_directives"] == [
        RevisionDirective(
            target_chapter_id="ch1", type="rewrite_only", instruction="把引言写得更简洁"
        ),
        RevisionDirective(
            target_chapter_id="ch2",
            type="evidence_augmented",
            instruction="为课程体系补充行业数据佐证",
        ),
    ]
    # 台账整值覆盖：保留旧轮次并追加新一轮。
    ledger = result["revision_ledger"]
    assert [entry.round_no for entry in ledger] == [1, 2]
    assert ledger[0].raw_feedback == "旧一轮意见"
    assert ledger[1].raw_feedback == "引言太啰嗦；课程体系缺数据支撑"
    assert ledger[1].directives == result["pending_directives"]
    assert result["iteration_round"] == 2
    assert result["citation_warnings"] == []
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    assert result["current_node_llm_config"] == {
        "unit": "human_review_gate",
        "model": "fake-llm",
        "base_url": "fake://",
    }
    assert len(fake.calls) == 1


def test_revise解析过滤_非法条目被丢弃仅保留合法条目() -> None:
    fake = FakeLLM(
        [
            json.dumps(
                [
                    {
                        "target_chapter_id": "ch99",
                        "type": "rewrite_only",
                        "instruction": "改写不存在的章节",
                    },
                    {
                        "target_chapter_id": "ch1",
                        "type": "polish",
                        "instruction": "非法类型",
                    },
                    {
                        "target_chapter_id": "ch1",
                        "type": "rewrite_only",
                        "instruction": "  ",
                    },
                    "不是对象",
                    {
                        "target_chapter_id": "ch2",
                        "type": "evidence_augmented",
                        "instruction": "补充佐证",
                    },
                ],
                ensure_ascii=False,
            )
        ]
    )
    result = _interrupt_then_resume(fake, {"action": "revise", "feedback": "意见"})

    assert result["pending_directives"] == [
        RevisionDirective(
            target_chapter_id="ch2", type="evidence_augmented", instruction="补充佐证"
        )
    ]


def test_revise全部条目非法_携错误说明重新中断() -> None:
    fake = FakeLLM(
        [
            json.dumps(
                [{"target_chapter_id": "ch99", "type": "灵感", "instruction": ""}],
                ensure_ascii=False,
            ),
            json.dumps(
                [
                    {
                        "target_chapter_id": "ch1",
                        "type": "rewrite_only",
                        "instruction": "重写引言",
                    }
                ],
                ensure_ascii=False,
            ),
        ]
    )
    graph = _build_graph(fake)
    config = _config()
    graph.invoke(_state(), config)

    # 安全汇点：解析不出有效指令不抛异常终止，携错误说明重新中断等待人工。
    result = graph.invoke(
        Command(resume={"action": "revise", "feedback": "意见"}), config
    )
    assert "修订指令" in result["__interrupt__"][0].value["error"]

    # 人工重新提交后流程照常继续。
    result = graph.invoke(
        Command(resume={"action": "revise", "feedback": "重写引言"}), config
    )
    assert [d.target_chapter_id for d in result["pending_directives"]] == ["ch1"]


@pytest.mark.parametrize(
    "decision",
    [
        "finalize",  # 非 dict
        {"action": "publish"},  # action 非法
        {"action": "revise"},  # revise 缺 feedback
        {"action": "revise", "feedback": "  "},  # feedback 空串
    ],
)
def test_恢复值契约不符_携错误说明重新中断(decision: Any) -> None:
    graph = _build_graph(FakeLLM())
    config = _config()
    graph.invoke(_state(), config)

    result = graph.invoke(Command(resume=decision), config)
    payload = result["__interrupt__"][0].value
    assert "恢复值" in payload["error"]

    # 错误后仍可定稿收束，永不卡死。
    result = graph.invoke(Command(resume={"action": "finalize"}), config)
    assert result["status"] == WorkflowStatus.FINISHED


def test_滑出保留窗口的早期轮次digest已持久化到State() -> None:
    """多轮迭代后，最近 K 轮之外的早期轮次在写回 State 时已落库一句话摘要。"""
    config = AssemblerConfig(
        summary_chain_max_chars=800,
        summary_digest_max_chars=60,
        ledger_keep_rounds=2,
        ledger_digest_max_chars=60,
        document_text_max_chars=30000,
    )
    fake = FakeLLM(
        [
            json.dumps(
                [
                    {
                        "target_chapter_id": "ch1",
                        "type": "rewrite_only",
                        "instruction": "收紧语气",
                    }
                ],
                ensure_ascii=False,
            )
        ]
    )
    graph = _build_graph(fake, config)
    thread = _config()
    existing = [
        RevisionRound(round_no=1, raw_feedback="第一轮原文。多余部分。"),
        RevisionRound(round_no=2, raw_feedback="第二轮原文。", digest="第二轮既有摘要"),
        RevisionRound(round_no=3, raw_feedback="第三轮原文。"),
    ]
    graph.invoke(_state(revision_ledger=existing, iteration_round=3), thread)
    result = graph.invoke(
        Command(resume={"action": "revise", "feedback": "第四轮意见"}), thread
    )

    ledger = result["revision_ledger"]
    assert [entry.round_no for entry in ledger] == [1, 2, 3, 4]
    # 保留窗口 K=2：第 1、2 轮滑出窗口，digest 落库；既有 digest 不被覆盖。
    assert ledger[0].digest == "第一轮原文。"
    assert ledger[1].digest == "第二轮既有摘要"
    assert ledger[2].digest is None
    assert ledger[3].digest is None

    # extract_revision_ledger 优先使用已持久化的 digest 装配摘要行。
    context = assemble(
        WritingAgentState(revision_ledger=ledger), "human_review_gate", config=config
    )
    assert "第1轮（摘要）：第一轮原文。" in context.text("revision_ledger")
    assert "第2轮（摘要）：第二轮既有摘要" in context.text("revision_ledger")
