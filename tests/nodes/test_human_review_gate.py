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
from nodes.human_review_gate import (
    make_human_review_gate_node,
    match_quote_chapters,
    needs_fanout_confirmation,
    resolve_directives,
)
from llm.llm_client import FakeLLM
from domain.state import (
    ChapterDraft,
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


def test_revise恢复_混合两类诉求解析为修订指令_经大扇出确认后执行() -> None:
    # 两章大纲全部受影响（2/2 > 一半）：先携解析清单重新中断待确认，confirm 后执行。
    # 解析应答备两份：confirm 恢复时节点从头重放，意见解析 LLM 调用重复执行一次。
    mixed = json.dumps(
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
    fake = FakeLLM([mixed, mixed])
    graph = _build_graph(fake)
    config = _config()
    graph.invoke(_state(), config)

    confirm_request = graph.invoke(
        Command(resume={"action": "revise", "feedback": "引言太啰嗦；课程体系缺数据支撑"}),
        config,
    )
    payload = confirm_request["__interrupt__"][0].value
    confirmation = payload["pending_confirmation"]
    assert confirmation["affected_chapter_ids"] == ["ch1", "ch2"]
    assert confirmation["total_chapters"] == 2
    assert [d["target_chapter_id"] for d in confirmation["directives"]] == ["ch1", "ch2"]

    result = dict(graph.invoke(Command(resume={"action": "confirm"}), config))
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
    # confirm 恢复时节点重放，意见解析共执行两次。
    assert len(fake.calls) == 2


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


# ---------- 定位增强（issue #49）：纯函数 ----------

DRAFTS = [
    ChapterDraft(
        chapter_id="ch1",
        text="引言正文铺陈培养定位。[m-1]",
        summary="第一章摘要。",
    ),
    ChapterDraft(
        chapter_id="ch2",
        text="课程体系对接 行业标准，产教融合。[m-2]",
        summary="第二章摘要。",
    ),
]


def test_引文匹配_归一化剔除角标与空白后唯一命中() -> None:
    # 引文无角标、空白与原文不同，归一化后仍确定性命中 ch2。
    assert match_quote_chapters("课程体系对接行业标准", DRAFTS) == ["ch2"]


def test_引文匹配_空引文与未命中返回空列表() -> None:
    assert match_quote_chapters("   ", DRAFTS) == []
    assert match_quote_chapters("正文中不存在的句子", DRAFTS) == []


def test_引文匹配_多章命中按草稿顺序返回全部() -> None:
    drafts = [
        ChapterDraft(chapter_id="ch1", text="同一句话。", summary=""),
        ChapterDraft(chapter_id="ch2", text="又是同一句话。", summary=""),
    ]
    assert match_quote_chapters("同一句话", drafts) == ["ch1", "ch2"]


def _directive(chapter_id: str) -> RevisionDirective:
    return RevisionDirective(
        target_chapter_id=chapter_id, type="rewrite_only", instruction="改写"
    )


def test_大扇出判定_严格超过大纲一半才触发() -> None:
    outline = [ChapterSpec(id=f"ch{i}", title=f"第{i}章") for i in range(1, 5)]
    # 2/4 恰为一半：不触发；3/4 超过一半：触发。
    assert not needs_fanout_confirmation([_directive("ch1"), _directive("ch2")], outline)
    assert needs_fanout_confirmation(
        [_directive("ch1"), _directive("ch2"), _directive("ch3")], outline
    )
    # 同章多条指令只计一章。
    assert not needs_fanout_confirmation(
        [_directive("ch1"), _directive("ch1"), _directive("ch2")], outline
    )


def test_解析归结_global扇出为逐章指令并去重() -> None:
    items = [
        {"locate": "global", "type": "rewrite_only", "instruction": "口吻更克制"},
        # 与扇出产物完全同形的显式条目被去重。
        {"locate": "chapter", "target_chapter_id": "ch2", "type": "rewrite_only", "instruction": "口吻更克制"},
    ]
    directives, questions = resolve_directives(items, OUTLINE, DRAFTS)
    assert questions == []
    assert [(d.target_chapter_id, d.instruction) for d in directives] == [
        ("ch1", "口吻更克制"),
        ("ch2", "口吻更克制"),
    ]


def test_解析归结_quote确定性命中优先于LLM章节判断() -> None:
    items = [
        {
            "locate": "quote",
            "quote": "课程体系对接行业标准",
            # LLM 判断给了错误章节，确定性命中应覆盖它。
            "target_chapter_id": "ch1",
            "type": "rewrite_only",
            "instruction": "该句表述更严谨",
        }
    ]
    directives, questions = resolve_directives(items, OUTLINE, DRAFTS)
    assert questions == []
    assert [d.target_chapter_id for d in directives] == ["ch2"]


def test_解析归结_quote未命中回退LLM章节判断() -> None:
    items = [
        {
            "locate": "quote",
            "quote": "正文中不存在的句子",
            "target_chapter_id": "ch1",
            "type": "rewrite_only",
            "instruction": "改写该句",
        }
    ]
    directives, questions = resolve_directives(items, OUTLINE, DRAFTS)
    assert questions == []
    assert [d.target_chapter_id for d in directives] == ["ch1"]


def test_解析归结_quote两级定位均失败生成回问且不产指令() -> None:
    items = [
        {
            "locate": "quote",
            "quote": "正文中不存在的句子",
            "target_chapter_id": None,
            "type": "rewrite_only",
            "instruction": "改写该句",
        }
    ]
    directives, questions = resolve_directives(items, OUTLINE, DRAFTS)
    assert directives == []
    assert len(questions) == 1
    assert "无法定位引文" in questions[0]


def test_解析归结_unclear收集回问且缺问题文本时兜底() -> None:
    items = [
        {"locate": "unclear", "question": "「写得再好点」指哪一章的哪方面？"},
        {"locate": "unclear"},
    ]
    directives, questions = resolve_directives(items, OUTLINE, DRAFTS)
    assert directives == []
    assert questions[0] == "「写得再好点」指哪一章的哪方面？"
    assert "请补充说明" in questions[1]


# ---------- 定位增强（issue #49）：经真实中断机制的节点行为 ----------


def test_引文命中直达目标章_不触发确认直接执行() -> None:
    response = json.dumps(
        [
            {
                "locate": "quote",
                "quote": "课程体系对接行业标准",
                "target_chapter_id": None,
                "type": "rewrite_only",
                "instruction": "该句表述更严谨",
            }
        ],
        ensure_ascii=False,
    )
    result = _interrupt_then_resume(
        FakeLLM([response]),
        {"action": "revise", "feedback": "「课程体系对接行业标准」这句要更严谨"},
        chapter_drafts=DRAFTS,
    )
    # 1/2 章受影响，不超过一半：无确认中断，指令直达 ch2。
    assert [d.target_chapter_id for d in result["pending_directives"]] == ["ch2"]


def test_引文定位失败_回问用户后重新提交可继续() -> None:
    quote_fail = json.dumps(
        [
            {
                "locate": "quote",
                "quote": "找不到的句子",
                "target_chapter_id": None,
                "type": "rewrite_only",
                "instruction": "改写该句",
            }
        ],
        ensure_ascii=False,
    )
    ok = json.dumps(
        [
            {
                "locate": "chapter",
                "target_chapter_id": "ch1",
                "type": "rewrite_only",
                "instruction": "重写引言",
            }
        ],
        ensure_ascii=False,
    )
    # 回问后再提交时节点重放：首轮解析重复执行一次，故失败应答备两份。
    fake = FakeLLM([quote_fail, quote_fail, ok])
    graph = _build_graph(fake)
    config = _config()
    graph.invoke(_state(chapter_drafts=DRAFTS), config)

    asked = graph.invoke(
        Command(resume={"action": "revise", "feedback": "「找不到的句子」改一下"}), config
    )
    payload = asked["__interrupt__"][0].value
    assert any("无法定位引文" in q for q in payload["clarification_questions"])
    assert "pending_confirmation" not in payload

    result = graph.invoke(
        Command(resume={"action": "revise", "feedback": "重写第一章引言"}), config
    )
    assert [d.target_chapter_id for d in result["pending_directives"]] == ["ch1"]


def test_意见含混_携回问问题重新中断而非猜测() -> None:
    unclear = json.dumps(
        [{"locate": "unclear", "question": "请说明希望修改哪一章、往什么方向改？"}],
        ensure_ascii=False,
    )
    graph = _build_graph(FakeLLM([unclear]))
    config = _config()
    graph.invoke(_state(chapter_drafts=DRAFTS), config)

    asked = graph.invoke(
        Command(resume={"action": "revise", "feedback": "整体写得再好一点"}), config
    )
    payload = asked["__interrupt__"][0].value
    assert payload["clarification_questions"] == [
        "请说明希望修改哪一章、往什么方向改？"
    ]

    # 回问后仍可定稿收束，永不卡死。
    result = graph.invoke(Command(resume={"action": "finalize"}), config)
    assert result["status"] == WorkflowStatus.FINISHED


def test_全局意见扇出为逐章指令_经大扇出确认后执行() -> None:
    global_response = json.dumps(
        [{"locate": "global", "type": "rewrite_only", "instruction": "全篇口吻更克制"}],
        ensure_ascii=False,
    )
    fake = FakeLLM([global_response, global_response])
    graph = _build_graph(fake)
    config = _config()
    graph.invoke(_state(chapter_drafts=DRAFTS), config)

    confirm_request = graph.invoke(
        Command(resume={"action": "revise", "feedback": "全篇口吻都要更克制"}), config
    )
    confirmation = confirm_request["__interrupt__"][0].value["pending_confirmation"]
    assert confirmation["affected_chapter_ids"] == ["ch1", "ch2"]
    assert confirmation["total_chapters"] == 2

    result = graph.invoke(Command(resume={"action": "confirm"}), config)
    assert [
        (d.target_chapter_id, d.instruction) for d in result["pending_directives"]
    ] == [("ch1", "全篇口吻更克制"), ("ch2", "全篇口吻更克制")]
    # 台账记录触发确认的那轮意见原文。
    assert result["revision_ledger"][-1].raw_feedback == "全篇口吻都要更克制"


def test_大扇出确认时改提意见_作废清单按新意见执行() -> None:
    global_response = json.dumps(
        [{"locate": "global", "type": "rewrite_only", "instruction": "全篇口吻更克制"}],
        ensure_ascii=False,
    )
    ch2_only = json.dumps(
        [
            {
                "locate": "chapter",
                "target_chapter_id": "ch2",
                "type": "rewrite_only",
                "instruction": "只收束第二章",
            }
        ],
        ensure_ascii=False,
    )
    # 改提意见时节点重放：全局解析重复执行一次，故全局应答备两份。
    fake = FakeLLM([global_response, global_response, ch2_only])
    graph = _build_graph(fake)
    config = _config()
    graph.invoke(_state(chapter_drafts=DRAFTS), config)
    graph.invoke(
        Command(resume={"action": "revise", "feedback": "全篇口吻都要更克制"}), config
    )

    result = graph.invoke(
        Command(resume={"action": "revise", "feedback": "算了，只改第二章"}), config
    )
    assert [d.target_chapter_id for d in result["pending_directives"]] == ["ch2"]
    assert result["revision_ledger"][-1].raw_feedback == "算了，只改第二章"


def test_无待确认清单时confirm按契约不符重新中断() -> None:
    graph = _build_graph(FakeLLM())
    config = _config()
    graph.invoke(_state(), config)

    result = graph.invoke(Command(resume={"action": "confirm"}), config)
    payload = result["__interrupt__"][0].value
    assert "confirm 不可用" in payload["error"]

    # 契约错误后仍可定稿收束。
    result = graph.invoke(Command(resume={"action": "finalize"}), config)
    assert result["status"] == WorkflowStatus.FINISHED
