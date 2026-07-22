"""writer 写作编排的契约测试：Fake 客户端驱动，覆盖纯写作两模式、空稿短路与进度事件。

自 ADR-0006 T3 起 rewriter_loop 收束为纯写作 + 空稿短路：一次写作调用即返回，
不再 lint / 不再自审 / 不触发第二次写作调用。质检已上移到 chapter_reviewer 与
循环层的修后 re-lint，故 self_check 恒退化——成稿引用通过、空稿引用不通过附退化说明。
本套用例只守当下真实行为：写作调用形状、返回字段、self_check 退化、单一进度事件对。
"""

import asyncio
from typing import Any

from agents.contracts import SubagentAdapter
from agents.rewriter_loop import (
    UNIT,
    FakeWriterLlmClient,
    WriterEnvelope,
    make_rewriter_loop,
    make_writer_run,
)
from domain.doc_types import tier_from_variant
from domain.events import SUBAGENT_END, SUBAGENT_PROGRESS, SUBAGENT_START
from llm.llm_client import LLM, FakeLLM


def _make_recorder() -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    """构造把 (事件类型, 载荷) 收进列表的挂钩。"""
    events: list[tuple[str, dict[str, Any]]] = []

    def record_hook(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    return events, record_hook


# 写作产物正文（角标随文即可，纯写作链路不再 lint）。
_TEXT = "本专业面向智能制造领域培养高素质人才。[m-h-1][m-h-2]"


def _envelope(text: str, summary: str = "一行摘要", **extra: Any) -> WriterEnvelope:
    return WriterEnvelope(chapter_text=text, chapter_summary=summary, **extra)


def test_写作编排_draft模式_一次写作调用且返回字段与自检退化通过(
    draft_task: dict[str, Any],
) -> None:
    fake = FakeWriterLlmClient(draft_script=[_envelope(_TEXT)])
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    # 恰一次 draft、零次 revise/audit；纯写作不再传 fix_violations。
    assert len(fake.draft_calls) == 1
    assert len(fake.revise_calls) == 0
    assert fake.audit_calls == []
    seen_task, fix = fake.draft_calls[0]
    assert fix is None
    # 素材与上一章摘要经任务包原样传达到注入点。
    assert seen_task["prev_chapter_summary"] == draft_task["prev_chapter_summary"]
    assert seen_task["materials"] == draft_task["materials"]

    assert set(result.keys()) == {
        "chapter_text",
        "chapter_summary",
        "self_check",
        "doc_type",
        "doc_variant",
    }
    assert result["chapter_text"] == _TEXT
    assert result["chapter_summary"] == "一行摘要"
    # 成稿：self_check 恒退化为引用通过、无 issues（终态质检交由评审与循环层）。
    assert result["self_check"] == {"citations_ok": True, "issues": []}


def test_写作编排_revise模式_走revise返回改写正文且自检退化通过(
    draft_task: dict[str, Any],
) -> None:
    fake = FakeWriterLlmClient(revise_script=[_envelope("改写后的正文。[m-h-1]", "修订摘要")])
    draft_task["mode"] = "revise"
    draft_task["revision_note"] = {
        "user_directives": "精简第一段",
        "rule_violations": [],
        "conflict_hints": [],
        "passed": True,
    }
    draft_task["current_text"] = "现有正文。[m-h-1]"
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    # revise 模式走 client.revise，不触 draft；纯写作不再传 fix_violations。
    assert len(fake.revise_calls) == 1
    assert len(fake.draft_calls) == 0
    seen_task, fix = fake.revise_calls[0]
    assert fix is None
    assert seen_task["current_text"] == draft_task["current_text"]

    assert set(result.keys()) == {
        "chapter_text",
        "chapter_summary",
        "self_check",
        "doc_type",
        "doc_variant",
    }
    assert result["chapter_text"] == "改写后的正文。[m-h-1]"
    assert result["chapter_summary"] == "修订摘要"
    assert result["self_check"] == {"citations_ok": True, "issues": []}


def test_写作编排_空正文退化_短路上报退化且引用不通过(draft_task: dict[str, Any]) -> None:
    events, record_hook = _make_recorder()
    fake = FakeWriterLlmClient(
        draft_script=[
            WriterEnvelope(
                chapter_text="", chapter_summary="退化占位摘要", attempts=3, degraded=True
            )
        ],
    )
    run = make_writer_run(fake, event_hook=record_hook)
    result = asyncio.run(run(draft_task))

    # 空稿短路：恰一次 draft、零次 audit；如实上报退化并判引用不通过。
    assert len(fake.draft_calls) == 1
    assert fake.audit_calls == []
    assert result["chapter_text"] == ""
    assert result["chapter_summary"] == "退化占位摘要"
    assert result["self_check"]["citations_ok"] is False
    # 退化说明携带重试轮次（来自信封 attempts）。
    assert result["self_check"]["issues"] == ["写作模型退化：正文为空（已重试 3 轮）"]


def test_写作编排_进度事件_恰一对写作事件且载荷完备无质检事件(
    draft_task: dict[str, Any],
) -> None:
    events, record_hook = _make_recorder()
    fake = FakeWriterLlmClient(draft_script=[_envelope(_TEXT, attempts=2, degraded=False)])
    run = make_writer_run(fake, event_hook=record_hook)
    asyncio.run(run(draft_task))

    progress = [payload for event_type, payload in events if event_type == SUBAGENT_PROGRESS]
    steps = [payload["step"] for payload in progress]
    # 纯写作链路只发唯一一对 llm_call 事件，绝不发 lint_done / audit_done 等质检事件。
    assert steps == ["llm_call_start", "llm_call_end"]
    assert "lint_done" not in steps
    assert "audit_done" not in steps
    assert "revise_triggered" not in steps
    assert "pre_lint_done" not in steps

    start, end = progress
    # 载荷统一带 unit / chapter_id / mode / step / call，只放元数据不放正文。
    for payload in progress:
        assert payload["unit"] == UNIT
        assert payload["chapter_id"] == "ch-1"
        assert payload["mode"] == "draft"
        assert payload["call"] == "draft"
        assert _TEXT not in str(payload)
    assert start["step"] == "llm_call_start"
    assert end["step"] == "llm_call_end"
    # llm_call_end 携带 attempts / text_chars / degraded。
    assert end["attempts"] == 2
    assert end["text_chars"] == len(_TEXT)
    assert end["degraded"] is False


def test_写作编排_空稿进度事件_同样只发一对写作事件且带退化标记(
    draft_task: dict[str, Any],
) -> None:
    events, record_hook = _make_recorder()
    fake = FakeWriterLlmClient(
        draft_script=[
            WriterEnvelope(
                chapter_text="", chapter_summary="退化占位摘要", attempts=3, degraded=True
            )
        ],
    )
    run = make_writer_run(fake, event_hook=record_hook)
    asyncio.run(run(draft_task))

    progress = [payload for event_type, payload in events if event_type == SUBAGENT_PROGRESS]
    assert [payload["step"] for payload in progress] == ["llm_call_start", "llm_call_end"]
    assert progress[1]["degraded"] is True
    assert progress[1]["attempts"] == 3
    assert progress[1]["text_chars"] == 0


def test_写作编排_经适配层_启动结束与进度事件共存(draft_task: dict[str, Any]) -> None:
    events, record_hook = _make_recorder()
    fake = FakeWriterLlmClient(draft_script=[_envelope(_TEXT)])
    run = make_writer_run(fake, event_hook=record_hook)
    adapter = SubagentAdapter(UNIT, run, record_hook)
    result = asyncio.run(adapter.run(draft_task))

    assert set(result.keys()) == {
        "chapter_text",
        "chapter_summary",
        "self_check",
        "doc_type",
        "doc_variant",
    }
    event_types = [event_type for event_type, _ in events]
    assert event_types[0] == SUBAGENT_START
    assert event_types[-1] == SUBAGENT_END
    assert SUBAGENT_PROGRESS in event_types[1:-1]


def test_工厂_请求单元名与适配器单元正确() -> None:
    seen_units: list[str] = []

    def factory(unit: str) -> LLM:
        seen_units.append(unit)
        return FakeLLM()

    adapter = make_rewriter_loop(factory)

    assert seen_units == ["rewriter_loop"]
    assert isinstance(adapter, SubagentAdapter)
    assert adapter.unit == "rewriter_loop"


def test_层次推导_人培两变体即层次() -> None:
    assert tier_from_variant("本科") == "本科"
    assert tier_from_variant("高职") == "高职"


def test_层次推导_无变体或非层次变体_回落缺省本科() -> None:
    assert tier_from_variant(None) == "本科"
    assert tier_from_variant("某未来变体") == "本科"


def test_写作编排_结果回带任务包的文种与变体(draft_task: dict[str, Any]) -> None:
    fake = FakeWriterLlmClient(draft_script=[_envelope(_TEXT)])
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    assert result["doc_type"] == "人才培养方案"
    assert result["doc_variant"] == "本科"


def test_写作编排_任务包缺文种字段_回带通用公文兑底(draft_task: dict[str, Any]) -> None:
    """过渡兼容：旧存档任务包无文种字段时按兑底文种处理，结果如实回带。"""
    del draft_task["doc_type"]
    del draft_task["doc_variant"]
    fake = FakeWriterLlmClient(draft_script=[_envelope(_TEXT)])
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    assert result["doc_type"] == "通用公文"
    assert result["doc_variant"] is None


def test_写作编排_空正文退化_同样回带文种与变体(draft_task: dict[str, Any]) -> None:
    fake = FakeWriterLlmClient(
        draft_script=[
            WriterEnvelope(
                chapter_text="", chapter_summary="退化占位摘要", attempts=3, degraded=True
            )
        ],
    )
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    assert result["doc_type"] == "人才培养方案"
    assert result["doc_variant"] == "本科"
