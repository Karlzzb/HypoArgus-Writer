"""writer 写作编排的契约测试：Fake 客户端驱动，覆盖两模式、修一次链路与进度事件。"""

import asyncio
from typing import Any

from agents.contracts import SubagentAdapter
from agents.rewriter_loop import (
    UNIT,
    AuditEnvelope,
    AuditIssue,
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


# 不触发任何 lint 规则的干净正文（角标在素材池内、无口语化/编号/意识形态违规）。
_CLEAN_TEXT = "本专业面向智能制造领域培养高素质人才。[m-h-1][m-h-2]"

# 含池外角标的违规正文：触发 unknown_material_marker（引用类违规）。
_MARKER_VIOLATION_TEXT = "本专业面向智能制造领域培养高素质人才。[m-x9]"


def _envelope(text: str, summary: str = "一行摘要") -> WriterEnvelope:
    return WriterEnvelope(chapter_text=text, chapter_summary=summary)


def test_写作编排_draft无违规_一次调用且自检通过(draft_task: dict[str, Any]) -> None:
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(_CLEAN_TEXT)],
        audit_script=[AuditEnvelope()],
    )
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    # 恰一次 draft、无 fix；素材与上一章摘要经任务包原样传达到注入点。
    assert len(fake.draft_calls) == 1
    seen_task, fix = fake.draft_calls[0]
    assert fix is None
    assert seen_task["prev_chapter_summary"] == draft_task["prev_chapter_summary"]
    assert seen_task["materials"] == draft_task["materials"]

    assert set(result.keys()) == {
        "chapter_text",
        "chapter_summary",
        "self_check",
        "doc_type",
        "doc_variant",
    }
    assert result["chapter_text"] == _CLEAN_TEXT
    assert result["self_check"] == {"citations_ok": True, "issues": []}


def test_写作编排_draft检出lint违规_修一次且修后复检出清(draft_task: dict[str, Any]) -> None:
    fixed_text = _CLEAN_TEXT
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(_MARKER_VIOLATION_TEXT), _envelope(fixed_text, "修后摘要")],
        audit_script=[AuditEnvelope(), AuditEnvelope()],
    )
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    # 恰好一次 fix：第二次 draft 带非空 fix_violations；修后复检（ADR-0004）：
    # 修前正文与修后产物各审一次。
    assert len(fake.draft_calls) == 2
    _, fix = fake.draft_calls[1]
    assert fix is not None and len(fix) > 0
    assert fake.audit_calls == [_MARKER_VIOLATION_TEXT, fixed_text]

    # 最终正文取第二次信封；self_check 折叠修后终态——违规已修净则引用通过。
    assert result["chapter_text"] == fixed_text
    assert result["chapter_summary"] == "修后摘要"
    assert result["self_check"] == {"citations_ok": True, "issues": []}


def test_写作编排_自审违规_修后复审出清_引用通过(draft_task: dict[str, Any]) -> None:
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(_CLEAN_TEXT), _envelope(_CLEAN_TEXT, "修后摘要")],
        audit_script=[
            AuditEnvelope(issues=[AuditIssue(material_id="m-h-1", excerpt="疑似片段")]),
            AuditEnvelope(),
        ],
    )
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    # 自审违规触发修一次；修后复审干净 → 终态引用通过、不留修前残迹。
    assert len(fake.draft_calls) == 2
    _, fix = fake.draft_calls[1]
    assert fix is not None and len(fix) == 1
    assert result["self_check"] == {"citations_ok": True, "issues": []}


def test_写作编排_自审违规_修后复审仍在_引用不通过(draft_task: dict[str, Any]) -> None:
    audit_issue = AuditEnvelope(
        issues=[AuditIssue(material_id="m-h-1", excerpt="疑似片段")]
    )
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(_CLEAN_TEXT), _envelope(_CLEAN_TEXT, "修后摘要")],
        audit_script=[audit_issue, audit_issue],
    )
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    # 修后复审仍报同类违规 → 终态引用不通过，issues 留的是修后结论。
    assert len(fake.draft_calls) == 2
    assert result["self_check"]["citations_ok"] is False
    assert any("m-h-1" in issue for issue in result["self_check"]["issues"])


def test_写作编排_revise模式_收到任务包且无违规不修(draft_task: dict[str, Any]) -> None:
    fake = FakeWriterLlmClient(
        revise_script=[_envelope(_CLEAN_TEXT)],
        audit_script=[AuditEnvelope()],
    )
    draft_task["mode"] = "revise"
    draft_task["revision_directives"] = [
        {"type": "rewrite_only", "instruction": "精简第一段"}
    ]
    draft_task["current_text"] = "现有正文。[m-h-1]"
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    # revise 模式走 client.revise，不触 draft；现有正文预 lint 干净 → 不带违规清单。
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
    assert result["self_check"] == {"citations_ok": True, "issues": []}


def test_写作编排_revise模式_既存违规并入唯一一次调用不再二次修(
    draft_task: dict[str, Any],
) -> None:
    events, record_hook = _make_recorder()
    fake = FakeWriterLlmClient(
        revise_script=[_envelope(_CLEAN_TEXT, "修订摘要")],
        audit_script=[AuditEnvelope()],
    )
    draft_task["mode"] = "revise"
    draft_task["revision_directives"] = [
        {"type": "rewrite_only", "instruction": "精简第一段"}
    ]
    # 现有正文含池外角标：预 lint 检出既存违规。
    draft_task["current_text"] = _MARKER_VIOLATION_TEXT
    run = make_writer_run(fake, event_hook=record_hook)
    result = asyncio.run(run(draft_task))

    # revise 与 fix 合并（ADR-0004）：既存违规并入唯一一次 revise 调用，
    # 调用后 lint + 自审即为终态质检，绝不触发第二次写作调用。
    assert len(fake.revise_calls) == 1
    assert len(fake.draft_calls) == 0
    _, fix = fake.revise_calls[0]
    assert fix is not None
    assert any(v.rule == "unknown_material_marker" for v in fix)
    # 修订产物干净 → 终态引用通过。
    assert result["self_check"] == {"citations_ok": True, "issues": []}

    # 事件流：预 lint 有事件、无 revise_triggered（不存在第二次调用）。
    progress = [payload for event_type, payload in events if event_type == SUBAGENT_PROGRESS]
    steps = [payload["step"] for payload in progress]
    assert steps == [
        "pre_lint_done",
        "llm_call_start",
        "llm_call_end",
        "lint_done",
        "llm_call_start",
        "llm_call_end",
        "audit_done",
    ]
    assert progress[0]["violations"] == 1


def test_写作编排_revise模式_修订产物仍违规_如实折叠不再补修(
    draft_task: dict[str, Any],
) -> None:
    fake = FakeWriterLlmClient(
        revise_script=[_envelope(_MARKER_VIOLATION_TEXT, "修订摘要")],
        audit_script=[AuditEnvelope()],
    )
    draft_task["mode"] = "revise"
    draft_task["revision_directives"] = [
        {"type": "rewrite_only", "instruction": "精简第一段"}
    ]
    draft_task["current_text"] = "现有正文。[m-h-1]"
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    # 修订产物仍有引用类违规 → 只如实折叠（交全局终审裁决），不触发二次调用。
    assert len(fake.revise_calls) == 1
    assert result["self_check"]["citations_ok"] is False
    assert any(
        "unknown_material_marker" in issue for issue in result["self_check"]["issues"]
    )


def test_写作编排_空正文退化_跳过质检直接上报(draft_task: dict[str, Any]) -> None:
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

    # 不 lint / 不自审 / 不修订：恰一次 draft、零次 audit。
    assert len(fake.draft_calls) == 1
    assert fake.audit_calls == []
    assert result["chapter_text"] == ""
    assert result["chapter_summary"] == "退化占位摘要"
    assert result["self_check"]["citations_ok"] is False
    assert result["self_check"]["issues"] == ["写作模型退化：正文为空（已重试 3 轮）"]

    # 短路路径的事件形状：只发首次写作调用的一对 llm_call 事件（含 degraded），
    # 不发 lint_done / audit_done 等后续步骤（刻意设计，事件流如实反映执行轨迹）。
    progress = [payload for event_type, payload in events if event_type == SUBAGENT_PROGRESS]
    assert [payload["step"] for payload in progress] == ["llm_call_start", "llm_call_end"]
    assert progress[1]["degraded"] is True
    assert progress[1]["attempts"] == 3


def test_写作编排_无pass素材_跳过自审仍发裁决事件(draft_task: dict[str, Any]) -> None:
    events, record_hook = _make_recorder()
    fake = FakeWriterLlmClient(
        draft_script=[_envelope("本专业面向智能制造领域培养高素质人才。")],
    )
    # 全部素材判 fail：pass 池为空 → 跳过自审的模型调用。
    for material in draft_task["materials"]:
        material["verdict"] = "fail"
    run = make_writer_run(fake, event_hook=record_hook)
    result = asyncio.run(run(draft_task))

    assert fake.audit_calls == []
    assert result["self_check"] == {"citations_ok": True, "issues": []}
    # audit_done 仍发出（issues=0），但没有 audit 的 llm_call 事件对。
    progress = [payload for event_type, payload in events if event_type == SUBAGENT_PROGRESS]
    steps = [payload["step"] for payload in progress]
    assert steps == ["llm_call_start", "llm_call_end", "lint_done", "audit_done"]
    audit_done = progress[steps.index("audit_done")]
    assert audit_done["issues"] == 0
    assert audit_done["degraded"] is False


def test_写作编排_修订产物退化为空_追加退化说明且引用不通过(draft_task: dict[str, Any]) -> None:
    fake = FakeWriterLlmClient(
        draft_script=[
            _envelope(_MARKER_VIOLATION_TEXT),
            WriterEnvelope(
                chapter_text="", chapter_summary="修订退化摘要", attempts=3, degraded=True
            ),
        ],
        audit_script=[AuditEnvelope()],
    )
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    # 保留修前违规明细，追加修订退化说明；引用一律判不通过。
    issues = result["self_check"]["issues"]
    assert any("unknown_material_marker" in issue for issue in issues)
    assert issues[-1] == "修订调用退化：正文为空（已重试 3 轮）"
    assert result["self_check"]["citations_ok"] is False
    assert result["chapter_text"] == ""


def test_写作编排_进度事件_步骤序列与载荷完备(draft_task: dict[str, Any]) -> None:
    events, record_hook = _make_recorder()
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(_MARKER_VIOLATION_TEXT), _envelope(_CLEAN_TEXT)],
        audit_script=[AuditEnvelope(), AuditEnvelope()],
    )
    run = make_writer_run(fake, event_hook=record_hook)
    asyncio.run(run(draft_task))

    progress = [payload for event_type, payload in events if event_type == SUBAGENT_PROGRESS]
    steps = [payload["step"] for payload in progress]
    # draft/audit/fix/复检 audit 四次真实调用各包一对 llm_call_start/llm_call_end；
    # 修后复检（ADR-0004）再走一轮 lint_done / audit_done。
    assert steps == [
        "llm_call_start",
        "llm_call_end",
        "lint_done",
        "llm_call_start",
        "llm_call_end",
        "audit_done",
        "revise_triggered",
        "llm_call_start",
        "llm_call_end",
        "lint_done",
        "llm_call_start",
        "llm_call_end",
        "audit_done",
    ]
    assert steps.count("revise_triggered") == 1
    for payload in progress:
        assert payload["unit"] == UNIT
        assert payload["chapter_id"] == "ch-1"
        assert payload["mode"] == "draft"
        # 载荷只放元数据，绝不放正文全文。
        assert _MARKER_VIOLATION_TEXT not in str(payload)


def test_写作编排_经适配层_启动结束与进度事件共存(draft_task: dict[str, Any]) -> None:
    events, record_hook = _make_recorder()
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(_CLEAN_TEXT)],
        audit_script=[AuditEnvelope()],
    )
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
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(_CLEAN_TEXT)],
        audit_script=[AuditEnvelope()],
    )
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    assert result["doc_type"] == "人才培养方案"
    assert result["doc_variant"] == "本科"


def test_写作编排_任务包缺文种字段_回带通用公文兑底(draft_task: dict[str, Any]) -> None:
    """过渡兼容：旧存档任务包无文种字段时按兑底文种处理，结果如实回带。"""
    del draft_task["doc_type"]
    del draft_task["doc_variant"]
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(_CLEAN_TEXT)],
        audit_script=[AuditEnvelope()],
    )
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


def test_写作编排_修前检出字数违规_修后复检达标不留残迹(
    draft_task: dict[str, Any],
) -> None:
    # 构造修前违规正文：约 1900 字章不足下限（2000）；修后产物达标（约 3000 字）。
    short_text = "## 一、总则\n\n" + "本专业面向智能制造领域培养高素质人才。" * 100  # 约 1900 字
    fixed_text = "## 一、总则\n\n" + "本专业面向智能制造领域培养高素质人才。" * 150  # 约 3000 字
    events, record_hook = _make_recorder()
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(short_text), _envelope(fixed_text)],
        audit_script=[AuditEnvelope(), AuditEnvelope()],
    )
    run = make_writer_run(fake, event_hook=record_hook)
    result = asyncio.run(run(draft_task))

    # 触发修订（修前检出字数违规）；修后全量复检达标 → 终态无残留 issues。
    assert len(fake.draft_calls) == 2
    assert result["self_check"] == {"citations_ok": True, "issues": []}

    # 事件流：修后复检再走一轮 lint_done。
    progress = [payload for event_type, payload in events if event_type == SUBAGENT_PROGRESS]
    steps = [payload["step"] for payload in progress]
    assert steps.count("lint_done") == 2
    assert progress[[i for i, s in enumerate(steps) if s == "lint_done"][1]]["violations"] == 0


def test_写作编排_修后复检仍违规_如实折入issues(draft_task: dict[str, Any]) -> None:
    # 修前不足下限（2000），修后仍不足（如实上报修后终态未达标）。
    # 标题「一、总则」= 3 字计入，故正文 1996 字，合计 1999 < 2000。
    short_text = "## 一、总则\n\n" + "字" * 1996
    # 修后正文 1897 字，合计 1900 仍不足。
    still_short = "## 一、总则\n\n" + "字" * 1897
    fake = FakeWriterLlmClient(
        draft_script=[_envelope(short_text), _envelope(still_short)],
        audit_script=[AuditEnvelope(), AuditEnvelope()],
    )
    run = make_writer_run(fake)
    result = asyncio.run(run(draft_task))

    issues = result["self_check"]["issues"]
    # 终态仍不足下限 → 修后复检结论折入 issues；字数不属引用类 → 引用仍通过。
    assert any("word_count" in issue and "不足下限" in issue for issue in issues)
    assert result["self_check"]["citations_ok"] is True

