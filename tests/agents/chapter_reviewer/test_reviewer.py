"""chapter_reviewer 评审编排单测（纯编排，FakeReviewLlmClient 替身）。

断言：进度步序、单次 LLM 调用（single-shot）、空裁决合法、降级不阻断主链、
revise 携用户意见原文进用户指令区、self_check 按引用类规则折叠。
"""

import asyncio
from typing import Any

from agents.chapter_reviewer import (
    FakeReviewLlmClient,
    ReviewConflict,
    ReviewEnvelope,
    ReviewIssue,
    make_reviewer_run,
)
from domain.events import SUBAGENT_PROGRESS


def _run(client: FakeReviewLlmClient, task: dict[str, Any], hook: Any = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if hook is not None:
        kwargs["event_hook"] = hook
    run = make_reviewer_run(client, **kwargs)
    return asyncio.run(run(task))


def test_评审_单次LLM调用且进度步序完整(review_task: dict[str, Any]) -> None:
    steps: list[str] = []

    def hook(event_type: str, payload: dict[str, Any]) -> None:
        if event_type == SUBAGENT_PROGRESS:
            steps.append(payload["step"])
            # 载荷只放元数据：带 unit/chapter_id/mode，绝不含正文全文。
            assert payload["unit"] == "chapter_reviewer"
            assert payload["chapter_id"] == "ch-1"
            assert review_task["chapter_text"] not in str(payload)

    client = FakeReviewLlmClient([ReviewEnvelope(issues=[ReviewIssue(item="intra_chapter_coherence")])])
    _run(client, review_task, hook)

    # single-shot：恰一次 review 调用，评审内部不迭代。
    assert len(client.review_calls) == 1
    # 步序：lint 完成 → 自审调用对 → 自审结论 → 修订说明生成。
    assert steps == [
        "lint_done",
        "llm_call_start",
        "llm_call_end",
        "audit_done",
        "revision_note_done",
    ]


def test_评审_空裁决合法_不阻断(review_task: dict[str, Any]) -> None:
    # 空 issues/conflicts 是合法非退化结果。
    client = FakeReviewLlmClient([ReviewEnvelope()])
    result = _run(client, review_task)

    # 干净短文本仍会被 lint 命中字数下限（error），此处只验空自审不报自审违规、
    # 且结果结构完整、self_check 折叠正常（无引用类违规 → citations_ok=True）。
    note = result["revision_note"]
    assert not any(e["rule"].startswith("self_audit_") for e in note["rule_violations"])
    assert result["self_check"]["citations_ok"] is True


def test_评审_模型降级不阻断主链(review_task: dict[str, Any]) -> None:
    # degraded 裁决（空）：不抛错、如实产出修订说明与自检。
    client = FakeReviewLlmClient([ReviewEnvelope(degraded=True)])
    result = _run(client, review_task)

    assert set(result.keys()) == {"revision_note", "self_check"}
    assert len(client.review_calls) == 1


def test_评审_revise携用户意见原文逐字进用户指令区(review_task: dict[str, Any]) -> None:
    review_task["mode"] = "revise"
    review_task["user_feedback"] = "保留第二段原句，勿改。"
    client = FakeReviewLlmClient([ReviewEnvelope(conflicts=[ReviewConflict(description="规则与用户意见冲突")])])

    result = _run(client, review_task)

    note = result["revision_note"]
    assert note["user_directives"] == "保留第二段原句，勿改。"
    assert note["conflict_hints"] == [{"description": "规则与用户意见冲突"}]


def test_评审_引用类自审违规_折叠为citations_not_ok(review_task: dict[str, Any]) -> None:
    # 派生未标属引用类规则：自审命中 → citations_ok=False（self_check 折叠口径）。
    client = FakeReviewLlmClient(
        [ReviewEnvelope(issues=[ReviewIssue(item="unmarked_derived_content", excerpt="改写未标")])]
    )
    result = _run(client, review_task)

    assert result["self_check"]["citations_ok"] is False
