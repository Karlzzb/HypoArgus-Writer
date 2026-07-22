"""chapter_reviewer 子智能体：章级评审，产出分区式修订说明 + 自检。

本文件持有单元名与打桩实现；真实现落地按 contracts.ReviewTask/ReviewResult
同一接口规范，工厂签名不变。打桩零副作用、确定性、瞬时完成——只用于显式注入。
"""

from typing import Any

from agents.contracts import RevisionNotePayload, SelfCheckPayload, SubagentAdapter
from domain.events import EventHook, noop_hook

UNIT = "chapter_reviewer"


async def stub_chapter_reviewer_run(task: dict[str, Any]) -> dict[str, Any]:
    """chapter_reviewer 打桩：产出「零违规、通过」的确定性修订说明与自检。

    用户指令区逐字回带 revise 模式的用户意见原文（draft 无意见为空串），
    保证分区式修订说明的接口可空转、且逐字保留语义在打桩层即成立。
    """
    revision_note = RevisionNotePayload(
        user_directives=task.get("user_feedback", ""),
        rule_violations=[],
        conflict_hints=[],
        passed=True,
    )
    return {
        "revision_note": revision_note,
        "self_check": SelfCheckPayload(citations_ok=True, issues=[]),
    }


def make_stub_chapter_reviewer(event_hook: EventHook = noop_hook) -> SubagentAdapter:
    """构造 chapter_reviewer 打桩适配器。"""
    return SubagentAdapter(UNIT, stub_chapter_reviewer_run, event_hook)
