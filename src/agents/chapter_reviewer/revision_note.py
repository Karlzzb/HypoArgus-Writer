"""revision_note：分区式修订说明的纯函数装配（无 LLM、无副作用）。

把确定性 lint 违规与四维 LLM 自审违规折成同形规则违规条目，连同逐字保留的
用户指令区与冲突提示区，装配成 ``RevisionNotePayload``。装配是纯函数，供单测
断言「用户指令逐字零改写」「error/warn 定级」「error 级为空即过」等口径。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agents.chapter_reviewer.review_client import ReviewConflict, ReviewIssue
from agents.contracts import (
    ConflictHintEntry,
    RevisionNotePayload,
    RuleViolationEntry,
)
from agents.rewriter_loop.style_linter import AUDIT_RULE_PREFIX, Violation


def _norm_severity(severity: str) -> str:
    """定级归一：非 warn 一律视为 error（缺省从严），保 Literal 口径。"""
    return "warn" if severity == "warn" else "error"


def _lint_to_entry(violation: Violation) -> RuleViolationEntry:
    """确定性 lint 违规折成规则违规条目：位置摘录留空（lint 不总能定位片段）。"""
    return RuleViolationEntry(
        rule=violation.rule,
        location_excerpt="",
        guidance=violation.message,
        severity=_norm_severity(violation.severity),  # type: ignore[typeddict-item]
    )


def _issue_to_entry(issue: ReviewIssue, severity: str) -> RuleViolationEntry:
    """四维自审违规折成规则违规条目：规则名 self_audit_<item>，定级取裁决项配置。

    定级权威来自裁决项配置（AuditItem.severity），模型不裁定 severity——
    保证「各文种开关定级」由 ssot-config 单一事实源掌控，不受模型漂移影响。
    """
    return RuleViolationEntry(
        rule=f"{AUDIT_RULE_PREFIX}{issue.item}",
        location_excerpt=issue.excerpt,
        guidance=issue.guidance or f"裁决项「{issue.item}」判定违规",
        severity=_norm_severity(severity),  # type: ignore[typeddict-item]
    )


def assemble_revision_note(
    user_feedback: str,
    lint_violations: Sequence[Violation],
    review_issues: Sequence[ReviewIssue],
    severity_by_item: Mapping[str, str],
    conflicts: Sequence[ReviewConflict],
) -> RevisionNotePayload:
    """装配分区式修订说明。

    - 用户指令区：``user_feedback`` **逐字保留**、零改写（评审绝不重写用户意见）。
    - 规则违规区：确定性 lint 违规 + 四维自审违规，各带位置摘录、修改指导与定级。
    - 冲突提示区：模型给出的「规则违规与用户指令冲突」提示（用户指令优先）。
    - passed：error 级违规为空即过（warn 级不阻断）。
    """
    rule_violations: list[RuleViolationEntry] = [
        _lint_to_entry(v) for v in lint_violations
    ]
    rule_violations.extend(
        _issue_to_entry(issue, severity_by_item.get(issue.item, "error"))
        for issue in review_issues
    )
    conflict_hints: list[ConflictHintEntry] = [
        ConflictHintEntry(description=c.description) for c in conflicts
    ]
    passed = not any(entry["severity"] == "error" for entry in rule_violations)
    return RevisionNotePayload(
        user_directives=user_feedback,
        rule_violations=rule_violations,
        conflict_hints=conflict_hints,
        passed=passed,
    )
