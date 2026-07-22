"""分区式修订说明纯装配函数单测：重点断言用户指令逐字零改写保留。

装配是纯函数（无 LLM、无副作用），断言四区口径：用户指令逐字保留、
规则违规区按 lint + 自审合并且定级正确、冲突提示原样进区、passed = error 级为空。
"""

from agents.chapter_reviewer import ReviewConflict, ReviewIssue, assemble_revision_note
from agents.rewriter_loop.style_linter import Violation


def test_装配_用户指令区逐字零改写保留() -> None:
    # 含标点、换行、特殊语气的原文：装配后须逐字一致，不做任何改写/裁剪/规范化。
    user_feedback = "请务必保留第三段的原句：“稳中求进”。\n另：第一段不要动。"

    note = assemble_revision_note(
        user_feedback,
        lint_violations=[],
        review_issues=[],
        severity_by_item={},
        conflicts=[],
    )

    assert note["user_directives"] == user_feedback


def test_装配_lint与自审合并且定级正确_passed按error判定() -> None:
    lint_violations = [
        Violation(rule="unknown_material_marker", message="正文出现池外角标 [m-x]"),
        Violation(rule="intra_chapter_coherence", message="警示级示意", severity="warn"),
    ]
    review_issues = [
        ReviewIssue(item="summary_chain_consistency", excerpt="衔接偏离", guidance="对齐上一章摘要"),
        ReviewIssue(item="weak_material_assertion", excerpt="弱素材写成断言", guidance="降口径"),
    ]
    severity_by_item = {
        "summary_chain_consistency": "warn",
        "weak_material_assertion": "error",
    }

    note = assemble_revision_note(
        "改一改", lint_violations, review_issues, severity_by_item, [ReviewConflict(description="与用户意见冲突")]
    )

    rows = {e["rule"]: e for e in note["rule_violations"]}
    # 确定性 lint 违规：severity 取 Violation.severity；位置摘录留空。
    assert rows["unknown_material_marker"]["severity"] == "error"
    assert rows["unknown_material_marker"]["location_excerpt"] == ""
    assert rows["intra_chapter_coherence"]["severity"] == "warn"
    # 四维自审违规：规则名 self_audit_<item>，定级取配置、位置摘录取片段、指导取 guidance。
    assert rows["self_audit_summary_chain_consistency"]["severity"] == "warn"
    assert rows["self_audit_summary_chain_consistency"]["location_excerpt"] == "衔接偏离"
    assert rows["self_audit_weak_material_assertion"]["severity"] == "error"
    assert rows["self_audit_weak_material_assertion"]["guidance"] == "降口径"
    # 冲突提示原样进区。
    assert note["conflict_hints"] == [{"description": "与用户意见冲突"}]
    # 存在 error 级违规 → 不过。
    assert note["passed"] is False


def test_装配_空裁决合法_passed为真() -> None:
    note = assemble_revision_note(
        "", lint_violations=[], review_issues=[], severity_by_item={}, conflicts=[]
    )

    assert note == {
        "user_directives": "",
        "rule_violations": [],
        "conflict_hints": [],
        "passed": True,
    }


def test_装配_仅warn级违规_仍判过() -> None:
    note = assemble_revision_note(
        "",
        lint_violations=[Violation(rule="intra_chapter_coherence", message="warn", severity="warn")],
        review_issues=[ReviewIssue(item="summary_chain_consistency", guidance="对齐")],
        severity_by_item={"summary_chain_consistency": "warn"},
        conflicts=[],
    )

    # warn 级不阻断：error 级为空即过。
    assert note["passed"] is True
    assert len(note["rule_violations"]) == 2
