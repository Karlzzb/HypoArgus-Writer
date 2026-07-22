"""llm_adapter：评审 LLM 注入点的真实适配器（单次调用，JSON-in-text）。

只依赖注入的 ``llm.invoke(messages) -> str``（``llm.llm_client.LLM`` 协议），
不触碰任何 SDK、API key 或环境变量。四维自审裁决项按任务包文种经
``audit_items_for`` 加载（与 rewriter 自审同源同机制），system 提示词逐任务拼装。

评审永不阻断主链：``issues: []``/``conflicts: []`` 是合法非退化结果（不重试）；
异常 / 解析失败 / 结构非法 → 重试至 ``max_attempts``，耗尽降级为空裁决
（``degraded=True``）；非法条目防御性丢弃。单次评审只发一次 LLM 调用（single-shot）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from agents.chapter_reviewer.review_client import (
    ReviewConflict,
    ReviewEnvelope,
    ReviewIssue,
)
from agents.contracts import MaterialPayload
from agents.rewriter_loop.style_linter import AuditItem, audit_items_for
from agents.rewriter_loop.writer_client import citable_materials
from domain.doc_types import carried_doc_facts
from llm.llm_client import LLM
from llm.llm_json import JSON_ONLY_RULE, parse_json

logger = logging.getLogger(__name__)

_REVIEW_TAG = "【章节评审】"


def build_review_system(items: Sequence[AuditItem]) -> str:
    """按适用裁决项拼装评审 system 提示词（裁决项按 doc_type 分派，ADR-0005/0006）。

    裁决项判定准则来自 ssot-config ``audit_items``（与 lint 同源）；本函数只
    负责固定框架：角色、逐项裁决口径、冲突提示口径与输出 JSON 契约。
    模型只判违规与冲突、给位置摘录与修改指导，不裁定 severity（定级由配置权威赋予）。
    """
    blocks = "\n".join(
        f"{idx}. 【{item.label}】（item={item.id}）判定准则：\n{item.criteria}"
        for idx, item in enumerate(items, 1)
    )
    return (
        "你是章节评审员，按下列裁决项逐项判断本章正文是否违规，只裁决下列各项，不扩大范围。\n\n"
        f"裁决项：\n{blocks}\n\n"
        "另需给出冲突提示：若某条规则违规的修改会与「用户意见」相抵触，"
        "列入 conflicts（用户意见优先，评审只提示不代改）；无用户意见或无冲突则为空数组。\n\n"
        "通用准则：无违规时 issues 返回空数组，不要臆造违规；不裁定 severity 定级。\n\n"
        "输出为一个 JSON 对象，字段如下：\n"
        "- issues：每条含 item（上列裁决项 id 之一）、excerpt（正文违规位置片段）、"
        "guidance（修改指导）；无违规时为空数组。\n"
        "- conflicts：每条含 description（冲突说明）；无冲突时为空数组。\n"
    ) + JSON_ONLY_RULE


def _format_materials(materials: Sequence[MaterialPayload]) -> str:
    if not materials:
        return "（无）"
    return "\n".join(
        f"- {m['id']}（支撑假说 {m['hypothesis_id']}，来源：{m['source']}）：{m['excerpt']}"
        for m in materials
    )


def build_review_user(task: dict[str, Any]) -> str:
    """评审 user 提示词：给素材池、上一章摘要、用户意见（revise）与本章正文。"""
    prev = task.get("prev_chapter_summary") or "（首章，无上一章摘要）"
    user_feedback = task.get("user_feedback") or "（无用户意见）"
    return (
        f"{_REVIEW_TAG}按 system 中的裁决项逐项判断下面的本章正文。\n"
        f"上一章摘要：{prev}\n"
        f"用户意见（如有，冲突时用户意见优先）：{user_feedback}\n"
        f"素材池（仅可引用池内 id）：\n{_format_materials(citable_materials(task))}\n\n"
        f"本章正文：\n{task['chapter_text']}\n\n"
        "判断并返回 issues 与 conflicts（无则为空数组，不要臆造）。"
    )


class LlmReviewClient:
    """评审 LLM 注入点的真实适配器：纯文本 JSON-in-text 单次调用注入的 LLM 协议。"""

    def __init__(self, llm: LLM, *, max_attempts: int = 3) -> None:
        self._llm = llm
        self._max_attempts = max_attempts

    def review(self, task: dict[str, Any]) -> ReviewEnvelope:
        """单次评审自审；``issues``/``conflicts`` 空为合法非退化（不重试）。

        裁决项按任务包文种加载并按素材池适用性过滤（与编排层跳过口径同源）。
        异常 / 解析失败 / 结构非法 → 重试；耗尽 → 空裁决 ``degraded=True``——
        评审永不阻断主链。非法条目（item 不在适用集、结构缺字段）防御性丢弃。
        """
        doc_type, _ = carried_doc_facts(task)
        items = audit_items_for(doc_type, has_materials=bool(citable_materials(task)))
        if not items:
            # 无适用裁决项（编排层通常已跳过）：防御性返回合法空裁决，不发无意义调用。
            return ReviewEnvelope()
        by_id: dict[str, AuditItem] = {item.id: item for item in items}
        messages = [
            {"role": "system", "content": build_review_system(items)},
            {"role": "user", "content": build_review_user(task)},
        ]
        for attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._llm.invoke(messages)
                payload = parse_json(raw, "review")
            except Exception as exc:
                logger.warning(
                    "chapter_reviewer 尝试 %d/%d 退化（异常）：%s: %s",
                    attempt, self._max_attempts, type(exc).__name__, exc,
                )
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("issues"), list):
                logger.warning(
                    "chapter_reviewer 尝试 %d/%d 退化：应答结构非法", attempt, self._max_attempts
                )
                continue
            issues = _parse_issues(payload["issues"], by_id)
            conflicts = _parse_conflicts(payload.get("conflicts"))
            return ReviewEnvelope(issues=issues, conflicts=conflicts, attempts=attempt)
        logger.warning(
            "chapter_reviewer 全部 %d 次尝试退化；降级为空裁决（评审不阻断）",
            self._max_attempts,
        )
        return ReviewEnvelope(attempts=self._max_attempts, degraded=True)


def _parse_issues(
    raw_issues: list[Any], by_id: dict[str, AuditItem]
) -> list[ReviewIssue]:
    """解析自审违规：裁决项不明或结构非法的条目防御性丢弃（不整体退化）。"""
    issues: list[ReviewIssue] = []
    for entry in raw_issues:
        if not isinstance(entry, dict):
            logger.warning("chapter_reviewer 丢弃非法自审条目：%r", entry)
            continue
        item_id = entry.get("item")
        if not isinstance(item_id, str) or item_id not in by_id:
            logger.warning("chapter_reviewer 丢弃裁决项不明的自审条目：%r", entry)
            continue
        excerpt = entry.get("excerpt")
        guidance = entry.get("guidance")
        issues.append(
            ReviewIssue(
                item=item_id,
                excerpt=excerpt if isinstance(excerpt, str) else "",
                guidance=guidance if isinstance(guidance, str) else "",
            )
        )
    return issues


def _parse_conflicts(raw_conflicts: Any) -> list[ReviewConflict]:
    """解析冲突提示：非列表或非法条目防御性忽略。"""
    if not isinstance(raw_conflicts, list):
        return []
    conflicts: list[ReviewConflict] = []
    for entry in raw_conflicts:
        if isinstance(entry, dict) and isinstance(entry.get("description"), str):
            conflicts.append(ReviewConflict(description=entry["description"]))
    return conflicts
