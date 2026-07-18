"""citation_validator 主节点：引文双层校验的第二层，全局终审门禁。

三步产生问题清单：
1. 纯程序对账（citation_reconciler，不调 LLM）；
2. 合并范围内章节的单章自检结果（rewriter_loop 产出的双层校验第一层）；
3. LLM 语义核查：逐章核对角标位置与素材观点是否对应。

核查范围：revised_chapter_ids 非空时增量核查（只重审这些章节），为空时全量核查。
终审失败递增 citation_retry_count；超过上限不再回退，携带未决引文警告
交人工裁决（回退路由在 graph.py，不在本节点）。
"""

import json
import os
from collections.abc import Mapping
from typing import Protocol

from citation_reconciler import MARKER_PATTERN, reconcile
from env_config import read_positive_int
from llm_client import LLM, LLMFactory
from llm_json import JSON_ONLY_RULE, invoke_json
from state import (
    ChapterDraft,
    CitationIssue,
    CitationReport,
    Material,
    WorkflowStatus,
    WritingAgentState,
)

# 重试上限环境变量名与缺省值。
_MAX_RETRIES_ENV = "CITATION_MAX_RETRIES"
_MAX_RETRIES_DEFAULT = 2


class CitationValidatorNode(Protocol):
    """节点函数类型：入参与返回均为图状态（state 具名，满足 LangGraph 节点协议）。"""

    def __call__(self, state: WritingAgentState) -> WritingAgentState: ...


def load_citation_max_retries(env: Mapping[str, str] | None = None) -> int:
    """读取终审失败重试上限：未设置或为空回落缺省值，非正整数抛 ValueError。"""
    if env is None:
        env = os.environ
    return read_positive_int(env, _MAX_RETRIES_ENV, _MAX_RETRIES_DEFAULT)


def _self_check_issues(drafts: list[ChapterDraft]) -> list[CitationIssue]:
    """步骤 2：把范围内章节的单章自检失败合并为 self_check_failed 问题。"""
    issues: list[CitationIssue] = []
    for draft in drafts:
        if draft.self_check.citations_ok:
            continue
        details = draft.self_check.issues or ["单章自检未通过且未给出具体问题。"]
        for detail in details:
            issues.append(
                CitationIssue(
                    kind="self_check_failed",
                    chapter_id=draft.chapter_id,
                    material_id="",
                    detail=detail,
                )
            )
    return issues


def _semantic_check_chapter(
    llm: LLM, draft: ChapterDraft, cited: list[Material]
) -> list[CitationIssue]:
    """步骤 3：单章一次 LLM 调用，核查每处角标位置与素材观点是否对应。"""
    cited_payload = [
        {"id": material.id, "excerpt": material.excerpt} for material in cited
    ]
    system = (
        "你是引文语义核查器。章节正文中形如 [素材id] 的角标是引文标注，"
        "逐条判断每个被引素材的摘录与其角标所在位置的观点是否对应。"
        "输出 JSON 数组，逐素材一项："
        '{"material_id": "素材id", "aligned": true|false, "reason": "一句话理由"}。'
        + JSON_ONLY_RULE
    )
    user = (
        f"章节 {draft.chapter_id} 正文：\n{draft.text}\n\n"
        f"该章被引素材：\n{json.dumps(cited_payload, ensure_ascii=False, indent=2)}"
    )
    payload = invoke_json(llm, "引文语义核查", system, user, list)

    cited_ids = {material.id for material in cited}
    issues: list[CitationIssue] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        material_id = item.get("material_id")
        # 素材 id 不在该章被引集合内的应答项（LLM 幻觉）直接丢弃。
        if not isinstance(material_id, str) or material_id not in cited_ids:
            continue
        if item.get("aligned") is not False:
            continue
        reason = item.get("reason")
        reason_text = reason.strip() if isinstance(reason, str) and reason.strip() else "未给出理由"
        issues.append(
            CitationIssue(
                kind="semantic_mismatch",
                chapter_id=draft.chapter_id,
                material_id=material_id,
                detail=f"素材 {material_id} 的标注位置与观点不对应：{reason_text}",
            )
        )
    return issues


def _ordered_failed_chapter_ids(
    issues: list[CitationIssue], outline_order: dict[str, int]
) -> list[str]:
    """问题所在章节去重，按大纲顺序排列（unused_material 已归其素材章节）。"""
    chapter_ids = list(dict.fromkeys(issue.chapter_id for issue in issues))
    return sorted(
        chapter_ids, key=lambda chapter_id: outline_order.get(chapter_id, len(outline_order))
    )


def make_citation_validator_node(
    llm_factory: LLMFactory, max_retries: int | None = None
) -> CitationValidatorNode:
    """构造 citation_validator 节点函数。

    max_retries 为 None 时在节点执行时读取环境变量 CITATION_MAX_RETRIES（缺省 2）。
    """

    def node(state: WritingAgentState) -> WritingAgentState:
        effective_max_retries = (
            max_retries if max_retries is not None else load_citation_max_retries()
        )
        llm = llm_factory("citation_validator")
        drafts = state.get("chapter_drafts", [])
        library = state.get("citation_library", [])
        revised_chapter_ids = state.get("revised_chapter_ids", [])
        scope = set(revised_chapter_ids) if revised_chapter_ids else None
        scoped_drafts = [
            draft for draft in drafts if scope is None or draft.chapter_id in scope
        ]

        # 步骤 1：纯程序对账。
        issues = reconcile(drafts, library, scope)
        # 步骤 2：单章自检合并。
        issues += _self_check_issues(scoped_drafts)
        # 步骤 3：逐章 LLM 语义核查；该章没有任何角标素材时跳过调用。
        materials_by_id = {material.id: material for material in library}
        for draft in scoped_drafts:
            cited = [
                materials_by_id[marker]
                for marker in dict.fromkeys(MARKER_PATTERN.findall(draft.text))
                if marker in materials_by_id
            ]
            if not cited:
                continue
            issues += _semantic_check_chapter(llm, draft, cited)

        llm_config = {"unit": "citation_validator", **llm.metadata}
        if not issues:
            # 终审通过即将进入人工中断点：中断暂停在 gate 节点内部、其更新
            # 尚未提交，等待人工期间对外可见的状态机值由本节点写入。
            return WritingAgentState(
                citation_report=CitationReport(passed=True),
                citation_retry_count=0,
                citation_warnings=[],
                revised_chapter_ids=[],
                status=WorkflowStatus.AWAIT_USER_REVIEW,
                current_node_llm_config=llm_config,
            )

        outline_order = {
            chapter.id: index for index, chapter in enumerate(state.get("outline", []))
        }
        retry = state.get("citation_retry_count", 0) + 1
        # 超限不再回退：携带未决引文警告交人工裁决（状态机同样置为等待人工）。
        exhausted = retry > effective_max_retries
        warnings = (
            [f"未决引文问题（{issue.kind}）：{issue.detail}" for issue in issues]
            if exhausted
            else []
        )
        return WritingAgentState(
            citation_report=CitationReport(
                passed=False,
                issues=issues,
                failed_chapter_ids=_ordered_failed_chapter_ids(issues, outline_order),
            ),
            citation_retry_count=retry,
            citation_warnings=warnings,
            revised_chapter_ids=[],
            status=(
                WorkflowStatus.AWAIT_USER_REVIEW
                if exhausted
                else WorkflowStatus.CITATION_CHECKING
            ),
            current_node_llm_config=llm_config,
        )

    return node
