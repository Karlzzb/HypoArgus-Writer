"""document_reviewer 主节点：篇级终审门禁，双层评审视野的上层。

评审视野切分（与 chapter_reviewer 章级评审互补）：章级评审只管单章内部质量，
篇级终审只裁「必须看全篇才能判」的维度，两级不重叠。

产生问题清单的步骤：
1. 纯程序对账（citation_reconciler，不调 LLM）；
2. 结构完整性（纯程序）：章节编号连续唯一校验 + 大纲章节缺稿检查，始终全量；
3. 合并范围内章节的单章自检结果（rewriter_loop 产出的双层校验第一层）；
4. LLM 引文语义核查：逐章核对角标位置与素材观点是否对应；
5. LLM 篇级评审：一次调用全篇评四维——跨章硬事实冲突（error 打回）、
   章间衔接 / 口径统一 / 跨章重复（warn 呈人工，不打回）。

严重级判定权在代码不在模型：模型只给维度与线索，error/warn 归属由代码固定。
结构完整性与跨章硬事实冲突为 error，进 issues/failed_chapter_ids 触发定向回退；
章间衔接 / 口径统一 / 跨章重复为 warn，进 review_warnings 每轮呈人工——warn 不打回
是刻意设计：篇级 warn 往往牵连多章，若打回易触发多章连锁重写雪崩。

核查范围：引用四步中 revised_chapter_ids 非空时增量核查（只重审这些章节），为空时
全量；结构完整性与篇级评审始终全量（是全文属性，增量检查无意义）。
LLM 引文语义核查各章互不依赖，并发执行（并发度由配置控制）。
终审失败递增 citation_retry_count；超过上限不再回退，携带未决终审警告
交人工裁决（回退路由在 graph.py，不在本节点）。
"""

import contextvars
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from assembly.assembler_config import AssemblerConfig
from domain.citation_reconciler import reconcile
from domain.chapter_numbering_validator import validate_chapter_numbering
from assembly.context_assembler import assemble
from domain.env_config import read_positive_int
from llm.llm_client import LLM, LLMFactory
from llm.llm_json import JSON_ONLY_RULE, invoke_json
from domain.state import (
    ChapterDraft,
    ChapterSpec,
    CitationIssue,
    CitationReport,
    WorkflowStatus,
    WritingAgentState,
)

# 重试上限环境变量名与缺省值。
_MAX_RETRIES_ENV = "DOCUMENT_REVIEW_MAX_RETRIES"
_MAX_RETRIES_DEFAULT = 2

# 引文语义核查章节并发度环境变量名与缺省值。
_MAX_CONCURRENT_ENV = "DOCUMENT_REVIEW_MAX_CONCURRENT_CHAPTERS"
_MAX_CONCURRENT_DEFAULT = 4

# 篇级评审四维 → 严重级与维度中文名；严重级判定权在代码，模型不参与。
_REVIEW_DIMENSIONS: dict[str, tuple[str, str]] = {
    "fact_conflict": ("error", "跨章硬事实冲突"),
    "transition": ("warn", "章间衔接"),
    "consistency": ("warn", "口径统一"),
    "duplication": ("warn", "跨章重复"),
}


@dataclass(frozen=True)
class ReviewerConfig:
    """篇级终审最终生效的配置。"""

    max_retries: int
    """终审失败重试上限：超限不再回退，携未决警告交人工。"""

    max_concurrent_chapters: int = _MAX_CONCURRENT_DEFAULT
    """引文语义核查的章节并发度：各章核查互不依赖，并发发起 LLM 调用。"""


class DocumentReviewerNode(Protocol):
    """节点函数类型：入参与返回均为图状态（state 具名，满足 LangGraph 节点协议）。"""

    def __call__(self, state: WritingAgentState) -> WritingAgentState: ...


def load_reviewer_config(env: Mapping[str, str] | None = None) -> ReviewerConfig:
    """读取篇级终审配置：未设置或为空回落缺省值，非正整数抛 ValueError。"""
    if env is None:
        env = os.environ
    return ReviewerConfig(
        max_retries=read_positive_int(env, _MAX_RETRIES_ENV, _MAX_RETRIES_DEFAULT),
        max_concurrent_chapters=read_positive_int(
            env, _MAX_CONCURRENT_ENV, _MAX_CONCURRENT_DEFAULT
        ),
    )


def _structure_issues(
    drafts: list[ChapterDraft], outline: list[ChapterSpec]
) -> list[CitationIssue]:
    """步骤 2：结构完整性校验（始终全量，是全文属性）。

    两项确定性检查：
    - 章节编号连续唯一（validate_chapter_numbering）；
    - 大纲章节缺稿：编号校验只遍历成稿，缺稿章不进其视野，须在此单独补检。
    """
    numbering_issues = validate_chapter_numbering(drafts, outline)
    issues = [
        CitationIssue(
            kind="numbering_broken",
            chapter_id=issue.chapter_id,
            material_id="",
            detail=issue.message,
        )
        for issue in numbering_issues
    ]
    drafted = {draft.chapter_id for draft in drafts}
    for chapter in outline:
        if chapter.id not in drafted:
            issues.append(
                CitationIssue(
                    kind="chapter_missing",
                    chapter_id=chapter.id,
                    material_id="",
                    detail="大纲章节缺少成稿。",
                )
            )
    return issues


def _self_check_issues(drafts: list[ChapterDraft]) -> list[CitationIssue]:
    """步骤 3：把范围内章节的单章自检失败合并为 self_check_failed 问题。"""
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


def _normalize_payload(payload: Any, item_key: str, step_name: str) -> list[Any]:
    """把 LLM 应答归一化为条目列表（引文语义核查与篇级评审共用）。

    要求的形态是顶层数组；开思考后模型偶发把数组包进对象
    （如 {"results": [...]}）或直接返回单个条目对象，兼容这两种偏差：
    - 顶层是数组：原样返回；
    - 顶层是单个条目对象（含判别键 item_key）：包成单元素列表；
    - 顶层对象恰含一个数组值：取该数组；
    其余形态视为无法归一化，抛含步骤名的 ValueError。
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get(item_key), str):
            return [payload]
        list_values = [value for value in payload.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return list_values[0]
    raise ValueError(
        f"步骤「{step_name}」的 LLM 应答无法归一化为条目数组："
        f"{json.dumps(payload, ensure_ascii=False)[:200]}"
    )


def _semantic_check_chapter(
    llm: LLM,
    chapter_id: str,
    chapter_text: str,
    cited: list[dict[str, Any]],
) -> list[CitationIssue]:
    """步骤 4：单章一次 LLM 调用，核查每处角标位置与素材观点是否对应。

    正文与被引素材（含 id/excerpt）均取自装配后的 chapter_text、cited_materials 段。
    """
    system = (
        "你是引文语义核查器。章节正文中形如 [素材id] 的角标是引文标注，"
        "逐条判断每个被引素材的摘录与其角标所在位置的观点是否对应。"
        "输出 JSON 数组，逐素材一项："
        '{"material_id": "素材id", "aligned": true|false, "reason": "一句话理由"}。'
        + JSON_ONLY_RULE
    )
    user = (
        f"章节 {chapter_id} 正文：\n{chapter_text}\n\n"
        f"该章被引素材：\n{json.dumps(cited, ensure_ascii=False, indent=2)}"
    )
    payload = invoke_json(llm, "引文语义核查", system, user, (list, dict))
    items = _normalize_payload(payload, "material_id", "引文语义核查")

    cited_ids = {material["id"] for material in cited}
    issues: list[CitationIssue] = []
    for item in items:
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
                chapter_id=chapter_id,
                material_id=material_id,
                detail=f"素材 {material_id} 的标注位置与观点不对应：{reason_text}",
            )
        )
    return issues


def _document_review(
    llm: LLM, document_text: str, outline_ids: set[str]
) -> tuple[list[CitationIssue], list[str]]:
    """步骤 5：一次 LLM 调用做全篇四维评审，返回（error 问题清单, warn 提示串）。

    严重级判定权在代码：模型只报维度与涉及章节，error（fact_conflict）与
    warn（其余三维）归属由 _REVIEW_DIMENSIONS 固定。幻觉防护：未知维度直接丢；
    涉及章节 id 不在大纲中的先剔除，剔空后整条丢弃。
    """
    system = (
        "你是篇级评审器，只评「必须看全篇才能判」的四个维度，逐条严格对照判定标准，"
        "拿不准一律不报：\n"
        "- fact_conflict 跨章硬事实冲突：仅当两章对同一事实给出确定矛盾的陈述"
        "（数字、日期、结论相反）才报；\n"
        "- transition 章间衔接：相邻章节承接生硬或断裂；\n"
        "- consistency 口径统一：同一概念/指标在不同章节口径或术语不一致；\n"
        "- duplication 跨章重复：不同章节大段重复论述。\n"
        "输出 JSON 数组，逐发现一项："
        '{"dimension": "fact_conflict|transition|consistency|duplication", '
        '"chapter_ids": ["涉及章节id"], "detail": "一句话说明"}。'
        "无任何发现输出 []。" + JSON_ONLY_RULE
    )
    user = f"全篇正文（按章）：\n{document_text}"
    payload = invoke_json(llm, "篇级评审", system, user, (list, dict))
    items = _normalize_payload(payload, "dimension", "篇级评审")

    error_issues: list[CitationIssue] = []
    warnings: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        dimension = item.get("dimension")
        if not isinstance(dimension, str) or dimension not in _REVIEW_DIMENSIONS:
            continue
        raw_ids = item.get("chapter_ids")
        chapter_ids = [
            chapter_id
            for chapter_id in (raw_ids if isinstance(raw_ids, list) else [])
            if isinstance(chapter_id, str) and chapter_id in outline_ids
        ]
        if not chapter_ids:
            continue
        detail_raw = item.get("detail")
        detail = detail_raw.strip() if isinstance(detail_raw, str) and detail_raw.strip() else "未给出说明"
        severity, dimension_cn = _REVIEW_DIMENSIONS[dimension]
        ids_text = "、".join(chapter_ids)
        if severity == "error":
            # 冲突牵涉多章时逐章问题都点名全部涉及章节，让重写侧看得到对方章。
            for chapter_id in chapter_ids:
                error_issues.append(
                    CitationIssue(
                        kind="fact_conflict",
                        chapter_id=chapter_id,
                        material_id="",
                        detail=f"跨章硬事实冲突（涉及章节 {ids_text}）：{detail}",
                    )
                )
        else:
            warnings.append(
                f"篇级评审提示（{dimension_cn}，涉及章节 {ids_text}）：{detail}"
            )
    return error_issues, warnings


def _ordered_failed_chapter_ids(
    issues: list[CitationIssue], outline_order: dict[str, int]
) -> list[str]:
    """问题所在章节去重，按大纲顺序排列（unused_material 已归其素材章节）。"""
    chapter_ids = list(dict.fromkeys(issue.chapter_id for issue in issues))
    return sorted(
        chapter_ids, key=lambda chapter_id: outline_order.get(chapter_id, len(outline_order))
    )


def make_document_reviewer_node(
    llm_factory: LLMFactory,
    config: ReviewerConfig | None = None,
    assembler_config: AssemblerConfig | None = None,
) -> DocumentReviewerNode:
    """构造 document_reviewer 节点函数。

    config 为 None 时在节点执行时读取环境变量 DOCUMENT_REVIEW_MAX_RETRIES（缺省 2）；
    assembler_config 为 None 时在节点执行时读取环境变量装配配置。
    """

    def node(state: WritingAgentState) -> WritingAgentState:
        effective_config = config if config is not None else load_reviewer_config()
        effective_max_retries = effective_config.max_retries
        llm = llm_factory("document_reviewer")
        drafts = state.get("chapter_drafts", [])
        library = state.get("citation_library", [])
        outline = state.get("outline", [])
        revised_chapter_ids = state.get("revised_chapter_ids", [])
        scope = set(revised_chapter_ids) if revised_chapter_ids else None
        scoped_drafts = [
            draft for draft in drafts if scope is None or draft.chapter_id in scope
        ]

        # 步骤 1：纯程序对账。
        issues = reconcile(drafts, library, scope)
        # 步骤 2：结构完整性（编号连续 + 缺稿，始终全量，是全文属性）。
        issues += _structure_issues(drafts, outline)
        # 步骤 3：单章自检合并。
        issues += _self_check_issues(scoped_drafts)
        # 步骤 4：逐章 LLM 引文语义核查；正文与被引素材经装配段取得，
        # 该章没有任何角标素材时跳过调用。各章核查互不依赖，并发执行，
        # 问题清单按 scoped_drafts 顺序合并保持确定性。
        def check_chapter(draft: ChapterDraft) -> list[CitationIssue]:
            """单章语义核查（供并发执行）。"""
            context = assemble(
                state,
                "document_reviewer",
                config=assembler_config,
                chapter_id=draft.chapter_id,
            )
            cited: list[dict[str, Any]] = json.loads(
                context.text("cited_materials", "[]")
            )
            if not cited:
                return []
            return _semantic_check_chapter(
                llm, draft.chapter_id, context.text("chapter_text"), cited
            )

        if scoped_drafts:
            max_workers = min(
                len(scoped_drafts), effective_config.max_concurrent_chapters
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 每个章节任务复制当前 contextvars 上下文执行，
                # 保证 Langfuse span 父子关系跨工作线程成立。
                futures = [
                    executor.submit(
                        contextvars.copy_context().run, check_chapter, draft
                    )
                    for draft in scoped_drafts
                ]
                for future in futures:
                    issues += future.result()

        # 步骤 5：一次 LLM 调用做全篇四维评审（始终全量）；error 并入问题清单驱动
        # 打回，warn 单列 review_warnings 每轮呈人工、不打回。全篇无草稿时跳过。
        review_warnings: list[str] = []
        document_text = assemble(
            state, "document_reviewer", config=assembler_config
        ).text("document_text")
        if document_text:
            outline_ids = {chapter.id for chapter in outline}
            review_errors, review_warnings = _document_review(
                llm, document_text, outline_ids
            )
            issues += review_errors

        llm_config = {"unit": "document_reviewer", **llm.metadata}
        if not issues:
            # 终审通过即将进入人工中断点：中断暂停在 gate 节点内部、其更新
            # 尚未提交，等待人工期间对外可见的状态机值由本节点写入。
            return WritingAgentState(
                citation_report=CitationReport(passed=True),
                citation_retry_count=0,
                citation_warnings=[],
                review_warnings=review_warnings,
                revised_chapter_ids=[],
                status=WorkflowStatus.AWAIT_USER_REVIEW,
                current_node_llm_config=llm_config,
            )

        outline_order = {
            chapter.id: index for index, chapter in enumerate(outline)
        }
        retry = state.get("citation_retry_count", 0) + 1
        # 超限不再回退：携带未决终审警告交人工裁决（状态机同样置为等待人工）；
        # 问题清单不止引文类，还含 fact_conflict/chapter_missing/numbering_broken。
        exhausted = retry > effective_max_retries
        warnings = (
            [f"未决终审问题（{issue.kind}）：{issue.detail}" for issue in issues]
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
            review_warnings=review_warnings,
            revised_chapter_ids=[],
            status=(
                WorkflowStatus.AWAIT_USER_REVIEW
                if exhausted
                else WorkflowStatus.CITATION_CHECKING
            ),
            current_node_llm_config=llm_config,
        )

    return node
