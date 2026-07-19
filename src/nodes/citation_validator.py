"""citation_validator 主节点：引文双层校验的第二层，全局终审门禁。

三步产生问题清单：
1. 纯程序对账（citation_reconciler，不调 LLM）；
2. 合并范围内章节的单章自检结果（rewriter_loop 产出的双层校验第一层）；
3. LLM 语义核查：逐章核对角标位置与素材观点是否对应。

核查范围：revised_chapter_ids 非空时增量核查（只重审这些章节），为空时全量核查。
LLM 语义核查各章互不依赖，并发执行（并发度由配置控制），问题清单按章节顺序合并。
终审失败递增 citation_retry_count；超过上限不再回退，携带未决引文警告
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
from assembly.context_assembler import assemble
from domain.env_config import read_positive_int
from llm.llm_client import LLM, LLMFactory
from llm.llm_json import JSON_ONLY_RULE, invoke_json
from domain.state import (
    ChapterDraft,
    CitationIssue,
    CitationReport,
    WorkflowStatus,
    WritingAgentState,
)

# 重试上限环境变量名与缺省值。
_MAX_RETRIES_ENV = "CITATION_MAX_RETRIES"
_MAX_RETRIES_DEFAULT = 2

# 语义核查章节并发度环境变量名与缺省值。
_MAX_CONCURRENT_ENV = "CITATION_MAX_CONCURRENT_CHAPTERS"
_MAX_CONCURRENT_DEFAULT = 4


@dataclass(frozen=True)
class ValidatorConfig:
    """引文终审最终生效的配置。"""

    max_retries: int
    """终审失败重试上限：超限不再回退，携未决警告交人工。"""

    max_concurrent_chapters: int = _MAX_CONCURRENT_DEFAULT
    """语义核查的章节并发度：各章核查互不依赖，并发发起 LLM 调用。"""


class CitationValidatorNode(Protocol):
    """节点函数类型：入参与返回均为图状态（state 具名，满足 LangGraph 节点协议）。"""

    def __call__(self, state: WritingAgentState) -> WritingAgentState: ...


def load_validator_config(env: Mapping[str, str] | None = None) -> ValidatorConfig:
    """读取引文终审配置：未设置或为空回落缺省值，非正整数抛 ValueError。"""
    if env is None:
        env = os.environ
    return ValidatorConfig(
        max_retries=read_positive_int(env, _MAX_RETRIES_ENV, _MAX_RETRIES_DEFAULT),
        max_concurrent_chapters=read_positive_int(
            env, _MAX_CONCURRENT_ENV, _MAX_CONCURRENT_DEFAULT
        ),
    )


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


def _normalize_semantic_payload(payload: Any) -> list[Any]:
    """把语义核查应答归一化为核查项列表。

    要求的形态是顶层数组；关思考后模型偶发把数组包进对象
    （如 {"results": [...]}）或直接返回单个核查项对象，兼容这两种偏差：
    - 顶层是数组：原样返回；
    - 顶层是单个核查项对象（含 material_id）：包成单元素列表；
    - 顶层对象恰含一个数组值：取该数组；
    其余形态视为无法归一化，抛含步骤名的 ValueError。
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("material_id"), str):
            return [payload]
        list_values = [value for value in payload.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return list_values[0]
    raise ValueError(
        "步骤「引文语义核查」的 LLM 应答无法归一化为核查项数组："
        f"{json.dumps(payload, ensure_ascii=False)[:200]}"
    )


def _semantic_check_chapter(
    llm: LLM,
    chapter_id: str,
    chapter_text: str,
    cited: list[dict[str, Any]],
) -> list[CitationIssue]:
    """步骤 3：单章一次 LLM 调用，核查每处角标位置与素材观点是否对应。

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
    items = _normalize_semantic_payload(payload)

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


def _ordered_failed_chapter_ids(
    issues: list[CitationIssue], outline_order: dict[str, int]
) -> list[str]:
    """问题所在章节去重，按大纲顺序排列（unused_material 已归其素材章节）。"""
    chapter_ids = list(dict.fromkeys(issue.chapter_id for issue in issues))
    return sorted(
        chapter_ids, key=lambda chapter_id: outline_order.get(chapter_id, len(outline_order))
    )


def make_citation_validator_node(
    llm_factory: LLMFactory,
    config: ValidatorConfig | None = None,
    assembler_config: AssemblerConfig | None = None,
) -> CitationValidatorNode:
    """构造 citation_validator 节点函数。

    config 为 None 时在节点执行时读取环境变量 CITATION_MAX_RETRIES（缺省 2）；
    assembler_config 为 None 时在节点执行时读取环境变量装配配置。
    """

    def node(state: WritingAgentState) -> WritingAgentState:
        effective_config = config if config is not None else load_validator_config()
        effective_max_retries = effective_config.max_retries
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
        # 步骤 3：逐章 LLM 语义核查；正文与被引素材经装配段取得，
        # 该章没有任何角标素材时跳过调用。各章核查互不依赖，并发执行，
        # 问题清单按 scoped_drafts 顺序合并保持确定性。
        def check_chapter(draft: ChapterDraft) -> list[CitationIssue]:
            """单章语义核查（供并发执行）。"""
            context = assemble(
                state,
                "citation_validator",
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
