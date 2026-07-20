"""writer：rewriter_loop 真实现的写作编排与工厂。

黑盒 dict 进/出的异步编排（ADR-0001 约束 3：禁止子图化），draft 与 revise
共享同一「校验-自审-修一次」链路：
写作调用 → lint → 自审 → 任一违规则恰好一次修订 → 折叠 self_check。

修后不复检（v1）：折叠进 ``self_check`` 的是**修前**质检结论——修订产物是否
真正规避了违规，由全局终审（citation_validator 等下游环节）兜底；单章内不做
二次校验，避免修订-复检循环失控。故 ``citations_ok=False`` 表示「本章修前检出过
引用类违规、已修一次但未复核」，交由下游裁决。

关键步骤经 ``SUBAGENT_PROGRESS`` 事件对外上报（ADR-0001 约束 2），载荷只放
元数据（unit / chapter_id / mode / step 与环节要点），绝不放正文全文。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any

from agents.contracts import SelfCheckPayload, SubagentAdapter
from agents.rewriter_loop.llm_adapter import LlmWriterClient
from agents.rewriter_loop.style_linter import (
    Violation,
    lint,
    load_prose,
    recheck_word_count,
)
from agents.rewriter_loop.stub import UNIT
from agents.rewriter_loop.writer_client import (
    AuditIssue,
    WriterEnvelope,
    WriterLlmClient,
    pass_materials,
)
from domain.events import SUBAGENT_PROGRESS, EventHook, noop_hook
from llm.llm_client import LLMFactory

_TIER_ENV = "REWRITER_LOOP_TIER"
_DOC_TYPE_ENV = "REWRITER_LOOP_DOC_TYPE"

_DEFAULT_TIER = "本科"
_DEFAULT_DOC_TYPE = "人才培养方案"

_VALID_TIERS = frozenset({"本科", "高职"})

# 自审违规的规则名：与 lint 侧规则同族命名，供 self_check 折叠时归为引用类。
_AUDIT_RULE = "self_audit_unmarked_derived_content"

# 引用类违规规则名：修前检出任一条则 citations_ok=False。
_CITATION_RULES = frozenset(
    {"unknown_material_marker", "unmarked_derived_content", _AUDIT_RULE}
)


def load_writer_settings(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """读取写作层次与文种：空值回落缺省；tier 非法抛 ValueError 指明变量名。

    doc_type 有意不校验：自由文本、只进提示词的上下文块，不参与任何规则分支，
    错值最多影响措辞而不破坏行为，故开放任意非空值。
    """
    if env is None:
        env = os.environ
    tier = env.get(_TIER_ENV, "").strip() or _DEFAULT_TIER
    doc_type = env.get(_DOC_TYPE_ENV, "").strip() or _DEFAULT_DOC_TYPE
    if tier not in _VALID_TIERS:
        raise ValueError(
            f"环境变量 {_TIER_ENV} 只接受 {sorted(_VALID_TIERS)}，当前值：{tier!r}"
        )
    return tier, doc_type


def audit_issues_to_violations(issues: Sequence[AuditIssue]) -> list[Violation]:
    """把自审条目折成 lint 同形的 ``Violation``，并入统一违规清单。

    公开导出：调测脚本（scripts/rewriter_debug.py）复用同一折叠逻辑，
    保证 --step 模式的违规口径与真编排零漂移。
    """
    out: list[Violation] = []
    for issue in issues:
        snippet = (issue.excerpt or "")[:80]
        out.append(
            Violation(
                rule=_AUDIT_RULE,
                message=(
                    f"自审发现正文疑似改写自素材「{issue.material_id}」原文"
                    f"却未挂 [{issue.material_id}] 角标"
                    + (f"：{snippet}" if snippet else "")
                ),
            )
        )
    return out


def make_writer_run(
    client: WriterLlmClient,
    *,
    tier: str,
    event_hook: EventHook = noop_hook,
) -> Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]:
    """构造写作编排的异步 run：黑盒 dict 进/出，供 ``SubagentAdapter`` 包装。

    文种（doc_type）只属客户端的提示词上下文，编排层不需要，故不收该参数。
    """

    async def run(task: dict[str, Any]) -> dict[str, Any]:
        spec = task["chapter_spec"]
        chapter_id = spec["id"]
        mode = task["mode"]
        materials = pass_materials(task)
        style_prose = load_prose()

        def progress(step: str, **extra: Any) -> None:
            """发进度事件：载荷统一带 unit / chapter_id / mode / step，只放元数据。"""
            event_hook(
                SUBAGENT_PROGRESS,
                {"unit": UNIT, "chapter_id": chapter_id, "mode": mode, "step": step, **extra},
            )

        def call_write(call: str, fix: list[Violation] | None = None) -> WriterEnvelope:
            """执行一次写作调用（draft/revise/fix），包 llm_call_start/end 事件对。"""
            progress("llm_call_start", call=call)
            if mode == "revise":
                envelope = client.revise(task, style_prose, fix_violations=fix)
            else:
                envelope = client.draft(task, style_prose, fix_violations=fix)
            progress(
                "llm_call_end",
                call=call,
                attempts=envelope.attempts,
                text_chars=len(envelope.chapter_text),
                degraded=envelope.degraded,
            )
            return envelope

        envelope = call_write(mode)

        # 退化诚实空稿短路：不做 lint / 自审 / 修订，如实上报退化，交由下游裁决。
        # 刻意设计：该路径只发首次写作的 llm_call_start / llm_call_end（含 degraded）
        # 一对事件，不发 lint_done / audit_done 等后续步骤——质检环节确实未执行，
        # 事件流如实反映执行轨迹。
        if not envelope.chapter_text.strip():
            return {
                "chapter_text": envelope.chapter_text,
                "chapter_summary": envelope.chapter_summary,
                "self_check": SelfCheckPayload(
                    citations_ok=False,
                    issues=[f"写作模型退化：正文为空（已重试 {envelope.attempts} 轮）"],
                ),
            }

        violations = lint(
            envelope.chapter_text,
            tier,
            materials=materials,
            hypotheses=spec["hypotheses"],
        )
        progress("lint_done", violations=len(violations))

        # 自审跳过分支（对齐源仓库「无引用池跳过自审」设计）：素材池为空则无
        # 「派生未标」可判，跳过模型调用省一次 LLM 花费；不发 llm_call 事件对
        # （没有真实调用），但仍发 audit_done（issues=0）保证步骤流完整可观测。
        if materials:
            progress("llm_call_start", call="audit")
            audit = client.audit(envelope.chapter_text, task)
            progress(
                "llm_call_end", call="audit", attempts=audit.attempts, degraded=audit.degraded
            )
            violations.extend(audit_issues_to_violations(audit.issues))
            progress("audit_done", issues=len(audit.issues), degraded=audit.degraded)
        else:
            progress("audit_done", issues=0, degraded=False)

        fix_degraded_empty = False
        word_count_recheck_issues: list[str] | None = None
        if violations:
            # 恰好一次修订，修后一般不复检（v1）：以修订产物为最终正文与摘要。
            progress("revise_triggered", violations=len(violations))
            envelope = call_write("fix", fix=violations)
            fix_degraded_empty = not envelope.chapter_text.strip()
            # 例外：修前检出过字数违规时，修后用纯函数复检字数（零 LLM 成本），
            # 结论无论达标与否均如实折入 self_check.issues，供终审与人工可见。
            if not fix_degraded_empty and any(v.rule == "word_count" for v in violations):
                recheck = recheck_word_count(envelope.chapter_text)
                progress("word_count_recheck", violations=len(recheck))
                if recheck:
                    word_count_recheck_issues = [
                        f"[word_count_recheck] 修后复检未达标：{v.message}" for v in recheck
                    ]
                else:
                    word_count_recheck_issues = ["[word_count_recheck] 修后复检：字数达标"]

        # self_check 折叠的是修前质检结论（引用类修后不复检），全局终审兜底；
        # 字数是唯一例外——修后复检结论一并折入（纯函数、不引入二次 LLM 修订）。
        issues = [f"[{v.rule}] {v.message}" for v in violations]
        if word_count_recheck_issues:
            issues.extend(word_count_recheck_issues)
        citations_ok = not any(v.rule in _CITATION_RULES for v in violations)
        if fix_degraded_empty:
            # 修订产物退化为空：保留修前违规明细，追加退化说明并判引用不通过。
            issues.append(f"修订调用退化：正文为空（已重试 {envelope.attempts} 轮）")
            citations_ok = False
        return {
            "chapter_text": envelope.chapter_text,
            "chapter_summary": envelope.chapter_summary,
            "self_check": SelfCheckPayload(citations_ok=citations_ok, issues=issues),
        }

    return run


def make_rewriter_loop(
    llm_factory: LLMFactory, event_hook: EventHook = noop_hook
) -> SubagentAdapter:
    """构造 rewriter_loop 真实现适配器：工厂内读取一次环境配置。"""
    tier, doc_type = load_writer_settings()
    client = LlmWriterClient(llm_factory(UNIT), tier=tier, doc_type=doc_type)
    run = make_writer_run(client, tier=tier, event_hook=event_hook)
    return SubagentAdapter(UNIT, run, event_hook)
