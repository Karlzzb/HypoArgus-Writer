"""writer：rewriter_loop 真实现的写作编排与工厂。

黑盒 dict 进/出的异步编排（ADR-0001 约束 3：禁止子图化），两模式链路
（ADR-0004：修后复检 v2 + revise/fix 合并）：

- draft：写作调用 → lint → 自审 → 任一违规则恰好一次修订 → **修后复检**
  （全量 lint 纯函数 + 一次轻量自审）→ 以修后终态折叠 self_check。
- revise：先对现有正文做预 lint（纯函数、零成本），把既存违规连同修订指令
  并入**唯一一次** revise 调用；调用后 lint + 自审即为终态质检，不再触发
  第二次写作调用（revise 即修，消除「revise 后必然再 fix」的固定开销）。

``self_check`` 折叠的是**修后终态**：``citations_ok=False`` 表示终态正文仍存
引用类违规（或产物退化为空），交由全局终审（citation_validator）裁决；
已被修一次真正规避的违规不再残留，避免门禁把已修好的章反复打回重写。

关键步骤经 ``SUBAGENT_PROGRESS`` 事件对外上报（ADR-0001 约束 2），载荷只放
元数据（unit / chapter_id / mode / step 与环节要点），绝不放正文全文。
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from agents.contracts import SelfCheckPayload, SubagentAdapter
from agents.rewriter_loop.llm_adapter import LlmWriterClient
from agents.rewriter_loop.style_linter import (
    Violation,
    lint,
    load_prose,
)
from agents.rewriter_loop.stub import UNIT
from agents.rewriter_loop.writer_client import (
    AuditIssue,
    WriterEnvelope,
    WriterLlmClient,
    pass_materials,
)
from domain.doc_types import carried_doc_facts
from domain.events import SUBAGENT_PROGRESS, EventHook, noop_hook
from llm.llm_client import LLMFactory

# 自审违规的规则名：与 lint 侧规则同族命名，供 self_check 折叠时归为引用类。
_AUDIT_RULE = "self_audit_unmarked_derived_content"

# 引用类违规规则名：修前检出任一条则 citations_ok=False。
_CITATION_RULES = frozenset(
    {"unknown_material_marker", "unmarked_derived_content", _AUDIT_RULE}
)


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
    event_hook: EventHook = noop_hook,
) -> Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]:
    """构造写作编排的异步 run：黑盒 dict 进/出，供 ``SubagentAdapter`` 包装。

    文种与变体逐任务取自任务包（ADR-0005：State 锚定后经契约携带），
    lint 与散文注入均按文种+变体两层加载，构造期不再固化任何写作场景配置。
    """

    async def run(task: dict[str, Any]) -> dict[str, Any]:
        spec = task["chapter_spec"]
        chapter_id = spec["id"]
        mode = task["mode"]
        doc_type, doc_variant = carried_doc_facts(task)
        materials = pass_materials(task)
        style_prose = load_prose(doc_type)

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

        def quality_check(text: str) -> list[Violation]:
            """终态质检：全量 lint（纯函数）+ 轻量自审，违规合并为统一清单。

            自审跳过分支（对齐源仓库「无引用池跳过自审」设计）：素材池为空则无
            「派生未标」可判，跳过模型调用省一次 LLM 花费；不发 llm_call 事件对
            （没有真实调用），但仍发 audit_done（issues=0）保证步骤流完整可观测。
            """
            found = lint(
                text, doc_type, doc_variant, materials=materials, hypotheses=spec["hypotheses"]
            )
            progress("lint_done", violations=len(found))
            if materials:
                progress("llm_call_start", call="audit")
                audit = client.audit(text, task)
                progress(
                    "llm_call_end",
                    call="audit",
                    attempts=audit.attempts,
                    degraded=audit.degraded,
                )
                found.extend(audit_issues_to_violations(audit.issues))
                progress("audit_done", issues=len(audit.issues), degraded=audit.degraded)
            else:
                progress("audit_done", issues=0, degraded=False)
            return found

        # revise 与 fix 合并（ADR-0004）：现有正文的既存违规经预 lint（纯函数、
        # 零 LLM 成本）得到，连同修订指令并入唯一一次 revise 调用，一次调用同时
        # 满足定向修订、引用规避与字数区间，消除「revise 后必然再 fix」的固定开销。
        pre_fix: list[Violation] = []
        if mode == "revise":
            pre_fix = lint(
                task.get("current_text", ""),
                doc_type,
                doc_variant,
                materials=materials,
                hypotheses=spec["hypotheses"],
            )
            progress("pre_lint_done", violations=len(pre_fix))

        envelope = call_write(mode, fix=pre_fix or None)

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
                "doc_type": doc_type,
                "doc_variant": doc_variant,
            }

        violations = quality_check(envelope.chapter_text)

        fix_degraded_empty = False
        if violations and mode != "revise":
            # draft 模式恰好一次修订；修后复检（quality_check）得到终态违规清单。
            # revise 模式不再二次调用——预 lint 违规已并入唯一一次 revise 调用，
            # 上面的 quality_check 即为终态质检。
            progress("revise_triggered", violations=len(violations))
            envelope = call_write("fix", fix=violations)
            fix_degraded_empty = not envelope.chapter_text.strip()
            if not fix_degraded_empty:
                violations = quality_check(envelope.chapter_text)

        # self_check 折叠修后终态（ADR-0004）：issues 只留终态仍存的违规，
        # 已被修一次真正规避的不再残留——门禁据此不把已修好的章打回重写。
        issues = [f"[{v.rule}] {v.message}" for v in violations]
        citations_ok = not any(v.rule in _CITATION_RULES for v in violations)
        if fix_degraded_empty:
            # 修订产物退化为空：保留修前违规明细，追加退化说明并判引用不通过。
            issues.append(f"修订调用退化：正文为空（已重试 {envelope.attempts} 轮）")
            citations_ok = False
        return {
            "chapter_text": envelope.chapter_text,
            "chapter_summary": envelope.chapter_summary,
            "self_check": SelfCheckPayload(citations_ok=citations_ok, issues=issues),
            "doc_type": doc_type,
            "doc_variant": doc_variant,
        }

    return run


def make_rewriter_loop(
    llm_factory: LLMFactory, event_hook: EventHook = noop_hook
) -> SubagentAdapter:
    """构造 rewriter_loop 真实现适配器：文种与变体逐任务取自任务包，工厂无环境配置。"""
    client = LlmWriterClient(llm_factory(UNIT))
    run = make_writer_run(client, event_hook=event_hook)
    return SubagentAdapter(UNIT, run, event_hook)
