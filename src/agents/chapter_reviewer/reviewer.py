"""reviewer：chapter_reviewer 真实现的评审编排与工厂。

黑盒 dict 进/出的异步编排（ADR-0001 约束 3：禁止子图化），单次评审链路
（ADR-0006）：确定性 lint（纯函数、零成本）→ 单次四维 LLM 自审（single-shot，
不在评审内部迭代）→ 装配分区式修订说明 → 按引用类规则折叠 self_check。

关键步骤经 ``SUBAGENT_PROGRESS`` 事件对外上报（ADR-0001 约束 2），载荷只放
元数据（unit / chapter_id / mode / step 与环节计数），绝不放正文全文：
lint 完成（lint_done）、自审调用（llm_call_start/end call=audit）、
自审结论（audit_done）、修订说明生成（revision_note_done）。

外部 LLM 调用经线程信号量限流（沿用检索子智能体同一机制，见 agents.concurrency）。
模型保持 plus：不设 CHAPTER_REVIEWER_LLM_MODEL 覆盖时回落全局 LLM_MODEL。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Coroutine
from typing import Any

from agents.chapter_reviewer.llm_adapter import LlmReviewClient
from agents.chapter_reviewer.review_client import ReviewLlmClient
from agents.chapter_reviewer.revision_note import assemble_revision_note
from agents.chapter_reviewer.stub import UNIT
from agents.citation_policy import citable_materials
from agents.concurrency import make_thread_permit
from agents.contracts import SelfCheckPayload, SubagentAdapter
from agents.rewriter_loop.style_linter import CITATION_RULES, audit_items_for, lint
from domain.doc_types import carried_doc_facts
from domain.env_config import read_positive_int
from domain.events import SUBAGENT_PROGRESS, EventHook, noop_hook
from llm.llm_client import LLMFactory

MAX_CONCURRENT_CALLS_ENV = "CHAPTER_REVIEWER_MAX_CONCURRENT_CALLS"
"""评审外部 LLM 调用总并发阈值的环境变量名。"""

DEFAULT_MAX_CONCURRENT_CALLS = 4
"""并发阈值缺省值：首写 8 章并行扇出时把章级评审压成 2 波而非 4 波，
仍远低于模型限流；与检索子智能体分口径（检索外层从紧、评审适度放宽），
可在 --real 运行时按模型限流实情经环境变量下调。"""


def make_reviewer_run(
    client: ReviewLlmClient,
    *,
    event_hook: EventHook = noop_hook,
    max_concurrent_calls: int | None = None,
) -> Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]:
    """构造评审编排的异步 run：黑盒 dict 进/出，供 ``SubagentAdapter`` 包装。

    文种与变体逐任务取自任务包（ADR-0005：State 锚定后经契约携带），
    lint 与四维自审裁决项均按文种加载，构造期不固化任何评审场景配置。
    """
    limit = (
        max_concurrent_calls
        if max_concurrent_calls is not None
        else read_positive_int(
            os.environ, MAX_CONCURRENT_CALLS_ENV, DEFAULT_MAX_CONCURRENT_CALLS
        )
    )
    permit = make_thread_permit(limit)

    async def run(task: dict[str, Any]) -> dict[str, Any]:
        spec = task["chapter_spec"]
        chapter_id = spec["id"]
        mode = task["mode"]
        doc_type, doc_variant = carried_doc_facts(task)
        materials = citable_materials(task)
        # 四维自审裁决项按文种加载 + 素材适用性过滤（与真实客户端内部同源）：
        # 编排层据此决定是否发起自审调用，并据裁决项配置权威赋予各违规 severity 定级。
        items = audit_items_for(doc_type, has_materials=bool(materials))
        severity_by_item = {item.id: item.severity for item in items}
        user_feedback = task.get("user_feedback", "")

        def progress(step: str, **extra: Any) -> None:
            """发进度事件：载荷统一带 unit / chapter_id / mode / step，只放元数据。"""
            event_hook(
                SUBAGENT_PROGRESS,
                {"unit": UNIT, "chapter_id": chapter_id, "mode": mode, "step": step, **extra},
            )

        # 确定性风格校验（跨包引用 rewriter_loop 纯函数、零 LLM 成本）。
        violations = lint(
            task["chapter_text"],
            doc_type,
            doc_variant,
            chapter_type=spec.get("chapter_type"),
            materials=materials,
            hypotheses=spec["hypotheses"],
        )
        progress("lint_done", violations=len(violations))

        review_issues = []
        conflicts = []
        if items:
            # 单次四维自审（single-shot）：信号量内一次 LLM 调用，评审内部不迭代。
            progress("llm_call_start", call="audit")
            async with permit.hold():
                envelope = client.review(task)
            progress(
                "llm_call_end",
                call="audit",
                attempts=envelope.attempts,
                degraded=envelope.degraded,
            )
            review_issues = envelope.issues
            conflicts = envelope.conflicts
            progress("audit_done", issues=len(review_issues), degraded=envelope.degraded)
        else:
            # 无适用裁决项（如素材池为空且仅依赖素材的裁决项适用）：跳过模型调用，
            # 不发 llm_call 事件对（没有真实调用），仍发 audit_done 保证步骤流可观测。
            progress("audit_done", issues=0, degraded=False)

        note = assemble_revision_note(
            user_feedback, violations, review_issues, severity_by_item, conflicts
        )
        progress(
            "revision_note_done",
            violations=len(note["rule_violations"]),
            passed=note["passed"],
        )

        # self_check 按引用类规则折叠（与 rewriter 同源 CITATION_RULES）：
        # 规则违规区任一条命中引用类规则则 citations_ok=False，交全局终审裁决。
        issues = [f"[{e['rule']}] {e['guidance']}" for e in note["rule_violations"]]
        citations_ok = not any(
            e["rule"] in CITATION_RULES for e in note["rule_violations"]
        )
        return {
            "revision_note": note,
            "self_check": SelfCheckPayload(citations_ok=citations_ok, issues=issues),
        }

    return run


def make_chapter_reviewer(
    llm_factory: LLMFactory,
    event_hook: EventHook = noop_hook,
    *,
    max_concurrent_calls: int | None = None,
) -> SubagentAdapter:
    """构造 chapter_reviewer 真实现适配器：按单元名取 LLM，文种逐任务取自任务包。"""
    client = LlmReviewClient(llm_factory(UNIT))
    run = make_reviewer_run(
        client, event_hook=event_hook, max_concurrent_calls=max_concurrent_calls
    )
    return SubagentAdapter(UNIT, run, event_hook)
