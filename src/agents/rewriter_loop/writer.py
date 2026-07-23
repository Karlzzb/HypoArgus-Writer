"""writer：rewriter_loop 真实现的写作编排与工厂。

黑盒 dict 进/出的异步编排（ADR-0001 约束 3：禁止子图化）。自 ADR-0006 T3 起
rewriter_loop 收束为**纯写作 + 空稿短路**：一次写作调用（draft 首写 / revise 定向
改写）拿到成稿即返回；正文退化为空则如实上报退化、判引用不通过，交由下游裁决。

质检职责已上移到 chapter_reviewer（确定性 lint + 四维自审）：章级「写→评→重写」
循环由 ``nodes.chapter_write_loop`` 编排，本单元不再自审、不再 lint、不再触发第二次
写作调用。``self_check`` 因此退化为：成稿恒 ``citations_ok=True``（终态质检由评审或
循环层的修后 re-lint 负责，见 chapter_write_loop）；空稿 ``citations_ok=False`` 附退化说明。

关键步骤经 ``SUBAGENT_PROGRESS`` 事件对外上报（ADR-0001 约束 2），载荷只放
元数据（unit / chapter_id / mode / step 与环节要点），绝不放正文全文。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from agents.contracts import SelfCheckPayload, SubagentAdapter
from agents.rewriter_loop.llm_adapter import LlmWriterClient
from agents.rewriter_loop.style_linter import AUDIT_RULE_PREFIX, Violation, load_prose
from agents.rewriter_loop.stub import UNIT
from agents.rewriter_loop.writer_client import (
    AuditIssue,
    WriterLlmClient,
)
from domain.doc_types import carried_doc_facts
from domain.env_config import read_nonnegative_int, read_positive_int
from domain.events import SUBAGENT_PROGRESS, EventHook, noop_hook
from llm.llm_client import LLMFactory


def audit_issues_to_violations(issues: Sequence[AuditIssue]) -> list[Violation]:
    """把自审条目折成 lint 同形的 ``Violation``，并入统一违规清单。

    规则名按裁决项分列（``self_audit_<item>``）；带 material_id 的条目
    （派生未标）沿用素材定位话术，语义级条目以裁决项名义描述。
    公开导出：调测脚本（scripts/rewriter_debug.py）复用同一折叠逻辑，
    保证 --step 模式的违规口径与真编排零漂移——本函数不再进主写作链路。
    """
    out: list[Violation] = []
    for issue in issues:
        snippet = (issue.excerpt or "")[:80]
        if issue.material_id:
            message = (
                f"自审发现正文疑似改写自素材「{issue.material_id}」原文"
                f"却未挂 [{issue.material_id}] 角标"
            )
        else:
            message = f"自审裁决项「{issue.label or issue.item}」判定违规"
        out.append(
            Violation(
                rule=AUDIT_RULE_PREFIX + issue.item,
                message=message + (f"：{snippet}" if snippet else ""),
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
    散文注入按文种加载。纯写作链路：一次写作调用即返回，不 lint / 不自审 /
    不二次调用；质检由 chapter_reviewer 与循环层承担（ADR-0006 T3）。
    """

    async def run(task: dict[str, Any]) -> dict[str, Any]:
        spec = task["chapter_spec"]
        chapter_id = spec["id"]
        mode = task["mode"]
        doc_type, doc_variant = carried_doc_facts(task)
        style_prose = load_prose(doc_type)

        def progress(step: str, **extra: Any) -> None:
            """发进度事件：载荷统一带 unit / chapter_id / mode / step，只放元数据。"""
            event_hook(
                SUBAGENT_PROGRESS,
                {"unit": UNIT, "chapter_id": chapter_id, "mode": mode, "step": step, **extra},
            )

        # 唯一一次写作调用（draft/revise），包 llm_call_start/end 事件对。
        progress("llm_call_start", call=mode)
        if mode == "revise":
            envelope = client.revise(task, style_prose)
        else:
            envelope = client.draft(task, style_prose)
        progress(
            "llm_call_end",
            call=mode,
            attempts=envelope.attempts,
            text_chars=len(envelope.chapter_text),
            degraded=envelope.degraded,
        )

        if not envelope.chapter_text.strip():
            # 退化诚实空稿短路：如实上报退化、判引用不通过，交由下游裁决。
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

        # 成稿：self_check 恒通过——终态质检交由 chapter_reviewer 或循环层的修后
        # re-lint 负责（ADR-0006），rewriter 自身不再折叠质检结论。
        return {
            "chapter_text": envelope.chapter_text,
            "chapter_summary": envelope.chapter_summary,
            "self_check": SelfCheckPayload(citations_ok=True, issues=[]),
            "doc_type": doc_type,
            "doc_variant": doc_variant,
        }

    return run


def make_rewriter_loop(
    llm_factory: LLMFactory, event_hook: EventHook = noop_hook
) -> SubagentAdapter:
    """构造 rewriter_loop 真实现适配器：文种与变体逐任务取自任务包，工厂无环境配置。

    逐字流合并粒度（字符数 / 时间窗口）经环境变量配置：``WRITER_DELTA_FLUSH_CHARS``
    为单帧字符数阈值（缺省 64，正整数），``WRITER_DELTA_FLUSH_MS`` 为时间窗口
    毫秒数（缺省 50，非负整数、设 0 关闭时间窗口）。同一 ``event_hook``
    同时承载 ``SUBAGENT_PROGRESS`` 进度事件与 ``CONTENT_DELTA`` 逐字流——
    后者由 ``LlmWriterClient`` 在 draft/revise 流式消费时直接经此钩子上网线，
    编排层 ``make_writer_run`` 仍零感知（只调 ``client.draft/revise``）。
    """
    flush_chars = read_positive_int(os.environ, "WRITER_DELTA_FLUSH_CHARS", 64)
    flush_ms = read_nonnegative_int(os.environ, "WRITER_DELTA_FLUSH_MS", 50)
    client = LlmWriterClient(
        llm_factory(UNIT),
        flush_chars=flush_chars,
        flush_ms=flush_ms,
        event_hook=event_hook,
    )
    run = make_writer_run(client, event_hook=event_hook)
    return SubagentAdapter(UNIT, run, event_hook)
