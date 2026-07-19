"""writer_client：rewriter_loop 包内 LLM 缝（协议、信封与确定性假客户端）。

编排层（``writer``）只依赖本模块的 ``WriterLlmClient`` 协议，不关心真实适配器
（``llm_adapter.LlmWriterClient``）还是测试假客户端（``FakeWriterLlmClient``）。
信封刻意扁平：正文/摘要之外仅带尝试轮次与退化标记两个元数据，供进度事件上报。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agents.contracts import MaterialPayload
from agents.rewriter_loop.style_linter import Violation


def pass_materials(task: dict[str, Any]) -> list[MaterialPayload]:
    """从任务包取判定通过（verdict=="pass"）的素材：fail 素材不进提示词与校验。

    过滤口径收敛于缝模块，供编排层与真实适配器共用，避免多处同形漂移。
    """
    return [m for m in task["materials"] if m["verdict"] == "pass"]


class WriterEnvelope(BaseModel):
    """draft 与 revise 共用产出信封。

    ``attempts`` 由真实适配器回填实际使用的尝试轮次，供进度事件上报；
    ``degraded`` 在重试耗尽、退化诚实返回（如空正文）时置位。
    """

    chapter_text: str
    chapter_summary: str
    attempts: int = 1
    degraded: bool = False


class AuditIssue(BaseModel):
    """自审单条违规：池内素材 id + 正文疑似派生却未标的片段。"""

    material_id: str
    excerpt: str = ""


class AuditEnvelope(BaseModel):
    """自审裁决信封。

    ``issues: []`` 是合法非退化结果——「未发现违规」不触发重试；
    ``degraded`` 仅在自审重试耗尽、降级为空裁决时置位。
    """

    issues: list[AuditIssue] = Field(default_factory=list)
    attempts: int = 1
    degraded: bool = False


class WriterLlmClient(Protocol):
    """写作 LLM 缝：三条同步调用，task 为 RewriteTask 任务包 dict。

    ``fix_violations`` 置位 = 「修一次」的修正口径：draft 按同一上下文重写并规避
    违规清单；revise 基于同一 current_text 重新执行定向改写并规避违规清单。
    """

    def draft(
        self,
        task: dict[str, Any],
        style_prose: str,
        *,
        fix_violations: Sequence[Violation] | None = None,
    ) -> WriterEnvelope: ...

    def revise(
        self,
        task: dict[str, Any],
        style_prose: str,
        *,
        fix_violations: Sequence[Violation] | None = None,
    ) -> WriterEnvelope: ...

    def audit(self, chapter_text: str, task: dict[str, Any]) -> AuditEnvelope: ...


class FakeWriterLlmClient:
    """离线写作 LLM 桩：确定性、可脚本化。

    ``draft_script`` / ``revise_script`` / ``audit_script`` 为按调用顺序消费的信封
    序列；每次调用弹出下一项并原样返回。序列耗尽时抛 ``IndexError``——
    测试须显式提供足够项，避免假阴性。
    调用记录供断言：``draft_calls`` / ``revise_calls`` 各记 (task, fix_violations)，
    ``audit_calls`` 记每次收到的正文。
    """

    def __init__(
        self,
        draft_script: Sequence[WriterEnvelope] | None = None,
        revise_script: Sequence[WriterEnvelope] | None = None,
        audit_script: Sequence[AuditEnvelope] | None = None,
    ) -> None:
        self._draft_script = list(draft_script or [])
        self._revise_script = list(revise_script or [])
        self._audit_script = list(audit_script or [])
        self.draft_calls: list[tuple[dict[str, Any], Sequence[Violation] | None]] = []
        self.revise_calls: list[tuple[dict[str, Any], Sequence[Violation] | None]] = []
        self.audit_calls: list[str] = []

    def draft(
        self,
        task: dict[str, Any],
        style_prose: str,
        *,
        fix_violations: Sequence[Violation] | None = None,
    ) -> WriterEnvelope:
        self.draft_calls.append((task, fix_violations))
        return self._draft_script.pop(0)

    def revise(
        self,
        task: dict[str, Any],
        style_prose: str,
        *,
        fix_violations: Sequence[Violation] | None = None,
    ) -> WriterEnvelope:
        self.revise_calls.append((task, fix_violations))
        return self._revise_script.pop(0)

    def audit(self, chapter_text: str, task: dict[str, Any]) -> AuditEnvelope:
        self.audit_calls.append(chapter_text)
        return self._audit_script.pop(0)
