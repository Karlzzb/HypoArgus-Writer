"""review_client：chapter_reviewer 包内 LLM 注入点（协议、信封与确定性假客户端）。

评审编排（``reviewer``）只依赖本模块的 ``ReviewLlmClient`` 协议，不关心真实适配器
（``llm_adapter.LlmReviewClient``）还是测试假客户端（``FakeReviewLlmClient``）。
单次评审调用一次 LLM（single-shot，不在评审内部迭代）；信封扁平：四维自审违规
（含位置摘录与修改指导）与用户指令冲突提示，外加尝试轮次与退化标记两个元数据。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ReviewIssue(BaseModel):
    """四维自审单条违规：命中的裁决项 + 位置摘录 + 修改指导。

    ``item`` 对应 ssot-config ``audit_items`` 的裁决项 id；``excerpt`` 给正文中
    违规位置片段（位置摘录），``guidance`` 给修改指导。severity 不在此信封内——
    由裁决项配置（AuditItem.severity）在装配时权威赋予，模型不裁定定级。
    """

    item: str
    excerpt: str = ""
    guidance: str = ""


class ReviewConflict(BaseModel):
    """冲突提示单条：某条规则违规与用户指令冲突（用户指令优先）。"""

    description: str


class ReviewEnvelope(BaseModel):
    """评审自审裁决信封（single-shot）。

    ``issues: []`` 与 ``conflicts: []`` 均为合法非退化结果——「未发现违规/冲突」
    不触发重试；``degraded`` 仅在自审重试耗尽、降级为空裁决时置位（评审不阻断主链）。
    """

    issues: list[ReviewIssue] = Field(default_factory=list)
    conflicts: list[ReviewConflict] = Field(default_factory=list)
    attempts: int = 1
    degraded: bool = False


class ReviewLlmClient(Protocol):
    """评审 LLM 注入点：单条同步调用，task 为 ReviewTask 任务包 dict。"""

    def review(self, task: dict[str, Any]) -> ReviewEnvelope: ...


class FakeReviewLlmClient:
    """离线评审 LLM 桩：确定性、可脚本化。

    ``review_script`` 为按调用顺序消费的信封序列；每次调用弹出下一项并原样返回。
    序列耗尽时抛 ``IndexError``——测试须显式提供足够项，避免假阴性。
    ``review_calls`` 记每次收到的任务包，供断言「恰一次调用」与入参。
    """

    def __init__(self, review_script: Sequence[ReviewEnvelope] | None = None) -> None:
        self._review_script = list(review_script or [])
        self.review_calls: list[dict[str, Any]] = []

    def review(self, task: dict[str, Any]) -> ReviewEnvelope:
        self.review_calls.append(task)
        return self._review_script.pop(0)
