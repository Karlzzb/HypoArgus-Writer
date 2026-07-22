"""chapter_reviewer 子智能体包：章级评审真实现、打桩与分区式修订说明装配。

真实现见 ``reviewer``（评审编排与工厂）、``review_client``（LLM 注入点协议与假客户端）、
``llm_adapter``（真实适配器）、``revision_note``（分区式修订说明纯函数装配）；
打桩见 ``stub``（同包共存，供空转与测试）。确定性风格校验与风格指南留在
rewriter_loop 包内，本包跨包引用其纯函数（style_linter.lint / audit_items_for /
CITATION_RULES），无循环依赖（ADR-0006）。
本包对外 re-export 工厂与装配入口，导入路径保持 ``agents.chapter_reviewer`` 不变。
"""

from agents.chapter_reviewer.llm_adapter import LlmReviewClient
from agents.chapter_reviewer.review_client import (
    FakeReviewLlmClient,
    ReviewConflict,
    ReviewEnvelope,
    ReviewIssue,
    ReviewLlmClient,
)
from agents.chapter_reviewer.reviewer import make_chapter_reviewer, make_reviewer_run
from agents.chapter_reviewer.revision_note import assemble_revision_note
from agents.chapter_reviewer.stub import (
    UNIT,
    make_stub_chapter_reviewer,
    stub_chapter_reviewer_run,
)

__all__ = [
    "FakeReviewLlmClient",
    "LlmReviewClient",
    "ReviewConflict",
    "ReviewEnvelope",
    "ReviewIssue",
    "ReviewLlmClient",
    "UNIT",
    "assemble_revision_note",
    "make_chapter_reviewer",
    "make_reviewer_run",
    "make_stub_chapter_reviewer",
    "stub_chapter_reviewer_run",
]
