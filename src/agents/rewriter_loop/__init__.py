"""rewriter_loop 子智能体包：真实现（写作编排 + LLM 注入点 + 真实适配器）、打桩与风格校验器。

真实现见 ``writer``（编排与工厂）、``writer_client``（LLM 注入点协议与假客户端）、
``llm_adapter``（真实适配器）；打桩实现见 ``stub``（同包共存，供空转与测试）；
风格校验器（纯函数，不依赖主图与 LLM）见 ``style_linter``，
其单一事实源为随包携带的 ``style_guide.md``。
本包对外 re-export 工厂与校验入口，导入路径保持 ``agents.rewriter_loop`` 不变。
"""

from agents.rewriter_loop.llm_adapter import LlmWriterClient
from agents.rewriter_loop.style_linter import (
    DEFAULT_STYLE_GUIDE_PATH,
    Fact,
    Violation,
    check_word_count,
    count_prose_words,
    detect_chapter_template,
    extract_facts,
    lint,
    load_config,
    load_prose,
    normalize_cjk_ws,
    resolve_ideology_chapter,
    resolve_template,
    word_count_prompt_block,
)
from agents.rewriter_loop.stub import (
    UNIT,
    make_stub_rewriter_loop,
    stub_rewriter_loop_run,
)
from agents.rewriter_loop.writer import (
    audit_issues_to_violations,
    load_writer_settings,
    make_rewriter_loop,
    make_writer_run,
)
from agents.rewriter_loop.writer_client import (
    AuditEnvelope,
    AuditIssue,
    FakeWriterLlmClient,
    WriterEnvelope,
    WriterLlmClient,
)

__all__ = [
    "AuditEnvelope",
    "AuditIssue",
    "DEFAULT_STYLE_GUIDE_PATH",
    "Fact",
    "FakeWriterLlmClient",
    "LlmWriterClient",
    "UNIT",
    "Violation",
    "WriterEnvelope",
    "WriterLlmClient",
    "audit_issues_to_violations",
    "check_word_count",
    "count_prose_words",
    "detect_chapter_template",
    "extract_facts",
    "lint",
    "load_config",
    "load_prose",
    "load_writer_settings",
    "make_rewriter_loop",
    "make_stub_rewriter_loop",
    "make_writer_run",
    "normalize_cjk_ws",
    "resolve_ideology_chapter",
    "resolve_template",
    "stub_rewriter_loop_run",
    "word_count_prompt_block",
]
