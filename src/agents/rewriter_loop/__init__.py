"""rewriter_loop 子智能体包：打桩实现与风格校验器。

打桩实现见 ``stub``；风格校验器（纯函数，不依赖主图与 LLM）见 ``style_linter``，
其单一事实源为随包携带的 ``style_guide.md``。
本包对外 re-export 打桩工厂与校验入口，导入路径保持 ``agents.rewriter_loop`` 不变。
"""

from agents.rewriter_loop.style_linter import (
    DEFAULT_STYLE_GUIDE_PATH,
    Fact,
    Violation,
    detect_chapter_template,
    extract_facts,
    lint,
    load_config,
    load_prose,
    normalize_cjk_ws,
    resolve_ideology_chapter,
    resolve_template,
)
from agents.rewriter_loop.stub import (
    UNIT,
    make_stub_rewriter_loop,
    stub_rewriter_loop_run,
)

__all__ = [
    "DEFAULT_STYLE_GUIDE_PATH",
    "Fact",
    "UNIT",
    "Violation",
    "detect_chapter_template",
    "extract_facts",
    "lint",
    "load_config",
    "load_prose",
    "make_stub_rewriter_loop",
    "normalize_cjk_ws",
    "resolve_ideology_chapter",
    "resolve_template",
    "stub_rewriter_loop_run",
]
