"""rewriter_loop 真实现的接口契约测试：镜像打桩契约的断言口径。

与 tests/agents/test_rewriter_loop.py（打桩契约）互为镜像：同一批验收点
（结果字段形状、素材角标原位出现且不含未过滤素材、承接前章摘要、
revise 保留原文并落实指令）在真实现链路上复验。

链路口径：真编排（make_writer_run）+ 真校验器（style_linter）+ 真解析路径
（LlmWriterClient JSON-in-text），仅最底层模型调用用 FakeLLM 替身；
经 make_rewriter_loop(lambda unit: fake) 构造，一并覆盖工厂路径
（单元名请求、文种与变体逐任务取自任务包）。
"""

import asyncio
from typing import Any

from agents.rewriter_loop import make_rewriter_loop
from llm.llm_client import FakeLLM
from tests.llm_response_plans import (
    AUDIT_EMPTY_RESPONSE,
    joined_prompt,
    writer_envelope,
)

# 不触发任何 lint 规则的干净正文：角标全在 pass 素材池内、承接前章摘要、
# 无口语化/编号/意识形态违规（无 ## 标题 → 不落入任何章型模板规则）。
_DRAFT_TEXT = (
    "承接上文：上一章摘要：已完成背景铺陈。"
    "本专业面向智能制造领域培养高素质人才。[m-h-1][m-h-2]"
)


def test_改写真实现_draft模式_返回字段与原位角标合规(
    draft_task: dict[str, Any],
) -> None:
    # 纯写作链路（ADR-0006 T3）：draft 只发一次写作调用，不再自审、不再 lint。
    fake = FakeLLM([writer_envelope(_DRAFT_TEXT, "一行公文摘要")])
    adapter = make_rewriter_loop(lambda unit: fake)
    result = asyncio.run(adapter.run(draft_task))

    assert set(result.keys()) == {
        "chapter_text",
        "chapter_summary",
        "self_check",
        "doc_type",
        "doc_variant",
    }
    chapter_text = result["chapter_text"]

    # 每条 pass 素材的原位角标出现在正文中；fail 素材的角标不出现。
    assert "[m-h-1]" in chapter_text
    assert "[m-h-2]" in chapter_text
    assert "[m-fail-x]" not in chapter_text

    # prev_chapter_summary 非空时正文承接该摘要文本。
    assert draft_task["prev_chapter_summary"] in chapter_text

    # 成稿：self_check 恒退化为引用通过、无 issues（终态质检交由评审与循环层）。
    assert set(result["self_check"].keys()) == {"citations_ok", "issues"}
    assert result["self_check"] == {"citations_ok": True, "issues": []}

    # 真链路验收：写作提示词携带前章摘要与 pass 素材，fail 素材已被过滤。
    write_prompt = joined_prompt(fake.calls[0])
    assert draft_task["prev_chapter_summary"] in write_prompt
    assert "m-h-1" in write_prompt and "m-h-2" in write_prompt
    assert "m-fail-x" not in write_prompt
    # 纯写作链路只发一次调用：无自审、无 lint 的第二次调用。
    assert len(fake.calls) == 1


def test_改写真实现_revise模式_保留原文并落实每条指令(
    draft_task: dict[str, Any],
) -> None:
    directives = [
        {"type": "rewrite_only", "instruction": "精简第一段"},
        {"type": "evidence_augmented", "instruction": "为论点乙补充数据佐证"},
    ]
    draft_task["mode"] = "revise"
    draft_task["revision_directives"] = directives
    draft_task["current_text"] = "现有正文：论点甲与论点乙的初稿。[m-h-1]"
    # 定向改写产物：保留原文与角标，逐条落实指令（干净正文，不触发修订）。
    revised_text = draft_task["current_text"] + "".join(
        f"（修订落实：{directive['instruction']}）" for directive in directives
    )
    fake = FakeLLM([writer_envelope(revised_text, "修订后摘要"), AUDIT_EMPTY_RESPONSE])
    adapter = make_rewriter_loop(lambda unit: fake)
    result = asyncio.run(adapter.run(draft_task))

    chapter_text = result["chapter_text"]
    assert set(result.keys()) == {
        "chapter_text",
        "chapter_summary",
        "self_check",
        "doc_type",
        "doc_variant",
    }
    assert draft_task["current_text"] in chapter_text
    for directive in directives:
        assert directive["instruction"] in chapter_text

    # 真链路验收：revise 提示词携带现有正文与全部定向指令（定向修改而非重写）。
    revise_prompt = joined_prompt(fake.calls[0])
    assert draft_task["current_text"] in revise_prompt
    for directive in directives:
        assert directive["instruction"] in revise_prompt


def test_改写真实现_工厂路径_请求单元名且回带任务包文种(
    draft_task: dict[str, Any],
) -> None:
    seen_units: list[str] = []
    fake = FakeLLM([writer_envelope(_DRAFT_TEXT, "一行公文摘要"), AUDIT_EMPTY_RESPONSE])

    def factory(unit: str) -> FakeLLM:
        seen_units.append(unit)
        return fake

    adapter = make_rewriter_loop(factory)
    result = asyncio.run(adapter.run(draft_task))

    # 工厂按单元名恰好取一次 LLM；文种与变体经任务包进出、结果如实回带。
    assert seen_units == ["rewriter_loop"]
    assert adapter.unit == "rewriter_loop"
    assert set(result.keys()) == {
        "chapter_text",
        "chapter_summary",
        "self_check",
        "doc_type",
        "doc_variant",
    }
    assert result["doc_type"] == "人才培养方案"
    assert result["doc_variant"] == "本科"
