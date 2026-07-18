"""子智能体打桩的接口契约测试：出入参字段与 issue #3 规范逐项对齐。

覆盖三块：search_agent 打桩的素材字段与假说回链、
rewriter_loop 打桩的 draft / revise 两种模式产物、
适配层的启动 / 结束事件挂钩。
"""

import asyncio
from typing import Any

from subagents import (
    SUBAGENT_END,
    SUBAGENT_START,
    make_stub_rewriter_loop,
    make_stub_search_agent,
)

# search_agent 任务包：含 2 条假说，字段按规范逐项给全。
SEARCH_TASK: dict[str, Any] = {
    "chapter_id": "ch-1",
    "hypotheses": [
        {
            "id": "h-1",
            "text": "示例假说一",
            "refute_condition": "若找不到任何佐证则证伪",
        },
        {
            "id": "h-2",
            "text": "示例假说二",
            "refute_condition": "若出现反例则证伪",
        },
    ],
    "genre": "行业白皮书",
    "existing_materials_digest": "",
}


def _make_draft_task() -> dict[str, Any]:
    """构造 draft 模式任务包：2 条 pass 素材 + 1 条 fail 素材。"""
    return {
        "mode": "draft",
        "chapter_spec": {
            "id": "ch-1",
            "title": "示例章节",
            "points": [
                {"id": "p-1", "text": "论点甲"},
                {"id": "p-2", "text": "论点乙"},
            ],
            "hypotheses": SEARCH_TASK["hypotheses"],
        },
        "materials": [
            {
                "id": "m-h-1",
                "hypothesis_id": "h-1",
                "source": "来源一",
                "excerpt": "摘录一",
                "relevance_score": 0.9,
                "verdict": "pass",
            },
            {
                "id": "m-h-2",
                "hypothesis_id": "h-2",
                "source": "来源二",
                "excerpt": "摘录二",
                "relevance_score": 0.8,
                "verdict": "pass",
            },
            {
                "id": "m-fail",
                "hypothesis_id": "h-2",
                "source": "来源三",
                "excerpt": "摘录三",
                "relevance_score": 0.1,
                "verdict": "fail",
            },
        ],
        "prev_chapter_summary": "上一章摘要：已完成背景铺陈。",
    }


def test_检索打桩_每条假说恰好一条素材且字段合规() -> None:
    adapter = make_stub_search_agent()
    result = asyncio.run(adapter.run(SEARCH_TASK))

    assert set(result.keys()) == {"materials"}
    materials = result["materials"]
    assert len(materials) == len(SEARCH_TASK["hypotheses"])

    expected_fields = {
        "id",
        "hypothesis_id",
        "source",
        "excerpt",
        "relevance_score",
        "verdict",
    }
    hypothesis_ids = [h["id"] for h in SEARCH_TASK["hypotheses"]]
    for material, hypothesis_id in zip(materials, hypothesis_ids, strict=True):
        assert set(material.keys()) == expected_fields
        assert material["hypothesis_id"] == hypothesis_id
        assert material["verdict"] == "pass"
        assert isinstance(material["relevance_score"], float)


def test_改写打桩_draft模式_返回字段与原位角标合规() -> None:
    adapter = make_stub_rewriter_loop()
    task = _make_draft_task()
    result = asyncio.run(adapter.run(task))

    assert set(result.keys()) == {"chapter_text", "chapter_summary", "self_check"}
    chapter_text = result["chapter_text"]

    # 每条 pass 素材的原位角标出现在正文中；fail 素材的角标不出现。
    assert "[m-h-1]" in chapter_text
    assert "[m-h-2]" in chapter_text
    assert "[m-fail]" not in chapter_text

    # prev_chapter_summary 非空时正文承接该摘要文本。
    assert task["prev_chapter_summary"] in chapter_text

    assert set(result["self_check"].keys()) == {"citations_ok", "issues"}


def test_改写打桩_revise模式_保留原文并落实每条指令() -> None:
    adapter = make_stub_rewriter_loop()
    directives = [
        {"type": "rewrite_only", "instruction": "精简第一段"},
        {"type": "evidence_augmented", "instruction": "为论点乙补充数据佐证"},
    ]
    task = _make_draft_task()
    task["mode"] = "revise"
    task["revision_directives"] = directives
    task["current_text"] = "现有正文：论点甲与论点乙的初稿。[m-h-1]"
    result = asyncio.run(adapter.run(task))

    chapter_text = result["chapter_text"]
    assert task["current_text"] in chapter_text
    for directive in directives:
        assert directive["instruction"] in chapter_text


def test_适配层_一次运行依次发出启动与结束事件() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def record_hook(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    adapter = make_stub_search_agent(record_hook)
    asyncio.run(adapter.run(SEARCH_TASK))

    assert [event_type for event_type, _ in events] == [SUBAGENT_START, SUBAGENT_END]
    for _, payload in events:
        assert payload["unit"] == "search_agent"
