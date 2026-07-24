"""rewriter_loop 打桩的接口契约测试：draft / revise 两种模式产物合规。"""

import asyncio
from typing import Any

from agents.rewriter_loop import make_stub_rewriter_loop


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
        },
        "materials": [
            {
                "id": "m-h-1",
                "hypothesis_id": "h-1",
                "source": "来源一",
                "url": "https://example.com/m-h-1",
                "source_kind": "web",
                "source_ref": {"url": "https://example.com/m-h-1"},
                "excerpt": "摘录一",
                "relevance_score": 0.9,
                "verdict": "pass",
            },
            {
                "id": "m-h-2",
                "hypothesis_id": "h-2",
                "source": "来源二",
                "url": None,
                "source_kind": "knowledge_base",
                "source_ref": {"knowledge_id": "kb", "file_id": "f2", "chunk_id": "c2"},
                "excerpt": "摘录二",
                "relevance_score": 0.8,
                "verdict": "pass",
            },
            {
                "id": "m-fail",
                "hypothesis_id": "h-2",
                "source": "来源三",
                "url": None,
                "source_kind": "structured_data",
                "source_ref": {"dataset_id": "ds", "row_id": "r3"},
                "excerpt": "摘录三",
                "relevance_score": 0.1,
                "verdict": "fail",
            },
        ],
        "prev_chapter_summary": "上一章摘要：已完成背景铺陈。",
    }


def test_改写打桩_draft模式_返回字段与原位角标合规() -> None:
    adapter = make_stub_rewriter_loop()
    task = _make_draft_task()
    result = asyncio.run(adapter.run(task))

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
    assert "[m-fail]" not in chapter_text

    # prev_chapter_summary 非空时正文承接该摘要文本。
    assert task["prev_chapter_summary"] in chapter_text

    assert set(result["self_check"].keys()) == {"citations_ok", "issues"}


def test_改写打桩_revise模式_保留原文并落实修订说明() -> None:
    adapter = make_stub_rewriter_loop()
    task = _make_draft_task()
    task["mode"] = "revise"
    task["revision_note"] = {
        "user_directives": "精简第一段",
        "rule_violations": [
            {
                "rule": "hypothesis_no_support",
                "location_excerpt": "",
                "guidance": "为论点乙补充数据佐证",
                "severity": "error",
            }
        ],
        "conflict_hints": [],
        "passed": False,
    }
    task["current_text"] = "现有正文：论点甲与论点乙的初稿。[m-h-1]"
    result = asyncio.run(adapter.run(task))

    chapter_text = result["chapter_text"]
    assert task["current_text"] in chapter_text
    # 打桩逐项附注修订说明的用户指令与违规修改指导。
    assert "精简第一段" in chapter_text
    assert "为论点乙补充数据佐证" in chapter_text
