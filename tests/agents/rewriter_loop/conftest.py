"""rewriter_loop 测试共享夹具：draft 模式任务包（两份契约测试共用，避免逐字重复）。"""

from typing import Any

import pytest


@pytest.fixture
def draft_task() -> dict[str, Any]:
    """draft 模式任务包：2 条 pass 素材 + 1 条 fail 素材（每个测试拿到全新副本）。"""
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
                "id": "m-fail-x",
                "hypothesis_id": "h-2",
                "source": "来源三",
                "excerpt": "摘录三",
                "relevance_score": 0.1,
                "verdict": "fail",
            },
        ],
        "prev_chapter_summary": "上一章摘要：已完成背景铺陈。",
    }
