"""chapter_reviewer 测试共享夹具：评审任务包（各测试拿到全新副本）。"""

from typing import Any

import pytest


@pytest.fixture
def review_task() -> dict[str, Any]:
    """review 模式评审任务包：2 条 pass 素材 + 1 条 fail 素材、含章骨架与摘要链。"""
    return {
        "mode": "review",
        "doc_type": "通用公文",
        "doc_variant": None,
        "chapter_spec": {
            "id": "ch-1",
            "title": "一、示例章节",
            "chapter_type": None,
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
        "chapter_text": "## 一、示例章节\n\n本章围绕论点甲与论点乙展开论述。[m-h-1][m-h-2]",
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
                "id": "m-fail-x",
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
