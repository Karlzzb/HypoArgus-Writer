"""search_agent 打桩的接口契约测试：出入参字段与 issue #3 规范逐项对齐。"""

import asyncio
from typing import Any

from agents.search_agent import make_stub_search_agent

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
