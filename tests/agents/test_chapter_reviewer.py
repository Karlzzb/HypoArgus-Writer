"""chapter_reviewer 打桩的接口契约测试：结果字段与分区式修订说明四区合规。

与 tests/agents/chapter_reviewer/test_real_contract.py（真实现契约）互为镜像：
同一批验收点（结果字段形状、修订说明四区键、passed、self_check 键、用户指令
逐字保留）在打桩与真实现两条链路上复验（stub 与真实现同形）。
"""

import asyncio
from typing import Any

from agents.chapter_reviewer import make_stub_chapter_reviewer


def _review_task(mode: str = "review", user_feedback: str | None = None) -> dict[str, Any]:
    task: dict[str, Any] = {
        "mode": mode,
        "doc_type": "通用公文",
        "doc_variant": None,
        "chapter_spec": {
            "id": "ch-1",
            "title": "一、示例章节",
            "chapter_type": None,
            "points": [{"id": "p-1", "text": "论点甲"}],
            "hypotheses": [
                {"id": "h-1", "text": "示例假说一", "refute_condition": "无佐证即证伪"}
            ],
        },
        "chapter_text": "## 一、示例章节\n\n本章论述。",
        "materials": [],
        "prev_chapter_summary": "上一章摘要。",
    }
    if user_feedback is not None:
        task["user_feedback"] = user_feedback
    return task


def test_评审打桩_返回字段与修订说明四区合规() -> None:
    adapter = make_stub_chapter_reviewer()
    result = asyncio.run(adapter.run(_review_task()))

    assert set(result.keys()) == {"revision_note", "self_check"}
    note = result["revision_note"]
    assert set(note.keys()) == {
        "user_directives",
        "rule_violations",
        "conflict_hints",
        "passed",
    }
    # 打桩零违规、判过；draft/无意见时用户指令区为空串。
    assert note["rule_violations"] == []
    assert note["conflict_hints"] == []
    assert note["passed"] is True
    assert note["user_directives"] == ""
    assert set(result["self_check"].keys()) == {"citations_ok", "issues"}
    assert result["self_check"] == {"citations_ok": True, "issues": []}


def test_评审打桩_revise携用户意见逐字回带用户指令区() -> None:
    adapter = make_stub_chapter_reviewer()
    result = asyncio.run(adapter.run(_review_task("revise", "保留原句，勿改。")))

    assert result["revision_note"]["user_directives"] == "保留原句，勿改。"
