"""chapter_reviewer 真实现的接口契约测试：镜像打桩契约的断言口径。

与 tests/agents/test_chapter_reviewer.py（打桩契约）互为镜像：同一批验收点
（结果字段形状、修订说明四区键、passed、self_check 键、用户指令逐字保留、
确定性 lint 与四维自审合并）在真实现链路上复验。

链路口径：真编排（make_reviewer_run）+ 真校验器（style_linter.lint）+ 真解析
路径（LlmReviewClient JSON-in-text），仅最底层模型调用用 FakeLLM 替身；
经 make_chapter_reviewer(lambda unit: fake) 构造，一并覆盖工厂路径（单元名请求）。
"""

import asyncio
import json
from typing import Any

from agents.chapter_reviewer import make_chapter_reviewer
from llm.llm_client import FakeLLM
from service.llm_response_plans import joined_prompt


def _review_envelope(issues: list[dict[str, str]], conflicts: list[dict[str, str]]) -> str:
    return json.dumps({"issues": issues, "conflicts": conflicts}, ensure_ascii=False)


def test_评审真实现_返回字段与四区合规_单次调用(review_task: dict[str, Any]) -> None:
    fake = FakeLLM([_review_envelope([{"item": "intra_chapter_coherence", "excerpt": "断裂处", "guidance": "补衔接"}], [])])
    adapter = make_chapter_reviewer(lambda unit: fake)
    result = asyncio.run(adapter.run(review_task))

    assert set(result.keys()) == {"revision_note", "self_check"}
    note = result["revision_note"]
    assert set(note.keys()) == {"user_directives", "rule_violations", "conflict_hints", "passed"}
    assert set(result["self_check"].keys()) == {"citations_ok", "issues"}

    # 四维自审违规经真解析进规则违规区（规则名 self_audit_<item>、定级取配置 warn）。
    rows = {e["rule"]: e for e in note["rule_violations"]}
    assert rows["self_audit_intra_chapter_coherence"]["severity"] == "warn"
    assert rows["self_audit_intra_chapter_coherence"]["location_excerpt"] == "断裂处"

    # single-shot：真链路恰一次 LLM 调用；调用为评审自审（携评审标签与本章正文）。
    assert len(fake.calls) == 1
    prompt = joined_prompt(fake.calls[0])
    assert "【章节评审】" in prompt
    assert review_task["chapter_text"] in prompt


def test_评审真实现_revise逐字保留用户意见并给冲突提示(review_task: dict[str, Any]) -> None:
    review_task["mode"] = "revise"
    review_task["user_feedback"] = "第二段原句必须保留，勿改。"
    fake = FakeLLM([_review_envelope([], [{"description": "规则要删的句子用户要求保留"}])])
    adapter = make_chapter_reviewer(lambda unit: fake)
    result = asyncio.run(adapter.run(review_task))

    note = result["revision_note"]
    # 用户指令区逐字零改写保留。
    assert note["user_directives"] == "第二段原句必须保留，勿改。"
    assert note["conflict_hints"] == [{"description": "规则要删的句子用户要求保留"}]
    # 真链路：用户意见进入评审提示词（供模型判冲突）。
    assert "第二段原句必须保留，勿改。" in joined_prompt(fake.calls[0])


def test_评审真实现_工厂路径_请求单元名(review_task: dict[str, Any]) -> None:
    seen_units: list[str] = []
    fake = FakeLLM([_review_envelope([], [])])

    def factory(unit: str) -> FakeLLM:
        seen_units.append(unit)
        return fake

    adapter = make_chapter_reviewer(factory)
    result = asyncio.run(adapter.run(review_task))

    assert seen_units == ["chapter_reviewer"]
    assert adapter.unit == "chapter_reviewer"
    assert set(result.keys()) == {"revision_note", "self_check"}
