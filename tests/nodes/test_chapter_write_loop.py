"""章级写→评→重写循环的单元测试（issue #46 / ADR-0006 T3）。

覆盖调用次数上限与三条退出路径：干净初稿短路为 2 次调用、脏初稿至多
``max_rewrites`` 次重写且最后一次重写后不再评审、首写与重写退化空稿均沿用
rewriter 退化自检诚实短路（重写空稿不 re-lint 洗白）、重写后 self_check 以
纯函数 re-lint 折出修后终态；另覆盖 ``resolve_max_rewrites`` 的解析优先级。
"""

import asyncio
from typing import Any

import pytest

from agents.contracts import (
    ChapterSpecPayload,
    MaterialPayload,
    RevisionNotePayload,
    RuleViolationEntry,
    SelfCheckPayload,
    SubagentAdapter,
)
from nodes.chapter_write_loop import (
    DEFAULT_MAX_REWRITES,
    MAX_REWRITES_ENV,
    resolve_max_rewrites,
    run_chapter_write_loop,
)

_SPEC = ChapterSpecPayload(
    id="ch1",
    title="第一章",
    chapter_type=None,
    points=[{"id": "ch1-p1", "text": "论点一"}],
    hypotheses=[
        {
            "id": "ch1-p1-h1",
            "text": "假说一",
            "refute_condition": "出现公开反例即证伪",
        }
    ],
)

_MATERIALS = [
    MaterialPayload(
        id="m-ch1",
        hypothesis_id="ch1-p1-h1",
        source="来源",
        url=None,
        source_kind="web",
        excerpt="摘录",
        relevance_score=0.9,
        verdict="pass",
    )
]


def _rewrite_result(text: str, *, degraded: bool = False) -> dict[str, Any]:
    self_check = (
        SelfCheckPayload(citations_ok=False, issues=["产物退化为空稿"])
        if degraded
        else SelfCheckPayload(citations_ok=True, issues=[])
    )
    return {
        "chapter_text": text,
        "chapter_summary": "摘要",
        "self_check": self_check,
        "doc_type": "通用公文",
        "doc_variant": None,
    }


def _review_result(*, passed: bool) -> dict[str, Any]:
    violations = (
        []
        if passed
        else [
            RuleViolationEntry(
                rule="word_count",
                location_excerpt="",
                guidance="扩写到目标字数",
                severity="error",
            )
        ]
    )
    return {
        "revision_note": RevisionNotePayload(
            user_directives="",
            rule_violations=violations,
            conflict_hints=[],
            passed=passed,
        ),
        "self_check": SelfCheckPayload(
            citations_ok=passed, issues=[] if passed else ["[word_count] 字数不足"]
        ),
    }


def _make_rewriter(
    results: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> SubagentAdapter:
    async def _run(task: dict[str, Any]) -> dict[str, Any]:
        tasks.append(task)
        return results[len(tasks) - 1]

    return SubagentAdapter("rewriter_loop", _run)


def _make_reviewer(
    results: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> SubagentAdapter:
    async def _run(task: dict[str, Any]) -> dict[str, Any]:
        tasks.append(task)
        return results[len(tasks) - 1]

    return SubagentAdapter("chapter_reviewer", _run)


def _run_loop(
    rewriter: SubagentAdapter, reviewer: SubagentAdapter, max_rewrites: int
) -> Any:
    return asyncio.run(
        run_chapter_write_loop(
            rewriter_loop=rewriter,
            chapter_reviewer=reviewer,
            max_rewrites=max_rewrites,
            doc_type="通用公文",
            doc_variant=None,
            chapter_spec=_SPEC,
            materials=_MATERIALS,
            prev_chapter_summary="",
        )
    )


def test_干净初稿_写加评共两次调用且自检取评审():
    rewrites: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    rewriter = _make_rewriter([_rewrite_result("正文 [m-ch1]")], rewrites)
    reviewer = _make_reviewer([_review_result(passed=True)], reviews)

    draft = _run_loop(rewriter, reviewer, max_rewrites=1)

    assert len(rewrites) == 1 and rewrites[0]["mode"] == "draft"
    assert len(reviews) == 1 and reviews[0]["mode"] == "review"
    assert reviews[0]["chapter_text"] == "正文 [m-ch1]"
    assert draft.self_check is not None
    assert draft.self_check.citations_ok is True


def test_脏初稿_重写一次后不再评审且修订任务带分区说明与当前正文():
    rewrites: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    rewriter = _make_rewriter(
        [_rewrite_result("初稿正文"), _rewrite_result("重写正文 [m-ch1]")], rewrites
    )
    reviewer = _make_reviewer([_review_result(passed=False)], reviews)

    draft = _run_loop(rewriter, reviewer, max_rewrites=1)

    # 调用次数：write + review + rewrite = 3 次子智能体调用，重写后无终态复审。
    assert [t["mode"] for t in rewrites] == ["draft", "revise"]
    assert len(reviews) == 1
    assert rewrites[1]["revision_note"]["passed"] is False
    assert rewrites[1]["current_text"] == "初稿正文"
    assert draft.text == "重写正文 [m-ch1]"


def test_脏初稿_重写后自检为重写正文的relint而非评审陈旧结果():
    rewrites: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    # 重写正文带未知素材角标：re-lint 应对新正文判 citations_ok=False。
    rewriter = _make_rewriter(
        [_rewrite_result("初稿正文 [m-ch1]"), _rewrite_result("重写正文 [m-unknown]")],
        rewrites,
    )
    reviewer = _make_reviewer([_review_result(passed=False)], reviews)

    draft = _run_loop(rewriter, reviewer, max_rewrites=1)

    assert draft.self_check is not None
    assert draft.self_check.citations_ok is False
    assert any("unknown_material_marker" in issue for issue in draft.self_check.issues)


def test_重写次数上限_评审恒不过时至多重写max次():
    rewrites: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    rewriter = _make_rewriter(
        [
            _rewrite_result("初稿正文"),
            _rewrite_result("重写一 [m-ch1]"),
            _rewrite_result("重写二 [m-ch1]"),
        ],
        rewrites,
    )
    reviewer = _make_reviewer(
        [_review_result(passed=False), _review_result(passed=False)], reviews
    )

    draft = _run_loop(rewriter, reviewer, max_rewrites=2)

    # write + 2×(review+rewrite)，最后一次重写后不再评审。
    assert [t["mode"] for t in rewrites] == ["draft", "revise", "revise"]
    assert len(reviews) == 2
    assert draft.text == "重写二 [m-ch1]"


def test_上限为零_只写不评():
    rewrites: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    rewriter = _make_rewriter([_rewrite_result("正文 [m-ch1]")], rewrites)
    reviewer = _make_reviewer([], reviews)

    draft = _run_loop(rewriter, reviewer, max_rewrites=0)

    assert len(rewrites) == 1
    assert reviews == []
    assert draft.text == "正文 [m-ch1]"


def test_首写空稿_短路不评审并沿用退化自检():
    rewrites: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    rewriter = _make_rewriter([_rewrite_result("", degraded=True)], rewrites)
    reviewer = _make_reviewer([], reviews)

    draft = _run_loop(rewriter, reviewer, max_rewrites=1)

    assert reviews == []
    assert draft.text == ""
    assert draft.self_check is not None
    assert draft.self_check.citations_ok is False
    assert draft.self_check.issues == ["产物退化为空稿"]


def test_重写空稿_沿用退化自检不被relint洗白():
    rewrites: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    rewriter = _make_rewriter(
        [_rewrite_result("初稿正文 [m-ch1]"), _rewrite_result("", degraded=True)],
        rewrites,
    )
    # max_rewrites=2：若重写空稿未短路，会再进一轮评审。
    reviewer = _make_reviewer(
        [_review_result(passed=False), _review_result(passed=True)], reviews
    )

    draft = _run_loop(rewriter, reviewer, max_rewrites=2)

    assert len(reviews) == 1
    assert draft.text == ""
    assert draft.self_check is not None
    assert draft.self_check.citations_ok is False
    assert draft.self_check.issues == ["产物退化为空稿"]


def test_解析上限_显式传入优先于环境变量(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MAX_REWRITES_ENV, "5")
    assert resolve_max_rewrites(2) == 2
    assert resolve_max_rewrites(0) == 0


def test_解析上限_环境变量与缺省(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(MAX_REWRITES_ENV, raising=False)
    assert resolve_max_rewrites(None) == DEFAULT_MAX_REWRITES
    monkeypatch.setenv(MAX_REWRITES_ENV, "3")
    assert resolve_max_rewrites(None) == 3
    monkeypatch.setenv(MAX_REWRITES_ENV, "0")
    assert resolve_max_rewrites(None) == 0
