"""document_reviewer 节点单元测试：用假 LLM 预置应答直接调用节点函数。

覆盖：引用四步（全部通过、程序对账失败、语义核查失败、自检失败合并、
重试超限携带警告、增量核查范围与 LLM 调用次数）、结构完整性（编号校验与
大纲缺稿）、篇级评审（fact_conflict 打回、warn 三维呈人工不打回、幻觉与
未知维度防护、失败轮同样写入 review_warnings）、环境变量读取。

应答消费顺序：引文语义核查逐章并发弹出顺序应答（各章应答须互为等价或
按内容可丢弃），篇级评审在全部语义核查完成后恰好消费最后一条应答。
"""

import json
from typing import Any

import pytest

from nodes.document_reviewer import (
    ReviewerConfig,
    load_reviewer_config,
    make_document_reviewer_node,
)
from llm.llm_client import FakeLLM
from domain.state import (
    ChapterDraft,
    ChapterSpec,
    Material,
    SelfCheck,
    WorkflowStatus,
    WritingAgentState,
    initial_state,
)

# 篇级评审「无任何发现」的放行应答。
REVIEW_PASS = "[]"


def _mat(material_id: str, chapter_id: str, verdict: str = "pass") -> Material:
    """构造一条引文库素材。"""
    return Material(
        id=material_id,
        hypothesis_id=f"{chapter_id}-p1-h1",
        chapter_id=chapter_id,
        source="来源",
        url=None,
        excerpt="摘录",
        relevance_score=0.9,
        verdict=verdict,  # type: ignore[arg-type]
    )


def _draft(
    chapter_id: str, text: str, self_check: SelfCheck | None = None
) -> ChapterDraft:
    """构造一章草稿。

    大纲标题（见 _state）与正文均不带编号，不触发章节编号校验；
    编号场景用例自行在 text 中写入带编号的 ## 标题。
    """
    return ChapterDraft(
        chapter_id=chapter_id,
        text=text,
        summary="摘要",
        self_check=self_check or SelfCheck(),
    )


def _state(
    drafts: list[ChapterDraft],
    library: list[Material],
    revised_chapter_ids: list[str] | None = None,
    citation_retry_count: int = 0,
) -> WritingAgentState:
    """构造带大纲与草稿的图状态。"""
    state = initial_state("需求", "身份", "trace-dr")
    state["outline"] = [
        ChapterSpec(id=draft.chapter_id, title=f"章 {draft.chapter_id}")
        for draft in drafts
    ]
    state["chapter_drafts"] = drafts
    state["citation_library"] = library
    state["revised_chapter_ids"] = revised_chapter_ids or []
    state["citation_retry_count"] = citation_retry_count
    return state


def _aligned(material_id: str, aligned: bool = True, reason: str = "对应") -> dict[str, Any]:
    """构造一条语义核查应答项。"""
    return {"material_id": material_id, "aligned": aligned, "reason": reason}


def _finding(
    dimension: str, chapter_ids: list[str], detail: str = "发现说明"
) -> dict[str, Any]:
    """构造一条篇级评审发现项。"""
    return {"dimension": dimension, "chapter_ids": chapter_ids, "detail": detail}


def _run(
    state: WritingAgentState,
    responses: list[Any],
    max_retries: int = 2,
) -> tuple[dict[str, Any], FakeLLM]:
    """预置应答序列（list/dict 自动转 JSON 文本）后执行一次节点。"""
    fake = FakeLLM(
        [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in responses
        ]
    )
    node = make_document_reviewer_node(
        lambda unit: fake, ReviewerConfig(max_retries=max_retries)
    )
    return dict(node(state)), fake


def test_全部通过_报告放行且计数归零() -> None:
    state = _state(
        [_draft("ch1", "引用[m1]。")],
        [_mat("m1", "ch1")],
        citation_retry_count=1,
    )
    result, fake = _run(state, [[_aligned("m1")], REVIEW_PASS])

    report = result["citation_report"]
    assert report.passed is True
    assert report.issues == []
    assert report.failed_chapter_ids == []
    assert result["citation_retry_count"] == 0
    assert result["citation_warnings"] == []
    assert result["review_warnings"] == []
    assert result["revised_chapter_ids"] == []
    # 通过即将进入人工中断点：等待人工期间的状态机值由本节点写入。
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    assert result["current_node_llm_config"]["unit"] == "document_reviewer"
    # 语义核查 1 次 + 篇级评审 1 次。
    assert len(fake.calls) == 2


def test_程序对账失败_定位章节且无语义误报() -> None:
    state = _state(
        [_draft("ch1", "引用[m1]与孤儿[m404]。"), _draft("ch2", "引用[m2]。")],
        [_mat("m1", "ch1"), _mat("m2", "ch2")],
    )
    result, _ = _run(state, [[_aligned("m1")], [_aligned("m2")], REVIEW_PASS])

    report = result["citation_report"]
    assert report.passed is False
    assert [issue.kind for issue in report.issues] == ["orphan_marker"]
    assert report.failed_chapter_ids == ["ch1"]
    assert result["citation_retry_count"] == 1
    assert result["citation_warnings"] == []
    assert result["revised_chapter_ids"] == []


def test_语义核查失败_生成semantic_mismatch() -> None:
    state = _state([_draft("ch1", "引用[m1]。")], [_mat("m1", "ch1")])
    result, _ = _run(
        state, [[_aligned("m1", aligned=False, reason="观点不符")], REVIEW_PASS]
    )

    report = result["citation_report"]
    assert report.passed is False
    assert [(issue.kind, issue.material_id) for issue in report.issues] == [
        ("semantic_mismatch", "m1")
    ]
    assert "观点不符" in report.issues[0].detail
    assert report.failed_chapter_ids == ["ch1"]


def test_语义核查应答中不在被引集合的素材项被丢弃() -> None:
    state = _state([_draft("ch1", "引用[m1]。")], [_mat("m1", "ch1")])
    result, _ = _run(
        state,
        [
            [_aligned("m1"), _aligned("m外来", aligned=False, reason="幻觉项")],
            REVIEW_PASS,
        ],
    )
    assert result["citation_report"].passed is True


def test_语义核查应答被对象包裹时同样解析() -> None:
    """关思考后模型偶发把核查项数组包进对象，应取出其中唯一的数组值。"""
    state = _state([_draft("ch1", "引用[m1]。")], [_mat("m1", "ch1")])
    result, _ = _run(
        state,
        [{"results": [_aligned("m1", aligned=False, reason="观点不符")]}, REVIEW_PASS],
    )
    report = result["citation_report"]
    assert report.passed is False
    assert [issue.kind for issue in report.issues] == ["semantic_mismatch"]


def test_语义核查应答为单个核查项对象时同样解析() -> None:
    """关思考后模型偶发直接返回单个核查项对象，应包成单元素数组。"""
    state = _state([_draft("ch1", "引用[m1]。")], [_mat("m1", "ch1")])
    result, _ = _run(
        state, [_aligned("m1", aligned=False, reason="观点不符"), REVIEW_PASS]
    )
    report = result["citation_report"]
    assert report.passed is False
    assert [issue.kind for issue in report.issues] == ["semantic_mismatch"]


def test_语义核查应答无法归一化时抛ValueError() -> None:
    state = _state([_draft("ch1", "引用[m1]。")], [_mat("m1", "ch1")])
    with pytest.raises(ValueError, match="引文语义核查"):
        _run(state, [{"a": [], "b": []}])


def test_自检失败合并为self_check_failed() -> None:
    state = _state(
        [
            _draft(
                "ch1",
                "引用[m1]。",
                self_check=SelfCheck(citations_ok=False, issues=["角标位置存疑"]),
            )
        ],
        [_mat("m1", "ch1")],
    )
    result, _ = _run(state, [[_aligned("m1")], REVIEW_PASS])

    report = result["citation_report"]
    assert report.passed is False
    assert [issue.kind for issue in report.issues] == ["self_check_failed"]
    assert report.issues[0].material_id == ""
    assert report.issues[0].detail == "角标位置存疑"


def test_自检失败但issues为空也生成一条() -> None:
    state = _state(
        [_draft("ch1", "无角标正文。", self_check=SelfCheck(citations_ok=False))],
        [],
    )
    result, _ = _run(state, [REVIEW_PASS])
    assert [issue.kind for issue in result["citation_report"].issues] == [
        "self_check_failed"
    ]


def test_重试超限_携带未决引文警告() -> None:
    state = _state(
        [_draft("ch1", "孤儿[m404]。")],
        [],
        citation_retry_count=2,
    )
    result, _ = _run(state, [REVIEW_PASS], max_retries=2)

    report = result["citation_report"]
    assert report.passed is False
    assert result["citation_retry_count"] == 3
    assert result["citation_warnings"]
    assert all(isinstance(warning, str) for warning in result["citation_warnings"])
    assert result["revised_chapter_ids"] == []


def test_增量核查_只审revised章节且语义核查只调一次() -> None:
    state = _state(
        [_draft("ch1", "孤儿[m404]。"), _draft("ch2", "引用[m2]。")],
        [_mat("m2", "ch2")],
        revised_chapter_ids=["ch2"],
    )
    result, fake = _run(state, [[_aligned("m2")], REVIEW_PASS])

    # ch1 的孤儿角标不在范围内，不报；语义核查只为 ch2 调用一次，
    # 篇级评审始终全量：其调用输入含未受审的 ch1 正文。
    assert result["citation_report"].passed is True
    assert len(fake.calls) == 2
    assert "m2" in fake.calls[0][1]["content"]
    assert "孤儿[m404]" in fake.calls[1][1]["content"]
    assert result["revised_chapter_ids"] == []


def test_章节没有角标素材时跳过语义核查_篇级评审仍执行() -> None:
    state = _state([_draft("ch1", "无角标正文。")], [])
    result, fake = _run(state, [REVIEW_PASS])
    assert result["citation_report"].passed is True
    # 唯一一次调用是篇级评审，不是语义核查。
    assert len(fake.calls) == 1
    assert "篇级评审器" in fake.calls[0][0]["content"]


def test_failed_chapter_ids按大纲顺序去重() -> None:
    state = _state(
        [_draft("ch1", "孤儿[m404]。"), _draft("ch2", "孤儿[m405]与[m406]。")],
        [],
    )
    result, _ = _run(state, [REVIEW_PASS])
    assert result["citation_report"].failed_chapter_ids == ["ch1", "ch2"]


def test_环境变量缺省为2() -> None:
    assert load_reviewer_config({}).max_retries == 2
    assert load_reviewer_config({"DOCUMENT_REVIEW_MAX_RETRIES": ""}).max_retries == 2
    assert load_reviewer_config({"DOCUMENT_REVIEW_MAX_RETRIES": "5"}).max_retries == 5


@pytest.mark.parametrize("raw", ["abc", "0", "-1"])
def test_环境变量非法值抛ValueError(raw: str) -> None:
    with pytest.raises(ValueError, match="DOCUMENT_REVIEW_MAX_RETRIES"):
        load_reviewer_config({"DOCUMENT_REVIEW_MAX_RETRIES": raw})


def test_章节编号重复_检出失败() -> None:
    """测试章节编号重复场景（issue #18）：两个「一、」。"""
    state = _state(
        [
            _draft("ch1", "## 一、专业名称及代码\n内容[m1]"),
            _draft("ch2", "## 一、入学要求\n内容[m2]"),  # 重复的「一、」
        ],
        [_mat("m1", "ch1"), _mat("m2", "ch2")],
    )
    result, fake = _run(state, [[_aligned("m1")], [_aligned("m2")], REVIEW_PASS])
    assert not result["citation_report"].passed
    issues = result["citation_report"].issues
    numbering_issues = [issue for issue in issues if issue.kind == "numbering_broken"]
    assert len(numbering_issues) == 2
    # ch2 预期「二、」，实际「一、」（断号）。
    断号_issues = [issue for issue in numbering_issues if "预期" in issue.detail]
    assert len(断号_issues) == 1
    assert 断号_issues[0].chapter_id == "ch2"
    # ch2 与 ch1 重复。
    重复_issues = [issue for issue in numbering_issues if "重复" in issue.detail]
    assert len(重复_issues) == 1
    assert 重复_issues[0].chapter_id == "ch2"


def test_章节编号断号_检出失败() -> None:
    """测试章节编号跳号场景（issue #18）：一、三、（缺二）。"""
    state = _state(
        [
            _draft("ch1", "## 一、专业名称及代码\n内容[m1]"),
            _draft("ch2", "## 三、学制学位\n内容[m2]"),  # 跳号：缺二
        ],
        [_mat("m1", "ch1"), _mat("m2", "ch2")],
    )
    result, _ = _run(state, [[_aligned("m1")], [_aligned("m2")], REVIEW_PASS])
    assert not result["citation_report"].passed
    issues = result["citation_report"].issues
    numbering_issues = [issue for issue in issues if issue.kind == "numbering_broken"]
    assert len(numbering_issues) == 1
    assert numbering_issues[0].chapter_id == "ch2"
    assert "预期「二」" in numbering_issues[0].detail
    assert "实际「三」" in numbering_issues[0].detail


def test_章节编号连续_全部通过() -> None:
    """测试章节编号正确的场景：一、二、三、连续。"""
    state = _state(
        [
            _draft("ch1", "## 一、专业名称及代码\n内容[m1]"),
            _draft("ch2", "## 二、入学要求\n内容[m2]"),
            _draft("ch3", "## 三、学制学位\n内容[m3]"),
        ],
        [_mat("m1", "ch1"), _mat("m2", "ch2"), _mat("m3", "ch3")],
    )
    result, _ = _run(
        state, [[_aligned("m1")], [_aligned("m2")], [_aligned("m3")], REVIEW_PASS]
    )
    assert result["citation_report"].passed
    issues = result["citation_report"].issues
    numbering_issues = [issue for issue in issues if issue.kind == "numbering_broken"]
    assert len(numbering_issues) == 0


def test_大纲章节缺稿_报chapter_missing() -> None:
    """结构完整性：大纲章节没有成稿时报 error 级 chapter_missing。"""
    state = _state([_draft("ch1", "无角标正文。")], [])
    state["outline"] = [
        ChapterSpec(id="ch1", title="章 ch1"),
        ChapterSpec(id="ch2", title="章 ch2"),
    ]
    result, _ = _run(state, [REVIEW_PASS])

    report = result["citation_report"]
    assert report.passed is False
    assert [(issue.kind, issue.chapter_id) for issue in report.issues] == [
        ("chapter_missing", "ch2")
    ]
    assert report.failed_chapter_ids == ["ch2"]
    assert result["status"] == WorkflowStatus.CITATION_CHECKING


def test_篇级评审fact_conflict_打回涉及章节() -> None:
    """跨章硬事实冲突是 error：进问题清单与 failed_chapter_ids 触发定向回退。"""
    state = _state(
        [_draft("ch1", "无角标正文一。"), _draft("ch2", "无角标正文二。")], []
    )
    result, _ = _run(
        state,
        [[_finding("fact_conflict", ["ch1", "ch2"], detail="两章结论相反")]],
    )

    report = result["citation_report"]
    assert report.passed is False
    assert [(issue.kind, issue.chapter_id) for issue in report.issues] == [
        ("fact_conflict", "ch1"),
        ("fact_conflict", "ch2"),
    ]
    # 每条问题都点名全部涉及章节，重写侧能看到冲突对方章。
    assert all(
        "跨章硬事实冲突（涉及章节 ch1、ch2）：两章结论相反" in issue.detail
        for issue in report.issues
    )
    assert report.failed_chapter_ids == ["ch1", "ch2"]
    assert result["citation_retry_count"] == 1
    assert result["status"] == WorkflowStatus.CITATION_CHECKING
    assert result["review_warnings"] == []


def test_篇级评审warn三维_不打回且写入review_warnings() -> None:
    """章间衔接/口径统一/跨章重复是 warn：呈人工不打回，重试计数归零。"""
    state = _state(
        [_draft("ch1", "无角标正文一。"), _draft("ch2", "无角标正文二。")],
        [],
        citation_retry_count=1,
    )
    result, _ = _run(
        state,
        [
            [
                _finding("transition", ["ch1", "ch2"], detail="承接生硬"),
                _finding("consistency", ["ch2"], detail="口径不一"),
                _finding("duplication", ["ch1", "ch2"], detail="大段重复"),
            ]
        ],
    )

    assert result["citation_report"].passed is True
    warnings = result["review_warnings"]
    assert len(warnings) == 3
    assert "章间衔接" in warnings[0] and "承接生硬" in warnings[0]
    assert "口径统一" in warnings[1] and "ch2" in warnings[1]
    assert "跨章重复" in warnings[2] and "大段重复" in warnings[2]
    assert result["citation_retry_count"] == 0
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW


def test_篇级评审幻觉章节与未知维度被丢弃() -> None:
    """涉及章节 id 不在大纲的剔除、剔空整条丢；未知维度直接丢。"""
    state = _state([_draft("ch1", "无角标正文。")], [])
    result, _ = _run(
        state,
        [
            [
                _finding("fact_conflict", ["ch99"], detail="全是幻觉章节"),
                _finding("nonsense", ["ch1"], detail="未知维度"),
                _finding("duplication", ["ch1", "ch幻觉"], detail="剔除幻觉后保留"),
            ]
        ],
    )

    assert result["citation_report"].passed is True
    # 前两条整条丢弃；第三条剔除幻觉 id 后仍有合法章节，保留为 warn。
    warnings = result["review_warnings"]
    assert len(warnings) == 1
    assert "剔除幻觉后保留" in warnings[0]
    assert "ch幻觉" not in warnings[0]
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW


def test_打回轮次review_warnings同样写入() -> None:
    """error 与 warn 同现：warn 每轮呈人工不因打回丢失，error 照常打回。"""
    state = _state(
        [_draft("ch1", "无角标正文一。"), _draft("ch2", "无角标正文二。")], []
    )
    result, _ = _run(
        state,
        [
            [
                _finding("fact_conflict", ["ch1"], detail="数字矛盾"),
                _finding("duplication", ["ch1", "ch2"], detail="大段重复"),
            ]
        ],
    )

    report = result["citation_report"]
    assert report.passed is False
    assert [(issue.kind, issue.chapter_id) for issue in report.issues] == [
        ("fact_conflict", "ch1")
    ]
    assert report.failed_chapter_ids == ["ch1"]
    assert result["status"] == WorkflowStatus.CITATION_CHECKING
    warnings = result["review_warnings"]
    assert len(warnings) == 1
    assert "跨章重复" in warnings[0] and "大段重复" in warnings[0]
