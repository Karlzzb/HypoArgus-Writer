"""citation_validator 节点单元测试：用假 LLM 预置语义核查应答直接调用节点函数。

覆盖：全部通过、程序对账失败、语义核查失败、自检失败合并、
重试超限携带警告、增量核查范围与 LLM 调用次数、环境变量读取。
"""

import json
from typing import Any

import pytest

from nodes.citation_validator import (
    ValidatorConfig,
    load_validator_config,
    make_citation_validator_node,
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
    """构造一章草稿。"""
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
    state = initial_state("需求", "身份", "trace-cv")
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


def _run(
    state: WritingAgentState,
    responses: list[Any],
    max_retries: int = 2,
) -> tuple[dict[str, Any], FakeLLM]:
    """预置应答序列（list 自动转 JSON 文本）后执行一次节点。"""
    fake = FakeLLM(
        [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in responses
        ]
    )
    node = make_citation_validator_node(
        lambda unit: fake, ValidatorConfig(max_retries=max_retries)
    )
    return dict(node(state)), fake


def test_全部通过_报告放行且计数归零() -> None:
    state = _state(
        [_draft("ch1", "引用[m1]。")],
        [_mat("m1", "ch1")],
        citation_retry_count=1,
    )
    result, fake = _run(state, [[_aligned("m1")]])

    report = result["citation_report"]
    assert report.passed is True
    assert report.issues == []
    assert report.failed_chapter_ids == []
    assert result["citation_retry_count"] == 0
    assert result["citation_warnings"] == []
    assert result["revised_chapter_ids"] == []
    # 通过即将进入人工中断点：等待人工期间的状态机值由本节点写入。
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    assert result["current_node_llm_config"]["unit"] == "citation_validator"
    assert len(fake.calls) == 1


def test_程序对账失败_定位章节且无语义误报() -> None:
    state = _state(
        [_draft("ch1", "引用[m1]与孤儿[m404]。"), _draft("ch2", "引用[m2]。")],
        [_mat("m1", "ch1"), _mat("m2", "ch2")],
    )
    result, _ = _run(state, [[_aligned("m1")], [_aligned("m2")]])

    report = result["citation_report"]
    assert report.passed is False
    assert [issue.kind for issue in report.issues] == ["orphan_marker"]
    assert report.failed_chapter_ids == ["ch1"]
    assert result["citation_retry_count"] == 1
    assert result["citation_warnings"] == []
    assert result["revised_chapter_ids"] == []


def test_语义核查失败_生成semantic_mismatch() -> None:
    state = _state([_draft("ch1", "引用[m1]。")], [_mat("m1", "ch1")])
    result, _ = _run(state, [[_aligned("m1", aligned=False, reason="观点不符")]])

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
        [[_aligned("m1"), _aligned("m外来", aligned=False, reason="幻觉项")]],
    )
    assert result["citation_report"].passed is True


def test_语义核查应答被对象包裹时同样解析() -> None:
    """关思考后模型偶发把核查项数组包进对象，应取出其中唯一的数组值。"""
    state = _state([_draft("ch1", "引用[m1]。")], [_mat("m1", "ch1")])
    result, _ = _run(
        state,
        [{"results": [_aligned("m1", aligned=False, reason="观点不符")]}],
    )
    report = result["citation_report"]
    assert report.passed is False
    assert [issue.kind for issue in report.issues] == ["semantic_mismatch"]


def test_语义核查应答为单个核查项对象时同样解析() -> None:
    """关思考后模型偶发直接返回单个核查项对象，应包成单元素数组。"""
    state = _state([_draft("ch1", "引用[m1]。")], [_mat("m1", "ch1")])
    result, _ = _run(state, [_aligned("m1", aligned=False, reason="观点不符")])
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
    result, _ = _run(state, [[_aligned("m1")]])

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
    result, _ = _run(state, [])
    assert [issue.kind for issue in result["citation_report"].issues] == [
        "self_check_failed"
    ]


def test_重试超限_携带未决引文警告() -> None:
    state = _state(
        [_draft("ch1", "孤儿[m404]。")],
        [],
        citation_retry_count=2,
    )
    result, _ = _run(state, [], max_retries=2)

    report = result["citation_report"]
    assert report.passed is False
    assert result["citation_retry_count"] == 3
    assert result["citation_warnings"]
    assert all(isinstance(warning, str) for warning in result["citation_warnings"])
    assert result["revised_chapter_ids"] == []


def test_增量核查_只审revised章节且LLM只调一次() -> None:
    state = _state(
        [_draft("ch1", "孤儿[m404]。"), _draft("ch2", "引用[m2]。")],
        [_mat("m2", "ch2")],
        revised_chapter_ids=["ch2"],
    )
    result, fake = _run(state, [[_aligned("m2")]])

    # ch1 的孤儿角标不在范围内，不报；LLM 只为 ch2 调用一次。
    assert result["citation_report"].passed is True
    assert len(fake.calls) == 1
    assert "m2" in fake.calls[0][1]["content"]
    assert result["revised_chapter_ids"] == []


def test_章节没有角标素材时跳过LLM调用() -> None:
    state = _state([_draft("ch1", "无角标正文。")], [])
    result, fake = _run(state, [])
    assert result["citation_report"].passed is True
    assert fake.calls == []


def test_failed_chapter_ids按大纲顺序去重() -> None:
    state = _state(
        [_draft("ch1", "孤儿[m404]。"), _draft("ch2", "孤儿[m405]与[m406]。")],
        [],
    )
    result, _ = _run(state, [])
    assert result["citation_report"].failed_chapter_ids == ["ch1", "ch2"]


def test_环境变量缺省为2() -> None:
    assert load_validator_config({}).max_retries == 2
    assert load_validator_config({"CITATION_MAX_RETRIES": ""}).max_retries == 2
    assert load_validator_config({"CITATION_MAX_RETRIES": "5"}).max_retries == 5


@pytest.mark.parametrize("raw", ["abc", "0", "-1"])
def test_环境变量非法值抛ValueError(raw: str) -> None:
    with pytest.raises(ValueError, match="CITATION_MAX_RETRIES"):
        load_validator_config({"CITATION_MAX_RETRIES": raw})
