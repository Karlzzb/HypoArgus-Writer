"""chapter_drafts / citation_library 合并 reducer 与 keep_last reducer 的语义测试。"""

from domain.state import (
    ChapterDraft,
    Material,
    keep_last,
    merge_chapter_drafts,
    merge_citation_library,
)


def _draft(chapter_id: str, text: str = "正文") -> ChapterDraft:
    return ChapterDraft(chapter_id=chapter_id, text=text, summary=f"{chapter_id}摘要")


def _material(
    mat_id: str, chapter_id: str, url: str | None = None, excerpt: str = "摘录"
) -> Material:
    return Material(
        id=mat_id,
        hypothesis_id=f"{chapter_id}-p1-h1",
        chapter_id=chapter_id,
        source=f"来源 {mat_id}",
        url=url,
        excerpt=excerpt,
        relevance_score=0.8,
        verdict="pass",
    )


def test_同id替换_新id插入() -> None:
    existing = [_draft("ch1", "旧一"), _draft("ch2", "旧二")]
    merged = merge_chapter_drafts(existing, [_draft("ch2", "新二"), _draft("ch3")])
    assert [draft.chapter_id for draft in merged] == ["ch1", "ch2", "ch3"]
    assert merged[1].text == "新二"
    assert merged[0].text == "旧一"


def test_并行完成顺序不影响排序() -> None:
    # 并行分支按完成先后到达：ch10 先于 ch2 到达，排序仍按数字后缀。
    merged = merge_chapter_drafts([], [_draft("ch10")])
    merged = merge_chapter_drafts(merged, [_draft("ch2")])
    merged = merge_chapter_drafts(merged, [_draft("ch1")])
    assert [draft.chapter_id for draft in merged] == ["ch1", "ch2", "ch10"]


def test_整值覆盖回写在合并语义下等价() -> None:
    # 串行节点回写完整列表（同 id 逐项替换）：结果与整值覆盖一致。
    existing = [_draft("ch1", "旧一"), _draft("ch2", "旧二")]
    full_rewrite = [_draft("ch1", "旧一"), _draft("ch2", "改二")]
    merged = merge_chapter_drafts(existing, full_rewrite)
    assert [draft.text for draft in merged] == ["旧一", "改二"]


def test_空值与None入参安全() -> None:
    assert merge_chapter_drafts(None, None) == []
    assert merge_chapter_drafts(None, [_draft("ch1")])[0].chapter_id == "ch1"
    assert merge_chapter_drafts([_draft("ch1")], None)[0].chapter_id == "ch1"


def test_非ch形态id靠后按字典序稳定排序() -> None:
    merged = merge_chapter_drafts(
        [], [_draft("附录b"), _draft("ch2"), _draft("附录a"), _draft("ch1")]
    )
    assert [draft.chapter_id for draft in merged] == ["ch1", "ch2", "附录a", "附录b"]


def test_keep_last_取最后写入值() -> None:
    assert keep_last("旧", "新") == "新"


def test_引文库同id替换_新id插入且按章排序() -> None:
    existing = [_material("m1", "ch1", excerpt="旧"), _material("m2", "ch2")]
    merged = merge_citation_library(
        existing, [_material("m1", "ch1", excerpt="新"), _material("m3", "ch3")]
    )
    assert [material.id for material in merged] == ["m1", "m2", "m3"]
    assert merged[0].excerpt == "新"


def test_引文库并行分支到达顺序不影响结果() -> None:
    ch1_mats = [_material("m1a", "ch1"), _material("m1b", "ch1")]
    ch2_mats = [_material("m2a", "ch2")]
    先一后二 = merge_citation_library(merge_citation_library([], ch1_mats), ch2_mats)
    先二后一 = merge_citation_library(merge_citation_library([], ch2_mats), ch1_mats)
    assert 先一后二 == 先二后一
    assert [material.id for material in 先一后二] == ["m1a", "m1b", "m2a"]


def test_引文库跨章按URL去重保留前章条目() -> None:
    url = "https://example.com/shared"
    ch1_mats = [_material("m1", "ch1", url=url)]
    ch3_mats = [_material("m3", "ch3", url=url)]
    # 无论分支到达顺序，同 URL 只保留章序靠前的条目。
    for first, second in ((ch1_mats, ch3_mats), (ch3_mats, ch1_mats)):
        merged = merge_citation_library(merge_citation_library([], first), second)
        assert [material.id for material in merged] == ["m1"]


def test_引文库url为None不参与去重() -> None:
    merged = merge_citation_library(
        [], [_material("m1", "ch1", url=None), _material("m2", "ch2", url=None)]
    )
    assert [material.id for material in merged] == ["m1", "m2"]


def test_引文库整值覆盖回写在合并语义下等价() -> None:
    # 修订轮增量检索路径回写完整列表：既有条目逐项同 id 替换、新条目插入。
    existing = [_material("m1", "ch1"), _material("m2", "ch2")]
    full_rewrite = [_material("m1", "ch1"), _material("m2", "ch2"), _material("m2b", "ch2")]
    merged = merge_citation_library(existing, full_rewrite)
    assert [material.id for material in merged] == ["m1", "m2", "m2b"]


def test_引文库空值与None入参安全() -> None:
    assert merge_citation_library(None, None) == []
    assert merge_citation_library(None, [_material("m1", "ch1")])[0].id == "m1"
    assert merge_citation_library([_material("m1", "ch1")], None)[0].id == "m1"
