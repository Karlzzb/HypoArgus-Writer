"""chapter_drafts 合并 reducer 与 keep_last reducer 的语义测试。"""

from domain.state import ChapterDraft, keep_last, merge_chapter_drafts


def _draft(chapter_id: str, text: str = "正文") -> ChapterDraft:
    return ChapterDraft(chapter_id=chapter_id, text=text, summary=f"{chapter_id}摘要")


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
