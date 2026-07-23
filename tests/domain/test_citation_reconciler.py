"""citation_reconciler 纯逻辑单元测试：不涉及 LLM，直接构造引文库与章节草稿对账。"""

from domain.citation_reconciler import MARKER_PATTERN, reconcile
from domain.state import ChapterDraft, Material


def _mat(
    material_id: str, chapter_id: str, verdict: str = "pass"
) -> Material:
    """构造一条引文库素材，默认校验结论为 pass。"""
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


def _draft(chapter_id: str, text: str) -> ChapterDraft:
    """构造一章草稿，摘要固定占位。"""
    return ChapterDraft(chapter_id=chapter_id, text=text, summary="摘要")


def test_角标模式只匹配字母数字下划线连字符() -> None:
    text = "合法角标[m-1_a2]，中文方括号[待补充：专业名称]不应匹配。"
    assert MARKER_PATTERN.findall(text) == ["m-1_a2"]


def test_臆造占位角标报orphan且不误匹配合法中文方括号() -> None:
    """issue #62 收口发现：writer 在素材池为空时臆造 [素材id-N] 占位角标，
    MARKER_PATTERN 对其隐形（正文残留角注、书目为空、无 orphan 警告）。
    纵深防御：独立扫描把 [素材...] 占位形报为 orphan_marker（warn），
    不触碰 ASCII MARKER_PATTERN——合法中文方括号内容仍不匹配。"""
    text = "无可引素材时模型臆造占位[素材id-1]，合法中文方括号[待补充：专业名称]不应报。"
    issues = reconcile([_draft("ch1", text)], [])
    orphans = [
        i for i in issues
        if i.kind == "orphan_marker" and i.material_id == "素材id-1"
    ]
    assert len(orphans) == 1
    assert orphans[0].chapter_id == "ch1"
    assert orphans[0].detail
    # 合法中文方括号内容不被误报。
    assert not any(
        i.kind == "orphan_marker" and "待补充" in i.material_id
        for i in issues
    )


def test_臆造占位角标与ASCII孤儿同章按id升序混合排序() -> None:
    """占位角标与 ASCII 孤角标同章并存时，按 material_id 升序确定性排序。"""
    issues = reconcile([_draft("ch1", "先[素材id-2]再[m404]后[素材id-1]")], [])
    assert [
        (i.kind, i.material_id) for i in issues if i.kind == "orphan_marker"
    ] == [
        ("orphan_marker", "m404"),
        ("orphan_marker", "素材id-1"),
        ("orphan_marker", "素材id-2"),
    ]


def test_识别无来源的标注() -> None:
    issues = reconcile([_draft("ch1", "正文[m404]结尾")], [])
    assert len(issues) == 1
    assert issues[0].kind == "orphan_marker"
    assert issues[0].chapter_id == "ch1"
    assert issues[0].material_id == "m404"
    assert issues[0].detail


def test_识别跨章误引() -> None:
    library = [_mat("m1", "ch1")]
    drafts = [_draft("ch1", "本章引用[m1]。"), _draft("ch2", "误引[m1]。")]
    issues = reconcile(drafts, library)
    assert [(issue.kind, issue.chapter_id, issue.material_id) for issue in issues] == [
        ("cross_chapter", "ch2", "m1")
    ]


def test_识别未被引用的素材() -> None:
    library = [_mat("m1", "ch1"), _mat("m2", "ch1", verdict="fail")]
    issues = reconcile([_draft("ch1", "正文没有任何角标。")], library)
    # verdict=fail 的素材本就不给写作用，不算未引用。
    assert [(issue.kind, issue.material_id) for issue in issues] == [
        ("unused_material", "m1")
    ]
    assert issues[0].chapter_id == "ch1"


def test_全部一致时返回空列表() -> None:
    library = [_mat("m1", "ch1"), _mat("m2", "ch2")]
    drafts = [_draft("ch1", "引用[m1]。"), _draft("ch2", "引用[m2]。")]
    assert reconcile(drafts, library) == []


def test_增量对账只报范围内章节的问题() -> None:
    library = [_mat("m1", "ch1"), _mat("m2", "ch2")]
    drafts = [
        _draft("ch1", "孤儿[m404]，且 m1 未被引用。"),
        _draft("ch2", "引用[m2]。"),
    ]
    assert reconcile(drafts, library, scope_chapter_ids={"ch2"}) == []


def test_同一章节同时命中多类错误() -> None:
    library = [_mat("m1", "ch1"), _mat("m2", "ch2"), _mat("m3", "ch2")]
    drafts = [
        _draft("ch1", "孤儿[m404]，跨章[m2]，m1 未被引用。"),
        _draft("ch2", "引用[m2]与[m3]。"),
    ]
    issues = reconcile(drafts, library)
    assert [(issue.kind, issue.chapter_id, issue.material_id) for issue in issues] == [
        ("cross_chapter", "ch1", "m2"),
        ("orphan_marker", "ch1", "m404"),
        ("unused_material", "ch1", "m1"),
    ]
