"""书目渲染纯逻辑测试：格式与内容解耦、按首次引用顺序重编号。"""

import pytest

from domain.bibliography import SUPPORTED_FORMATS, render_article
from domain.state import ChapterDraft, Material, SourceKind


def _material(
    material_id: str,
    chapter_id: str,
    url: str | None,
    source_kind: SourceKind = "web",
) -> Material:
    return Material(
        id=material_id,
        hypothesis_id=f"h-{material_id}",
        chapter_id=chapter_id,
        source=f"来源{material_id}",
        url=url,
        source_kind=source_kind,
        excerpt="摘录",
        relevance_score=0.9,
        verdict="pass",
    )


LIBRARY = [
    _material("m1", "ch1", "https://example.com/1"),
    _material("m2", "ch1", None),
    _material("m3", "ch2", "https://example.com/3"),
]

DRAFTS = [
    ChapterDraft(chapter_id="ch1", text="观点甲[m2]，观点乙[m1]。", summary="s1"),
    ChapterDraft(chapter_id="ch2", text="观点丙[m3]，再证甲[m2]。", summary="s2"),
]


def test_按首次引用顺序重编号且重复引用共用同一序号():
    rendered = render_article(DRAFTS, LIBRARY, "markdown")

    # m2 首现于 ch1 在 m1 之前 → m2 是 [1]，m1 是 [2]，m3 是 [3]。
    assert rendered.chapters[0].text == "观点甲[1]，观点乙[2]。"
    assert rendered.chapters[1].text == "观点丙[3]，再证甲[1]。"
    assert [entry.material_id for entry in rendered.entries] == ["m2", "m1", "m3"]
    assert [entry.index for entry in rendered.entries] == [1, 2, 3]


def test_未被正文引用的素材不进入书目():
    library = [*LIBRARY, _material("m9", "ch2", None)]
    rendered = render_article(DRAFTS, library, "markdown")
    assert all(entry.material_id != "m9" for entry in rendered.entries)


def test_三种内置格式渲染同一引文库产出不同文本():
    texts = {
        fmt: [entry.text for entry in render_article(DRAFTS, LIBRARY, fmt).entries]
        for fmt in SUPPORTED_FORMATS
    }
    assert set(texts) == {"gbt7714", "apa", "markdown"}
    # 同一素材在不同格式下呈现不同，但都包含来源；有链接时链接在场。
    for entries in texts.values():
        assert "来源m2" in entries[0]
        assert "https://example.com/1" in entries[1]
    assert len({entries[1] for entries in texts.values()}) == 3


def test_gbt7714按来源通道输出类型标识():
    """联网 [EB/OL]、知识库 [DB/OL]、结构化数据 [DS]，联网来源带真实链接位。"""
    library = [
        _material("mw", "ch1", "https://example.com/w", source_kind="web"),
        _material("mk", "ch1", None, source_kind="knowledge_base"),
        _material("ms", "ch1", None, source_kind="structured_data"),
    ]
    drafts = [ChapterDraft(chapter_id="ch1", text="甲[mw]乙[mk]丙[ms]。", summary="s")]
    rendered = render_article(drafts, library, "gbt7714")
    assert rendered.entries[0].text == "[1] 来源mw[EB/OL]. https://example.com/w."
    assert rendered.entries[1].text == "[2] 来源mk[DB/OL]."
    assert rendered.entries[2].text == "[3] 来源ms[DS]."


def test_gbt7714条目带方括号序号_markdown条目为有序列表():
    rendered = render_article(DRAFTS, LIBRARY, "gbt7714")
    assert rendered.entries[0].text.startswith("[1] ")
    markdown = render_article(DRAFTS, LIBRARY, "markdown")
    assert markdown.entries[0].text.startswith("1. ")


def test_未知格式报错并列出支持的格式():
    with pytest.raises(ValueError) as excinfo:
        render_article(DRAFTS, LIBRARY, "chicago")
    for fmt in SUPPORTED_FORMATS:
        assert fmt in str(excinfo.value)


def test_正文角标未落库时保留原样并不产生条目():
    drafts = [ChapterDraft(chapter_id="ch1", text="孤角标[mx]。", summary="s")]
    rendered = render_article(drafts, LIBRARY, "markdown")
    assert rendered.chapters[0].text == "孤角标[mx]。"
    assert rendered.entries == []
