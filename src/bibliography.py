"""书目渲染：从结构化引文库按任意书目格式渲染最终交付，格式与内容解耦。

正文只嵌轻量角标（素材 ID）；交付时按全文首次引用顺序统一重编号，
角标替换为数字序号，书目条目由所选格式的渲染器生成。
新增格式只需在渲染器注册表登记一个条目渲染函数，内容层完全不动。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from citation_reconciler import MARKER_PATTERN
from state import ChapterDraft, Material

EntryRenderer = Callable[[int, Material], str]
"""条目渲染器：（序号，素材）→ 单条书目文本。"""


def _render_gbt7714(index: int, material: Material) -> str:
    """GB/T 7714 风格：[序号] 来源[EB/OL]. 链接."""
    link = f" {material.url}." if material.url else ""
    return f"[{index}] {material.source}[EB/OL].{link}"


def _render_apa(index: int, material: Material) -> str:
    """APA 风格：来源. Retrieved from 链接

    APA 参考文献本身不编号，故忽略序号；素材字段只有来源与链接，
    作者与年份等要素待真实检索素材补充后再充实。
    """
    del index
    link = f" Retrieved from {material.url}" if material.url else ""
    return f"{material.source}.{link}"


def _render_markdown(index: int, material: Material) -> str:
    """Markdown 有序列表：序号. [来源](链接)"""
    label = f"[{material.source}]({material.url})" if material.url else material.source
    return f"{index}. {label}"


_RENDERERS: dict[str, EntryRenderer] = {
    "gbt7714": _render_gbt7714,
    "apa": _render_apa,
    "markdown": _render_markdown,
}

SUPPORTED_FORMATS: tuple[str, ...] = tuple(_RENDERERS)


@dataclass(frozen=True)
class BibliographyEntry:
    """一条书目：全文统一序号、素材 ID 与格式化文本。"""

    index: int
    material_id: str
    text: str


@dataclass(frozen=True)
class RenderedChapter:
    """重编号后的章节正文：角标已从素材 ID 替换为数字序号。"""

    chapter_id: str
    text: str


@dataclass(frozen=True)
class RenderedArticle:
    """最终交付：重编号正文 + 按格式渲染的书目列表。"""

    format: str
    chapters: list[RenderedChapter]
    entries: list[BibliographyEntry]


def render_article(
    drafts: Sequence[ChapterDraft],
    citation_library: Sequence[Material],
    format: str,
) -> RenderedArticle:
    """按书目格式渲染最终交付。

    只有被正文实际引用且在引文库中登记的素材才进入书目；
    序号按全文首次引用顺序分配，重复引用共用同一序号；
    未落库的孤角标原样保留（终审门禁负责拦截，渲染不做裁决）。
    """
    renderer = _RENDERERS.get(format)
    if renderer is None:
        raise ValueError(
            f"不支持的书目格式：{format!r}，支持的格式：{'、'.join(SUPPORTED_FORMATS)}"
        )

    materials_by_id = {material.id: material for material in citation_library}
    order: dict[str, int] = {}
    for draft in drafts:
        for material_id in MARKER_PATTERN.findall(draft.text):
            if material_id in materials_by_id and material_id not in order:
                order[material_id] = len(order) + 1

    chapters = [
        RenderedChapter(
            chapter_id=draft.chapter_id,
            text=MARKER_PATTERN.sub(
                lambda match: (
                    f"[{order[match.group(1)]}]"
                    if match.group(1) in order
                    else match.group(0)
                ),
                draft.text,
            ),
        )
        for draft in drafts
    ]
    entries = [
        BibliographyEntry(
            index=index,
            material_id=material_id,
            text=renderer(index, materials_by_id[material_id]),
        )
        for material_id, index in order.items()
    ]
    return RenderedArticle(format=format, chapters=chapters, entries=entries)
