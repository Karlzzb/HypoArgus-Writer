"""书目渲染：从结构化引文库按任意书目格式渲染最终交付，格式与内容解耦。

正文只嵌轻量角标（素材 ID）；交付时按全文首次引用顺序统一重编号，
角标替换为数字序号，书目条目由所选格式的渲染器生成。
新增格式只需在渲染器注册表登记一个条目渲染函数，内容层完全不动。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

from domain.citation_reconciler import MARKER_PATTERN
from domain.state import ChapterDraft, Material, SourceKind

EntryRenderer = Callable[[int, Material], str]
"""条目渲染器：（序号，素材）→ 单条书目文本。"""


# GB/T 7714—2015 表 1 文献类型标识按来源通道选取：联网来源是在线电子公告
# [EB/OL]、知识库来源是在线数据库 [DB/OL]、结构化数据来源是数据集 [DS]。
# 键类型收紧到 SourceKind：新增通道漏配码表在类型检查期暴露。
_GBT7714_TYPE_CODES: dict[SourceKind, str] = {
    "web": "EB/OL",
    "knowledge_base": "DB/OL",
    "structured_data": "DS",
}


def _render_gbt7714(index: int, material: Material) -> str:
    """GB/T 7714 风格：[序号] 来源[类型标识]. 链接."""
    type_code = _GBT7714_TYPE_CODES[material.source_kind]
    link = f" {material.url}." if material.url else ""
    return f"[{index}] {material.source}[{type_code}].{link}"


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

_REFERENCE_SECTION_HEADING_PATTERN = re.compile(
    r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:参考文献|参考资料|references?)\s*[:：]?\s*$"
)
"""模型生成的章内参考文献段起始行。"""

_REFERENCE_ENTRY_LEAD_PATTERN = re.compile(
    r"^\s*(?:\[\d+\]|\[[A-Za-z0-9_\-]+\]|\d+[.)]|[-*])\s+"
)
"""参考文献段标题后的首个非空行需呈现列表条目形态。"""


@dataclass(frozen=True)
class BibliographyEntry:
    """一条书目：全文统一序号、素材 ID 与格式化文本。"""

    index: int
    material_id: str
    material: Material
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


def _strip_generated_reference_section(text: str) -> str:
    """剔除模型生成的章内参考文献段，最终书目只由渲染器统一产出。"""
    for match in _REFERENCE_SECTION_HEADING_PATTERN.finditer(text):
        for line in text[match.end() :].splitlines():
            if not line.strip():
                continue
            if _REFERENCE_ENTRY_LEAD_PATTERN.match(line):
                return text[: match.start()].rstrip()
            break
    return text


def _material_marker_ids(text: str) -> list[str]:
    """提取可作为素材 ID 的正文角标，排除最终展示编号。"""
    return [
        material_id
        for material_id in MARKER_PATTERN.findall(text)
        if not material_id.isdecimal()
    ]


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
    draft_bodies = [
        RenderedChapter(
            chapter_id=draft.chapter_id,
            text=_strip_generated_reference_section(draft.text),
        )
        for draft in drafts
    ]
    order: dict[str, int] = {}
    for draft in draft_bodies:
        for material_id in _material_marker_ids(draft.text):
            if material_id in materials_by_id and material_id not in order:
                order[material_id] = len(order) + 1

    chapters = [
        RenderedChapter(
            chapter_id=draft.chapter_id,
            text=MARKER_PATTERN.sub(
                lambda match: (
                    f"[{order[match.group(1)]}]"
                    if not match.group(1).isdecimal() and match.group(1) in order
                    else match.group(0)
                ),
                draft.text,
            ),
        )
        for draft in draft_bodies
    ]
    entries = [
        BibliographyEntry(
            index=index,
            material_id=material_id,
            material=materials_by_id[material_id],
            text=renderer(index, materials_by_id[material_id]),
        )
        for material_id, index in order.items()
    ]
    return RenderedArticle(format=format, chapters=chapters, entries=entries)
