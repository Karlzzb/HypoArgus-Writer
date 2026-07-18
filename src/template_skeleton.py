"""模板骨架解析纯逻辑：输入 Markdown 模板文本，输出章节层级与填充变量。

零 LLM、零 IO 依赖。只识别行首 H1/H2/H3 标题；
三级标题归属其前最近的二级标题，围栏代码块内的井号行不算标题。
"""

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SectionSkeleton:
    """单个标题节点：序号原文 + 去序号后的标题文本。"""

    numbering: str
    """序号原文，如 "一"、"（一）"、"1"；无序号为 ""。"""
    title: str
    """去除序号与分隔符后的标题文本（可含 {填充变量}）。"""
    subsections: tuple["SectionSkeleton", ...] = ()
    """仅二级标题节点携带其三级子节点。"""


@dataclass(frozen=True)
class TemplateSkeleton:
    """整篇模板的骨架解析结果。"""

    doc_title: str
    """首个一级标题原文（去掉 "# " 前缀）；无则为 ""。"""
    chapters: tuple[SectionSkeleton, ...] = ()
    """全文二级标题按文档顺序。"""
    variables: tuple[str, ...] = ()
    """全文 {xxx} 填充变量名，去重且保持首次出现顺序。"""


_CHINESE_DIGITS = "〇零一二三四五六七八九十百千"

# 中文数字 + 顿号或点号，如 "一、"、"十一、"。
_CHINESE_NUMBERING = re.compile(rf"^([{_CHINESE_DIGITS}]+)[、.．]\s*")
# 全角括号中文数字，如 "（一）"；序号原文含括号。
_PAREN_NUMBERING = re.compile(rf"^（[{_CHINESE_DIGITS}]+）\s*")
# 阿拉伯数字 + 点号或顿号，如 "1."、"2、"。
_ARABIC_NUMBERING = re.compile(r"^(\d+)[.、．]\s*")

# 填充变量：花括号内不含嵌套花括号、不跨行。公开供占位替换等场景复用。
VARIABLE_PATTERN = re.compile(r"\{([^{}\n]+)\}")

_FENCE = re.compile(r"^\s*```")


def _split_numbering(heading_text: str) -> tuple[str, str]:
    """把标题文本拆成（序号原文, 标题正文），无序号时序号为空串。"""
    match = _PAREN_NUMBERING.match(heading_text)
    if match:
        return match.group(0).strip(), heading_text[match.end() :].strip()
    for pattern in (_CHINESE_NUMBERING, _ARABIC_NUMBERING):
        match = pattern.match(heading_text)
        if match:
            return match.group(1), heading_text[match.end() :].strip()
    return "", heading_text.strip()


@dataclass
class _ChapterBuilder:
    """解析过程中的可变二级标题累积器。"""

    numbering: str
    title: str
    subsections: list[SectionSkeleton] = field(default_factory=list)

    def build(self) -> SectionSkeleton:
        return SectionSkeleton(
            numbering=self.numbering,
            title=self.title,
            subsections=tuple(self.subsections),
        )


def parse_template_skeleton(markdown_text: str) -> TemplateSkeleton:
    """解析模板文本，返回文档标题、二/三级标题层级与填充变量。

    规则：doc_title 取首个 H1；H3 归属其前最近的 H2，孤儿 H3 忽略；
    围栏代码块内的行不参与标题识别，但其中的 {变量} 照常收集。
    """
    doc_title = ""
    chapters: list[_ChapterBuilder] = []
    variables: dict[str, None] = {}
    in_fence = False

    for line in markdown_text.splitlines():
        for name in VARIABLE_PATTERN.findall(line):
            variables.setdefault(name)

        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if line.startswith("# ") and not doc_title:
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            numbering, title = _split_numbering(line[3:].strip())
            chapters.append(_ChapterBuilder(numbering=numbering, title=title))
        elif line.startswith("### ") and chapters:
            numbering, title = _split_numbering(line[4:].strip())
            chapters[-1].subsections.append(
                SectionSkeleton(numbering=numbering, title=title)
            )

    return TemplateSkeleton(
        doc_title=doc_title,
        chapters=tuple(builder.build() for builder in chapters),
        variables=tuple(variables),
    )
