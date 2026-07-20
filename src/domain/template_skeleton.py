"""模板骨架解析纯逻辑：输入 Markdown 模板文本，输出章节层级与填充变量。

零 LLM、零 IO 依赖。只识别行首 H1/H2/H3 标题；
三级标题归属其前最近的二级标题，围栏代码块内的井号行不算标题。

可重复章标记（ADR-0005 三条封顶）：标题行尾的 <!-- repeat: 1..N -->
只允许标在章级（H2），一份模板至多一个可重复章位；
标在一级/小节级标题或出现多个重复位时解析直接报错，
不引入条件章、章间引用等任何其他结构化标记。
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
    repeatable: bool = False
    """可重复章位（repeat: 1..N）：实例化时按用户需求展开为 N 个具体章。"""


@dataclass(frozen=True)
class TemplateSkeleton:
    """整篇模板的骨架解析结果。"""

    doc_title: str
    """首个一级标题原文（去掉 "# " 前缀）；无则为 ""。"""
    chapters: tuple[SectionSkeleton, ...] = ()
    """全文二级标题按文档顺序。"""
    variables: tuple[str, ...] = ()
    """全文 {xxx} 填充变量名，去重且保持首次出现顺序。"""


CHINESE_DIGITS = "〇零一二三四五六七八九十百千"
"""中文数字字符集：标题序号识别的统一口径，公开供编号校验等场景复用。"""

# 中文数字 + 顿号或点号，如 "一、"、"十一、"；公开供编号校验等场景复用。
CHINESE_NUMBERING = re.compile(rf"^([{CHINESE_DIGITS}]+)[、.．]\s*")
# 全角括号中文数字，如 "（一）"；序号原文含括号。
_PAREN_NUMBERING = re.compile(rf"^（[{CHINESE_DIGITS}]+）\s*")
# 阿拉伯数字 + 点号或顿号，如 "1."、"2、"。
_ARABIC_NUMBERING = re.compile(r"^(\d+)[.、．]\s*")

# 填充变量：花括号内不含嵌套花括号、不跨行。公开供占位替换等场景复用。
VARIABLE_PATTERN = re.compile(r"\{([^{}\n]+)\}")

_FENCE = re.compile(r"^\s*```")

# 可重复章标记：标题行尾的 HTML 注释，字面 1..N（N 在实例化时由用户需求决定）。
_REPEAT_MARKER = re.compile(r"\s*<!--\s*repeat:\s*1\.\.N\s*-->\s*$")
# 疑似标记的宽口径：拦下 repeat: 1..3 之类的非规范写法，显式报错而非静默当标题文本。
_REPEAT_LIKE = re.compile(r"\s*<!--\s*repeat\b[^>]*-->\s*$")


def _strip_repeat_marker(heading_text: str) -> tuple[str, bool]:
    """剥离标题行尾的可重复章标记，返回（剥离后文本, 是否带标记）。

    形似 repeat 标记但不是规范 repeat: 1..N 写法的抛 ValueError，不静默放过。
    """
    match = _REPEAT_MARKER.search(heading_text)
    if match is not None:
        return heading_text[: match.start()].strip(), True
    like = _REPEAT_LIKE.search(heading_text)
    if like is not None:
        raise ValueError(
            f"可重复标记仅支持 repeat: 1..N 一种写法：{heading_text!r}"
        )
    return heading_text, False


def _split_numbering(heading_text: str) -> tuple[str, str]:
    """把标题文本拆成（序号原文, 标题正文），无序号时序号为空串。"""
    match = _PAREN_NUMBERING.match(heading_text)
    if match:
        return match.group(0).strip(), heading_text[match.end() :].strip()
    for pattern in (CHINESE_NUMBERING, _ARABIC_NUMBERING):
        match = pattern.match(heading_text)
        if match:
            return match.group(1), heading_text[match.end() :].strip()
    return "", heading_text.strip()


@dataclass
class _ChapterBuilder:
    """解析过程中的可变二级标题累积器。"""

    numbering: str
    title: str
    repeatable: bool = False
    subsections: list[SectionSkeleton] = field(default_factory=list)

    def build(self) -> SectionSkeleton:
        return SectionSkeleton(
            numbering=self.numbering,
            title=self.title,
            subsections=tuple(self.subsections),
            repeatable=self.repeatable,
        )


def parse_template_skeleton(markdown_text: str) -> TemplateSkeleton:
    """解析模板文本，返回文档标题、二/三级标题层级与填充变量。

    规则：doc_title 取首个 H1；H3 归属其前最近的 H2，孤儿 H3 忽略；
    围栏代码块内的行不参与标题识别，但其中的 {变量} 照常收集。
    可重复章标记违反封顶（标在非 H2、或一份模板多个重复位）抛 ValueError。
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

        if line.startswith("# "):
            heading, repeatable = _strip_repeat_marker(line[2:].strip())
            if repeatable:
                raise ValueError(
                    f"可重复标记只允许标在章级（H2）标题：{heading!r}"
                )
            if not doc_title:
                doc_title = heading
        elif line.startswith("## "):
            heading, repeatable = _strip_repeat_marker(line[3:].strip())
            numbering, title = _split_numbering(heading)
            chapters.append(
                _ChapterBuilder(
                    numbering=numbering, title=title, repeatable=repeatable
                )
            )
        elif line.startswith("### "):
            heading, repeatable = _strip_repeat_marker(line[4:].strip())
            if repeatable:
                raise ValueError(
                    f"可重复标记只允许标在章级（H2）标题，不允许小节级或嵌套：{heading!r}"
                )
            # 孤儿三级标题（首个 H2 之前）无所属章可挂，剥离并校验标记后忽略。
            if chapters:
                numbering, title = _split_numbering(heading)
                chapters[-1].subsections.append(
                    SectionSkeleton(numbering=numbering, title=title)
                )

    repeatable_titles = [c.title for c in chapters if c.repeatable]
    if len(repeatable_titles) > 1:
        raise ValueError(
            f"一份模板至多一个可重复章位，实际标记了 {len(repeatable_titles)} 个："
            f"{repeatable_titles}"
        )

    return TemplateSkeleton(
        doc_title=doc_title,
        chapters=tuple(builder.build() for builder in chapters),
        variables=tuple(variables),
    )
