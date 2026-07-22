"""章节编号连续唯一校验：纯程序逻辑，不调 LLM。

检查全文各章正文中的二级标题编号（形如「## 一、」「## 二、」）连续且不重复，
消除拼接痕迹（issue #18）。

编号来源根因（两个层面，均在本校验覆盖范围内）：
1. 骨架层注入：编号由模板骨架确定（template_skeleton.py 解析模板 H2 标题的
   中文数字序号），经 framework_orchestrator 实例化后保留在 ChapterSpec.title 中；
   大纲实例化裁剪章节（applicable=false）时可能产生断号。
2. 模型自生成：rewriter_loop 的 LLM 被要求在 chapter_text 中原样输出含编号的
   ## 标题，但可能自行生成或修改编号，导致重复或断号。

校验层级：全局校验层（document_reviewer 调用本模块），不修改 rewriter_loop
单章边界内的任何职责（遵守 ADR-0001 非子图边界与章级 checkpoint 约束）。

自由结构模式的大纲标题可以没有中文数字编号，不带编号的章节不参与连续性校验，
不误报；仅当大纲标题带编号而正文标题缺失或不一致、或正文编号序列重复断号时报问题。
"""

import re
from dataclasses import dataclass

from domain.state import ChapterDraft, ChapterSpec
from domain.template_skeleton import CHINESE_DIGITS, CHINESE_NUMBERING

# 标题文本行首的中文数字编号 + 顿号/点号，如「一、专业名称及代码」；
# 复用 template_skeleton 的序号识别口径。
_TITLE_NUMBERING = CHINESE_NUMBERING

# 二级标题行首的中文数字编号，如「## 一、专业名称及代码」。
_H2_CHINESE_NUMBERING = re.compile(rf"^##\s*([{CHINESE_DIGITS}]+)[、.．]\s*")


@dataclass(frozen=True)
class NumberingIssue:
    """章节编号问题：章 id + 正文中的实际编号 + 问题描述。"""

    chapter_id: str
    actual_numbering: str | None
    """正文中提取到的编号原文；无编号时为 None。"""
    message: str


def _h2_heading_lines(chapter_text: str) -> list[str]:
    """返回正文中所有二级标题行（去首尾空白）；围栏代码块内的 ## 不算标题。"""
    headings: list[str] = []
    in_fence = False
    for line in chapter_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("## ") and not stripped.startswith("### "):
            headings.append(stripped)
    return headings


def _extract_h2_numbering(chapter_text: str) -> str | None:
    """从正文提取首个二级标题的中文数字编号原文；无编号时返回 None。

    只取首个 ## 标题行（模型偶发在正文中误用 ## 时以首行为准）；
    围栏代码块内的 ## 不算标题。
    """
    headings = _h2_heading_lines(chapter_text)
    if not headings:
        return None
    match = _H2_CHINESE_NUMBERING.match(headings[0])
    return match.group(1) if match else None


def _expected_numbering(title: str) -> str | None:
    """从大纲章节标题解析预期编号原文；标题不带编号时返回 None。"""
    match = _TITLE_NUMBERING.match(title.strip())
    return match.group(1) if match else None


# 中文数字到阿拉伯数字的转换表：一到二十，
# 覆盖公文模板的实际章节编号范围（模板最多十余章）。
_CHINESE_TO_ARABIC: dict[str, int] = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
    "十三": 13,
    "十四": 14,
    "十五": 15,
    "十六": 16,
    "十七": 17,
    "十八": 18,
    "十九": 19,
    "二十": 20,
}

_ARABIC_TO_CHINESE = {value: key for key, value in _CHINESE_TO_ARABIC.items()}


def _parse_chinese_numeral(text: str) -> int | None:
    """中文数字转阿拉伯数字；不在转换表中时返回 None（非标准编号或超出范围）。"""
    return _CHINESE_TO_ARABIC.get(text)


def _arabic_to_chinese(num: int) -> str:
    """阿拉伯数字转中文数字（一到二十）；超出范围返回阿拉伯数字原文。"""
    return _ARABIC_TO_CHINESE.get(num, str(num))


def validate_chapter_numbering(
    drafts: list[ChapterDraft], outline: list[ChapterSpec]
) -> list[NumberingIssue]:
    """校验全文章节编号连续唯一性，返回问题列表（按章节顺序）。

    检查项：
    1. 单章草稿只允许一个二级标题（模型把多章内容并入一章的根因场景，
       真实 E2E 复跑 issue #19 发现）；
    2. 大纲标题带编号的章节，正文首个 ## 标题必须携带同一编号
       （缺失或不一致均报问题——模型自生成编号的根因场景）；
    3. 正文中带编号的章节按出现顺序构成的编号序列必须从「一」起连续递增、
       不重复（骨架层裁剪断号与模型重复编号的根因场景）；
    4. 编号必须是一到二十的标准中文数字。

    大纲与正文均不带编号的章节不参与校验（自由结构模式不误报）。
    无问题时返回空列表。
    """
    titles = {chapter.id: chapter.title for chapter in outline}
    issues: list[NumberingIssue] = []
    numbered: list[tuple[str, str]] = []  # 正文带编号的章：（chapter_id, 编号原文）。

    for draft in drafts:
        headings = _h2_heading_lines(draft.text)
        if len(headings) > 1:
            # 单章草稿内多出二级标题＝模型在一章里凭空多写了一章，
            # 拼接后破坏全篇章节结构与编号连续性；只看首个 ## 会漏检。
            issues.append(
                NumberingIssue(
                    chapter_id=draft.chapter_id,
                    actual_numbering=None,
                    message=(
                        f"章节 {draft.chapter_id} 正文含多个二级标题"
                        f"（{len(headings)} 个，单章应只有一个），"
                        "疑将多章内容并入一章，须拆分或删除多余标题。"
                    ),
                )
            )
        actual = _extract_h2_numbering(draft.text)
        expected = _expected_numbering(titles.get(draft.chapter_id, ""))
        if expected is not None and actual is None:
            issues.append(
                NumberingIssue(
                    chapter_id=draft.chapter_id,
                    actual_numbering=None,
                    message=(
                        f"章节 {draft.chapter_id} 大纲标题编号为「{expected}」，"
                        "但正文二级标题缺少编号。"
                    ),
                )
            )
            continue
        if expected is not None and actual != expected:
            issues.append(
                NumberingIssue(
                    chapter_id=draft.chapter_id,
                    actual_numbering=actual,
                    message=(
                        f"章节 {draft.chapter_id} 正文编号「{actual}」"
                        f"与大纲标题编号「{expected}」不一致。"
                    ),
                )
            )
        if actual is not None:
            numbered.append((draft.chapter_id, actual))

    seen: dict[int, str] = {}  # 已出现的编号数值 → 首次出现的 chapter_id。
    for position, (chapter_id, actual) in enumerate(numbered, start=1):
        value = _parse_chinese_numeral(actual)
        if value is None:
            issues.append(
                NumberingIssue(
                    chapter_id=chapter_id,
                    actual_numbering=actual,
                    message=(
                        f"章节 {chapter_id} 编号「{actual}」无法识别"
                        "（仅支持一到二十的标准中文数字）。"
                    ),
                )
            )
            continue
        if value in seen:
            issues.append(
                NumberingIssue(
                    chapter_id=chapter_id,
                    actual_numbering=actual,
                    message=(
                        f"章节 {chapter_id} 编号「{actual}」"
                        f"与章节 {seen[value]} 重复。"
                    ),
                )
            )
        else:
            seen[value] = chapter_id
        if value != position:
            issues.append(
                NumberingIssue(
                    chapter_id=chapter_id,
                    actual_numbering=actual,
                    message=(
                        f"章节 {chapter_id} 编号断号或乱序："
                        f"预期「{_arabic_to_chinese(position)}」，实际「{actual}」。"
                    ),
                )
            )
    return issues
