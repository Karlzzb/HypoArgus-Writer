"""章节编号连续唯一校验测试（issue #18）。"""

from domain.chapter_numbering_validator import (
    _extract_h2_numbering,
    _parse_chinese_numeral,
    validate_chapter_numbering,
)
from domain.state import ChapterDraft, ChapterSpec, SelfCheck


def _draft(chapter_id: str, text: str) -> ChapterDraft:
    """构造最小章节草稿。"""
    return ChapterDraft(
        chapter_id=chapter_id, text=text, summary="摘要", self_check=SelfCheck()
    )


def _spec(chapter_id: str, title: str) -> ChapterSpec:
    """构造最小大纲章节。"""
    return ChapterSpec(id=chapter_id, title=title)


class TestExtractH2Numbering:
    """测试从正文提取首个二级标题的中文数字编号。"""

    def test_extract_standard_numbering(self):
        text = "## 一、专业名称及代码\n- 专业名称：计算机科学"
        assert _extract_h2_numbering(text) == "一"

    def test_extract_double_digit_numbering(self):
        text = "## 十一、附录\n附录内容"
        assert _extract_h2_numbering(text) == "十一"

    def test_extract_with_dot_separator(self):
        text = "## 二.入学要求\n要求内容"
        assert _extract_h2_numbering(text) == "二"

    def test_extract_with_fullwidth_dot(self):
        text = "## 三．学制学位\n学制内容"
        assert _extract_h2_numbering(text) == "三"

    def test_no_numbering_returns_none(self):
        text = "## 专业名称及代码\n内容"
        assert _extract_h2_numbering(text) is None

    def test_only_takes_first_h2(self):
        text = "## 一、标题一\n内容\n## 五、标题五"
        assert _extract_h2_numbering(text) == "一"

    def test_ignores_h3_headings(self):
        text = "### （一）子标题\n## 二、主标题"
        assert _extract_h2_numbering(text) == "二"

    def test_ignores_fenced_code_blocks(self):
        text = """```markdown
## 九、假标题
```
## 三、真标题"""
        assert _extract_h2_numbering(text) == "三"

    def test_no_h2_returns_none(self):
        text = "# 一级标题\n正文内容"
        assert _extract_h2_numbering(text) is None

    def test_empty_text_returns_none(self):
        assert _extract_h2_numbering("") is None


class TestParseChineseNumeral:
    """测试中文数字转阿拉伯数字。"""

    def test_parse_single_digits(self):
        assert _parse_chinese_numeral("一") == 1
        assert _parse_chinese_numeral("五") == 5
        assert _parse_chinese_numeral("九") == 9

    def test_parse_ten(self):
        assert _parse_chinese_numeral("十") == 10

    def test_parse_teens(self):
        assert _parse_chinese_numeral("十一") == 11
        assert _parse_chinese_numeral("十五") == 15
        assert _parse_chinese_numeral("十九") == 19

    def test_parse_twenty(self):
        assert _parse_chinese_numeral("二十") == 20

    def test_unsupported_returns_none(self):
        assert _parse_chinese_numeral("二十一") is None
        assert _parse_chinese_numeral("三十") is None
        assert _parse_chinese_numeral("百") is None

    def test_invalid_returns_none(self):
        assert _parse_chinese_numeral("abc") is None
        assert _parse_chinese_numeral("1") is None


class TestValidateChapterNumbering:
    """测试全文章节编号连续唯一性校验。"""

    def test_valid_consecutive_numbering(self):
        outline = [
            _spec("ch1", "一、专业名称及代码"),
            _spec("ch2", "二、入学要求"),
            _spec("ch3", "三、学制学位"),
        ]
        drafts = [
            _draft("ch1", "## 一、专业名称及代码\n内容"),
            _draft("ch2", "## 二、入学要求\n内容"),
            _draft("ch3", "## 三、学制学位\n内容"),
        ]
        assert validate_chapter_numbering(drafts, outline) == []

    def test_duplicate_numbering(self):
        outline = [_spec("ch1", "一、标题一"), _spec("ch2", "二、标题二")]
        drafts = [
            _draft("ch1", "## 一、标题一\n内容"),
            _draft("ch2", "## 一、标题二\n内容"),
        ]
        issues = validate_chapter_numbering(drafts, outline)
        # ch2：与大纲不一致 + 与 ch1 重复 + 断号（预期二实际一）。
        assert all(issue.chapter_id == "ch2" for issue in issues)
        assert any("不一致" in issue.message for issue in issues)
        assert any(
            "重复" in issue.message and "ch1" in issue.message for issue in issues
        )
        assert any("预期「二」" in issue.message for issue in issues)

    def test_skipped_numbering(self):
        outline = [_spec("ch1", "一、标题一"), _spec("ch2", "二、标题二")]
        drafts = [
            _draft("ch1", "## 一、标题一\n内容"),
            _draft("ch2", "## 三、标题三\n内容"),
        ]
        issues = validate_chapter_numbering(drafts, outline)
        assert any(
            "预期「二」" in issue.message and "实际「三」" in issue.message
            for issue in issues
        )
        assert all(issue.chapter_id == "ch2" for issue in issues)

    def test_outline_numbered_body_missing(self):
        """大纲标题带编号而正文缺编号：模型自生成根因场景。"""
        outline = [_spec("ch1", "一、标题一"), _spec("ch2", "二、标题二")]
        drafts = [
            _draft("ch1", "## 一、标题一\n内容"),
            _draft("ch2", "## 标题二\n内容"),
        ]
        issues = validate_chapter_numbering(drafts, outline)
        assert len(issues) == 1
        assert issues[0].chapter_id == "ch2"
        assert "缺少编号" in issues[0].message
        assert issues[0].actual_numbering is None

    def test_outline_numbered_body_inconsistent(self):
        """正文编号与大纲不一致：即便正文序列自身连续也要报。"""
        outline = [_spec("ch1", "一、标题一"), _spec("ch2", "三、标题三")]
        drafts = [
            _draft("ch1", "## 一、标题一\n内容"),
            _draft("ch2", "## 二、标题三\n内容"),
        ]
        issues = validate_chapter_numbering(drafts, outline)
        assert any(
            "不一致" in issue.message and issue.chapter_id == "ch2"
            for issue in issues
        )

    def test_unnumbered_chapters_not_validated(self):
        """大纲与正文均无编号的章不参与校验：自由结构模式不误报。"""
        outline = [_spec("ch1", "背景介绍"), _spec("ch2", "主要参与人列表")]
        drafts = [
            _draft("ch1", "## 背景介绍\n内容"),
            _draft("ch2", "正文没有二级标题"),
        ]
        assert validate_chapter_numbering(drafts, outline) == []

    def test_mixed_numbered_and_unnumbered(self):
        """无编号章夹在带编号章之间：带编号序列仍须连续。"""
        outline = [
            _spec("ch1", "一、标题一"),
            _spec("ch2", "主要参与人列表"),
            _spec("ch3", "二、标题二"),
        ]
        drafts = [
            _draft("ch1", "## 一、标题一\n内容"),
            _draft("ch2", "## 主要参与人列表\n内容"),
            _draft("ch3", "## 二、标题二\n内容"),
        ]
        assert validate_chapter_numbering(drafts, outline) == []

    def test_unrecognized_numbering(self):
        outline = [_spec("ch1", "一、标题一"), _spec("ch2", "二十一、标题")]
        drafts = [
            _draft("ch1", "## 一、标题一\n内容"),
            _draft("ch2", "## 二十一、标题\n内容"),
        ]
        issues = validate_chapter_numbering(drafts, outline)
        assert any(
            "无法识别" in issue.message and issue.actual_numbering == "二十一"
            for issue in issues
        )

    def test_draft_without_outline_entry_still_checked_in_sequence(self):
        """大纲缺失该章（或标题无编号）但正文自带编号：仍参与序列连续性校验。"""
        drafts = [
            _draft("ch1", "## 一、标题一\n内容"),
            _draft("ch2", "## 三、标题三\n内容"),
        ]
        issues = validate_chapter_numbering(drafts, [])
        assert len(issues) == 1
        assert issues[0].chapter_id == "ch2"
        assert "预期「二」" in issues[0].message

    def test_empty_drafts_returns_no_issues(self):
        assert validate_chapter_numbering([], []) == []

    def test_complex_scenario_two_duplicates_one_skip(self):
        """复现 issue #18 描述的「两个一、两个三」场景。"""
        outline = [
            _spec("ch1", "一、专业名称及代码"),
            _spec("ch2", "二、入学要求"),
            _spec("ch3", "三、学制学位"),
            _spec("ch4", "四、适用年级"),
        ]
        drafts = [
            _draft("ch1", "## 一、专业名称及代码\n内容"),
            _draft("ch2", "## 一、入学要求\n内容"),  # 重复的一
            _draft("ch3", "## 三、学制学位\n内容"),
            _draft("ch4", "## 三、适用年级\n内容"),  # 重复的三
        ]
        issues = validate_chapter_numbering(drafts, outline)
        chapter_ids_with_issues = {issue.chapter_id for issue in issues}
        assert chapter_ids_with_issues == {"ch2", "ch4"}
        duplicate_issues = [issue for issue in issues if "重复" in issue.message]
        assert len(duplicate_issues) == 2
