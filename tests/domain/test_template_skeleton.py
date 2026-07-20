"""模板骨架解析纯逻辑的单元测试与真实模板冒烟测试。"""

from pathlib import Path

import pytest

from domain.template_skeleton import SectionSkeleton, TemplateSkeleton, parse_template_skeleton

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "docs_templates"


def test_空文档():
    skeleton = parse_template_skeleton("")
    assert skeleton == TemplateSkeleton(doc_title="", chapters=(), variables=())


def test_中文序号二级与全角括号三级():
    text = "\n".join(
        [
            "# 方案标题",
            "## 一、专业名称及代码",
            "## 五、职业面向",
            "### （一）职业领域",
            "### （二）工作岗位",
        ]
    )
    skeleton = parse_template_skeleton(text)
    assert skeleton.doc_title == "方案标题"
    assert skeleton.chapters == (
        SectionSkeleton(numbering="一", title="专业名称及代码"),
        SectionSkeleton(
            numbering="五",
            title="职业面向",
            subsections=(
                SectionSkeleton(numbering="（一）", title="职业领域"),
                SectionSkeleton(numbering="（二）", title="工作岗位"),
            ),
        ),
    )


def test_复合中文数字序号():
    skeleton = parse_template_skeleton("## 十一、附录")
    assert skeleton.chapters == (SectionSkeleton(numbering="十一", title="附录"),)


def test_阿拉伯序号点号与顿号():
    text = "\n".join(
        [
            "## 课程设置",
            "### 1. 思政要求",
            "### 2、课程结构",
        ]
    )
    skeleton = parse_template_skeleton(text)
    assert skeleton.chapters[0].subsections == (
        SectionSkeleton(numbering="1", title="思政要求"),
        SectionSkeleton(numbering="2", title="课程结构"),
    )


def test_无序号标题原样保留():
    text = "\n".join(["## 培养目标", "### 培养规格说明"])
    skeleton = parse_template_skeleton(text)
    assert skeleton.chapters == (
        SectionSkeleton(
            numbering="",
            title="培养目标",
            subsections=(SectionSkeleton(numbering="", title="培养规格说明"),),
        ),
    )


def test_二级标题之前的三级标题被忽略():
    text = "\n".join(["### 孤儿三级标题", "## 一、正式章节"])
    skeleton = parse_template_skeleton(text)
    assert skeleton.chapters == (SectionSkeleton(numbering="一", title="正式章节"),)


def test_多个H1只取首个且后续章节照常收集():
    text = "\n".join(
        [
            "# 主标题",
            "## 一、正文章节",
            "# 附录标题",
            "## 二、附录章节",
            "### （一）附录小节",
        ]
    )
    skeleton = parse_template_skeleton(text)
    assert skeleton.doc_title == "主标题"
    assert [c.title for c in skeleton.chapters] == ["正文章节", "附录章节"]
    assert skeleton.chapters[1].subsections == (
        SectionSkeleton(numbering="（一）", title="附录小节"),
    )


def test_围栏代码块内的井号行不算标题():
    text = "\n".join(
        [
            "## 一、真章节",
            "```",
            "## 二、假章节",
            "### （一）假小节",
            "```",
            "## 三、又一真章节",
        ]
    )
    skeleton = parse_template_skeleton(text)
    assert [c.title for c in skeleton.chapters] == ["真章节", "又一真章节"]


def test_变量去重且保持首次出现顺序():
    text = "\n".join(
        [
            "# {专业名称}专业方案（{XXXX}级）",
            "## 一、概况",
            "正文含 {专业名称} 与 {6位数字代码}，还有 {XXXX}。",
            "### （一）小节含 {3/4} 变量",
        ]
    )
    skeleton = parse_template_skeleton(text)
    assert skeleton.variables == ("专业名称", "XXXX", "6位数字代码", "3/4")


def test_标题中的变量与加粗原样保留():
    text = "## 一、**{专业名称}**简介"
    skeleton = parse_template_skeleton(text)
    assert skeleton.chapters[0].title == "**{专业名称}**简介"
    assert skeleton.variables == ("专业名称",)


def test_变量不跨行不含嵌套花括号():
    text = "正文 {跨行\n变量} 与 {外{内}层} 混排"
    skeleton = parse_template_skeleton(text)
    assert skeleton.variables == ("内",)


def test_章级repeat标记解析为可重复章且标题不含标记():
    text = "\n".join(
        [
            "## 一、监测概述",
            "## 二、维度章 <!-- repeat: 1..N -->",
            "## 三、结论与建议",
        ]
    )
    skeleton = parse_template_skeleton(text)
    assert [c.repeatable for c in skeleton.chapters] == [False, True, False]
    assert skeleton.chapters[1] == SectionSkeleton(
        numbering="二", title="维度章", repeatable=True
    )


def test_repeat标记标在小节级被拒绝():
    text = "\n".join(["## 一、章", "### （一）小节 <!-- repeat: 1..N -->"])
    with pytest.raises(ValueError, match="章级"):
        parse_template_skeleton(text)


def test_repeat标记嵌套于可重复章的小节被拒绝():
    text = "\n".join(
        [
            "## 一、维度章 <!-- repeat: 1..N -->",
            "### （一）子维度 <!-- repeat: 1..N -->",
        ]
    )
    with pytest.raises(ValueError, match="章级"):
        parse_template_skeleton(text)


def test_repeat标记标在一级标题被拒绝():
    with pytest.raises(ValueError, match="章级"):
        parse_template_skeleton("# 报告标题 <!-- repeat: 1..N -->")


def test_一份模板出现多个可重复章位被拒绝():
    text = "\n".join(
        [
            "## 一、维度章 <!-- repeat: 1..N -->",
            "## 二、专题章 <!-- repeat: 1..N -->",
        ]
    )
    with pytest.raises(ValueError, match="至多一个"):
        parse_template_skeleton(text)


def test_repeat标记标在首个H2之前的孤儿小节同样被拒绝():
    text = "\n".join(["### 孤儿小节 <!-- repeat: 1..N -->", "## 一、正式章节"])
    with pytest.raises(ValueError, match="章级"):
        parse_template_skeleton(text)


@pytest.mark.parametrize(
    "marker",
    ["<!-- repeat: 1..3 -->", "<!-- repeat -->", "<!-- repeat: N -->"],
)
def test_非规范repeat写法显式报错不静默当标题文本(marker: str):
    with pytest.raises(ValueError, match="仅支持 repeat: 1..N"):
        parse_template_skeleton(f"## 一、维度章 {marker}")


def test_围栏代码块内的repeat标记不生效():
    text = "\n".join(
        [
            "## 一、真章节",
            "```",
            "## 二、假章节 <!-- repeat: 1..N -->",
            "```",
        ]
    )
    skeleton = parse_template_skeleton(text)
    assert [c.repeatable for c in skeleton.chapters] == [False]


@pytest.mark.parametrize(
    "filename",
    [
        "人才培养方案总结（汇报）模版.md",
        "学院级多专业培养方案模版.md",
        "本科职业教育人才培养方案模版.md",
        "调研报告模版.md",
        "高职专科人才培养方案模版.md",
    ],
)
def test_真实模板解析不抛错(filename: str):
    text = (TEMPLATES_DIR / filename).read_text(encoding="utf-8")
    skeleton = parse_template_skeleton(text)
    assert isinstance(skeleton, TemplateSkeleton)


def test_调研报告模板已知结构断言():
    """骨架事实：首章固定四小节、单一可重复维度章位、首尾固定章齐全（issue #24）。"""
    text = (TEMPLATES_DIR / "调研报告模版.md").read_text(encoding="utf-8")
    skeleton = parse_template_skeleton(text)
    首章 = skeleton.chapters[0]
    assert 首章.title == "监测概述与数据说明"
    assert [sub.title for sub in 首章.subsections] == [
        "监测背景",
        "数据来源与可靠性",
        "核心指标界定",
        "核心评估维度",
    ]
    repeatable = [c for c in skeleton.chapters if c.repeatable]
    assert [c.title for c in repeatable] == ["维度章"]
    assert [c.title for c in skeleton.chapters[-2:]] == [
        "主要发现与问题诊断",
        "结论与对策建议",
    ]


def test_高职专科模板已知结构断言():
    text = (TEMPLATES_DIR / "高职专科人才培养方案模版.md").read_text(encoding="utf-8")
    skeleton = parse_template_skeleton(text)
    assert skeleton.doc_title == "{专业名称}专业人才培养方案（{XXXX}级）"
    numberings = [c.numbering for c in skeleton.chapters]
    assert "一" in numberings
    assert "十一" in numberings
    职业面向 = next(c for c in skeleton.chapters if c.title == "职业面向")
    assert 职业面向.numbering == "五"
    assert 职业面向.subsections[0] == SectionSkeleton(numbering="（一）", title="职业领域")
    assert "专业名称" in skeleton.variables
    assert len(skeleton.variables) == len(set(skeleton.variables))
