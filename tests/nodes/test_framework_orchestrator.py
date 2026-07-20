"""framework_orchestrator 节点单元测试：用假 LLM 预置 JSON 应答序列直接调用节点函数。

不跑全图，只验证节点的外部行为：品类识别与模板匹配、大纲生成（模板骨架实例化
与自由结构两条路径）、全文论点单次调用、逐论点假说、ID 规则、占位标记、
上限截断与筛选、配额预分配。

多章场景假说按章并发生成，调用顺序不确定，
假说应答一律用键控方式（按「待发散的论点：xxx」提示词片段）绑定到论点。
"""

import json
from pathlib import Path
from typing import Any

import pytest

from domain.framework_config import FrameworkLimits
from nodes.framework_orchestrator import (
    _allocate_hypothesis_caps,
    make_framework_orchestrator_node,
)
from llm.llm_client import FakeLLM
from domain.state import WorkflowStatus, initial_state

LIMITS = FrameworkLimits(
    max_points_per_chapter=4, max_hypotheses_per_point=3, max_hypotheses_total=60
)


def _hyp(text: str = "假说", angle: str = "预言", retrievable: bool = True) -> dict[str, Any]:
    """构造一条合法的假说应答项。"""
    return {
        "text": text,
        "refute_condition": f"若检索到与「{text}」相反的公开证据则被证伪",
        "angle": angle,
        "evidence_retrievable": retrievable,
    }


def _points_all(*points_per_chapter: list[Any]) -> list[dict[str, Any]]:
    """构造全文论点单次调用的应答：按位置生成 chapter_index（项允许混入非法形态）。"""
    return [
        {"chapter_index": index, "points": points}
        for index, points in enumerate(points_per_chapter, start=1)
    ]


def _hyp_key(point_text: str) -> str:
    """假说应答的键：绑定到假说提示词中的论点片段。"""
    return f"待发散的论点：{point_text}"


@pytest.fixture()
def templates_dir(tmp_path: Path) -> Path:
    """构造一个只含单个小模板的临时模板目录。"""
    (tmp_path / "index.md").write_text(
        "模板\t典型调用场景\n测试模版.md\t人才培养方案撰写\n", encoding="utf-8"
    )
    (tmp_path / "测试模版.md").write_text(
        "# {专业名称}人才培养方案\n\n"
        "## 一、培养目标\n### （一）目标定位\n\n"
        "## 二、课程体系\n### （一）课程结构\n### （二）实践环节\n\n"
        "## 三、附则\n",
        encoding="utf-8",
    )
    return tmp_path


def _encode(item: Any) -> str:
    """dict/list 应答自动转 JSON 文本，str 原样保留。"""
    return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)


def _run_node(
    responses: list[Any],
    templates_dir: Path,
    limits: FrameworkLimits = LIMITS,
    keyed: dict[str, list[Any]] | None = None,
) -> tuple[dict[str, Any], FakeLLM]:
    """预置顺序应答与键控应答后执行一次节点。"""
    fake = FakeLLM(
        [_encode(item) for item in responses],
        keyed_responses={
            key: [_encode(item) for item in values]
            for key, values in (keyed or {}).items()
        },
    )
    node = make_framework_orchestrator_node(
        lambda unit: fake, templates_dir=templates_dir, limits=limits
    )
    result = node(
        initial_state("写一篇软件工程专业人才培养方案", "专业撰稿人", "trace-fw")
    )
    return dict(result), fake


def test_模板路径_产出合规State且章节裁剪生效(templates_dir: Path) -> None:
    result, fake = _run_node(
        [
            {"genre": "人才培养方案", "template_file": "测试模版.md"},
            [
                {
                    "index": 1,
                    "applicable": True,
                    "title": "软件工程专业培养目标",
                    "subsections": ["目标定位"],
                },
                {
                    "index": 2,
                    "applicable": True,
                    "title": "软件工程课程体系",
                    "subsections": ["课程结构", "实践环节"],
                },
                {"index": 3, "applicable": False, "title": "附则", "subsections": []},
            ],
            _points_all([{"text": "论点甲"}], [{"text": "论点乙"}]),
        ],
        templates_dir,
        keyed={
            _hyp_key("论点甲"): [[_hyp("假说甲", angle="假设")]],
            _hyp_key("论点乙"): [[_hyp("假说乙", angle="失效模式")]],
        },
    )

    assert result["genre"] == "人才培养方案"
    assert result["template_id"] == "测试模版.md"
    assert result["status"] == WorkflowStatus.FRAMEWORK_BUILDING
    assert result["current_node_llm_config"]["unit"] == "framework_orchestrator"

    outline = result["outline"]
    # applicable=false 的第三章被裁剪，其余章节沿用模板骨架顺序。
    assert [chapter.title for chapter in outline] == [
        "软件工程专业培养目标",
        "软件工程课程体系",
    ]
    assert outline[1].subsections == ["课程结构", "实践环节"]
    # ID 规则由程序生成：章节 ch1…、论点 ch1-p1…、假说 ch1-p1-h1…。
    assert [chapter.id for chapter in outline] == ["ch1", "ch2"]
    assert outline[0].points[0].id == "ch1-p1"
    assert outline[0].points[0].text == "论点甲"
    # 并发下假说仍与所属论点正确绑定。
    assert outline[0].points[0].hypotheses[0].id == "ch1-p1-h1"
    assert outline[0].points[0].hypotheses[0].text == "假说甲"
    assert outline[0].points[0].hypotheses[0].angle == "假设"
    assert outline[1].points[0].hypotheses[0].id == "ch2-p1-h1"
    assert outline[1].points[0].hypotheses[0].text == "假说乙"
    # 识别 + 大纲 + 全文论点各一次，每论点各一次假说调用。
    assert len(fake.calls) == 5


def test_论点应答为字符串数组时同样解析(templates_dir: Path) -> None:
    """真实模型常把 points 直接给成字符串数组而非 {"text": ...} 对象，两种形态都要认。"""
    result, _ = _run_node(
        [
            {"genre": "行业评论", "template_file": None},
            [{"title": "引言", "subsections": ["背景"]}],
            _points_all(["字符串论点", {"text": "对象论点"}, "  ", 42]),
            [_hyp()],
        ],
        templates_dir,
        keyed={
            _hyp_key("字符串论点"): [[_hyp("假说甲")]],
            _hyp_key("对象论点"): [[_hyp("假说乙")]],
        },
    )
    points = result["outline"][0].points
    # 空白串与非法类型项被丢弃，字符串与对象两种形态都被解析。
    assert [point.text for point in points] == ["字符串论点", "对象论点"]


def test_自由结构路径_识别应答无模板时正常产出(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "行业评论", "template_file": None},
            [{"title": "引言", "subsections": ["背景"]}],
            _points_all([{"text": "论点"}]),
            [_hyp()],
        ],
        templates_dir,
    )

    assert result["template_id"] is None
    assert result["genre"] == "行业评论"
    assert result["status"] == WorkflowStatus.FRAMEWORK_BUILDING
    outline = result["outline"]
    assert [chapter.id for chapter in outline] == ["ch1"]
    assert outline[0].subsections == ["背景"]
    assert outline[0].points[0].hypotheses[0].id == "ch1-p1-h1"


def test_识别应答给出不存在的模板文件_回落自由结构(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "未知", "template_file": "不存在的模版.md"},
            [{"title": "自由章节", "subsections": []}],
            _points_all([{"text": "论点"}]),
            [_hyp()],
        ],
        templates_dir,
    )

    assert result["template_id"] is None
    assert result["outline"][0].title == "自由章节"


def test_残留填充变量被程序替换为待补充占位(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            # 围栏包裹的 JSON 也必须能解析。
            '```json\n{"genre": "方案", "template_file": null}\n```',
            [{"title": "{专业名称}的课程改革", "subsections": ["{专业名称}师资建设"]}],
            _points_all([{"text": "论点"}]),
            [_hyp()],
        ],
        templates_dir,
    )

    chapter = result["outline"][0]
    assert chapter.title == "【待补充：专业名称】的课程改革"
    assert chapter.subsections == ["【待补充：专业名称】师资建设"]


def test_模板路径_变量缺失时标题含待补充占位且流程不阻塞(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "方案", "template_file": "测试模版.md"},
            # LLM 未代入 {专业名称}，标题残留原样变量 → 程序兜底替换为占位标记。
            [
                {
                    "index": 1,
                    "applicable": True,
                    "title": "{专业名称}培养目标",
                    "subsections": ["{专业名称}目标定位"],
                },
                {"index": 2, "applicable": False, "title": "", "subsections": []},
                {"index": 3, "applicable": False, "title": "", "subsections": []},
            ],
            _points_all([{"text": "论点"}]),
            [_hyp()],
        ],
        templates_dir,
    )

    chapter = result["outline"][0]
    assert chapter.title == "【待补充：专业名称】培养目标"
    assert chapter.subsections == ["【待补充：专业名称】目标定位"]
    # 占位不阻塞：论点与假说照常生成。
    assert chapter.points[0].hypotheses[0].id == "ch1-p1-h1"


def test_模板路径_LLM缺失章节项时回落骨架标题并保留占位(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "方案", "template_file": "测试模版.md"},
            # 只返回第 1 章，第 2、3 章缺失 → 回落骨架标题与子标题，视为适用。
            [{"index": 1, "applicable": True, "title": "培养目标", "subsections": []}],
            _points_all(
                [{"text": "论点1"}], [{"text": "论点2"}], [{"text": "论点3"}]
            ),
        ],
        templates_dir,
        keyed={
            _hyp_key("论点1"): [[_hyp("假说1")]],
            _hyp_key("论点2"): [[_hyp("假说2")]],
            _hyp_key("论点3"): [[_hyp("假说3")]],
        },
    )

    outline = result["outline"]
    assert [chapter.title for chapter in outline] == ["培养目标", "课程体系", "附则"]
    assert outline[1].subsections == ["课程结构", "实践环节"]
    # 三章并发生成的假说各归其位。
    assert [chapter.points[0].hypotheses[0].text for chapter in outline] == [
        "假说1",
        "假说2",
        "假说3",
    ]


def test_论点应答缺失章节项时该章论点为空且不发起假说调用(templates_dir: Path) -> None:
    result, fake = _run_node(
        [
            {"genre": "方案", "template_file": None},
            [
                {"title": "第一章", "subsections": []},
                {"title": "第二章", "subsections": []},
            ],
            # 只返回第 2 章的论点，第 1 章缺失 → 空论点，无假说调用。
            [{"chapter_index": 2, "points": [{"text": "论点乙"}]}],
            [_hyp("假说乙")],
        ],
        templates_dir,
    )

    outline = result["outline"]
    assert outline[0].points == []
    assert outline[1].points[0].text == "论点乙"
    assert outline[1].points[0].hypotheses[0].id == "ch2-p1-h1"
    assert len(fake.calls) == 4


def test_上限截断与总数配额耗尽后跳过假说调用(templates_dir: Path) -> None:
    small = FrameworkLimits(
        max_points_per_chapter=2, max_hypotheses_per_point=2, max_hypotheses_total=2
    )
    result, fake = _run_node(
        [
            {"genre": "方案", "template_file": None},
            [{"title": "唯一章", "subsections": []}],
            # 3 条论点 → 截断到 2 条。
            _points_all([{"text": "论点1"}, {"text": "论点2"}, {"text": "论点3"}]),
            # 3 条假说 → 截断到 2 条，同时耗尽全文总数配额。
            [
                _hyp("假说1", angle="假设"),
                _hyp("假说2", angle="边界条件"),
                _hyp("假说3", angle="预言"),
            ],
        ],
        templates_dir,
        limits=small,
    )

    points = result["outline"][0].points
    assert [point.id for point in points] == ["ch1-p1", "ch1-p2"]
    assert [hyp.id for hyp in points[0].hypotheses] == ["ch1-p1-h1", "ch1-p1-h2"]
    # 配额预分配耗尽：第二个论点不发起假说调用，全程共 4 次 LLM 调用。
    assert points[1].hypotheses == []
    assert len(fake.calls) == 4


def test_配额预分配_按论点顺序扣减且不回补() -> None:
    limits = FrameworkLimits(
        max_points_per_chapter=4, max_hypotheses_per_point=3, max_hypotheses_total=4
    )
    # 两章各 2 论点，总配额 4：p1 得 3、p2 得 1，第二章两论点均为 0。
    assert _allocate_hypothesis_caps([2, 2], limits) == [[3, 1], [0, 0]]


def test_证据不可检索与非法角度的假说被过滤(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "方案", "template_file": None},
            [{"title": "章", "subsections": []}],
            _points_all([{"text": "论点"}]),
            [
                _hyp("不可检索", retrievable=False),
                {
                    "text": "非法角度",
                    "refute_condition": "条件",
                    "angle": "灵感",
                    "evidence_retrievable": True,
                },
                _hyp("合规假说", angle="竞争解释"),
            ],
        ],
        templates_dir,
    )

    hypotheses = result["outline"][0].points[0].hypotheses
    assert [hyp.text for hyp in hypotheses] == ["合规假说"]
    assert hypotheses[0].id == "ch1-p1-h1"


def test_应答不是合法JSON时抛ValueError并指明步骤(templates_dir: Path) -> None:
    with pytest.raises(ValueError, match="品类识别"):
        _run_node(["这不是 JSON"], templates_dir)


def test_缺省模板目录指向仓库根docs_templates且含模板文件() -> None:
    """回归覆盖：缺省路径曾少一级 parent 指向 src/docs_templates（提交 eacb335）。

    现有测试都显式传 templates_dir，缺省解析错误不会暴露，故直接断言
    缺省目录存在、位于仓库根、且含索引文件与至少一个模板文件。
    """
    from nodes.framework_orchestrator import (
        _default_templates_dir,
        _list_template_files,
    )

    default_dir = _default_templates_dir()
    repo_root = Path(__file__).resolve().parent.parent.parent
    assert default_dir == repo_root / "docs_templates"
    assert default_dir.is_dir()
    assert (default_dir / "index.md").is_file()
    assert _list_template_files(default_dir), "缺省模板目录应含至少一个模板文件"


def test_识别应答文件名差扩展名或括号空白时归一化命中(templates_dir: Path) -> None:
    """正式健壮化（issue #14）：不再要求 LLM 逐字符答对模板文件名。"""
    (templates_dir / "培养方案（专科）模版.md").write_text(
        "# 标题\n\n## 一、章\n", encoding="utf-8"
    )
    for answered in (
        "培养方案（专科）模版",          # 差 .md 扩展名
        "培养方案(专科)模版.md",         # 全角括号答成半角
        " 培养方案（专科） 模版.md ",    # 混入空白
    ):
        result, _ = _run_node(
            [
                {"genre": "培养方案", "template_file": answered},
                [{"index": 1, "applicable": True, "title": "章", "subsections": []}],
                _points_all([]),
            ],
            templates_dir,
        )
        assert result["template_id"] == "培养方案（专科）模版.md", answered


def test_识别应答文件名归一化后仍无匹配时回落自由结构(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "方案", "template_file": "完全不存在的模版.md"},
            [{"title": "章", "subsections": []}],
            _points_all([]),
        ],
        templates_dir,
    )
    assert result["template_id"] is None


@pytest.fixture()
def survey_templates_dir(tmp_path: Path) -> Path:
    """构造含可重复维度章位的调研报告模板目录（用注册表已登记的文件名）。"""
    (tmp_path / "index.md").write_text(
        "模板\t典型调用场景\n调研报告模版.md\t年度质量报告、专项调研\n",
        encoding="utf-8",
    )
    (tmp_path / "调研报告模版.md").write_text(
        "# {学校名称}调研报告\n\n"
        "## 一、监测概述与数据说明\n### （一）监测背景\n\n"
        "## 二、维度章 <!-- repeat: 1..N -->\n\n"
        "## 三、结论与对策建议\n### （一）核心结论\n",
        encoding="utf-8",
    )
    return tmp_path


def _survey_identify() -> dict[str, Any]:
    return {"genre": "调研报告", "template_file": "调研报告模版.md"}


def test_可重复章展开_维度章按应答展开且首尾固定章对齐(
    survey_templates_dir: Path,
) -> None:
    """可重复章实例化（issue #24）：重复位共用骨架序号展开为 N 章，固定章按序号对齐。"""
    result, _ = _run_node(
        [
            _survey_identify(),
            [
                {
                    "index": 1,
                    "applicable": True,
                    "title": "监测概述与数据说明",
                    "subsections": ["监测背景"],
                },
                {
                    "index": 2,
                    "applicable": True,
                    "title": "空间下沉：精准破局基层人才荒",
                    "subsections": ["县域就业分布"],
                },
                {
                    "index": 2,
                    "applicable": True,
                    "title": "行业赋能：技术岗位契合度攀升",
                    "subsections": ["行业流向"],
                },
                {
                    "index": 2,
                    "applicable": True,
                    "title": "留才效应：本地就业留存走高",
                    "subsections": ["留存率对比"],
                },
                {
                    "index": 3,
                    "applicable": True,
                    "title": "结论与对策建议",
                    "subsections": ["核心结论"],
                },
            ],
            _points_all([], [], [], [], []),
        ],
        survey_templates_dir,
    )

    outline = result["outline"]
    assert [chapter.title for chapter in outline] == [
        "监测概述与数据说明",
        "空间下沉：精准破局基层人才荒",
        "行业赋能：技术岗位契合度攀升",
        "留才效应：本地就业留存走高",
        "结论与对策建议",
    ]
    # 章 ID 由程序按展开后的最终顺序连续生成。
    assert [chapter.id for chapter in outline] == ["ch1", "ch2", "ch3", "ch4", "ch5"]
    # 「章序号 → 章型」映射作为骨架事实随章写入 State（ADR-0005）。
    assert [chapter.chapter_type for chapter in outline] == [
        "监测概述与数据说明",
        "维度章",
        "维度章",
        "维度章",
        "结论与对策建议",
    ]
    assert outline[2].subsections == ["行业流向"]
    assert result["doc_type"] == "调研报告"
    assert result["doc_variant"] is None


def test_可重复章展开_应答缺失重复位时回落单章骨架标题(
    survey_templates_dir: Path,
) -> None:
    """repeat 下限为 1：应答未展开重复位（缺失或全部标记不适用）时保底一章。"""
    result, _ = _run_node(
        [
            _survey_identify(),
            [
                {
                    "index": 1,
                    "applicable": True,
                    "title": "监测概述与数据说明",
                    "subsections": [],
                },
                {"index": 2, "applicable": False, "title": "", "subsections": []},
                {
                    "index": 3,
                    "applicable": True,
                    "title": "结论与对策建议",
                    "subsections": [],
                },
            ],
            _points_all([], [], []),
        ],
        survey_templates_dir,
    )

    outline = result["outline"]
    assert [chapter.title for chapter in outline] == [
        "监测概述与数据说明",
        "维度章",
        "结论与对策建议",
    ]
    assert outline[1].chapter_type == "维度章"


def test_章型_非重复模板各章携带骨架标题为章型(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "人才培养方案", "template_file": "测试模版.md"},
            [
                {
                    "index": 1,
                    "applicable": True,
                    "title": "软件工程专业培养目标",
                    "subsections": [],
                },
                {"index": 2, "applicable": True, "title": "课程体系", "subsections": []},
                {"index": 3, "applicable": False, "title": "", "subsections": []},
            ],
            _points_all([], []),
        ],
        templates_dir,
    )
    # 章型取骨架章标题原文，不随 LLM 实例化后的标题变化。
    assert [chapter.chapter_type for chapter in result["outline"]] == [
        "培养目标",
        "课程体系",
    ]


def test_章型_自由结构模式为None(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "行业评论", "template_file": None},
            [{"title": "引言", "subsections": []}],
            _points_all([]),
        ],
        templates_dir,
    )
    assert [chapter.chapter_type for chapter in result["outline"]] == [None]


def test_文种锚定_命中注册表模板_State写入文种与变体(tmp_path: Path) -> None:
    """注册表映射进 State（ADR-0005）：模板命中经注册表锚定文种与变体。"""
    (tmp_path / "index.md").write_text(
        "模板\t典型调用场景\n高职专科人才培养方案模版.md\t高职人才培养方案撰写\n",
        encoding="utf-8",
    )
    (tmp_path / "高职专科人才培养方案模版.md").write_text(
        "# {专业名称}人才培养方案\n\n## 一、培养目标\n", encoding="utf-8"
    )
    result, _ = _run_node(
        [
            {"genre": "人才培养方案", "template_file": "高职专科人才培养方案模版.md"},
            [{"index": 1, "applicable": True, "title": "培养目标", "subsections": []}],
            _points_all([]),
        ],
        tmp_path,
    )
    assert result["template_id"] == "高职专科人才培养方案模版.md"
    assert result["doc_type"] == "人才培养方案"
    assert result["doc_variant"] == "高职"


def test_文种锚定_自由结构无模板_落通用公文兑底(templates_dir: Path) -> None:
    result, _ = _run_node(
        [
            {"genre": "行业评论", "template_file": None},
            [{"title": "引言", "subsections": []}],
            _points_all([]),
        ],
        templates_dir,
    )
    assert result["template_id"] is None
    assert result["doc_type"] == "通用公文"
    assert result["doc_variant"] is None


def test_文种锚定_命中未登记模板_同样落通用公文兑底(templates_dir: Path) -> None:
    """临时/未登记模板不臆断文种：确定性落兑底，登记后才升为专属文种。"""
    result, _ = _run_node(
        [
            {"genre": "人才培养方案", "template_file": "测试模版.md"},
            [{"index": 1, "applicable": True, "title": "章", "subsections": []}],
            _points_all([]),
        ],
        templates_dir,
    )
    assert result["template_id"] == "测试模版.md"
    assert result["doc_type"] == "通用公文"
    assert result["doc_variant"] is None
